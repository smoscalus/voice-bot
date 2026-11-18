#!/usr/bin/env bash
# Запуск бота одной командой через Docker (Linux / macOS): ./start.sh
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '%s\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*"; exit 1; }
get_env() { grep -E "^$1=" .env | head -1 | cut -d= -f2- || true; }
set_env() {
  if grep -qE "^#?\s*$1=" .env; then
    python3 - "$1" "$2" <<'PY' 2>/dev/null || sed -i.bak -E "s|^#?\s*$1=.*|$1=$2|" .env
import re, sys
key, val = sys.argv[1], sys.argv[2]
s = open(".env", encoding="utf-8").read()
s = re.sub(rf"^#?\s*{re.escape(key)}=.*$", lambda m: f"{key}={val}", s, count=1, flags=re.M)
open(".env", "w", encoding="utf-8").write(s)
PY
  else
    printf '%s=%s\n' "$1" "$2" >> .env
  fi
}

say "=== Голосовой бот: запуск ==="
command -v docker >/dev/null || fail "Не найден Docker: https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1 || fail "Docker не запущен (или нужен sudo / группа docker)."

vram_gb() {  # видеопамять самой большой видеокарты, ГБ (NVIDIA или AMD); 0 — не определил
  local best=0 v
  if command -v nvidia-smi >/dev/null 2>&1; then
    v=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -n "$v" ] && best=$(( v / 1024 ))
  fi
  for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v=$(( $(cat "$f") / 1073741824 )); [ "$v" -gt "$best" ] && best=$v
  done
  echo "$best"
}

ask() {  # ask "вопрос" "текущее значение" → ответ (Enter — оставить текущее)
  local hint="" answer
  [ -n "$2" ] && hint=" [Enter — оставить]"
  read -rp "$1$hint: " answer
  echo "${answer:-$2}"
}

run_setup() {  # $1 = 1 при первом запуске (значения из шаблона не предлагаем «оставить»)
  local first="$1" cur token choice provider key vram rec pick model cur_model
  say ""; say "--- Настройка ---"
  token=$(ask "Токен Discord-бота (discord.com/developers/applications → Bot → Reset Token)" "$(get_env DISCORD_TOKEN)")
  [ -n "$token" ] || fail "Без токена бот не запустится."
  set_env DISCORD_TOKEN "$token"

  cur=""; [ "$first" = 1 ] || cur=$(get_env PROVIDER)
  say ""; say "Где будут работать мозги бота?"
  say "  1 — на этом компьютере (Ollama): бесплатно, без лимитов, но глупее и медленнее"
  say "  2 — в облаке Groq: умнее и быстрее, бесплатно, но с дневными лимитами (нужен ключ)"
  choice=$(ask "Выбери 1 или 2" "$([ "$cur" = groq ] && echo 2 || { [ "$cur" = local ] && echo 1; } || true)")
  provider=local; [ "$choice" = 2 ] && provider=groq
  set_env PROVIDER "$provider"

  say ""
  if [ "$provider" = groq ]; then
    say "Ключ Groq: console.groq.com → API Keys (бесплатно, начинается с gsk_)"
    key=$(ask "Ключ Groq" "$(get_env GROQ_API_KEY)")
    [ -n "$key" ] || fail "Для облачных мозгов нужен ключ Groq."
  else
    say "Ключ Groq — НЕОБЯЗАТЕЛЬНО. С ним распознавание речи пойдёт через облако (быстро и точно),"
    say "а мозги останутся на компе без лимитов. Без него всё будет на компе (медленнее)."
    key=$(ask "Ключ Groq (Enter — пропустить)" "$(get_env GROQ_API_KEY)")
  fi
  set_env GROQ_API_KEY "$key"

  if [ "$provider" = local ]; then
    vram=$(vram_gb)
    if [ "$vram" -ge 12 ]; then rec=3; elif [ "$vram" -ge 6 ]; then rec=2; else rec=1; fi
    say ""; say "Какую модель взять? (видеопамять: $([ "$vram" -gt 0 ] && echo "$vram ГБ" || echo 'не определил'))"
    say "  1 — qwen2.5:3b   слабый компьютер или без видеокарты"
    say "  2 — qwen2.5:7b   видеокарта 6–8 ГБ"
    say "  3 — qwen2.5:14b  видеокарта 12 ГБ и больше (заметно умнее)"
    say "  4 — своя (название из ollama.com/library)"
    cur_model=""; [ "$first" = 1 ] || cur_model=$(get_env OLLAMA_MODEL)
    case "$cur_model" in qwen2.5:3b) cur=1;; qwen2.5:7b) cur=2;; qwen2.5:14b) cur=3;; "") cur=$rec;; *) cur=4;; esac
    pick=$(ask "Выбери 1–4 (советую $rec)" "$cur")
    case "$pick" in
      1) model=qwen2.5:3b;; 2) model=qwen2.5:7b;; 3) model=qwen2.5:14b;;
      4) model=$(ask "Название модели (например, gemma3:12b)" "$cur_model");;
      *) model=$([ "$rec" = 3 ] && echo qwen2.5:14b || { [ "$rec" = 2 ] && echo qwen2.5:7b; } || echo qwen2.5:3b);;
    esac
    set_env OLLAMA_MODEL "$model"
    say "Модель: $model — скачается сама при первом запуске."
  fi
  say ""; say "Готово. Поменять потом: ./start.sh config"
}

if [ ! -f .env ]; then
  cp .env.example .env
  say "Первый запуск — пара вопросов."
  run_setup 1
elif [ "${1:-}" = "config" ]; then
  run_setup 0
fi

provider="$(get_env PROVIDER)"; provider="${provider:-local}"
args=(compose)
if [ "$provider" = "local" ]; then
  if curl -fs -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || curl -fs -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 || pgrep -x ollama >/dev/null 2>&1; then
    say "Нашёл Ollama на этом компьютере — использую её."
    [ -n "${OLLAMA_CONTEXT_LENGTH:-}" ] || say "Совет: запусти Ollama с OLLAMA_CONTEXT_LENGTH=8192 — иначе память бота не влезет в её окно."
    set_env DOCKER_OLLAMA_URL "http://host.docker.internal:11434"
  else
    set_env DOCKER_OLLAMA_URL "http://ollama:11434"
    args+=(-f docker-compose.yml)
    # nvidia-smi бывает и без рабочей видеокарты — проверяем, что он её реально видит
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      say "Видеокарта NVIDIA найдена — попробую считать на ней."
      gpu=(-f docker-compose.gpu.yml)
    else
      say "Видеокарты NVIDIA нет — Ollama на процессоре (медленнее). Совет: поставь Ollama прямо на систему (ollama.com)"
      gpu=()
    fi
    local_profile=(--profile local)
  fi
fi

say "Собираю и запускаю (первый раз — несколько минут)…"
if [ -n "${local_profile+x}" ]; then
  if ! docker "${args[@]}" ${gpu[@]+"${gpu[@]}"} "${local_profile[@]}" up -d --build; then
    [ ${#gpu[@]} -gt 0 ] || fail "Не удалось запустить. Посмотри сообщение выше."
    say "Docker не видит видеокарту — запускаю Ollama на процессоре (медленнее)."
    docker compose --profile local down >/dev/null 2>&1 || true
    docker "${args[@]}" "${local_profile[@]}" up -d --build || fail "Не удалось запустить. Посмотри сообщение выше."
  fi
else
  docker "${args[@]}" up -d --build || fail "Не удалось запустить. Посмотри сообщение выше."
fi
say "Бот запущен. Зайди в голосовой канал и напиши /join. Остановить: docker compose --profile local stop"
docker compose logs -f --tail 50 bot
