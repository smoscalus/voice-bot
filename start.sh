#!/usr/bin/env bash
# Запуск бота одной командой (Linux / macOS): ./start.sh        — запустить (при первом запуске — настройка)
#                                             ./start.sh config — заново ответить на вопросы
# Умеет два способа: напрямую (Python на этом компе, как удобнее на Linux) и через Docker.
set -euo pipefail
cd "$(dirname "$0")"

say()  { printf '%s\n' "$*"; }
ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*"; exit 1; }
get_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- || true; }
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
ask() {  # ask "вопрос" "значение по умолчанию" → ответ (Enter — по умолчанию)
  local hint="" answer
  [ -n "$2" ] && hint=" [$2]"
  read -rp "$1$hint: " answer
  echo "${answer:-$2}"
}
yes_no() {  # yes_no "вопрос" y|n → 0 — да
  local def="$2" a
  a=$(ask "$1 (y/n)" "$def")
  case "$a" in y|Y|д|Д|да|Да|yes) return 0;; *) return 1;; esac
}

# ---------------------------------------------------------------- железо
gpu_info() {  # → "vendor|имя|ГБ" (vendor: nvidia / amd / none)
  local name="" gb=0 v vendor=none
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    vendor=nvidia
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    v=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -n "$v" ] && gb=$(( (v + 512) / 1024 ))
  fi
  for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    [ -r "$f" ] || continue
    v=$(( ($(cat "$f") + 536870912) / 1073741824 ))
    if [ "$v" -gt "$gb" ]; then
      gb=$v; vendor=amd
      name=$(lspci 2>/dev/null | grep -iE 'vga|display|3d' | grep -iE 'amd|ati|radeon' | head -1 | sed -E 's/.*: //' || true)
    fi
  done
  if [ "$(uname)" = Darwin ]; then  # Mac: видеопамять общая с оперативной
    vendor=apple; name="Apple Silicon"; gb=$(( $(sysctl -n hw.memsize) / 1073741824 * 2 / 3 ))
  fi
  printf '%s|%s|%s' "$vendor" "${name:-видеокарта}" "$gb"
}

# Профили видеокарт — держать в согласии со start.ps1 (Windows):
#   профиль → окно Ollama, слотов, сжатие кэша, где считает Whisper
profile_for_gb() {
  local gb=$1
  if [ "$gb" -ge 16 ]; then echo 16gb; elif [ "$gb" -ge 10 ]; then echo 12gb; elif [ "$gb" -ge 6 ]; then echo 8gb; else echo cpu; fi
}
apply_profile() {
  case "$1" in
    16gb) set_env OLLAMA_CONTEXT_LENGTH 8192; set_env OLLAMA_NUM_PARALLEL 2; set_env OLLAMA_KV_CACHE_TYPE f16;  set_env LOCAL_WHISPER_DEVICE auto;;
    12gb) set_env OLLAMA_CONTEXT_LENGTH 8192; set_env OLLAMA_NUM_PARALLEL 1; set_env OLLAMA_KV_CACHE_TYPE q8_0; set_env LOCAL_WHISPER_DEVICE auto;;
    8gb)  set_env OLLAMA_CONTEXT_LENGTH 6144; set_env OLLAMA_NUM_PARALLEL 1; set_env OLLAMA_KV_CACHE_TYPE q8_0; set_env LOCAL_WHISPER_DEVICE cpu;;
    *)    set_env OLLAMA_CONTEXT_LENGTH 4096; set_env OLLAMA_NUM_PARALLEL 1; set_env OLLAMA_KV_CACHE_TYPE q8_0; set_env LOCAL_WHISPER_DEVICE cpu;;
  esac
  set_env OLLAMA_FLASH_ATTENTION 1
  set_env GPU_PROFILE "$1"
}

# ---------------------------------------------------------------- вопросы
run_setup() {  # $1 = 1 при первом запуске
  local first="$1" token choice provider key info vendor gname gb answer profile rec pick model cur_model name voice speed mode def
  say ""; say "--- Настройка ---"
  token=$(ask "Токен Discord-бота (discord.com/developers/applications → Bot → Reset Token)" "$(get_env DISCORD_TOKEN)")
  [ -n "$token" ] || fail "Без токена бот не запустится."
  set_env DISCORD_TOKEN "$token"

  say ""; say "Где будут работать мозги бота?"
  say "  1 — на этом компьютере (Ollama): бесплатно, без лимитов, но глупее и медленнее"
  say "  2 — в облаке Groq: умнее и быстрее, бесплатно, но с дневными лимитами (нужен ключ)"
  def=1
  if [ "$first" != 1 ] && [ "$(get_env PROVIDER)" = groq ]; then def=2; fi
  choice=$(ask "Выбери 1 или 2" "$def")
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
    info=$(gpu_info); vendor=${info%%|*}; gname=${info#*|}; gname=${gname%|*}; gb=${info##*|}
    say ""
    if [ "$vendor" = none ] || [ "$gb" -eq 0 ]; then
      say "Видеокарту не нашёл."
    else
      say "Видеокарта: $gname, $gb ГБ."
    fi
    answer=$(ask "Сколько ГБ видеопамяти? (0 — без видеокарты)" "$gb")
    [[ "$answer" =~ ^[0-9]+$ ]] && gb=$answer
    profile=$(profile_for_gb "$gb")
    apply_profile "$profile"
    ok "Профиль видеокарты: $profile (окно $(get_env OLLAMA_CONTEXT_LENGTH), слотов $(get_env OLLAMA_NUM_PARALLEL))"

    case "$profile" in cpu) rec=1;; *) rec=3;; esac
    say ""; say "Какую модель взять?"
    say "  1 — qwen2.5:3b   без видеокарты или совсем слабая (быстро, но туповато)"
    say "  2 — qwen2.5:7b   видеокарта 6 ГБ"
    say "  3 — qwen3.5:9b   видеокарта 8 ГБ и больше (основная)"
    say "  4 — qwen2.5:14b  видеокарта 12 ГБ и больше"
    say "  5 — своя (название из ollama.com/library)"
    cur_model=""; [ "$first" = 1 ] || cur_model=$(get_env OLLAMA_MODEL)
    case "$cur_model" in qwen2.5:3b) def=1;; qwen2.5:7b) def=2;; qwen3.5:9b) def=3;; qwen2.5:14b) def=4;; "") def=$rec;; *) def=5;; esac
    pick=$(ask "Выбери 1–5 (советую $rec)" "$def")
    case "$pick" in
      1) model=qwen2.5:3b;; 2) model=qwen2.5:7b;; 4) model=qwen2.5:14b;;
      5) model=$(ask "Название модели (например, gemma3:12b)" "$cur_model");;
      *) model=qwen3.5:9b;;
    esac
    set_env OLLAMA_MODEL "$model"
    ok "Модель: $model — скачается сама при первом запуске."
  fi

  # имя
  say ""
  say "Как зовут бота? Новое имя — новый «человек» с чистой памятью (старый останется в data/ и вернётся с именем)."
  name=$(get_env BOT_NAME)
  name=$(ask "Имя" "${name:-Петрык}")
  [ -n "$name" ] || name=Петрык
  set_env BOT_NAME "$name"
  set_env BOT_NAME_ALIASES "$(ask "Как ещё его зовут, через запятую (Enter — пропустить)" "$(get_env BOT_NAME_ALIASES)")"

  # голос
  say ""; say "Голос:"
  say "  1 — Дмитрий   2 — Денис   3 — Руслан   4 — Ирина (женский)    — на компе, мгновенно"
  say "  5 — Дмитрий онлайн   6 — Светлана онлайн (женский)              — живее, нужен интернет"
  case "$(get_env TTS_BACKEND)/$(get_env PIPER_VOICE)/$(get_env EDGE_VOICE)" in
    edge/*/ru-RU-SvetlanaNeural) def=6;; edge/*) def=5;; */ru_RU-denis-medium/*) def=2;;
    */ru_RU-ruslan-medium/*) def=3;; */ru_RU-irina-medium/*) def=4;; *) def=1;;
  esac
  voice=$(ask "Выбери 1–6" "$def")
  case "$voice" in
    2) set_env TTS_BACKEND piper; set_env PIPER_VOICE ru_RU-denis-medium;;
    3) set_env TTS_BACKEND piper; set_env PIPER_VOICE ru_RU-ruslan-medium;;
    4) set_env TTS_BACKEND piper; set_env PIPER_VOICE ru_RU-irina-medium;;
    5) set_env TTS_BACKEND edge;  set_env EDGE_VOICE ru-RU-DmitryNeural;;
    6) set_env TTS_BACKEND edge;  set_env EDGE_VOICE ru-RU-SvetlanaNeural;;
    *) set_env TTS_BACKEND piper; set_env PIPER_VOICE ru_RU-dmitri-medium;;
  esac
  def=1; [ "$(get_env PIPER_SPEED)" = 1.15 ] && def=2
  speed=$(ask "Темп речи: 1 — обычный, 2 — чуть быстрее" "$def")
  set_env PIPER_SPEED "$([ "$speed" = 2 ] && echo 1.15 || echo 1.0)"

  # перебивание голосом
  say ""
  def=y; [ "$(get_env BARGE_IN_MS)" = 0 ] && [ "$first" != 1 ] && def=n
  if yes_no "Можно перебивать его голосом? (говоришь поверх ~1 с — он замолкает; в шумной компании иногда будет обрываться)" "$def"; then
    set_env BARGE_IN_MS 900
  else
    set_env BARGE_IN_MS 0
  fi

  # как запускать
  say ""; say "Как запускать?"
  say "  1 — напрямую (Python на этом компе) — проще всего на Linux"
  say "  2 — через Docker"
  def=1; [ "$(get_env RUN_MODE)" = docker ] && def=2
  mode=$(ask "Выбери 1 или 2" "$def")
  if [ "$mode" = 2 ] && [ "$provider" = local ] && [ "${vendor:-none}" != nvidia ]; then
    # Ollama в контейнере видит только NVIDIA (AMD — нужен особый образ и настройка), иначе — процессор
    warn "У тебя не NVIDIA: если Ollama поднимется в Docker, модель будет считать на процессоре (медленно)."
    yes_no "Точно Docker? (напрямую Ollama возьмёт и Radeon)" n || mode=1
  fi
  set_env RUN_MODE "$([ "$mode" = 2 ] && echo docker || echo direct)"

  # для любопытных
  if [ "$provider" = local ] && yes_no "Расширенные настройки (размышления модели)?" n; then
    say "Размышления: модель сначала думает «про себя», потом отвечает. Умнее, но медленнее."
    set_env THINK_REPLY "$(yes_no "Думать в живом ответе? (медленно: +3–10 с перед каждой фразой)" n && echo on || echo off)"
    set_env THINK_BACKGROUND "$(yes_no "Думать в фоновых делах (досье, сны, книжка)?" y && echo on || echo off)"
  fi
  say ""; ok "Готово. Поменять потом: ./start.sh config"
}

# ---------------------------------------------------------------- Ollama на этом компе
ollama_up() { curl -fs -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; }

configure_host_ollama() {  # настройки самой Ollama — под профиль из .env
  local ctx par kv fa unit dir conf want cur
  ctx=$(get_env OLLAMA_CONTEXT_LENGTH); par=$(get_env OLLAMA_NUM_PARALLEL)
  kv=$(get_env OLLAMA_KV_CACHE_TYPE); fa=$(get_env OLLAMA_FLASH_ATTENTION)
  want=$(printf 'Environment="OLLAMA_CONTEXT_LENGTH=%s"\nEnvironment="OLLAMA_NUM_PARALLEL=%s"\nEnvironment="OLLAMA_KV_CACHE_TYPE=%s"\nEnvironment="OLLAMA_FLASH_ATTENTION=%s"' \
    "${ctx:-8192}" "${par:-1}" "${kv:-q8_0}" "${fa:-1}")
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service >/dev/null 2>&1 \
     && systemctl cat ollama >/dev/null 2>&1; then
    dir=/etc/systemd/system/ollama.service.d; conf=$dir/voice-bot.conf
    cur=$(cat "$conf" 2>/dev/null | grep '^Environment' || true)
    [ "$cur" = "$want" ] && return 0
    say ""; say "Ollama работает как служба. Чтобы бот не терял характер, ей нужны настройки под твою видеокарту:"
    say "$want" | sed 's/^/    /'
    if yes_no "Записать их в $conf и перезапустить Ollama? (спросит пароль sudo)" y; then
      printf '[Service]\n%s\n' "$want" | sudo tee "$conf" >/dev/null && sudo systemctl daemon-reload \
        && sudo systemctl restart ollama && ok "Ollama перезапущена с настройками бота." \
        || warn "Не получилось — пропиши это сам: sudo systemctl edit ollama"
      for _ in $(seq 20); do ollama_up && break; sleep 1; done
    else
      warn "Ок, без меня: sudo systemctl edit ollama → вставь строки выше под [Service] → sudo systemctl restart ollama"
    fi
  else
    # Ollama запущена руками (ollama serve) — службу не трогаем
    if [ "${OLLAMA_CONTEXT_LENGTH:-}" != "${ctx:-8192}" ]; then
      warn "Ollama запущена не как служба. Перезапусти её с настройками бота:"
      say "    OLLAMA_CONTEXT_LENGTH=${ctx:-8192} OLLAMA_NUM_PARALLEL=${par:-1} OLLAMA_KV_CACHE_TYPE=${kv:-q8_0} OLLAMA_FLASH_ATTENTION=${fa:-1} ollama serve"
    fi
  fi
}

ensure_host_ollama() {
  if ! command -v ollama >/dev/null 2>&1 && ! ollama_up; then
    say ""; warn "Ollama не установлена."
    yes_no "Поставить её сейчас (официальный установщик с ollama.com, спросит пароль sudo)?" y \
      || fail "Без Ollama мозги на компе не работают. Поставь с ollama.com или выбери облако: ./start.sh config"
    curl -fsSL https://ollama.com/install.sh | sh || fail "Установка Ollama не удалась — смотри сообщение выше."
  fi
  configure_host_ollama
  if ! ollama_up; then
    say "Запускаю Ollama…"
    (OLLAMA_CONTEXT_LENGTH=$(get_env OLLAMA_CONTEXT_LENGTH) OLLAMA_NUM_PARALLEL=$(get_env OLLAMA_NUM_PARALLEL) \
     OLLAMA_KV_CACHE_TYPE=$(get_env OLLAMA_KV_CACHE_TYPE) OLLAMA_FLASH_ATTENTION=$(get_env OLLAMA_FLASH_ATTENTION) \
     nohup ollama serve >logs/ollama.log 2>&1 &)
    for _ in $(seq 30); do ollama_up && break; sleep 1; done
    ollama_up || fail "Ollama не поднялась — смотри logs/ollama.log"
  fi
}

# ---------------------------------------------------------------- запуск напрямую
run_direct() {
  local py req_hash
  py=$(command -v python3.12 || command -v python3.11 || command -v python3 || true)
  [ -n "$py" ] || fail "Не найден Python 3.11+. Поставь: sudo apt install python3 python3-venv (или аналог для твоей системы)"
  "$py" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || fail "Нужен Python 3.11 или новее (сейчас $("$py" -V))."
  if [ "$(uname)" = Linux ] && ! { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null || true; } | grep -q libopus; then
    warn "Нет libopus (кодек голоса Discord). Поставь: sudo apt install libopus0  (Fedora: sudo dnf install opus)"
  fi
  if [ ! -x .venv/bin/python ]; then
    say "Создаю окружение .venv…"
    "$py" -m venv .venv || fail "Не создалось окружение. На Debian/Ubuntu: sudo apt install python3-venv"
  fi
  req_hash=$(sha1sum requirements.txt 2>/dev/null | cut -d' ' -f1 || shasum requirements.txt | cut -d' ' -f1)
  if [ "$(cat .venv/.req-hash 2>/dev/null)" != "$req_hash" ]; then
    say "Ставлю зависимости (первый раз — несколько минут)…"
    .venv/bin/python -m pip install -q --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt || fail "Не встали зависимости — смотри сообщение выше."
    echo "$req_hash" > .venv/.req-hash
  fi
  ok "Запускаю бота. Зайди в голосовой канал и напиши /join. Остановить: Ctrl+C"
  exec .venv/bin/python -m bot
}

# ---------------------------------------------------------------- запуск через Docker
run_docker() {
  local provider=$1 args=(compose) gpu=() local_profile=()
  command -v docker >/dev/null || fail "Не найден Docker: https://docs.docker.com/get-docker/ (или выбери запуск напрямую: ./start.sh config)"
  docker info >/dev/null 2>&1 || fail "Docker не запущен (или нужен sudo / группа docker)."
  if [ "$provider" = local ]; then
    if ollama_up || pgrep -x ollama >/dev/null 2>&1; then
      say "Нашёл Ollama на этом компьютере — использую её."
      configure_host_ollama
      set_env DOCKER_OLLAMA_URL "http://host.docker.internal:11434"
    else
      set_env DOCKER_OLLAMA_URL "http://ollama:11434"
      args+=(-f docker-compose.yml)
      if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        say "Видеокарта NVIDIA найдена — попробую считать на ней."
        gpu=(-f docker-compose.gpu.yml)
      else
        warn "Видеокарты NVIDIA нет — Ollama в Docker будет на процессоре. Быстрее: запуск напрямую (./start.sh config)"
      fi
      local_profile=(--profile local)
    fi
  fi
  say "Собираю и запускаю (первый раз — несколько минут)…"
  if [ ${#local_profile[@]} -gt 0 ]; then
    if ! docker "${args[@]}" ${gpu[@]+"${gpu[@]}"} "${local_profile[@]}" up -d --build; then
      [ ${#gpu[@]} -gt 0 ] || fail "Не удалось запустить. Посмотри сообщение выше."
      warn "Docker не видит видеокарту — запускаю Ollama на процессоре (медленнее)."
      docker compose --profile local down >/dev/null 2>&1 || true
      docker "${args[@]}" "${local_profile[@]}" up -d --build || fail "Не удалось запустить. Посмотри сообщение выше."
    fi
  else
    docker "${args[@]}" up -d --build || fail "Не удалось запустить. Посмотри сообщение выше."
  fi
  ok "Бот запущен. Зайди в голосовой канал и напиши /join. Остановить: docker compose --profile local stop"
  docker compose logs -f --tail 50 bot
}

# ---------------------------------------------------------------- поехали
say "=== Голосовой бот: запуск ==="
mkdir -p logs data models
if [ ! -f .env ]; then
  cp .env.example .env
  say "Первый запуск — пара вопросов."
  run_setup 1
elif [ "${1:-}" = "config" ]; then
  run_setup 0
fi

provider="$(get_env PROVIDER)"; provider="${provider:-local}"
mode="$(get_env RUN_MODE)"
if [ -z "$mode" ]; then  # .env от старой версии: раньше был только Docker
  mode=direct; command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && mode=docker
fi
if [ "$mode" = docker ]; then
  run_docker "$provider"
else
  [ "$provider" = local ] && ensure_host_ollama
  run_direct
fi
