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

if [ ! -f .env ]; then
  cp .env.example .env
  say "Первый запуск — пара вопросов."
  read -rp "Токен Discord-бота: " token
  [ -n "$token" ] || { rm .env; fail "Без токена бот не запустится."; }
  set_env DISCORD_TOKEN "$token"
  say "Где будут работать мозги бота?"
  say "  1 — на этом компьютере (Ollama): бесплатно, без лимитов, но глупее и медленнее"
  say "  2 — в облаке Groq: умнее и быстрее, бесплатно, но с дневными лимитами (нужен ключ)"
  read -rp "Выбери 1 или 2: " choice
  if [ "$choice" = "2" ]; then
    read -rp "Ключ Groq (console.groq.com → API Keys): " key
    set_env PROVIDER groq; set_env GROQ_API_KEY "$key"
  else
    set_env PROVIDER local
  fi
fi

provider="$(get_env PROVIDER)"; provider="${provider:-local}"
args=(compose)
if [ "$provider" = "local" ]; then
  if curl -fs -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 || curl -fs -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 || pgrep -x ollama >/dev/null 2>&1; then
    say "Нашёл Ollama на этом компьютере — использую её."
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
