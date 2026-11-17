# Запуск бота одной кнопкой через Docker (Windows). Запускается из start.bat.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Fail($text) { Say $text "Red"; Read-Host "Нажми Enter, чтобы закрыть"; exit 1 }

function Get-EnvValue($key) {
    $line = Get-Content .env -Encoding UTF8 | Where-Object { $_ -match "^$key=" } | Select-Object -First 1
    if ($line) { return ($line -split "=", 2)[1].Trim() } else { return "" }
}
function Set-EnvValue($key, $value) {
    $lines = @(Get-Content .env -Encoding UTF8)
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^#?\s*$key=") { $lines[$i] = "$key=$value"; $found = $true; break }
    }
    if (-not $found) { $lines += "$key=$value" }
    [IO.File]::WriteAllText((Join-Path $PSScriptRoot ".env"), ($lines -join "`n") + "`n", $Utf8NoBom)
}

Say "=== Голосовой бот: запуск ===" "Cyan"

# 1. Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "Не найден Docker. Поставь Docker Desktop: https://www.docker.com/products/docker-desktop/ и запусти снова.`n(Или без Docker: setup.bat, потом run.bat)"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker установлен, но не запущен. Открой Docker Desktop, дождись зелёного статуса и запусти снова." }

# 2. Первый запуск — создаём .env и спрашиваем главное
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Say "`nПервый запуск — пара вопросов." "Yellow"
    $token = Read-Host "Токен Discord-бота (discord.com/developers/applications → Bot → Reset Token)"
    if (-not $token) { Remove-Item .env; Fail "Без токена бот не запустится." }
    Set-EnvValue "DISCORD_TOKEN" $token.Trim()
    Say "`nГде будут работать мозги бота?"
    Say "  1 — на этом компьютере (Ollama): бесплатно, без лимитов, но глупее и медленнее"
    Say "  2 — в облаке Groq: умнее и быстрее, бесплатно, но с дневными лимитами (нужен ключ)"
    $choice = Read-Host "Выбери 1 или 2"
    if ($choice -eq "2") {
        $key = Read-Host "Ключ Groq (console.groq.com → API Keys, начинается с gsk_)"
        Set-EnvValue "PROVIDER" "groq"
        Set-EnvValue "GROQ_API_KEY" $key.Trim()
    } else {
        Set-EnvValue "PROVIDER" "local"
    }
}

$provider = Get-EnvValue "PROVIDER"
if (-not $provider) { $provider = "local" }
$composeArgs = @("compose")

# 3. Локальный режим: своя Ollama на компе или контейнер (с видеокартой, если есть)
function Test-HostOllama {
    # Стучимся напрямую в порт: «localhost» Windows часто пробует сначала по IPv6, а Ollama слушает только IPv4
    foreach ($h in @("127.0.0.1", "localhost")) {
        $c = New-Object System.Net.Sockets.TcpClient
        try {
            $ar = $c.BeginConnect($h, 11434, $null, $null)
            if ($ar.AsyncWaitHandle.WaitOne(2000) -and $c.Connected) { return "порт 11434 на $h отвечает" }
        } catch {} finally { $c.Close() }
    }
    # порт пока молчит (Ollama ещё запускается), но процесс есть — всё равно используем её, бот подождёт
    $p = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($p) { return "запущен процесс $($p.ProcessName)" }
    return $null
}

if ($provider -eq "local") {
    $found = Test-HostOllama
    $hostOllama = [bool]$found
    if ($found) { Say "Ollama на компьютере: $found" "Gray" } else { Say "Ollama на этом компьютере не запущена." "Gray" }
    if ($hostOllama) {
        Say "Нашёл Ollama на этом компьютере — использую её." "Green"
        Set-EnvValue "DOCKER_OLLAMA_URL" "http://host.docker.internal:11434"
    } else {
        Set-EnvValue "DOCKER_OLLAMA_URL" "http://ollama:11434"
        $composeArgs += @("-f", "docker-compose.yml")
        # nvidia-smi может лежать в системе и без рабочей видеокарты NVIDIA — проверяем, что он её реально видит
        $nvidia = $false
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            nvidia-smi -L *> $null
            if ($LASTEXITCODE -eq 0) { $nvidia = $true }
        }
        if ($nvidia) {
            Say "Видеокарта NVIDIA найдена — попробую считать на ней." "Green"
            $gpuArgs = @("-f", "docker-compose.gpu.yml")
        } else {
            Say "Видеокарты NVIDIA нет — Ollama в Docker будет на процессоре (медленно)." "Yellow"
            Say "Совет: поставь Ollama прямо на Windows с ollama.com (умеет и AMD) и запусти start.bat снова." "Yellow"
            $gpuArgs = @()
        }
        $localArgs = @("--profile", "local")
    }
}

# 4. Поехали
Say "`nСобираю и запускаю (первый раз — несколько минут: скачиваются зависимости и модели)…" "Cyan"
if ($provider -eq "local" -and $localArgs) {
    & docker @composeArgs @gpuArgs @localArgs up -d --build
    if ($LASTEXITCODE -ne 0 -and $gpuArgs.Count -gt 0) {
        # Docker не смог отдать видеокарту контейнеру (частая история на Windows) — запускаемся без неё
        Say "`nDocker не видит видеокарту — запускаю Ollama на процессоре (медленнее)." "Yellow"
        Say "Совет: поставь Ollama прямо на Windows с ollama.com — она возьмёт видеокарту сама." "Yellow"
        & docker compose --profile local down *> $null
        & docker @composeArgs @localArgs up -d --build
    }
} else {
    & docker @composeArgs up -d --build
}
if ($LASTEXITCODE -ne 0) { Fail "Не удалось запустить. Посмотри сообщение выше." }

Say "`nБот запущен в фоне. Зайди в голосовой канал и напиши /join" "Green"
Say "Остановить: stop.bat   Переключить мозги на ходу: /brain в Discord`n"
Say "Логи (закрыть окно — бот продолжит работать):" "Gray"
& docker compose logs -f --tail 50 bot
