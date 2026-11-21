# Запуск бота одной кнопкой (Windows). Запускается из start.bat (или config.bat — заново ответить на вопросы).
# Умеет два способа: напрямую (Python на этом компе) и через Docker.
param([switch]$Configure)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Fail($text) { Say $text "Red"; Read-Host "Нажми Enter, чтобы закрыть"; exit 1 }

function Get-EnvValue($key) {
    if (-not (Test-Path .env)) { return "" }
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
function Ask($question, $default) {
    $hint = if ($default) { " [$default]" } else { "" }
    $answer = (Read-Host "$question$hint").Trim()
    if (-not $answer) { return $default } else { return $answer }
}
function Ask-YesNo($question, $default) {
    $a = Ask "$question (y/n)" $default
    return @("y", "Y", "д", "Д", "да", "Да", "yes") -contains $a
}

# ---------------------------------------------------------------- железо
function Get-Gpu {
    # Самая большая видеокарта: имя, производитель, видеопамять. Точный объём — в реестре драйвера
    # (Win32_VideoController не видит больше 4 ГБ).
    $best = @{ Name = ""; Vendor = "none"; GB = 0 }
    try {
        $root = "HKLM:\SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        Get-ChildItem $root -ErrorAction SilentlyContinue | ForEach-Object {
            $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue
            $v = $p."HardwareInformation.qwMemorySize"
            if ($v -and [double]$v -gt $best.GB * 1GB) {
                $best.GB = [Math]::Round([double]$v / 1GB)
                $best.Name = "$($p.DriverDesc)"
            }
        }
    } catch {}
    if ($best.GB -eq 0) {
        try {
            Get-CimInstance Win32_VideoController | ForEach-Object {
                $gb = [Math]::Round([double]$_.AdapterRAM / 1GB)
                if ($gb -gt $best.GB) { $best.GB = $gb; $best.Name = $_.Name }
            }
        } catch {}
    }
    if ($best.Name -match "NVIDIA|GeForce|RTX|GTX|Quadro") { $best.Vendor = "nvidia" }
    elseif ($best.Name -match "AMD|Radeon|ATI") { $best.Vendor = "amd" }
    elseif ($best.Name -match "Intel") { $best.Vendor = "intel" }
    return $best
}

# Профили видеокарт — держать в согласии со start.sh (Linux):
#   профиль → окно Ollama, слотов, сжатие кэша, где считает Whisper
function Get-ProfileForGB($gb) {
    if ($gb -ge 16) { "16gb" } elseif ($gb -ge 10) { "12gb" } elseif ($gb -ge 6) { "8gb" } else { "cpu" }
}
$Profiles = @{
    "16gb" = @{ Ctx = "8192"; Par = "2"; Kv = "f16";  Whisper = "auto" }
    "12gb" = @{ Ctx = "8192"; Par = "1"; Kv = "q8_0"; Whisper = "auto" }
    "8gb"  = @{ Ctx = "6144"; Par = "1"; Kv = "q8_0"; Whisper = "cpu" }
    "cpu"  = @{ Ctx = "4096"; Par = "1"; Kv = "q8_0"; Whisper = "cpu" }
}
function Set-Profile($name, $vendor) {
    $p = $Profiles[$name]
    Set-EnvValue "GPU_PROFILE" $name
    Set-EnvValue "OLLAMA_CONTEXT_LENGTH" $p.Ctx
    Set-EnvValue "OLLAMA_NUM_PARALLEL" $p.Par
    Set-EnvValue "OLLAMA_KV_CACHE_TYPE" $p.Kv
    Set-EnvValue "OLLAMA_FLASH_ATTENTION" "1"
    Set-EnvValue "LOCAL_WHISPER_DEVICE" $p.Whisper
    # AMD на Windows Ollama видит только через Vulkan
    Set-EnvValue "OLLAMA_VULKAN" $(if ($vendor -eq "amd") { "1" } else { "" })
}

# ---------------------------------------------------------------- вопросы
function Run-Setup([bool]$first) {
    Say "`n--- Настройка ---" "Yellow"

    $token = Ask "Токен Discord-бота (discord.com/developers/applications → Bot → Reset Token)" (Get-EnvValue "DISCORD_TOKEN")
    if (-not $token) { Fail "Без токена бот не запустится." }
    Set-EnvValue "DISCORD_TOKEN" $token

    Say "`nГде будут работать мозги бота?"
    Say "  1 — на этом компьютере (Ollama): бесплатно, без лимитов, но глупее и медленнее"
    Say "  2 — в облаке Groq: умнее и быстрее, бесплатно, но с дневными лимитами (нужен ключ)"
    $def = if (-not $first -and (Get-EnvValue "PROVIDER") -eq "groq") { "2" } else { "1" }
    $provider = if ((Ask "Выбери 1 или 2" $def) -eq "2") { "groq" } else { "local" }
    Set-EnvValue "PROVIDER" $provider

    Say ""
    if ($provider -eq "groq") {
        Say "Ключ Groq: console.groq.com → API Keys (бесплатно, начинается с gsk_)"
        $key = Ask "Ключ Groq" (Get-EnvValue "GROQ_API_KEY")
        if (-not $key) { Fail "Для облачных мозгов нужен ключ Groq." }
    } else {
        Say "Ключ Groq — НЕОБЯЗАТЕЛЬНО. С ним распознавание речи пойдёт через облако: быстро и точно,"
        Say "а мозги останутся на компе без лимитов. Без него всё будет на компе (медленнее)."
        $key = Ask "Ключ Groq (Enter — пропустить)" (Get-EnvValue "GROQ_API_KEY")
    }
    Set-EnvValue "GROQ_API_KEY" $key

    if ($provider -eq "local") {
        $gpu = Get-Gpu
        Say ""
        if ($gpu.GB -gt 0) { Say "Видеокарта: $($gpu.Name), $($gpu.GB) ГБ." } else { Say "Видеокарту не нашёл." }
        $answer = Ask "Сколько ГБ видеопамяти? (0 — без видеокарты)" "$($gpu.GB)"
        $gb = 0
        if ([int]::TryParse($answer, [ref]$gb)) { } else { $gb = $gpu.GB }
        $gpuProfile = Get-ProfileForGB $gb
        Set-Profile $gpuProfile $gpu.Vendor
        Say "Профиль видеокарты: $gpuProfile (окно $($Profiles[$gpuProfile].Ctx), слотов $($Profiles[$gpuProfile].Par))" "Green"
        if ($gpu.Vendor -eq "amd") { Say "AMD: включу Ollama режим Vulkan (иначе она не видит Radeon на Windows)." "Gray" }

        $rec = if ($gpuProfile -eq "cpu") { "1" } else { "3" }
        $models = @{ "1" = "qwen2.5:3b"; "2" = "qwen2.5:7b"; "3" = "qwen3.5:9b"; "4" = "qwen2.5:14b" }
        Say "`nКакую модель взять?"
        Say "  1 — qwen2.5:3b   без видеокарты или совсем слабая (быстро, но туповато)"
        Say "  2 — qwen2.5:7b   видеокарта 6 ГБ"
        Say "  3 — qwen3.5:9b   видеокарта 8 ГБ и больше (основная)"
        Say "  4 — qwen2.5:14b  видеокарта 12 ГБ и больше"
        Say "  5 — своя (название из ollama.com/library)"
        $curModel = if ($first) { "" } else { Get-EnvValue "OLLAMA_MODEL" }
        $curNum = ($models.GetEnumerator() | Where-Object { $_.Value -eq $curModel } | Select-Object -First 1).Key
        $def = if ($curNum) { $curNum } elseif ($curModel) { "5" } else { $rec }
        $pick = Ask "Выбери 1–5 (советую $rec)" $def
        if ($pick -eq "5") { $model = Ask "Название модели (например, gemma3:12b)" $curModel }
        elseif ($models.ContainsKey($pick)) { $model = $models[$pick] }
        else { $model = $models["3"] }
        Set-EnvValue "OLLAMA_MODEL" $model
        Say "Модель: $model — скачается сама при первом запуске." "Green"
    }

    # имя
    Say "`nКак зовут бота? Новое имя — новый «человек» с чистой памятью (старый останется в data\ и вернётся с именем)."
    $cur = Get-EnvValue "BOT_NAME"
    $name = Ask "Имя" $(if ($cur) { $cur } else { "Петрык" })
    Set-EnvValue "BOT_NAME" $name
    Set-EnvValue "BOT_NAME_ALIASES" (Ask "Как ещё его зовут, через запятую (Enter — пропустить)" (Get-EnvValue "BOT_NAME_ALIASES"))

    # голос
    Say "`nГолос:"
    Say "  1 — Дмитрий   2 — Денис   3 — Руслан   4 — Ирина (женский)    — на компе, мгновенно"
    Say "  5 — Дмитрий онлайн   6 — Светлана онлайн (женский)              — живее, нужен интернет"
    $voices = @{
        "1" = @("piper", "ru_RU-dmitri-medium"); "2" = @("piper", "ru_RU-denis-medium")
        "3" = @("piper", "ru_RU-ruslan-medium"); "4" = @("piper", "ru_RU-irina-medium")
        "5" = @("edge", "ru-RU-DmitryNeural");   "6" = @("edge", "ru-RU-SvetlanaNeural")
    }
    $curVoice = if ((Get-EnvValue "TTS_BACKEND") -eq "edge") { Get-EnvValue "EDGE_VOICE" } else { Get-EnvValue "PIPER_VOICE" }
    $def = ($voices.GetEnumerator() | Where-Object { $_.Value[1] -eq $curVoice } | Select-Object -First 1).Key
    $v = $voices[(Ask "Выбери 1–6" $(if ($def) { $def } else { "1" }))]
    if (-not $v) { $v = $voices["1"] }
    Set-EnvValue "TTS_BACKEND" $v[0]
    if ($v[0] -eq "edge") { Set-EnvValue "EDGE_VOICE" $v[1] } else { Set-EnvValue "PIPER_VOICE" $v[1] }
    $def = if ((Get-EnvValue "PIPER_SPEED") -eq "1.15") { "2" } else { "1" }
    Set-EnvValue "PIPER_SPEED" $(if ((Ask "Темп речи: 1 — обычный, 2 — чуть быстрее" $def) -eq "2") { "1.15" } else { "1.0" })

    # как запускать
    Say "`nКак запускать?"
    Say "  1 — напрямую (Python на этом компе)"
    Say "  2 — через Docker"
    $cur = Get-EnvValue "RUN_MODE"
    # Docker на Windows отдаёт контейнеру только видеокарту NVIDIA: с AMD/Intel модель в Docker считала бы
    # на процессоре — советуем «напрямую» (Ollama на самом Windows видит и Radeon)
    $noDockerGpu = $provider -eq "local" -and $gpu.Vendor -ne "nvidia"
    if ($noDockerGpu) {
        Say "  У тебя не NVIDIA: в Docker модель считала бы на процессоре (медленно). Советую 1 — напрямую." "Yellow"
        $def = "1"
    } else {
        $def = if ($cur -eq "docker") { "2" } elseif ($cur -eq "direct") { "1" } elseif (Test-Docker) { "2" } else { "1" }
    }
    $mode = Ask "Выбери 1 или 2" $def
    if ($mode -eq "2" -and $noDockerGpu) {
        $mode = if (Ask-YesNo "Точно Docker? Модель будет на процессоре — если Ollama не запущена на самом компе" "n") { "2" } else { "1" }
    }
    Set-EnvValue "RUN_MODE" $(if ($mode -eq "2") { "docker" } else { "direct" })

    # для любопытных
    if ($provider -eq "local" -and (Ask-YesNo "Расширенные настройки (размышления модели)?" "n")) {
        Say "Размышления: модель сначала думает «про себя», потом отвечает. Умнее, но медленнее."
        Set-EnvValue "THINK_REPLY" $(if (Ask-YesNo "Думать в живом ответе? (медленно: +3–10 с перед каждой фразой)" "n") { "on" } else { "off" })
        Set-EnvValue "THINK_BACKGROUND" $(if (Ask-YesNo "Думать в фоновых делах (досье, сны, книжка)?" "y") { "on" } else { "off" })
    }
    Say "`nГотово. Поменять это потом можно через config.bat." "Green"
}

function Test-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { return $false }
    try { docker info *> $null } catch { return $false }
    return $LASTEXITCODE -eq 0
}

# ---------------------------------------------------------------- Ollama на этом компе
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

function Set-HostOllamaVars {
    # Настройки самой Ollama — под профиль из .env (переменные пользователя Windows; Ollama читает их при старте)
    $want = [ordered]@{
        "OLLAMA_CONTEXT_LENGTH"  = (Get-EnvValue "OLLAMA_CONTEXT_LENGTH")
        "OLLAMA_NUM_PARALLEL"    = (Get-EnvValue "OLLAMA_NUM_PARALLEL")
        "OLLAMA_KV_CACHE_TYPE"   = (Get-EnvValue "OLLAMA_KV_CACHE_TYPE")
        "OLLAMA_FLASH_ATTENTION" = (Get-EnvValue "OLLAMA_FLASH_ATTENTION")
        "OLLAMA_VULKAN"          = (Get-EnvValue "OLLAMA_VULKAN")
    }
    if (-not $want["OLLAMA_CONTEXT_LENGTH"]) { $want["OLLAMA_CONTEXT_LENGTH"] = "8192" }  # .env от старой версии
    $changed = $false
    foreach ($k in $want.Keys) {
        $cur = [Environment]::GetEnvironmentVariable($k, "User")
        $val = $want[$k]
        if (-not $val) { continue }  # не задано профилем — то, что человек выставил сам, не трогаем
        if ($cur -ne $val) { [Environment]::SetEnvironmentVariable($k, $val, "User"); $changed = $true }
    }
    return $changed
}

function Restart-HostOllama {
    Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
    if (Test-Path $app) { Start-Process $app } else { Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden }
    for ($i = 0; $i -lt 30; $i++) { if (Test-HostOllama) { return $true }; Start-Sleep -Seconds 1 }
    return $false
}

function Ensure-HostOllama {
    if (-not (Test-HostOllama) -and -not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Say "`nOllama не установлена." "Yellow"
        if ((Get-Command winget -ErrorAction SilentlyContinue) -and (Ask-YesNo "Поставить её сейчас (winget, официальная Ollama)?" "y")) {
            winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        } else {
            Fail "Поставь Ollama с ollama.com и запусти start.bat снова (или выбери облако: config.bat)."
        }
    }
    $changed = Set-HostOllamaVars
    if ($changed -or -not (Test-HostOllama)) {
        if ($changed) { Say "Настроил Ollama под твою видеокарту (окно $(Get-EnvValue 'OLLAMA_CONTEXT_LENGTH'))." "Yellow" }
        if (-not (Test-HostOllama) -or (Ask-YesNo "Перезапустить Ollama, чтобы она взяла настройки?" "y")) {
            if (-not (Restart-HostOllama)) { Fail "Ollama не запустилась. Открой её вручную и запусти start.bat снова." }
            Say "Ollama запущена с настройками бота." "Green"
        } else {
            Say "Не забудь перезапустить Ollama (значок у часов → Quit, и открыть снова)." "Yellow"
        }
    }
}

# ---------------------------------------------------------------- запуск напрямую
function Run-Direct {
    $py = $null
    foreach ($v in @("-3.12", "-3.11", "-3.13")) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            try { & py $v -c "import sys" *> $null } catch {}  # нет такой версии — py пишет в stderr
            if ($LASTEXITCODE -eq 0) { $py = @("py", $v); break }
        }
    }
    if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) {
        try { & python -c "import sys; sys.exit(sys.version_info < (3, 11))" *> $null } catch {}
        if ($LASTEXITCODE -eq 0) { $py = @("python") }
    }
    if (-not $py) { Fail "Не найден Python 3.11+. Поставь с python.org (галочка «Add to PATH») или выбери Docker: config.bat" }
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $venvPy)) {
        Say "Создаю окружение .venv…"
        & $py[0] @($py | Select-Object -Skip 1) -m venv .venv
        if ($LASTEXITCODE -ne 0) { Fail "Не создалось окружение .venv — смотри сообщение выше." }
    }
    $hash = (Get-FileHash requirements.txt -Algorithm SHA1).Hash
    $stamp = Join-Path $PSScriptRoot ".venv\.req-hash"
    if (-not (Test-Path $stamp) -or (Get-Content $stamp) -ne $hash) {
        Say "Ставлю зависимости (первый раз — несколько минут)…"
        & $venvPy -m pip install -q --upgrade pip
        & $venvPy -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { Fail "Не встали зависимости — смотри сообщение выше." }
        Set-Content -Path $stamp -Value $hash
    }
    Say "`nЗапускаю бота. Зайди в голосовой канал и напиши /join. Остановить: закрой окно или Ctrl+C`n" "Green"
    & $venvPy -m bot
    Read-Host "Бот остановился. Нажми Enter, чтобы закрыть"
}

# ---------------------------------------------------------------- запуск через Docker
function Run-Docker($provider) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail "Не найден Docker. Поставь Docker Desktop: https://www.docker.com/products/docker-desktop/`n(Или без Docker: config.bat → «напрямую»)"
    }
    try { docker info *> $null } catch {}
    if ($LASTEXITCODE -ne 0) { Fail "Docker установлен, но не запущен. Открой Docker Desktop, дождись зелёного статуса и запусти снова." }
    $composeArgs = @("compose"); $gpuArgs = @(); $localArgs = @()
    if ($provider -eq "local") {
        $found = Test-HostOllama
        if ($found) {
            Say "Нашёл Ollama на этом компьютере ($found) — использую её." "Green"
            Set-EnvValue "DOCKER_OLLAMA_URL" "http://host.docker.internal:11434"
            Ensure-HostOllama
        } else {
            Set-EnvValue "DOCKER_OLLAMA_URL" "http://ollama:11434"
            $composeArgs += @("-f", "docker-compose.yml")
            # nvidia-smi может лежать в системе и без рабочей видеокарты NVIDIA — проверяем, что он её реально видит
            $nvidia = $false
            if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
                try { nvidia-smi -L *> $null } catch {}
                if ($LASTEXITCODE -eq 0) { $nvidia = $true }
            }
            if ($nvidia) {
                Say "Видеокарта NVIDIA найдена — попробую считать на ней." "Green"
                $gpuArgs = @("-f", "docker-compose.gpu.yml")
            } else {
                Say "Видеокарты NVIDIA нет — Ollama в Docker будет на процессоре (медленно)." "Yellow"
                Say "Совет: поставь Ollama прямо на Windows с ollama.com (умеет и AMD) и запусти start.bat снова." "Yellow"
            }
            $localArgs = @("--profile", "local")
        }
    }
    Say "`nСобираю и запускаю (первый раз — несколько минут: скачиваются зависимости и модели)…" "Cyan"
    if ($localArgs.Count -gt 0) {
        & docker @composeArgs @gpuArgs @localArgs up -d --build
        if ($LASTEXITCODE -ne 0 -and $gpuArgs.Count -gt 0) {
            # Docker не смог отдать видеокарту контейнеру (частая история на Windows) — запускаемся без неё
            Say "`nDocker не видит видеокарту — запускаю Ollama на процессоре (медленнее)." "Yellow"
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
}

# ---------------------------------------------------------------- поехали
Say "=== Голосовой бот: запуск ===" "Cyan"
foreach ($d in @("logs", "data", "models")) { New-Item -ItemType Directory -Force $d | Out-Null }
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Say "`nПервый запуск — пара вопросов." "Yellow"
    Run-Setup $true
} elseif ($Configure) {
    Run-Setup $false
}

$provider = Get-EnvValue "PROVIDER"
if (-not $provider) { $provider = "local" }
$mode = Get-EnvValue "RUN_MODE"
if (-not $mode) { $mode = "docker" }  # .env от старой версии: раньше был только Docker

# Docker + мозги на компе + не NVIDIA + Ollama на самом компе не запущена → Ollama поднимется в контейнере
# на процессоре (видеокарту AMD/Intel Docker на Windows не отдаёт). Предлагаем запуск напрямую.
if ($mode -eq "docker" -and $provider -eq "local" -and -not (Test-HostOllama) -and (Get-Gpu).Vendor -ne "nvidia") {
    Say "`nВнимание: в Docker модель будет считать на процессоре — видеокарту AMD/Intel Docker на Windows не видит." "Yellow"
    Say "Напрямую (Ollama на самом Windows) она возьмёт видеокарту и будет в разы быстрее." "Yellow"
    if (Ask-YesNo "Перейти на запуск напрямую?" "y") {
        Set-EnvValue "RUN_MODE" "direct"
        $mode = "direct"
    }
}

if ($mode -eq "docker") {
    Run-Docker $provider
} else {
    if ($provider -eq "local") { Ensure-HostOllama }
    Run-Direct
}
