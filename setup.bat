@echo off
REM ==== DiscordCodexVoiceMode: ustanovka ====
cd /d "%~dp0"
echo [1/4] Sozdayu virtualnoe okruzhenie .venv ...
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 ( echo [ERROR] Ne naiden Python 3.11/3.12. Ustanovi s python.org i povtori. & pause & exit /b 1 )
call ".venv\Scripts\activate"
echo [2/4] Obnovlyayu pip ...
python -m pip install --upgrade pip
echo [3/4] Stavlyu zavisimosti (eto dolgo) ...
pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] pip install upal. Smotri soobshenie vyshe. & pause & exit /b 1 )
echo [4/4] Tyanu model Ollama qwen2.5:7b (~4.7 GB) ...
ollama pull qwen2.5:7b
if errorlevel 1 ( echo [WARN] ollama pull ne proshel. Ubedis chto Ollama ustanovlena i zapushena. )
echo.
echo Gotovo. Teper vpishi DISCORD_TOKEN v fayl .env i zapusti run.bat
pause
