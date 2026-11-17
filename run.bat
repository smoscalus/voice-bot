@echo off
REM ==== DiscordCodexVoiceMode: zapusk bota ====
cd /d "%~dp0"
if not exist ".venv\Scripts\activate" ( echo Snachala zapusti setup.bat & pause & exit /b 1 )
call ".venv\Scripts\activate"
python -m bot
pause
