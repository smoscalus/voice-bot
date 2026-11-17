@echo off
cd /d "%~dp0"
docker compose --profile local stop
echo Bot ostanovlen. Pamyat sohranena v papke data.
pause
