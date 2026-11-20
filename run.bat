@echo off
REM Zapusk bota (to zhe, chto start.bat)
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
