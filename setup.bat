@echo off
REM Ustanovka i nastroika: voprosy mastera (to zhe, chto config.bat)
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Configure
