@echo off
setlocal
cd /d "%~dp0"
"runtime\python.exe" -X utf8 launcher.py start
if errorlevel 1 pause
