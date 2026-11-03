@echo off
setlocal
cd /d "%~dp0.."
set "STATION_PYTHON=%~dp0..\windows_vision\.venv\Scripts\python.exe"
if not exist "%STATION_PYTHON%" set "STATION_PYTHON=%~dp0..\..\FPGA2026\windows_vision\.venv\Scripts\python.exe"
if not exist "%STATION_PYTHON%" set "STATION_PYTHON=%~dp0..\..\windows_vision\.venv\Scripts\python.exe"
if not exist "%STATION_PYTHON%" (
  echo Python environment missing. See demo_station\README.md.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
"%STATION_PYTHON%" -m demo_station.app %*
if errorlevel 1 pause
endlocal
