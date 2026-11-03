@echo off
setlocal
cd /d "%~dp0.."
set "BVH_PYTHON=E:\FPGAchina\FPGA2026\windows_vision\.venv\Scripts\python.exe"
if not exist "%BVH_PYTHON%" (
  echo Python environment not found: %BVH_PYTHON%
  pause
  exit /b 1
)
"%BVH_PYTHON%" -m demo_station.bvh_app --native %*
if errorlevel 1 pause
