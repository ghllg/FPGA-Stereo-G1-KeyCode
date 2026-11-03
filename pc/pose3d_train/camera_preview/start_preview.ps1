$ErrorActionPreference = 'Stop'
$PreviewPython = 'E:\FPGAchina\FPGA2026\windows_vision\.venv\Scripts\python.exe'
& $PreviewPython "$PSScriptRoot\preview.py" --serial COM22 --board 192.168.137.2
exit $LASTEXITCODE
