param([switch]$Cpu)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$visionPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$torchIndex = if ($Cpu) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu126' }
$torchSuffix = if ($Cpu) { '+cpu' } else { '+cu126' }
& $visionPython -m pip install "torch==2.14.1$torchSuffix" "torchvision==0.29.1$torchSuffix" --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $visionPython -m pip install -r requirements.txt
exit $LASTEXITCODE
