param([switch]$Cli, [Parameter(ValueFromRemainingArguments=$true)][string[]]$Remaining)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$visionPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $visionPython)) {
    Write-Host 'Missing environment. Run setup.ps1 first.'
    exit 1
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($Cli) { & $visionPython -m mocap @Remaining }
else { & $visionPython -m mocap gui }
exit $LASTEXITCODE

