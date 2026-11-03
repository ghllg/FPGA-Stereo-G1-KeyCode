$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 before starting this application.' }
}
$taskDependenciesOk = $false
try {
    & $taskPython -c 'import av, PySide6, imageio_ffmpeg' 2>$null
    $taskDependenciesOk = $LASTEXITCODE -eq 0
} catch {
    $taskDependenciesOk = $false
}
if (!$taskDependenciesOk) {
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your network and retry.' }
}
& $taskPython app.py @args
exit $LASTEXITCODE
