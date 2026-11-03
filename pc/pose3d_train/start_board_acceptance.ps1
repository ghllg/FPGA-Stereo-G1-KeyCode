$ErrorActionPreference = 'Stop'
$AcceptancePython = Join-Path (Split-Path -Parent $PSScriptRoot) 'windows_vision\.venv\Scripts\python.exe'
$AcceptanceOldPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Split-Path -Parent $PSScriptRoot) + ';' + (Join-Path (Split-Path -Parent $PSScriptRoot) 'windows_vision')
    & $AcceptancePython -X utf8 -m pose3d_train.board_live_acceptance run
    if ($LASTEXITCODE -ne 0) { throw '本段发生错误，保留窗口中的报错，先不要重复运行。' }
} finally {
    $env:PYTHONPATH = $AcceptanceOldPath
}
