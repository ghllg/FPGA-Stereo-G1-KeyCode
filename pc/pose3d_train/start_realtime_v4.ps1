param(
    [ValidateSet('arms','range','squat','stability')][string]$Action='arms',
    [ValidateRange(5,300)][int]$Seconds=25
)
$ErrorActionPreference='Stop'
$PosePython=Join-Path (Split-Path -Parent $PSScriptRoot) 'windows_vision\.venv\Scripts\python.exe'
$PoseOldPath=$env:PYTHONPATH
try {
    $env:PYTHONPATH=(Split-Path -Parent $PSScriptRoot)+';'+(Join-Path (Split-Path -Parent $PSScriptRoot) 'windows_vision')
    & $PosePython -X utf8 -m pose3d_train.board_live_acceptance run --action $Action --seconds $Seconds --show
    if ($LASTEXITCODE -ne 0) { throw '本段未正常完成。保留报错，先不要拍下一段。' }
} finally { $env:PYTHONPATH=$PoseOldPath }
