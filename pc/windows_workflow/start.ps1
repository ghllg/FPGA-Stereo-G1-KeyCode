param([switch]$Demo)
$ErrorActionPreference = 'Stop'
$workflowRoot = $PSScriptRoot
$workflowPython = Join-Path $workflowRoot '../windows_vision/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $workflowPython)) {
    throw '未找到 windows_vision 的 Python 环境。请先按 windows_vision/README.md 安装环境。'
}
Push-Location $workflowRoot
try {
    if ($Demo) { & $workflowPython -m workflow --demo }
    else { & $workflowPython -m workflow }
    if ($LASTEXITCODE -ne 0) { throw "流程退出码：$LASTEXITCODE" }
} finally { Pop-Location }
