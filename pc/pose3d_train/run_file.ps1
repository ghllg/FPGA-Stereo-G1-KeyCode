param(
    [string]$Pose = 'windows_vision/outputs/20261004_194029_recovered_windows/pose2d_fused.jsonl',
    [string]$Calibration = 'windows_vision/outputs/calibration_world_2.json',
    [string]$Checkpoint = 'pose3d_train/runs/20261013_mlp_v1/checkpoint.pt',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$posePython = Join-Path (Get-Location) 'windows_vision/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $posePython)) { throw 'Create the project Python environment first; see pose3d_train/README.md.' }
if (-not $Output) { $Output = Join-Path $PSScriptRoot ('runs/replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss')) }
if (Test-Path -LiteralPath $Output) { throw 'Output already exists. Use a new directory.' }
$skeleton = Join-Path $Output 'skeleton3d.jsonl'
$bvh = Join-Path $Output 'bvh'
& $posePython -X utf8 -m pose3d_train.infer --pose $Pose --calibration $Calibration --checkpoint $Checkpoint --output $skeleton
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $posePython -X utf8 -c 'import sys, pose3d_train; from mocap.lafan import export_lafan; export_lafan(sys.argv[1], sys.argv[2])' $skeleton $bvh
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Output "RESULTS=$Output"
