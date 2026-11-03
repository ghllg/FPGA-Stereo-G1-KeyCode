$ErrorActionPreference = 'Stop'
$CapturePython = 'E:\FPGAchina\FPGA2026\windows_vision\.venv\Scripts\python.exe'
$CaptureDemo = 'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000'

Write-Host '本轮只拍静止棋盘样片：每路 150 帧，约 5 秒。' -ForegroundColor Cyan
Write-Host '先关闭实时预览和占用 COM22 的串口终端，摄像头保持固定。'
Write-Host '棋盘长边横放，拿在胸前，朝向两只摄像头中间；完整图案可见，手握边框。'
Write-Host '听到开始声后保持棋盘不动，直到听到结束声再回来。'
$CaptureReady = Read-Host '准备好后输入 START；其他输入不采集'
if ($CaptureReady -cne 'START') {
    Write-Host '没有开始采集。'
    return
}

for ($CaptureCountdown = 10; $CaptureCountdown -ge 1; $CaptureCountdown--) {
    Write-Host "请拿棋盘就位，倒计时 $CaptureCountdown 秒"
    Start-Sleep -Seconds 1
}
[Console]::Beep(900, 250)
Write-Host '请保持棋盘不动。正在准备、拍摄并回传；等待结束声。' -ForegroundColor Cyan
& $CapturePython "$CaptureDemo\pc\demo.py" capture-calibration --serial COM22 --board 192.168.137.2 --frames 150
$CaptureExitCode = $LASTEXITCODE
if ($CaptureExitCode -eq 0) {
    Write-Host '棋盘样片已拍完并回传，可以回来；保持摄像头位置不动。' -ForegroundColor Green
} else {
    Write-Host '本轮没有正常完成，请保留上面的报错，不要接着拍下一段。' -ForegroundColor Yellow
}
[Console]::Beep(1200, 800)
exit $CaptureExitCode
