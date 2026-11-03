param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('near_left', 'near_right', 'near_yaw_left', 'near_yaw_right', 'near_pitch_top', 'near_pitch_bottom', 'near_position_left', 'near_position_right', 'near_position_high', 'near_position_low', 'closer_center', 'closer_yaw_left', 'closer_yaw_right', 'closer_pitch_top', 'closer_yaw_left_mild', 'closer_yaw_right_mild', 'farther_center', 'farther_yaw_left', 'farther_yaw_right', 'farther_pitch_bottom', 'validation_mid_center', 'validation_mid_left_high', 'validation_mid_right_low', 'validation_mid_yaw', 'validation_far_pitch_top', 'world_ground_center', 'sync_horizontal_1', 'sync_vertical_2', 'sync_horizontal_3', 'body_motion_baseline', 'validation_final_center', 'current_yaw_left', 'current_pitch_top', 'validation_current_left', 'world_ground_current')]
    [string]$Pose
)

$ErrorActionPreference = 'Stop'
$CapturePython = 'E:\FPGAchina\FPGA2026\windows_vision\.venv\Scripts\python.exe'
$CaptureDemo = 'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000'
$PoseInstructions = @{
    current_yaw_left = '相对位置修复1：从能看清鞋的靠后中间站位向两摄中间走近约50厘米。棋盘放胸前、长边水平，身体朝两摄中间；双手同高，右手一边不动，左手握的边缘向胸口收约10厘米。两路都要看见完整棋盘，保持到结束声。'
    current_pitch_top = '相对位置修复2：保持上一段走近50厘米后的站位。棋盘放胸前，先恢复竖直、长边水平；上边沿向胸口倾、下边沿向前送，让上下边沿前后错开约10厘米。整块板仍在胸前，两路看见完整棋盘，保持到结束声。'
    validation_current_left = '新独立验证：回到原来能看清鞋的靠后中间位置，再向自己的左边侧移约20厘米。棋盘在胸前，竖直、长边水平、朝两摄中间，不倾斜。保持到结束声；本段不用于拟合。'
    world_ground_current = '当前地面参考：在原来靠后中间站位的脚边，把棋盘平放在平整地面，图案朝上、长边左右横放，确保两路能看到完整棋盘。输入START后站到棋盘后侧，不遮挡图案。不要移动棋盘或摄像头，保持到结束声。'
    validation_final_center = '最终独立几何检查：站在能完整看到头和鞋的靠后中间位置。棋盘竖直放在胸前，长边水平，朝向两摄中间；双手只握边缘，不遮黑白格。整段保持不动，不前后倾斜。保持到结束提示音。本段仅验收，不参与标定拟合。'
    body_motion_baseline = '人体样例：不用棋盘。站在刚才能完整看到头和鞋的靠后中间位置，正对两摄中间，双脚自然分开、身体不转身不走动。短提示音后反复做：双臂自然下垂停1秒；双臂缓慢向两侧抬到斜下方约45度（用2秒），停1秒；缓慢放下（用2秒）；轻轻屈膝再站直（共2秒）。循环到结束提示音。全身始终留在两路画面内。'
    sync_horizontal_1 = '同步测试1：站在刚才能看到全身的靠后位置，身体不动。棋盘竖直放在胸前、长边水平、朝向两摄中间，双手握边框。开始提示音后，整块板向自己左边平移约20厘米（约1秒），停1秒；再向右移动约40厘米（约1秒），停2秒；回中间。循环到结束提示音，不旋转棋盘。'
    sync_vertical_2 = '同步测试2：保持相同靠后站位。棋盘竖直放在胸前、长边水平、朝向两摄中间。开始提示音后，整块板向上平移约20厘米（约1秒），停1秒；再向下移动约40厘米到腹部（约1秒），停2秒；回胸前。循环到结束提示音，不前后倾斜棋盘。'
    sync_horizontal_3 = '同步测试3：保持相同靠后站位。棋盘竖直放在胸前、长边水平、朝向两摄中间。开始提示音后，整块板向自己右边平移约20厘米（约1秒），停2秒；再向左移动约40厘米（约1秒），停1秒；回中间。循环到结束提示音，不旋转棋盘。' 
    near_left = '向两只摄像头中间靠近约半米到一米；棋盘在胸前，长边横放，拿板时左手一侧向身体靠近，绕竖直方向转约20度，整段保持。'
    near_right = '保持与上一段相同的近处站位；棋盘在胸前，长边横放，改为右手一侧向身体靠近，绕竖直方向反向转约20度，整段保持。'
    near_yaw_left = '保持这次靠近后的站位。棋盘长边水平，双手同高；右手那一边留在原处，把左手握住的边缘往胸前收约15厘米，使两边前后错开。不要把一只手抬高、另一只手放低；整段保持。'
    near_yaw_right = '保持同一近处站位，先把棋盘恢复正对前方。长边水平，双手同高；左手那一边留在原处，把右手握住的边缘往胸前收约15厘米，使两边前后反向错开。不要把一只手抬高、另一只手放低；整段保持。'
    near_pitch_top = '保持这次近处站位，先把棋盘恢复朝向两只摄像头中间，长边水平。双手握左右边框中部，把上边沿向胸口倾、下边沿向前送，使上边沿比下边沿靠近身体约10厘米（约20度）。这是上下边缘前后错开，整块板仍在胸前；整段保持。'
    near_pitch_bottom = '保持同一近处站位，先把棋盘恢复竖直、朝向两只摄像头中间，长边水平。双手握左右边框中部，把下边沿向胸口收、上边沿向前送，使下边沿比上边沿靠近身体约10厘米（约20度）。这是上下边缘反向前后错开，整块板仍在胸前；整段保持。'
    closer_yaw_left_mild = '站在刚才走近40厘米后的中间位置，身体朝向两摄中间，不跟着棋盘转身。先把棋盘竖直放在胸前、长边水平、正面朝向两摄中间。双手同高，右手不动，只把左手握住的边缘向身体收约5厘米；轻轻转一点即可，两路都要看到完整棋盘。保持到结束声。'
    closer_yaw_right_mild = '保持同一个更近的中间站位，身体朝向两摄中间，不跟着棋盘转身。先把棋盘恢复竖直、长边水平、正面朝向两摄中间。双手同高，左手不动，只把右手握住的边缘向身体收约5厘米；轻轻转一点即可，两路都要看到完整棋盘。保持到结束声。'
    validation_far_pitch_top = '独立验证第5段：从刚才验证组的中间站位向后退约50厘米。棋盘放在胸前、长边水平，先正对两摄中间，再把上边沿向身体倾、下边沿向前送，让上下边沿前后错开约5厘米。整段保持；仅验证，不参与拟合。'
    world_ground_center = '地面参考：先在预览里确认棋盘平放在本次验证组中间站位脚前约20厘米的地面，两路均看见完整棋盘。图案朝上、长边左右横放，一条长边朝摄像头，摆平不弯曲。按Q关闭预览后运行本段。输入START后退出棋盘所在区域，不遮挡棋盘、不移动摄像头。只建立世界坐标，不加入内外参拟合。'
    validation_mid_center = '独立验证第1段：从刚才较远的中间站位，向两只摄像头中间走近约50厘米。棋盘竖直放在胸前，长边水平、正面朝向两摄中间。记住这个新的中间站位；本组只用于检验标定，不加入拟合。'
    validation_mid_left_high = '独立验证第2段：从本组新的中间站位，向你自己的左边侧移约25厘米，把棋盘从胸前举高约20厘米，到下巴附近。保持竖直、长边水平、正面朝向两摄中间，两路看见完整棋盘。'
    validation_mid_right_low = '独立验证第3段：先回到本组新的中间站位，再向你自己的右边侧移约25厘米，把棋盘放到腹部，比胸前低约20厘米。保持竖直、长边水平、正面朝向两摄中间。'
    validation_mid_yaw = '独立验证第4段：回到本组新的中间站位，棋盘恢复胸前、竖直、长边水平、正面朝向两摄中间。身体不转、双手同高；右手不动，只把左手握住的边缘向身体收约5厘米，整段保持。'
    farther_center = '从刚才两段补拍的位置，沿两摄中间方向向后退约1米，离摄像头更远。身体正对两摄中间，棋盘竖直放在胸前、长边水平、正面朝向两摄中间；两路都要看见完整棋盘。记住这个较远的位置，后面三段都站这里。'
    farther_yaw_left = '保持刚才后退1米后的较远站位，身体不转。棋盘先恢复竖直、长边水平、正面朝向两摄中间；双手同高，右手不动，只把左手握住的边缘向身体收约5厘米，轻轻转一点，整段保持。'
    farther_yaw_right = '保持同一较远站位，身体不转。棋盘先恢复竖直、长边水平、正面朝向两摄中间；双手同高，左手不动，只把右手握住的边缘向身体收约5厘米，反向轻轻转一点，整段保持。'
    farther_pitch_bottom = '保持同一较远站位，先把棋盘恢复竖直、长边水平、正面朝向两摄中间。棋盘仍在胸前，把下边沿向身体收、上边沿向前送，让上下边沿前后错开约5厘米；不要把整块板降低，整段保持。'
    closer_center = '先回到上组两摄中间的站位，再向两只摄像头中间走近约40厘米。棋盘竖直放在胸前，长边水平，正面朝向两摄中间；两路必须看见完整棋盘，整段保持。记住这个更近的位置，后面三段都站在这里。'
    closer_yaw_left = '保持刚才走近40厘米后的站位。棋盘在胸前，长边水平，双手同高；右手那边留在原处，把左手握住的边缘向胸口收约10厘米，让左右两边前后错开，像开门；不要一只手高一只手低，整段保持。'
    closer_yaw_right = '保持同一个更近的站位，先把棋盘恢复正对两摄中间。长边水平，双手同高；左手那边留在原处，把右手握住的边缘向胸口收约10厘米，反向像开门；不要一只手高一只手低，整段保持。'
    closer_pitch_top = '保持同一个更近的站位，先把棋盘恢复竖直、朝向两摄中间、长边水平。棋盘仍在胸前，把上边沿向胸口倾、下边沿向前送，让上边沿比下边沿靠近身体约10厘米。不是把整块板举高，整段保持。'
    near_position_left = '从刚才中间的近处站位，向你自己的左边侧移约30厘米。棋盘竖直放在胸前，长边水平，正面朝向两只摄像头中间；整段保持，两路都要看见完整棋盘。'
    near_position_right = '先回到刚才中间的近处站位，再向你自己的右边侧移约30厘米。棋盘竖直放在胸前，长边水平，正面朝向两只摄像头中间；整段保持，两路都要看见完整棋盘。'
    near_position_high = '回到刚才中间的近处站位，把整块棋盘从胸前向上举约25厘米，到下巴至脸部高度。棋盘保持竖直、长边水平、正面朝向两只摄像头中间；挡住脸没有关系，两路都要看见完整棋盘，整段保持。'
    near_position_low = '保持中间的近处站位，把整块棋盘放到比原来胸前位置低约30厘米的腹部位置。棋盘保持竖直、长边水平、正面朝向两只摄像头中间；两路都要看见完整棋盘，整段保持。'
}

$CaptureBody = $Pose -eq 'body_motion_baseline'
$CaptureSync = $Pose.StartsWith('sync_')
$CaptureDynamic = $CaptureSync -or $CaptureBody
$CaptureFrames = if ($CaptureDynamic) { 300 } else { 150 }
$CaptureHeldOut = $Pose.StartsWith('validation_')
$CaptureExcludedFromFit = $CaptureDynamic -or $CaptureHeldOut -or ($Pose.StartsWith('world_ground_'))
$CapturePurpose = if ($CaptureBody) { 'stereo_body_motion_baseline' } elseif ($CaptureSync) { 'stereo_content_synchronization_trial' } elseif ($CaptureHeldOut) { 'stereo_calibration_held_out_validation' } elseif ($Pose.StartsWith('world_ground_')) { 'stereo_world_reference' } else { 'stereo_calibration_fit_candidate' }
Write-Host "本轮姿态：$Pose；每路$CaptureFrames 帧，拍摄约$($CaptureFrames / 30)秒。" -ForegroundColor Cyan
Write-Host $PoseInstructions[$Pose]
if ($CaptureBody) { Write-Host '不用棋盘，两路必须能看到全身及鞋，手臂不要抬到水平或头顶；摄像头位置不动。' } else { Write-Host '两路必须都能看到完整棋盘，手握边框，避免反光；摄像头位置不动。' }
Write-Host '先关闭实时预览和占用COM22的串口终端。输入START后有10秒就位。'
if ($CaptureDynamic) { Write-Host '开始提示音后持续重复上述移动和停顿，直到结束提示音；准备和回传也会占用时间。不要只做一遍。' } else { Write-Host '保持到结束声。' }
$CaptureReady = Read-Host '准备好请输入 START；其他输入不采集'
if ($CaptureReady -cne 'START') {
    Write-Host '没有开始采集。'
    return
}
for ($CaptureCountdown = 10; $CaptureCountdown -ge 1; $CaptureCountdown--) {
    if ($CaptureBody) { Write-Host "请站到靠后中间位置，倒计时 $CaptureCountdown 秒" } else { Write-Host "请拿棋盘就位，倒计时 $CaptureCountdown 秒" }
    Start-Sleep -Seconds 1
}
[Console]::Beep(900, 250)
if ($CaptureDynamic) { Write-Host '现在开始反复做本段动作，直到结束提示音。正在准备、拍摄和回传。' -ForegroundColor Cyan } else { Write-Host '请保持当前棋盘姿态不动，正在准备、拍摄和回传。' -ForegroundColor Cyan }
$CaptureLines = @()
& $CapturePython "$CaptureDemo\pc\demo.py" capture-calibration --serial COM22 --board 192.168.137.2 --frames $CaptureFrames | Tee-Object -Variable CaptureLines
$CaptureExitCode = $LASTEXITCODE
if ($CaptureExitCode -eq 0) {
    $CaptureResults = @($CaptureLines | Where-Object { $_ -like 'RESULTS *' })
    if ($CaptureResults.Count -eq 1) {
        $CaptureResultPath = $CaptureResults[0].Substring(8).Trim()
        $CaptureRegistry = Join-Path $PSScriptRoot 'runs\20261014_chessboard_setup\capture_registry'
        New-Item -ItemType Directory -Path $CaptureRegistry -Force | Out-Null
        $CaptureNote = [ordered]@{
            source = $CaptureResultPath
            purpose = $CapturePurpose
            excluded_from_calibration_fit = $CaptureExcludedFromFit
            requested_pose = $Pose
            requested_pose_description = $PoseInstructions[$Pose]
            actual_pose_verified = $false
            board_kind = 'checkerboard'
            inner_corners = @(9, 6)
            square_m = 0.04
            frames_requested_per_camera = $CaptureFrames
            capture_command_exit_code = $CaptureExitCode
            exposure_sync_verified = $false
            dynamic_motion_requested = $CaptureDynamic
        }
        if ($CaptureBody) {
            foreach ($CaptureBoardField in @('board_kind', 'inner_corners', 'square_m')) { $CaptureNote.Remove($CaptureBoardField) }
            $CaptureNote['pose_inference_performed'] = $false
        }
        $CaptureNotePath = Join-Path $CaptureRegistry ((Split-Path -Leaf $CaptureResultPath) + '.json')
        [System.IO.File]::WriteAllText($CaptureNotePath, ($CaptureNote | ConvertTo-Json -Depth 4), [System.Text.UTF8Encoding]::new($false))
    } else {
        Write-Warning '拍摄程序已成功，但未提取到唯一结果目录。请保留输出，不要为了登记而重拍。'
    }
    Write-Host '本段已拍完并回传，可以回来。下一段由你另行运行命令启动。' -ForegroundColor Green
} else {
    Write-Host '本段未正常完成，请保留报错，先不要拍下一段。' -ForegroundColor Yellow
}
[Console]::Beep(1200, 800)
exit $CaptureExitCode
