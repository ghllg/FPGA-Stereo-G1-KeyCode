param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskOutput = Join-Path $PSScriptRoot '..\reports\pagefile_configuration'
$taskOutput = [IO.Path]::GetFullPath($taskOutput)
$taskSettings = @(Get-CimInstance Win32_PageFileSetting)
$taskUsage = @(Get-CimInstance Win32_PageFileUsage)
$taskSystem = Get-CimInstance Win32_ComputerSystem
$taskDisk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='D:'"
$taskPlan = [ordered]@{
    apply = [bool]$Apply
    preserve_c = $true
    update_existing_d_setting_only = $true
    target = 'D:\pagefile.sys'
    initial_mib = 32768
    maximum_mib = 32768
    d_free_bytes = $taskDisk.FreeSpace
    reboot_automatically = $false
    existing_settings = @($taskSettings | Select-Object Name,InitialSize,MaximumSize)
    currently_active = @($taskUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage)
    automatic_managed = $taskSystem.AutomaticManagedPagefile
}
$taskPlan | ConvertTo-Json -Depth 5
if (-not $Apply) { exit 0 }
$taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskPrincipal = New-Object Security.Principal.WindowsPrincipal($taskIdentity)
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator required' }
if ($taskDisk.FreeSpace -lt 48GB) { throw 'Need 48 GiB free on D to retain build space at maximum paging size' }
if ($taskSystem.AutomaticManagedPagefile) { throw 'Automatic management changed; review current settings first' }
$taskD = @($taskSettings | Where-Object Name -ieq 'D:\pagefile.sys')
if ($taskD.Count -ne 1 -or $taskD[0].InitialSize -ne 32768 -or $taskD[0].MaximumSize -ne 65536) { throw 'D configuration changed; review again' }
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskBackup = Join-Path $taskOutput ('before_' + [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fff') + '.json')
$taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $taskBackup -Encoding UTF8
# User selected D. Bound it to 32 GiB, leaving roughly 23 GiB on D.
$taskD[0] | Set-CimInstance -Property @{InitialSize=[uint32]32768; MaximumSize=[uint32]32768}
$taskAfter = @(Get-CimInstance Win32_PageFileSetting | Select-Object Name,InitialSize,MaximumSize)
$taskAfter | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath ($taskBackup + '.after.json') -Encoding UTF8
$taskAfter | ConvertTo-Json -Depth 4
Write-Output 'Configuration saved. No restart requested. Verify active pagefile and commit limit after an explicitly authorized PC restart.'
