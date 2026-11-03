$ErrorActionPreference = 'Stop'
$PreviewCompiler = 'E:\FPGAchina\Xilinx\Vitis\2020.2\gnu\aarch64\nt\aarch64-linux\bin\aarch64-linux-gnu-g++.exe'
$CaptureSource = 'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000\src\ultrascale_plus\u02\inference\host'
$PreviewBuild = Join-Path $PSScriptRoot '.build'
New-Item -ItemType Directory -Path $PreviewBuild -Force | Out-Null
# The vendor compiler wrapper splits paths containing spaces. Copy only these
# two unchanged source files to our build directory; preserve the demo package.
Copy-Item -LiteralPath "$CaptureSource\camera_capture.h", "$CaptureSource\camera_capture.cpp" -Destination $PreviewBuild -Force
& $PreviewCompiler -std=c++14 -O2 -Wall -Wextra -pthread -s -march=armv8-a -mtune=cortex-a53 "-I$PreviewBuild" "$PSScriptRoot\board_preview.cpp" "$PreviewBuild\camera_capture.cpp" -o "$PSScriptRoot\board_preview"
if ($LASTEXITCODE -ne 0) { throw 'Board preview build failed.' }
$PreviewSysroot = 'E:\FPGAchina\Xilinx\Vitis\2020.2\gnu\aarch64\nt\aarch64-linux\aarch64-xilinx-linux'
$PreviewRuntime = Join-Path $PSScriptRoot 'runtime'
New-Item -ItemType Directory -Path $PreviewRuntime -Force | Out-Null
# The recovery root filesystem has no shared libraries. Keep this runtime private
# to our temporary preview folder; no board system libraries are installed.
$PreviewLibraries = @{
    'lib\ld-2.30.so' = 'ld-linux-aarch64.so.1'
    'lib\libc-2.30.so' = 'libc.so.6'
    'lib\libm-2.30.so' = 'libm.so.6'
    'lib\libpthread-2.30.so' = 'libpthread.so.0'
    'lib\libgcc_s.so.1' = 'libgcc_s.so.1'
    'usr\lib\libstdc++.so.6.0.27' = 'libstdc++.so.6'
}
foreach ($Library in $PreviewLibraries.GetEnumerator()) {
    Copy-Item -LiteralPath (Join-Path $PreviewSysroot $Library.Key) -Destination (Join-Path $PreviewRuntime $Library.Value) -Force
}
Get-FileHash -LiteralPath "$PSScriptRoot\board_preview" -Algorithm SHA256
