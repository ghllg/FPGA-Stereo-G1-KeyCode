#!/usr/bin/env bash
# Build/test isolated ABI8 only. Never install/load a module or access a board.
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
stamp="$(date +%s%N)"
work="$base/module_build_dual_fp32_$stamp"
reports="$base/../../reports/stream_dual_fp32_module_$stamp"
mkdir -p "$work" "$reports"
fixture="$base/../../reports/fp32_output_fixtures_1791794443783245700"
for name in u02_stream_dual_fp32.c u02_stream_dual_fp32_api.h u02_stream_dual_api.h \
    u02_stream_fp32.c u02_stream_fp32_api.h u02_stream_fp32_validation.h \
    u02_stream_resident_api.h u02_stream_api.h u02_pose_api.h; do
    cp "$base/$name" "$work/"
    cp "$base/$name" "$reports/"
done
cp "$base/test_stream_fp32_validation.c" "$reports/"
cp "$base/prepare_dual_fp32_test.py" "$base/build_stream_dual_fp32_module.sh" "$reports/"
python3 "$base/prepare_dual_fp32_test.py" --output "$reports/test_stream_dual_fp32_validation.c"
gcc -std=c11 -O2 -Wall -Wextra -I"$work" "$reports/test_stream_fp32_validation.c" -o "$reports/test_legacy" 2>&1 | tee "$reports/legacy_compile.log"
"$reports/test_legacy" "$fixture" | tee "$reports/legacy_validation.log"
gcc -std=c11 -O2 -Wall -Wextra -I"$work" "$reports/test_stream_dual_fp32_validation.c" -o "$reports/test_dual" 2>&1 | tee "$reports/dual_compile.log"
"$reports/test_dual" "$fixture" | tee "$reports/dual_validation.log"
gcc -std=c11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I"$work" \
    "$reports/test_stream_dual_fp32_validation.c" -o "$reports/test_dual_sanitized"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$reports/test_dual_sanitized" "$fixture" | tee "$reports/dual_sanitized.log"
printf '%s\n' 'obj-m := u02_stream_dual_fp32.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j2 modules 2>&1 | tee "$reports/build.log"
cp "$work/u02_stream_dual_fp32.ko" "$reports/"
modinfo "$reports/u02_stream_dual_fp32.ko" | tee "$reports/modinfo.txt"
sha256sum "$reports/u02_stream_dual_fp32.ko" | tee "$reports/module.sha256"
printf 'STREAM_DUAL_FP32_MODULE %s\n' "$reports"
