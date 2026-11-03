#!/usr/bin/env bash
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
stamp="$(date +%s%N)"
work="$base/module_build_dual_$stamp"
reports="$base/../../reports/stream_dual_module_$stamp"
mkdir -p "$work" "$reports"
fixture="$base/../../reports/stream_fixtures_1791657207786097100/fixture_0.json.fixture"
printf '%s  %s\n' bc995b679066df250bfd2b8254d9b0b7e2f5408d648225fdafc2ac1c991c36a8 "$fixture" | sha256sum -c -
for kind in dual resident; do
    g++ -std=c++11 -O2 -Wall -Wextra "$base/../../inference/host/test_stream_${kind}_validation.cpp" \
        -o "$reports/test_stream_${kind}_validation" 2>&1 | tee "$reports/${kind}_compile.log"
    "$reports/test_stream_${kind}_validation" "$fixture" | tee "$reports/${kind}_validation.log"
done
g++ -std=c++11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
    "$base/../../inference/host/test_stream_dual_validation.cpp" -o "$reports/test_stream_dual_sanitized"
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
    "$reports/test_stream_dual_sanitized" "$fixture" | tee "$reports/dual_sanitized.log"
cp "$base/u02_stream_dual.c" "$base/u02_stream_dual_api.h" \
   "$base/u02_stream_resident.c" "$base/u02_stream_resident_api.h" \
   "$base/u02_stream_resident_validation.h" "$base/u02_stream_api.h" "$base/u02_pose_api.h" "$work/"
printf '%s\n' 'obj-m := u02_stream_dual.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules 2>&1 | tee "$reports/build.log"
cp "$work/u02_stream_dual.ko" "$reports/"
cp "$work/"*.c "$work/"*.h "$reports/"
cp "$base/build_stream_dual_module.sh" "$reports/"
cp "$base/../../inference/host/test_stream_dual_validation.cpp" \
   "$base/../../inference/host/test_stream_resident_validation.cpp" "$reports/"
modinfo "$reports/u02_stream_dual.ko" | tee "$reports/modinfo.txt"
sha256sum "$reports/u02_stream_dual.ko" | tee "$reports/module.sha256"
printf 'STREAM_DUAL_MODULE %s\n' "$reports"
