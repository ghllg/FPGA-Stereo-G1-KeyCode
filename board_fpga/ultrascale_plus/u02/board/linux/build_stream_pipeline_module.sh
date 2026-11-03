#!/usr/bin/env bash
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
stamp="$(date +%s%N)"
work="$base/module_build_pipeline_$stamp"
reports="$base/../../reports/stream_pipeline_module_$stamp"
fixture="$base/../../milestones/stream_hls_csim/stream_fixtures_1791657207786097100/fixture_0.json.fixture"
mkdir -p "$work" "$reports"
printf '%s  %s\n' bc995b679066df250bfd2b8254d9b0b7e2f5408d648225fdafc2ac1c991c36a8 "$fixture" | sha256sum -c -
for kind in shared_packet_pool pipeline_validation; do
    source="$base/../../inference/host/test_stream_${kind}.cpp"
    g++ -std=c++11 -O2 -Wall -Wextra -pthread "$source" -o "$reports/test_$kind" 2>&1 | tee "$reports/${kind}_compile.log"
    "$reports/test_$kind" "$fixture" | tee "$reports/${kind}_validation.log"
    g++ -std=c++11 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer "$source" -o "$reports/test_${kind}_sanitized"
    ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 "$reports/test_${kind}_sanitized" "$fixture" | tee "$reports/${kind}_sanitized.log"
done
for name in u02_stream_pipeline.c u02_stream_pipeline_api.h u02_stream_resident.c u02_stream_resident_api.h u02_stream_resident_validation.h u02_stream_api.h u02_pose_api.h; do
    cp "$base/$name" "$work/"
done
printf '%s\n' 'obj-m := u02_stream_pipeline.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules 2>&1 | tee "$reports/build.log"
cp "$work/u02_stream_pipeline.ko" "$reports/"
modinfo "$reports/u02_stream_pipeline.ko" | tee "$reports/modinfo.txt"
sha256sum "$reports/u02_stream_pipeline.ko" | tee "$reports/module.sha256"
printf 'STREAM_PIPELINE_MODULE %s\n' "$reports"
