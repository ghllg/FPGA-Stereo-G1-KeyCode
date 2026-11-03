#!/usr/bin/env bash
# Compile an external candidate module only; never install or load it.
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
work="$base/module_build_fp32"
reports="$base/../../reports/stream_fp32_candidate_module"
test -s "$kernel_out/Module.symvers"
test -f "$kernel_out/include/generated/autoconf.h"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
test ! -e "$work"
test ! -e "$reports"
mkdir -p "$work" "$reports"
cp "$base/u02_stream_fp32.c" "$base/u02_stream_fp32_api.h" \
   "$base/u02_stream_fp32_validation.h" "$base/u02_stream_resident_api.h" \
   "$base/u02_stream_api.h" "$base/u02_pose_api.h" "$work/"
printf '%s\n' 'obj-m := u02_stream_fp32.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j2 modules 2>&1 | tee "$reports/build.log"
cp "$work/u02_stream_fp32.ko" "$reports/"
modinfo "$reports/u02_stream_fp32.ko" | tee "$reports/modinfo.txt"
sha256sum "$reports/u02_stream_fp32.ko" | tee "$reports/SHA256SUMS"
