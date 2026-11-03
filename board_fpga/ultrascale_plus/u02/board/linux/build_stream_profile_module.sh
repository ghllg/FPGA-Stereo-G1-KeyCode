#!/usr/bin/env bash
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
stamp="$(date +%s%N)"
work="$base/module_build_profile_$stamp"
reports="$base/../../reports/stream_sync_profile_module_$stamp"
mkdir -p "$work" "$reports"
cp "$base/u02_stream_resident_profile.c" "$base/u02_stream_resident.c" \
   "$base/u02_stream_resident_api.h" "$base/u02_stream_resident_validation.h" \
   "$base/u02_stream_api.h" "$base/u02_pose_api.h" "$work/"
printf '%s\n' 'obj-m := u02_stream_resident_profile.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules 2>&1 | tee "$reports/build.log"
cp "$work/u02_stream_resident_profile.ko" "$reports/"
cp "$work/"*.c "$work/"*.h "$reports/"
modinfo "$reports/u02_stream_resident_profile.ko" | tee "$reports/modinfo.txt"
sha256sum "$reports/u02_stream_resident_profile.ko" | tee "$reports/module.sha256"
printf 'STREAM_SYNC_PROFILE_MODULE %s\n' "$reports"
