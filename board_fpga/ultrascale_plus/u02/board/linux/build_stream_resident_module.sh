#!/usr/bin/env bash
# External module only. Use the already prepared Linux 6.6.40 kernel build.
# No kernel/image rebuild, startup changes, installs or legacy output overwrite.
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
work="$base/module_build_resident"
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
test -f "$kernel_out/include/generated/autoconf.h"
mkdir -p "$work"
cp "$base/u02_stream_resident.c" "$base/u02_stream_resident_api.h" \
   "$base/u02_stream_resident_validation.h" "$base/u02_stream_api.h" "$base/u02_pose_api.h" "$work/"
printf '%s\n' 'obj-m := u02_stream_resident.o' > "$work/Makefile"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules
reports="$base/../../reports/stream_resident_guard_module"
mkdir -p "$reports"
test ! -e "$reports/u02_stream_resident.ko"
cp "$work/u02_stream_resident.ko" "$reports/"
modinfo "$reports/u02_stream_resident.ko"
sha256sum "$reports/u02_stream_resident.ko"
