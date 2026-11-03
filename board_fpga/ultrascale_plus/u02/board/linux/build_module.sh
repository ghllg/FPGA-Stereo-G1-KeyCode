#!/usr/bin/env bash
# External module only, using the existing running-kernel build. No kernel/image rebuild.
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
out=/home/gby/axu5evb_e_boot/kernel-build
work="$base/module_build"
test -f "$out/vmlinux.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$out/include/generated/utsrelease.h"
mkdir -p "$work"
cp "$base/u02_ddr_test.c" "$base/u02_rpll_restore.c" "$base/u02_test_api.h" "$base/Makefile" "$work/"
cp "$base/u02_pose.c" "$base/u02_pose_api.h" "$work/"
cp "$base/u02_stream.c" "$base/u02_stream_api.h" "$base/u02_stream_validation.h" "$work/"
# Image-only builds have vmlinux.symvers; external modpost expects this filename.
if [ ! -f "$out/Module.symvers" ]; then cp "$out/vmlinux.symvers" "$out/Module.symvers"; fi
make -C "$out" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules_prepare
make -C "$out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules
cp "$work/u02_ddr_test.ko" "$base/../../reports/"
cp "$work/u02_rpll_restore.ko" "$base/../../reports/"
cp "$work/u02_pose.ko" "$base/../../reports/"
cp "$work/u02_stream.ko" "$base/../../reports/"
modinfo "$work/u02_ddr_test.ko"
sha256sum "$work/u02_ddr_test.ko"
