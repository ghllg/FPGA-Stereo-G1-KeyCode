#!/usr/bin/env bash
# Build an already frozen candidate snapshot. No installation or board access.
set -euo pipefail
record="$1"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
test -s "$kernel_out/Module.symvers"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
grep -q '^CONFIG_ARM64=y$' "$kernel_out/.config"
test -s "$record/source/u02_region_dual.c"
test -s "$record/source/source_manifest.json"
test ! -e "$record/build.log"
sha256sum "$record"/source/* > "$record/source_hashes.txt"
make -C "$kernel_out" M="$record/source" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j2 modules 2>&1 | tee "$record/build.log"
cp "$record/source/u02_region_dual.ko" "$record/"
modinfo "$record/u02_region_dual.ko" | tee "$record/modinfo.txt"
sha256sum "$record/u02_region_dual.ko" > "$record/SHA256SUMS"
printf '%s\n' U02_REGION_DUAL_MODULE_COMPILED_NOT_LOADED
