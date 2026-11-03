#!/usr/bin/env bash
# Build only against the existing 6.6.40 tree. No install, insmod or board access.
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
record="$1"
test -s "$kernel_out/Module.symvers"
test -f "$kernel_out/include/generated/autoconf.h"
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
grep -q '^CONFIG_ARM64=y$' "$kernel_out/.config"
test ! -e "$record"
mkdir -p "$record/source"
cp "$base/u02_region.c" "$base/u02_region_api.h" "$base/u02_region_hw.h" \
   "$base/u02_region_model_validation.h" "$base/u02_region_program_validation.h" \
   "$base/u02_region_binding_validation.h" "$base/u02_region_session.h" \
   "$base/u02_region_linux_memory.h" "$record/source/"
cp "$base/Makefile.region" "$record/source/Makefile"
sha256sum "$record"/source/* > "$record/source_hashes.txt"
make -C "$kernel_out" M="$record/source" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- -j2 modules 2>&1 | tee "$record/build.log"
cp "$record/source/u02_region.ko" "$record/"
modinfo "$record/u02_region.ko" | tee "$record/modinfo.txt"
sha256sum "$record/u02_region.ko" > "$record/SHA256SUMS"
printf '%s\n' U02_REGION_KERNEL_MODULE_COMPILED_NOT_LOADED
