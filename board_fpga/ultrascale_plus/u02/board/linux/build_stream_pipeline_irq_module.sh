#!/usr/bin/env bash
set -euo pipefail
base="$(cd "$(dirname "$0")" && pwd)"
kernel_out=/home/gby/axu5evb_e_boot/kernel-build
grep -q 'UTS_RELEASE "6.6.40"' "$kernel_out/include/generated/utsrelease.h"
stamp="$(date +%s%N)"
work="$base/module_build_irq_$stamp"
report="$base/../../reports/stream_irq_module_$stamp"
mkdir -p "$report"
cp "$base/prepare_stream_pipeline_irq.py" "$report/executed_generator.py"
cp "$base/build_stream_pipeline_irq_module.sh" "$report/executed_build.sh"
python3 "$base/prepare_stream_pipeline_irq.py" --output "$work" > "$report/generate.log"
make -C "$kernel_out" M="$work" ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- modules 2>&1 | tee "$report/build.log"
cp "$work/u02_stream_pipeline_irq.ko" "$report/"
cp "$work/transformation.json" "$report/"
modinfo "$report/u02_stream_pipeline_irq.ko" | tee "$report/modinfo.txt"
sha256sum "$report/u02_stream_pipeline_irq.ko" | tee "$report/module.sha256"
printf 'STREAM_IRQ_MODULE %s\n' "$report"
