"""Generate an isolated IRQ module from exact reviewed polling-driver anchors.

The original .c and previously frozen .ko files are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

POLL_SOURCE_SHA256 = '710b63664a97a4d6f6fcc3960b8ac166f923597ddae95d292080edc60b5359a9'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def generate(source):
    assert hashlib.sha256(source.encode()).hexdigest()==POLL_SOURCE_SHA256, 'Polling baseline changed; IRQ derivation needs review'
    changes=[]
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
        changes.append(dict(old=old,new=new))
    original=source
    replace('#include "u02_stream_resident_validation.h"',
            '#include "u02_stream_resident_validation.h"\n#include "u02_stream_pipeline_irq_completion.h"')
    replace('    u64 start, deadline, spin_deadline, now;', '    u64 start, deadline;')
    replace('        else { dma_rmb(); finish(); }',
            '        else { rc = u02_irq_recover(); if (!rc) { dma_rmb(); finish(); } }')
    replace('    if ((ctrl & 1) || !(ctrl & 4)) { rc = -EBUSY; goto out; }\n    dma_wmb();',
            '    if ((ctrl & 1) || !(ctrl & 4)) { rc = -EBUSY; goto out; }\n'
            '    rc = u02_irq_prepare();\n    if (rc) goto out;\n    dma_wmb();')
    polling='''    spin_deadline = min(deadline, start + (u64)poll_spin_us * 1000);
    writel(1, regs);
    for (;;) {
        ctrl = readl(regs); ++j.polls;
        if (ctrl & 2) break;
        now = ktime_get_ns();
        if (now >= deadline) { rc = -ETIMEDOUT; break; }
        if (j.timeout_us >= 1000 && j.polls > 64 && now >= spin_deadline)
            usleep_range(10, 20);
        else cpu_relax();
    }'''
    replace(polling,'    rc = u02_irq_start_wait(deadline);')
    replace('    if (!bridge_checked) return -EPERM;',
            '    if (!bridge_checked || !irq_route_checked) return -EPERM;')
    replace('MODULE_PARM_DESC(poll_spin_us, "Optional initial active-poll window, 0..1000 us; deadline and timeout ownership unchanged");',
            'MODULE_PARM_DESC(poll_spin_us, "Compatibility value only; isolated IRQ variant never polls");')
    replace('MODULE_DESCRIPTION("U02 fused TinyPose streaming transport using guarded resident packet starts in kernel-owned DMA");',
            'MODULE_DESCRIPTION("U02 ABI9 exclusive IRQ completion; guarded kernel DMA and timeout ownership retained");')
    replace('    device.parent = &pdev->dev;',
            '    rc = u02_irq_init();\n    if (rc) goto mapping;\n'
            '    device.groups = u02_irq_groups;\n    device.parent = &pdev->dev;')
    replace('    rc = misc_register(&device);\n    if (rc) goto mapping;',
            '    rc = misc_register(&device);\n    if (rc) goto irq_cleanup;')
    replace('mapping: iounmap(regs);','irq_cleanup: u02_irq_free();\nmapping: iounmap(regs);')
    replace('    misc_deregister(&device); iounmap(regs);',
            '    misc_deregister(&device); u02_irq_free(); iounmap(regs);')
    # Byte-exact preservation of DMA/cache ownership and userspace mapping.
    preserved={}
    for begin,end in (
        ('static void sync_job(', '#include "u02_stream_resident_validation.h"'),
        ('static int pose_open(', 'static const struct file_operations'),
    ):
        a=original[original.index(begin):original.index(end)]
        b=source[source.index(begin):source.index(end)]
        assert a==b,begin
        preserved[begin]=hashlib.sha256(a.encode()).hexdigest()
    prefix='''/* Generated isolated IRQ variant; see transformation.json for exact changes. */
#include "u02_stream_pipeline_api.h"
#define U02_RESIDENT_DEVICE_NAME "u02_stream_pipeline"
#define U02_RESIDENT_ABI U02_STREAM_PIPELINE_ABI
#define U02_RESIDENT_SLOTS_VALID(j) u02_stream_pipeline_slots_valid(j)
'''
    return prefix+source,dict(changes=changes,preserved_blocks_sha256=preserved)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    args=p.parse_args();base=Path(__file__).resolve().parent
    output=args.output or base/f'module_build_irq_{time.time_ns()}'
    output.mkdir(exist_ok=False,parents=True)
    source=base/'u02_stream_resident.c'
    text,audit=generate(source.read_text(encoding='utf-8'))
    (output/'u02_stream_pipeline_irq.c').write_text(text,encoding='utf-8')
    files=('u02_stream_pipeline_irq_completion.h','u02_stream_pipeline_api.h','u02_stream_resident_api.h',
           'u02_stream_resident_validation.h','u02_stream_api.h','u02_pose_api.h')
    for name in files:shutil.copyfile(base/name,output/name)
    (output/'Makefile').write_text('obj-m := u02_stream_pipeline_irq.o\n',encoding='ascii')
    audit.update(source=str(source),source_sha256=sha(source),generator_sha256=sha(Path(__file__)),
        generated_sha256=sha(output/'u02_stream_pipeline_irq.c'),
        dependencies_sha256={name:sha(output/name) for name in files},board_accessed=False)
    (output/'transformation.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(output,flush=True)


if __name__=='__main__':main()
