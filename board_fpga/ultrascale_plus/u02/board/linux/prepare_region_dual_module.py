"""Derive a separate dual candidate module without changing the ABI5 source."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--register-header',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.output_dir.exists():raise ValueError('Fresh dual module snapshot required')
    args.output_dir.mkdir(parents=True)
    source=(root/'u02_region.c').read_text()
    def replace(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('ABI5 derivation anchor changed: '+old)
        source=source.replace(old,new)
    replace('#include "u02_region_hw.h"','#include "u02_region_hw.h"\n#include "u02_region_dual_api.h"')
    replace('static long region_ioctl(', '#include "u02_region_dual_legacy.h"\nstatic long region_ioctl(')
    replace('command!=U02_REGION_RUN && command!=U02_REGION_RECOVER)',
            'command!=U02_REGION_RUN && command!=U02_REGION_RECOVER && command!=U02_REGION_DUAL_LEGACY_RUN)')
    replace('if (command==U02_REGION_INFO) {',
            'if (command==U02_REGION_DUAL_LEGACY_RUN) {status=dual_legacy_ioctl(argument);goto out;}\n    if (command==U02_REGION_INFO) {')
    replace('.abi=U02_REGION_ABI,','.abi=U02_REGION_DUAL_ABI,')
    replace('regs=ioremap(U02_REGION_BASE,U02_REGION_MMIO_BYTES);',
            'status=dual_allocate_private();if (status) goto dma;\n    regs=ioremap(U02_REGION_BASE,U02_REGION_MMIO_BYTES);')
    replace('dma:dma_free_coherent(', 'dma:dual_release_private();dma_free_coherent(')
    replace('iounmap(regs);dma_free_coherent(', 'dual_release_private();iounmap(regs);dma_free_coherent(')
    source=source.replace('"u02_region"','"u02_region_dual"').replace('private configuration ABI5','private configuration ABI6')
    (args.output_dir/'u02_region_dual.c').write_text(source)
    (args.output_dir/'Makefile').write_text('obj-m := u02_region_dual.o\n')
    dependencies=['u02_region_api.h','u02_region_dual_api.h','u02_region_dual_legacy.h',
                  'u02_region_dual_packet_validation.h','u02_region_model_validation.h',
                  'u02_region_program_validation.h','u02_region_binding_validation.h',
                  'u02_region_session.h','u02_region_linux_memory.h']
    for name in dependencies:(args.output_dir/name).write_bytes((root/name).read_bytes())
    (args.output_dir/'u02_region_hw.h').write_bytes(args.register_header.read_bytes())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof=dict(board_accessed=False,compiled=False,loaded=False,
        source_sha256={str(p):digest(p) for p in [root/'u02_region.c',Path(__file__),args.register_header,
                                                *[root/name for name in dependencies]]},
        generated_sha256={p.name:digest(p) for p in args.output_dir.iterdir()},
        limitations=['Register header must come from accepted dual HLS RTL audit before build/deployment.',
                     'Private copies on ordinary calls are a correctness baseline; their full-network cost is unmeasured.'])
    (args.output_dir/'source_manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('U02_REGION_DUAL_MODULE_PREPARED_NOT_LOADED',args.output_dir)


if __name__=='__main__':main()
