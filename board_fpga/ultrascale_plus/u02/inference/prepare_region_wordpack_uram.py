"""Bind unchanged, validated boundary code to a separate weight-URAM candidate."""
import argparse
import json
from pathlib import Path
import shutil
import time
from prepare_region_boundary_pipeline import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-validation',type=Path,required=True)
    parser.add_argument('--complete-reference',type=Path,required=True)
    args=parser.parse_args()
    native_path=args.boundary_validation/'result.json';native=json.loads(native_path.read_text())
    full_path=args.complete_reference/'result.json';full=json.loads(full_path.read_text())
    tails_path=args.boundary_validation/'tail_validation.json';tails=json.loads(tails_path.read_text())
    if not native.get('complete') or native.get('real_boundaries')!=70 or native.get('nonfinite_rejections')!=18:
        raise ValueError('Completed real wordpack numerics required')
    if not full.get('complete') or not full.get('full_network_reference_verified') or full.get('pipeline_boundary_calls')!=90:
        raise ValueError('Completed full-graph wordpack reference required')
    if not tails.get('complete') or tails.get('cases')!=32:
        raise ValueError('Completed tail/guard proof required')
    frozen={**native['input_sha256'],**full['input_sha256'],**full['generated_sha256'],**tails['input_sha256']}
    reused_binaries={str(args.boundary_validation/name):digest for name,digest in native['binary_sha256'].items()}
    frozen.update(reused_binaries)
    for path,digest in frozen.items():
        if sha(Path(path))!=digest:raise ValueError('Verified source changed: '+path)
    for proof,root in ((native,args.boundary_validation),(full,args.complete_reference)):
        for name,digest in proof['log_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Numeric evidence changed')
    if sha(args.boundary_validation/'tail_test.log')!=tails['log_sha256']:
        raise ValueError('Tail evidence changed')
    parent=Path(native['preparation']);manifest=parent/'preparation.json';plan=json.loads(manifest.read_text())
    generator=Path(plan['preparation_script'])
    if sha(generator)!=plan['preparation_script_sha256']:raise ValueError('Wordpack generator changed')
    frozen.update({str(p):sha(p) for p in (Path(__file__),native_path,full_path,tails_path,manifest,generator)})
    record=U02/('reports/region_wordpack_uram_preparation_'+str(time.time_ns()));record.mkdir()
    needle=b'#pragma HLS RESOURCE variable=weight_words core=RAM_2P_BRAM'
    replacement=b'#pragma HLS RESOURCE variable=weight_words core=RAM_2P_URAM'
    files={}
    for name,entry in plan['files'].items():
        source=parent/name
        if sha(source)!=entry['candidate_sha256']:raise ValueError('Wordpack candidate changed')
        raw=source.read_bytes()
        if name=='region_dual_compute.h':
            if raw.count(needle)!=1 or b'static uint64_t weight_words[12800];' not in raw:
                raise ValueError('Unexpected weight staging')
            raw=raw.replace(needle,replacement)
        target=record/name;target.write_bytes(raw)
        frozen[str(target)]=sha(target)
        files[name]=dict(baseline_sha256=entry['baseline_sha256'],candidate_sha256=sha(target))
    if sha(record/'region_boundary_words.h')!=sha(parent/'region_boundary_words.h'):
        raise ValueError('Boundary conversion changed during memory preparation')
    preparation=dict(plan)
    preparation.update(preparation_script=str(Path(__file__).resolve()),preparation_script_sha256=sha(Path(__file__)),
                       files=files,weight_uram=True,arithmetic_changed=False,
                       memory_change=dict(file='region_dual_compute.h',before=needle.decode(),after=replacement.decode()),
                       scope='Same validated local word packing and pipeline; only weight staging RAM resource changed. Hardware initialization, latency, II, resource and numeric RTL remain unverified.')
    prepared_manifest=record/'preparation.json'
    prepared_manifest.write_text(json.dumps(preparation,indent=2)+'\n')
    frozen[str(prepared_manifest)]=sha(prepared_manifest)
    for name in native['log_sha256']:
        shutil.copyfile(args.boundary_validation/name,record/name)
    result=dict(native)
    result.pop('binary_sha256',None)
    result.update(preparation=str(record),input_sha256=frozen,boundary_numerics_reused=True,
                  reused_binary_sha256=reused_binaries,
                  reused_boundary_validation=str(args.boundary_validation.resolve()),
                  reused_full_network_reference=str(args.complete_reference.resolve()),
                  full_reference_boundary_calls=90,tail_cases=32,
                  synthesized=False,rtl_cosim_verified=False,memory_binding_verified=False,
                  scope='Native boundary proof reused only because header bytes are identical; full graph and tail evidence are source-bound. New weight-URAM implementation still requires whole-group HLS and RTL/mapping/board gates.')
    (record/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('U02_REGION_WORDPACK_URAM_PREPARATION',record)


if __name__=='__main__':main()
