"""Prepare/run the established DW clock/bit owner with ABI7 pair verification.

Only the child selector/arguments and artifact labels differ from the verified
owner. Its independent gates, DMA ownership and original restoration remain.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from retest_original_after_region import U02,ROOT,sha
from validate_dw_stem_single import accepted_artifacts
from pair_dw_comparison_source import prepare_pair_dw_comparison


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--restored-record',type=Path,default=U02/'reports/post_reboot_dw_ab_recovered_1792104845038960500')
    args=ap.parse_args()
    implementation=U02/'reports/formal_dw_implementation_1792087086862103800'
    candidate,bit=accepted_artifacts(implementation)
    out=U02/'reports'/('pair_dw118_prepared_'+str(time.time_ns()));out.mkdir()
    # Compile and exercise --help before the owner can touch hardware.
    child,provenance=prepare_pair_dw_comparison(out,candidate,bit)
    subprocess.run([sys.executable,str(child),'--help'],cwd=ROOT,check=True,capture_output=True)
    prepared=subprocess.run([sys.executable,str(child),'--execute','--prepare-only','--restored-record',str(args.restored_record.resolve())],cwd=ROOT,check=True,capture_output=True,text=True)
    (out/'child_offline_preparation.log').write_text(prepared.stdout+prepared.stderr)
    base=U02/'inference/validate_dw_stem_single.py';text=base.read_text()
    def once(before,after):
        nonlocal text
        assert text.count(before)==1,(before,text.count(before));text=text.replace(before,after)
    once('from retest_original_after_region import','sys.path.insert(0,'+repr(str(U02/'inference'))+')\nfrom retest_original_after_region import')
    once('from dw_stem_comparison_source import prepare_dw_stem_comparison','from pair_dw_comparison_source import prepare_pair_dw_comparison')
    once('prepare_dw_stem_comparison(out,candidate,bit)','prepare_pair_dw_comparison(out,candidate,bit)')
    once("[sys.executable,str(derived),'--restored-record'","[sys.executable,str(derived),'--execute','--restored-record'")
    once('DW_STEM_FULL_AB_RECORD','PAIR_DW118_RECORD')
    text=text.replace('formal_dw_stem_single_','formal_dw_pair_').replace('FORMAL_DW_STEM_SINGLE','FORMAL_DW_PAIR')
    once('Isolated full-model A/B of the accepted DW118 and exact ARM stem.','Isolated full-model paired execution of accepted DW118 and the ABI7 contexts.')
    script=out/'validate_dw_pair_owned.py';compile(text,str(script),'exec');script.write_text(text)
    subprocess.run([sys.executable,str(script),'--help'],cwd=ROOT,check=True,capture_output=True)
    state=dict(complete=True,scope='Offline immutable owner/child preparation and help validation only',
        source=str(base),source_sha256=sha(base),derived_sha256=sha(script),child_provenance=provenance,
        generator_sha256=sha(Path(__file__)),implementation=str(implementation),bit_sha256=sha(bit),
        board_execution_requested=args.execute)
    (out/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('PAIR_DW118_PREPARED',out,flush=True)
    if args.execute:
        code=subprocess.call([sys.executable,str(script),str(implementation),'--execute','--restored-record',str(args.restored_record.resolve())],cwd=ROOT)
        raise SystemExit(code)


if __name__=='__main__':main()
