"""Isolated guarded-bounded Host A/B, retaining the accepted polling transport."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from retest_original_after_region import U02, ROOT, sha
from prepare_pipeline_board import generate as generate_pipeline
from prepare_pipeline_irq_board import EXTRA,replace_once


RUN=r'''
        load('pipeline');mapping_probe('poll_transport');cohort('baseline')
        pipeline_case('smoke',2,0,2,2)
        smoke=out/'smoke';smoke.mkdir()
        state['smoke_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories['smoke'],pipeline_names(2,0),smoke)
        check_pipeline(smoke,2,0,2,2);state['first_four_context_whole_network_exact']=True;save()
        cohort('candidate');cohort('baseline_after')
        unload('idle_after_final_baseline');rollback()
        state.update(complete=True,phase='finished',all_full_heatmaps_and_keypoints_exact=True)
'''


def generate():
    text,evidence=generate_pipeline()
    text=replace_once(text,"ap.add_argument('--execute',action='store_true')",
        "ap.add_argument('--execute',action='store_true')\n    ap.add_argument('--quant-prepared',type=Path,required=True)")
    text=replace_once(text,"args=ap.parse_args()", "args=ap.parse_args()\n    assert not args.reuse_baseline,'Fresh matched controls required'")
    text=replace_once(text,"    elf={n:verify_pair_host_elf",r'''    quant_prepared=args.quant_prepared.resolve()
    assert quant_prepared.is_relative_to((U02/'milestones').resolve());verify_frozen(quant_prepared)
    quant_contract=json.loads((quant_prepared/'summary.json').read_text())
    assert quant_contract['cpu_numeric_verified'] and not quant_contract['hardware_acceptance']
    files['guarded_host']=quant_prepared/'binaries/pose_pipeline_network_bounded_guarded'
    files['recover_pipeline']=U02/'milestones/stream_irq_prepared_20261018/binaries/recover_pipeline_transport'
    assert sha(files['guarded_host'])==quant_contract['pipeline_host_sha256']
    elf={n:verify_pair_host_elf''')
    text=replace_once(text,"('pair_host','pipeline_host')}","('pair_host','pipeline_host','guarded_host')}")
    text=replace_once(text,"('pipeline_board_'+str(time.time_ns()))","('pipeline_quant_board_'+str(time.time_ns()))")
    text=replace_once(text,"candidate_kind='shared_model_abi9_whole_network'","candidate_kind='bounded_guarded_quantization_host'")
    text=replace_once(text,'chmod 700 pair_host pipeline_host dma_probe','chmod 700 pair_host pipeline_host dma_probe guarded_host recover_pipeline')
    text=replace_once(text,'warmup_pairs_per_input_pattern=6,measured_pairs_per_input_pattern=10',
        'warmup_pairs_per_input_pattern=12,measured_pairs_per_input_pattern=20,inflight_pairs=2')
    text=replace_once(text,"        if not c:connect()\n        if state['driver_state']!='none':unload",
        "        if not c:connect()\n        if state['driver_state']=='pipeline' and state['dma_ambiguous']:recover_observed_pipeline_failure()\n        if state['driver_state']!='none':unload")
    text=replace_once(text,"        inputs=' '.join(f'samples/{i}.f32' for i in indices)",
        "        inputs=' '.join(f'samples/{i}.f32' for i in indices)\n        host='guarded_host' if stage in ('smoke','candidate') else 'pipeline_host'")
    text=replace_once(text,'./{tools}/pipeline_host pl-stream','./{tools}/{host} pl-stream')
    helpers=EXTRA[:EXTRA.index('    def irq_stats')]+EXTRA[EXTRA.index('    def mapping_probe'):]
    text=replace_once(text,'    try:\n        connect();before=snapshot_cpu_state(c)',helpers+'\n    try:\n        connect();before=snapshot_cpu_state(c)')
    start=text.index("        if not args.reuse_baseline:\n            load('dual')")
    end=text.index('    except Exception as error:',start)
    text=text[:start]+RUN+text[end:]
    text=text.replace('DW128 /9 full-network ABI7 vs ABI9 pipeline A/B with original /11 rollback',
        'Same polling ABI9 DW128 /9 bounded quantization Host before/candidate/after')
    ast.parse(text)
    evidence.update(generator_sha256=sha(Path(__file__)),
        recovery_source_sha256=sha(U02/'inference/prepare_pipeline_irq_board.py'),
        adaptations=['Separate candidate Host only; same loaded polling driver/DMA/bit/clock across cohorts',
                     'Observed-exit and external-idle kernel recovery retained',
                     'Fresh complete-network depth2 before/candidate/after; original /11 rollback'])
    return text,evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--restored-record',type=Path,required=True)
    ap.add_argument('--quant-prepared',type=Path,required=True)
    args=ap.parse_args();text,evidence=generate()
    out=U02/'reports'/('pipeline_quant_owner_prepared_'+str(time.time_ns()));out.mkdir()
    script=out/'validate_pipeline_quant_owned.py';script.write_text(text,encoding='utf-8')
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,check=True)
    (out/'offline_help.log').write_bytes(result.stdout+result.stderr)
    evidence.update(complete=True,board_accessed=False,generated_sha256=sha(script))
    (out/'state.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PIPELINE_QUANT_OWNER_PREPARED',out,flush=True)
    if args.execute or args.preflight:
        command=[sys.executable,str(script),'--restored-record',str(args.restored_record.resolve()),
            '--prepared',str(U02/'milestones/shared_model_pipeline_roi_prepared_20261018'),
            '--quant-prepared',str(args.quant_prepared.resolve())]
        if args.execute:command+=['--execute']
        raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
