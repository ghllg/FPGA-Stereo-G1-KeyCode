"""Isolated same-Host polling/IRQ/polling full-network owner generator."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from retest_original_after_region import U02, ROOT, sha
from prepare_pipeline_board import generate as generate_pipeline


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


EXTRA = r'''
    def recover_observed_pipeline_failure():
        nonlocal c
        assert state['driver_state']=='pipeline' and state['dma_ambiguous']
        assert not state['command_uncertain'] and not state['programming_pending']
        assert state['bit_identity']=='candidate'
        no_fds()  # Application exit was observed; inflight module pin may remain.
        label='recover_external_idle_'+uuid.uuid4().hex[:8]
        c.port.close();c=None
        verify_core_idle_readonly(hardware['candidate']['project'],hardware['candidate']['reports'],out/label)
        connect();no_fds()
        # Do not clear DMA ownership until kernel RECOVER succeeds and refs are zero.
        state.update(command_uncertain=True,recovery_idle_proof=label);save()
        code,text=c.run(f'{remote}/{tools}/recover_pipeline --recover-after-independent-idle',120)
        state['command_uncertain']=False;state['recovery_output']=text;save()
        assert code==0,text
        assert '"complete":true' in text and '"run_submissions":0' in text and '"dma_mapped":false' in text
        no_fds();module_idle();state['dma_ambiguous']=False;state['recovered_for_rollback_only']=True;save()
    def irq_stats(label,expected):
        assert modules['pipeline']=='u02_stream_pipeline_irq' and state['driver_state']=='pipeline'
        assert not state['command_uncertain'] and not state['dma_ambiguous']
        path=f'{remote}/{tools}/irq_{label}.json'
        checked(f'cat /sys/class/misc/u02_stream_pipeline/irq_stats > {path}')
        local=out/('irq_'+label+'.json');c.download(path,local)
        assert sha(local) in checked(f'sha256sum {path}')
        d=json.loads(local.read_text())
        state.setdefault('irq_observations',{})[label]=dict(stats=d,sha256=sha(local),
            interrupts=checked('cat /proc/interrupts'));save()
        assert d['linux_irq']>0 and d['gic_hwirq']==121 and d['spi']==89
        assert all(d[k]==expected for k in ('submitted','delivered','completed')),d
        assert all(d[k]==0 for k in ('unexpected','timeouts','recoveries','failed')),d
        return d
    def mapping_probe(label):
        run(f'{remote}/{tools}/dma_probe {remote}/{tools}/dma_probe.json',False)
        local=out/(label+'_dma_probe.json');c.download(remote+'/'+tools+'/dma_probe.json',local)
        d=json.loads(local.read_text())
        assert d['complete'] and d['abi']==9 and d['arena_bytes']==8388608 and d['checked_regions']==24
        assert d['cpu_mapping_disjoint'] and d['original_bytes_restored'] and d['pl_submissions']==0
        state.setdefault('mapping_probes',{})[label]=dict(sha256=sha(local),result=d);save()
    def cohort(stage):
        names=[]
        for index in range(0,10,2):
            pipeline_case(stage,2,index,12,20);names+=pipeline_names(2,index)
            print('IRQ_AB_QUARTET_DONE',stage,index,flush=True)
        dest=out/stage;dest.mkdir()
        state[stage+'_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories[stage],names,dest)
        rows=[];elapsed=0
        for index in range(0,10,2):
            d=check_pipeline(dest,2,index,12,20)
            rows+=d['rows'];elapsed+=d['measured_end_ns']-d['measured_start_ns']
        state['summary'][stage]=summarize(rows,elapsed,True);save()
'''

RUN = r'''
        load('pipeline');mapping_probe('poll_before');cohort('baseline')
        unload('idle_before_irq');modules['pipeline']='u02_stream_pipeline_irq';load('pipeline')
        irq_stats('loaded_disabled',0);mapping_probe('irq');irq_stats('mapping_no_run',0)
        pipeline_case('smoke',2,0,2,2)
        smoke=out/'smoke';smoke.mkdir()
        state['smoke_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories['smoke'],pipeline_names(2,0),smoke)
        check_pipeline(smoke,2,0,2,2);irq_stats('smoke',880)
        state['first_four_context_whole_network_exact']=True;save()
        cohort('candidate');irq_stats('complete',36080)
        unload('idle_after_irq');modules['pipeline']='u02_stream_pipeline';load('pipeline')
        mapping_probe('poll_after');cohort('baseline_after')
        unload('idle_after_final_baseline');rollback()
        state.update(complete=True,phase='finished',all_full_heatmaps_and_keypoints_exact=True,
                     irq_full_network_calls_verified=36080,formal_promoted=False)
'''


def generate():
    text, evidence = generate_pipeline()
    text = replace_once(text, "ap.add_argument('--execute',action='store_true')", "ap.add_argument('--execute',action='store_true')\n    ap.add_argument('--irq-prepared',type=Path,default=U02/'milestones/stream_irq_prepared_20261018')")
    text = replace_once(text, "args=ap.parse_args()", "args=ap.parse_args()\n    assert not args.reuse_baseline,'IRQ A/B requires fresh matched polling controls'")
    text = replace_once(text, "    elf={n:verify_pair_host_elf", """    irq_prepared=args.irq_prepared.resolve()
    assert irq_prepared.is_relative_to((U02/'milestones').resolve());verify_frozen(irq_prepared)
    irq_contract=json.loads((irq_prepared/'summary.json').read_text())
    assert irq_contract['offline_ready'] and not irq_contract['hardware_acceptance']
    assert irq_contract['bit_sha256']==BIT_SHA256
    route=U02/'milestones/pl_completion_irq_preflight_20261018';verify_frozen(route)
    route_contract=json.loads((route/'summary.json').read_text())
    assert route_contract['bit_sha256']==BIT_SHA256 and route_contract['gic_hardware_id']==121
    assert route_contract['gic_dt_spi_number']==89 and route_contract['irq_sink']=='PS8 PLPSIRQ0[0]'
    files['u02_stream_pipeline_irq.ko']=irq_prepared/'binaries/u02_stream_pipeline_irq.ko'
    files['recover_pipeline']=irq_prepared/'binaries/recover_pipeline_transport'
    assert sha(files['u02_stream_pipeline_irq.ko'])==irq_contract['module_sha256']
    elf={n:verify_pair_host_elf"""
    )
    text = replace_once(text, "('pipeline_board_'+str(time.time_ns()))", "('pipeline_irq_board_'+str(time.time_ns()))")
    text = replace_once(text, "candidate_kind='shared_model_abi9_whole_network'", "candidate_kind='same_host_abi9_irq_completion'")
    text = replace_once(text, 'warmup_pairs_per_input_pattern=6,measured_pairs_per_input_pattern=10', 'warmup_pairs_per_input_pattern=12,measured_pairs_per_input_pattern=20,inflight_pairs=2')
    text = replace_once(text, "chmod 700 pair_host pipeline_host dma_probe", "chmod 700 pair_host pipeline_host dma_probe recover_pipeline")
    text = replace_once(text, "module_command(f'insmod {path} bridge_checked=1 cached_dma=1 poll_spin_us=1000',kind)", "extra=' irq_route_checked=1' if modules.get(kind)=='u02_stream_pipeline_irq' else ''\n        module_command(f'insmod {path} bridge_checked=1 cached_dma=1 poll_spin_us=1000'+extra,kind)")
    text = replace_once(text, "        if not c:connect()\n        if state['driver_state']!='none':unload", "        if not c:connect()\n        if state['driver_state']=='pipeline' and state['dma_ambiguous']:recover_observed_pipeline_failure()\n        if state['driver_state']!='none':unload")
    marker = "    try:\n        connect();before=snapshot_cpu_state(c)"
    text = replace_once(text, marker, EXTRA+'\n'+marker)
    start = text.index('        if not args.reuse_baseline:\n            load(\'dual\')')
    end = text.index('    except Exception as error:', start)
    text = text[:start]+RUN+text[end:]
    text = text.replace('DW128 /9 full-network ABI7 vs ABI9 pipeline A/B with original /11 rollback', 'DW128 /9 same-Host ABI9 polling/IRQ/polling full-network A/B with original /11 rollback')
    ast.parse(text)
    evidence.update(irq_owner_generator_sha256=sha(Path(__file__)),
        preserved_baseline_generator_sha256=sha(U02/'inference/prepare_pipeline_board.py'),
        adaptations=['isolated IRQ payload/load flag', 'observed-exit independent-idle RECOVER before rollback',
                     'fresh depth2 polling/IRQ/polling cohorts', 'zero-RUN mapping and exact IRQ counter gates'])
    return text,evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--restored-record',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,default=U02/'milestones/shared_model_pipeline_roi_prepared_20261018')
    ap.add_argument('--irq-prepared',type=Path,default=U02/'milestones/stream_irq_prepared_20261018')
    args=ap.parse_args();text,evidence=generate()
    out=U02/'reports'/('pipeline_irq_owner_prepared_'+str(time.time_ns()));out.mkdir()
    script=out/'validate_pipeline_irq_owned.py';script.write_text(text,encoding='utf-8')
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,check=True)
    (out/'offline_help.log').write_bytes(result.stdout+result.stderr)
    evidence.update(complete=True,board_accessed=False,generated_sha256=sha(script))
    (out/'state.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PIPELINE_IRQ_OWNER_PREPARED',out,flush=True)
    if args.execute or args.preflight:
        command=[sys.executable,str(script),'--restored-record',str(args.restored_record.resolve()),
                 '--prepared',str(args.prepared.resolve()),'--irq-prepared',str(args.irq_prepared.resolve())]
        if args.execute:command+=['--execute']
        raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
