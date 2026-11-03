"""Prepare/run an isolated DW118 hardware owner around the real stereo child.

The existing hardware/rollback helpers retain their ownership/write logic.
Helpers have explicitly audited lookup/clock-observation/ownership adaptations.
All new orchestration is explicit. No broad field-name substitutions are used.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from retest_original_after_region import U02, ROOT, sha


MAIN_HEAD = r'''def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',required=True)
    parser.add_argument('--seconds',type=int,choices=(20,300,1800),default=20)
    parser.add_argument('--threads',type=int,choices=(1,2),default=1)
    parser.add_argument('--source-mode',choices=('720p','1080p'),default='720p')
    parser.add_argument('--uvc1080-module',type=Path)
    parser.add_argument('--uvc-hwtimestamps',type=int,choices=(0,1),default=0)
    parser.add_argument('--spin-count',type=int,choices=(0,300000),default=300000)
    parser.add_argument('--pl-poll-spin-us',type=int,choices=(0,1000),default=1000)
    parser.add_argument('--stop-after-pairs',type=int,choices=(0,128),default=0)
    parser.add_argument('--strict-image-reference',action='store_true')
    parser.add_argument('--bounded-clock-check',action='store_true')
    parser.add_argument('--clock-capture-samples',type=int,choices=(7,13),default=7)
    parser.add_argument('--restored-record',type=Path,required=True)
    parser.add_argument('--portable-hardware',type=Path)
    parser.add_argument('--dw128-v2',action='store_true',help='Experimental board-A/B-verified DW128 hardware; no formal promotion')
    parser.add_argument('--record-thread-cpu',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--candidate-host',type=Path)
    parser.add_argument('--candidate-host-sha256')
    parser.add_argument('--pipeline-pairs',type=int,choices=(1,2),default=0)
    parser.add_argument('--pair-limit-us',type=int,choices=(10000,16667),default=10000)
    parser.add_argument('--ioctl-cpu-profile',action='store_true')
    args=parser.parse_args()
    if args.ioctl_cpu_profile:assert args.pipeline_pairs and args.candidate_host and args.pair_limit_us==16667
    if args.pair_limit_us!=10000:assert args.pipeline_pairs and args.uvc_hwtimestamps==1
    if args.pipeline_pairs:
        assert args.dw128_v2 and args.source_mode=='1080p' and args.threads==1 and args.strict_image_reference
        assert args.candidate_host and not args.stop_after_pairs
    assert args.source_mode=='720p' or args.candidate_host, '1080p requires the isolated Host'
    from uvc_timestamp_mode import validate_mode
    validate_mode(args.uvc_hwtimestamps,args.source_mode,args.uvc1080_module)
    uvc_module_sha256=None
    if args.uvc1080_module:
        assert args.source_mode=='1080p'
        module_proof=json.loads((args.uvc1080_module/'state.json').read_text())
        assert module_proof['complete'] and module_proof['terminal'] and module_proof['module_name']=='u02_uvc1080'
        uvc_module_sha256=sha(args.uvc1080_module/'u02_uvc1080.ko')
        assert uvc_module_sha256==module_proof['module_sha256']
    if args.bounded_clock_check:
        from bounded_pl_clock_check import instrument_capture,verify_capture
    if args.stop_after_pairs:assert args.seconds>=300
    implementation=U02/'reports/formal_dw_implementation_1792087086862103800'
    portable=None
    if args.portable_hardware:
        from prepare_portable_hardware import load_prepared
        portable=load_prepared(args.portable_hardware)
        implementation=U02/'milestones/formal_dw_routed_118m_20261015'
        candidate={key:str(portable['dw118'][key]) for key in ('project','reports')};bit=portable['dw118']['bit']
    else:candidate,bit=accepted_artifacts(implementation)
    if args.dw128_v2:
        assert portable, 'DW128 stereo requires portable original rollback assets'
        from dw128_v2_stereo_hardware import accepted_candidate
        selected=accepted_candidate()
        candidate={key:str(selected[key]) for key in ('project','reports')};bit=selected['bit']
        implementation=U02/'milestones/dw128_v2_board_ab_verified_20261017'
    formal=json.loads((U02/'inference/stream_ingress_parallel_affine_runtime.json').read_text())
    original_project=U02/'inference/stream_ingress_project';original_reports=U02/'reports/stream_ingress_implementation'
    baseline_meta=None;sample0=ROOT/'board_light_pose/outputs/samples/0.f32'
    if portable:
        original_project=portable['original']['project'];original_reports=portable['original']['reports']
        assets=U02/'milestones/fresh_formal_stage_verified_20261015/assets'
        formal.update(bit=str(portable['original']['bit']),host=str(assets/'original_pose'),driver=str(assets/'original_resident.ko'))
        sample0=assets/'samples/0.f32'
        from restored_board_session import load_restored_baseline_metadata
        baseline_meta=load_restored_baseline_metadata(args.restored_record,formal['bit_sha256'])
    for name in ('bit','host','driver'):assert sha(U02/formal[name])==formal[name+'_sha256']
    from restored_board_session import load_restored_board_session
    prior=load_restored_board_session(args.restored_record,formal['bit_sha256'])
    runtime=json.loads((U02/'inference/dual_context_stereo_thread1_5min_verified_runtime_20261015.json').read_text())
    host=U02/runtime['host'];assert sha(host)==runtime['host_sha256']
    assert bool(args.candidate_host)==bool(args.candidate_host_sha256)
    if args.candidate_host:
        assert re.fullmatch('[a-f0-9]{64}',args.candidate_host_sha256)
        host=args.candidate_host.resolve();assert sha(host)==args.candidate_host_sha256
        from pair_host_elf import verify_pair_host_elf
        verify_pair_host_elf(host)
        runtime=dict(runtime,host_sha256=args.candidate_host_sha256)
        if args.ioctl_cpu_profile:assert b'ioctl_thread_cpu_profile' in host.read_bytes()
    if args.strict_image_reference:
        assert sha(U02/'milestones/strict_image_reference_replay_20261015/verified/tested_strict_image_reference')=='d61e65848373bbefc12fc9da6d981d2c8c5885965a2fc1a2795d9068d5431309'
    if args.preflight_only:
        print(json.dumps(dict(board_accessed=False,portable_hardware=bool(portable),bit_sha256=sha(bit),
            original_project=str(original_project),original_reports=str(original_reports),candidate_project=candidate['project'],
            candidate_reports=candidate['reports'],source_remote=prior['remote'],sample0_sha256=sha(sample0),
            strict_image_reference=args.strict_image_reference,bounded_clock_check=args.bounded_clock_check,
            host_sha256=runtime['host_sha256'],candidate_host=bool(args.candidate_host),source_mode=args.source_mode,
            uvc_module_sha256=uvc_module_sha256,pipeline_pairs=args.pipeline_pairs,pair_limit_us=args.pair_limit_us,ioctl_cpu_profile=args.ioctl_cpu_profile),indent=2));return
    remote=prior['remote'];results='dw_ab_'+uuid.uuid4().hex
    out=U02/'reports'/('formal_dw_dual_stereo_'+str(time.time_ns()));out.mkdir()
    state=dict(complete=False,terminal=False,phase='preflight',remote=remote,results=results,boot_id=prior['boot_id'],
        implementation=str(implementation),candidate_bit_sha256=sha(bit),formal_bit_sha256=formal['bit_sha256'],
        host_sha256=runtime['host_sha256'],driver_sha256=formal['driver_sha256'],restored_session=prior,
        driver_loaded=True,clock_changed=False,formal_restored=False,dma_ambiguous=False,
        startup_files_modified=False,reboot=False,measurements=[],threads_per_network=args.threads,gomp_spincount=args.spin_count,
        requested_seconds=args.seconds)
    state['ioctl_cpu_profile']=args.ioctl_cpu_profile
    state['pair_limit_us']=args.pair_limit_us
    state['pipeline_pairs']=args.pipeline_pairs
    state['source_mode']=args.source_mode
    state['hardware_variant']='dw128_v2' if args.dw128_v2 else 'dw118'
    state['uvc_module_sha256']=uvc_module_sha256
    state['uvc_hwtimestamps_requested']=args.uvc_hwtimestamps
    state['stop_after_pc_pairs']=args.stop_after_pairs
    state['pl_poll_spin_us']=args.pl_poll_spin_us
    state['bounded_clock_check']=args.bounded_clock_check
    state['clock_capture_samples']=args.clock_capture_samples
    if portable:
        state['portable_hardware']=str(args.portable_hardware.resolve())
        state['portable_hardware_descriptor_sha256']=portable['descriptor_sha256']
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    state['script_sha256']=sha(out/'executed_script.py')
    if args.bounded_clock_check:
        (out/'helpers').mkdir()
        state['clock_helper_sha256']={}
        for name in ('bounded_pl_clock_check.py','observe_pl_frequency.py'):
            (out/'helpers'/name).write_bytes((U02/'inference'/name).read_bytes())
            state['clock_helper_sha256'][name]=sha(out/'helpers'/name)
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('FORMAL_DW_DUAL_STEREO_RECORD',out,flush=True)
    c=None;baseline=None;hardware_owned=False;log=(out/'uart.log').open('wb')
'''

MAIN_TAIL = r'''
    def no_fds_and_refs():
        opened=checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/u02*|/dev/video*) printf "OPEN:%s\\n" "$t";; esac; done; true')
        assert not re.search(r'(?m)^OPEN:',opened)
        assert re.search(r'(?m)^u02_stream_resident\s+\d+\s+0\s',checked('cat /proc/modules'))
    try:
        connect();state['cpu_before']=snapshot_cpu_state(c);no_fds_and_refs()
        assert state['cpu_before']['online']=='0-3'
        assert 'clk_ignore_unused' in state['cpu_before']['kernel_cmdline'] and 'cpuidle.off=1' in state['cpu_before']['kernel_cmdline']
        settings=checked('cat /sys/module/u02_stream_resident/parameters/cached_dma; cat /sys/module/u02_stream_resident/parameters/poll_spin_us')
        assert re.search(r'(?m)^Y\r?$',settings) and re.search(r'(?m)^1000\r?$',settings)
        for name,digest in (('original_pose',formal['host_sha256']),('original_resident.ko',formal['driver_sha256']),
                            ('model.stream.param',formal['stream_param_sha256']),('model.opt.bin',formal['model_weights_sha256'])):
            assert digest in checked(f'sha256sum {remote}/{name}')
        assert sha(sample0) in checked(f'sha256sum {remote}/samples/0.f32')
        state['original_governor']=state['cpu_before']['governor'];assert re.fullmatch('[a-z]+',state['original_governor'])
        checked('printf %s performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
        state['governor_changed']=True;time.sleep(.1)
        baseline=registers();assert baseline['ff5e00c0']==0x01010b02 and baseline['ff5e0030']==0x14000 and baseline['ff5e0040']&2
        state['registers_baseline']=baseline;save();checked(f'mkdir {remote}/{results}')
        unload();hardware_owned=True
        gate('original_before',original_project,original_reports,True,11);load()
        measure('original',0,'pl-stream')
        unload();gate('candidate_same_clock',Path(candidate['project']),Path(candidate['reports']),True,11)
        clock(9);gate('candidate_high_clock',Path(candidate['project']),Path(candidate['reports']),False,9);load()
        child_source=U02/'inference/validate_dual_stereo.py'
        handoff=dict(kind='u02_dual_stereo_dw118_volatile_handoff_v1',owner_pid=os.getpid(),owner_record=str(out),
            owner_script_sha256=state['script_sha256'],child_script_sha256=sha(child_source),boot_id=prior['boot_id'],
            implementation=str(implementation),bit_sha256=sha(bit),pl_clock_divisor=9)
        if args.pipeline_pairs:handoff.update(pipeline_pairs=args.pipeline_pairs,pair_limit_us=args.pair_limit_us)
        if args.dw128_v2:
            from dw128_v2_stereo_hardware import HANDOFF_KIND
            handoff['kind']=HANDOFF_KIND
        if portable:
            handoff.update(portable_hardware=str(args.portable_hardware.resolve()),portable_hardware_descriptor_sha256=portable['descriptor_sha256'])
        handoff_path=out/'child_handoff.json';handoff_path.write_text(json.dumps(handoff,indent=2)+'\n')
        state.update(phase='child_running',handoff_sha256=sha(handoff_path),dma_ambiguous=True);save()
        c.port.close();c=None
        command=[sys.executable,str(child_source),'--execute','--backend','pl-stream','--seconds',str(args.seconds),
            '--restored-record',str(args.restored_record.resolve()),
            '--threads',str(args.threads),'--spin-count',str(args.spin_count),'--candidate-host',str(host),'--candidate-host-sha256',runtime['host_sha256'],
            '--pl-poll-spin-us',str(args.pl_poll_spin_us),
            '--wait-client-ms','10000','--connect-timeout','2','--dw118-owner-handoff',str(handoff_path)]
        if args.pipeline_pairs:command+=['--pipeline-pairs',str(args.pipeline_pairs)]
        command+=['--pair-limit-us',str(args.pair_limit_us)]
        if args.ioctl_cpu_profile:command+=['--ioctl-cpu-profile']
        if args.stop_after_pairs:command+=['--stop-after-pairs',str(args.stop_after_pairs)]
        if args.strict_image_reference:command+=['--strict-image-reference']
        command+=['--source-mode',args.source_mode]
        if args.record_thread_cpu:command+=['--record-thread-cpu']
        if args.uvc1080_module:command+=['--uvc1080-module',str(args.uvc1080_module.resolve())]
        command+=['--uvc-hwtimestamps',str(args.uvc_hwtimestamps)]
        with (out/'comparison.log').open('wb') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['comparison_pid']=proc.pid;save();code=proc.wait()
        text=(out/'comparison.log').read_text(errors='replace')
        match=re.search(r'DUAL_STEREO_NETWORK_RECORD (.+)',text);assert match,text
        child_record=Path(match.group(1).strip());child=json.loads((child_record/'state.json').read_text())
        state.update(comparison_record=str(child_record),comparison_exit_code=code,child_state_sha256=sha(child_record/'state.json'));save()
        assert child['terminal'] and child['resident_driver_restored'] and child['governor_restored'] and child['cameras_restored']
        assert not child['dma_ambiguous'] and not child['command_uncertain']
        assert not child['formal_restored'] and child['external_hardware_owner']
        assert child['hardware_bit_sha256']==sha(bit) and child['pl_clock_divisor']==9
        assert child['owner_handoff_sha256']==sha(handoff_path)
        assert child['observed_pl_poll_spin_us']==child['pl_poll_spin_us']==args.pl_poll_spin_us
        connect();no_fds_and_refs();c.port.close();c=None
        from verify_core_idle_readonly import verify_core_idle_readonly
        verify_core_idle_readonly(Path(candidate['project']),Path(candidate['reports']),out/'idle_after_child')
        connect();no_fds_and_refs();state['dma_ambiguous']=False;state['phase']='restoring';save()
        # A failed child with independently proven idle is still safely rolled back.
        restore();measure('restored',0,'pl-stream');assert registers()==baseline
        state['rollback_outputs_sha256']={}
        for label in ('original','restored'):
            for suffix in ('.json','.json.heatmap'):
                name=label+'_0'+suffix;c.download(remote+'/'+results+'/'+name,out/name)
                state['rollback_outputs_sha256'][name]=sha(out/name)
        assert (out/'original_0.json.heatmap').read_bytes()==(out/'restored_0.json.heatmap').read_bytes()
        assert code==0 and child['complete'],child
        if args.stop_after_pairs:assert child['signal_stop_complete_pair_verified'] and child['stop_signal_sent']
        state['camera_summary']={k:child[k] for k in ('completed_pairs','camera_elapsed_s','actual_pairs_per_s','total_pl_calls',
            'network_received_pairs','retained_pc_pairs_matched','intentional_disconnect_recovered','final_both_views_replay_exact')}
        state['automatic_controller_completed']=True;state['complete']=True;state['phase']='finished'
    except Exception as error:state['error']=repr(error);raise
    finally:
        try:
            if hardware_owned and not state['formal_restored']:restore()
            if state.get('governor_changed') and c:
                checked('printf %s '+state['original_governor']+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
                state['governor_restored']=True
            if c:
                state['cpu_after']=snapshot_cpu_state(c)
                assert state['cpu_after']['boot_id']==prior['boot_id'] and state['cpu_after']['online']=='0-3'
                if state.get('governor_changed'):assert state['cpu_after']['governor']==state['original_governor']
                if state['formal_restored']:assert registers()['ff5e00c0']==0x01010b02
            if not state['formal_restored']:state['complete']=False
        except Exception as error:state['complete']=False;state['restore_error']=repr(error)
        if c:c.port.close()
        log.close();state['terminal']=True;save();print('FORMAL_DW_DUAL_STEREO_TERMINAL',out,flush=True)

if __name__=='__main__':main()
'''


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--seconds',type=int,choices=(20,300,1800),default=20)
    ap.add_argument('--threads',type=int,choices=(1,2),default=1)
    ap.add_argument('--source-mode',choices=('720p','1080p'),default='720p')
    ap.add_argument('--uvc1080-module',type=Path)
    ap.add_argument('--uvc-hwtimestamps',type=int,choices=(0,1),default=0)
    ap.add_argument('--spin-count',type=int,choices=(0,300000),default=300000)
    ap.add_argument('--pl-poll-spin-us',type=int,choices=(0,1000),default=1000)
    ap.add_argument('--stop-after-pairs',type=int,choices=(0,128),default=0)
    ap.add_argument('--strict-image-reference',action='store_true')
    ap.add_argument('--bounded-clock-check',action='store_true')
    ap.add_argument('--clock-capture-samples',type=int,choices=(7,13),default=7)
    ap.add_argument('--restored-record',type=Path,default=U02/'reports/formal_dw_pair_finished_1792116056736822900')
    ap.add_argument('--portable-hardware',type=Path)
    ap.add_argument('--dw128-v2',action='store_true')
    ap.add_argument('--record-thread-cpu',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--candidate-host',type=Path)
    ap.add_argument('--candidate-host-sha256')
    ap.add_argument('--pipeline-pairs',type=int,choices=(1,2),default=0)
    ap.add_argument('--pair-limit-us',type=int,choices=(10000,16667),default=10000)
    ap.add_argument('--ioctl-cpu-profile',action='store_true')
    args=ap.parse_args()
    base=U02/'inference/validate_dw_stem_single.py';source=base.read_text();tree=ast.parse(source)
    parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    helpers={n.name:n for n in parent.body if isinstance(n,ast.FunctionDef)}
    names=('connect','checked','registers','unload','load','clock','gate','measure','restore')
    lines=source.splitlines(keepends=True)
    blocks=[''.join(lines[helpers[n].lineno-1:helpers[n].end_lineno]) for n in names]
    # Explicit lookups and optional clock observation differ. Ownership,
    # DMA, clock register writes and rollback operations remain identical.
    adaptations={'restore':[("U02/'inference/stream_ingress_project'",'original_project'),
                            ("U02/'reports/stream_ingress_implementation'",'original_reports'),
                            ('        if not c:connect()',
                             "        if state.get('gate_pending') or state.get('command_uncertain'):raise RuntimeError('Preserve unobserved JTAG/command ownership; no concurrent recovery')\n        if not c:connect()")],
        'checked':[("        code,text=c.run(command,timeout)",
                    "        state['command_uncertain']=True;save()\n        code,text=c.run(command,timeout)\n        state['command_uncertain']=False;save()")],
        'measure':[("json.loads((ROOT/f'board_light_pose/outputs/samples/{index}.json').read_text())",
                    "(baseline_meta if baseline_meta is not None else json.loads((ROOT/f'board_light_pose/outputs/samples/{index}.json').read_text()))")],
        'gate':[("        artifact_label=label;attempt=1",
                 "        if args.bounded_clock_check:text=instrument_capture(text,args.clock_capture_samples)\n        artifact_label=label;attempt=1"),
                ("len(samples)==3 and all(len(v)==3 for v in samples.values())",
                 "len(samples)==3 and all(len(v)==(args.clock_capture_samples if args.bounded_clock_check else 3) for v in samples.values())"),
                ("assert resets==[[3,3,3]]", "assert resets==[[3]*(args.clock_capture_samples if args.bounded_clock_check else 3)]"),
                ("for i in range(2)]", "for i in range(args.clock_capture_samples-1 if args.bounded_clock_check else 2)]"),
                ("        if label!='original_before':",
                 "        clock_observation=None\n        if args.bounded_clock_check:clock_observation=verify_capture(text,divisor)\n        elif label!='original_before':"),
                ("state[label]=dict(pl_debug_ratios=ratios,",
                 "state[label]=dict(clock_observation=clock_observation,pl_debug_ratios=ratios,"),
                ("        with (execution_log).open('wb') as stream:",
                 "        state['gate_pending']=True;save()\n        with (execution_log).open('wb') as stream:"),
                ("            try:code=proc.wait(timeout=180)\n            except subprocess.TimeoutExpired:\n                subprocess.run(['taskkill.exe','/PID',str(proc.pid),'/T','/F'],capture_output=True)\n                proc.wait(timeout=30);raise",
                 "            code=proc.wait()\n        state['gate_pending']=False;save()") ]}
    for name,replacements in adaptations.items():
        index=names.index(name)
        for before,after in replacements:
            assert blocks[index].count(before)==1,(name,before)
            blocks[index]=blocks[index].replace(before,after)
    prefix=source[:source.index('def main():')]
    prefix=prefix.replace('from retest_original_after_region import',
        'sys.path.insert(0,'+repr(str(U02/'inference'))+')\nfrom retest_original_after_region import',1)
    generated=prefix+MAIN_HEAD+'\n'+'\n'.join(blocks)+'\n'+MAIN_TAIL
    out=U02/'reports'/('dual_stereo_dw118_prepared_'+str(time.time_ns()));out.mkdir()
    script=out/'validate_dw_dual_stereo_owned.py';compile(generated,str(script),'exec');script.write_text(generated)
    child_tree=ast.parse(generated)
    generated_main=next(n for n in child_tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    generated_helpers={n.name:n for n in generated_main.body if isinstance(n,ast.FunctionDef)}
    for name,block in zip(names,blocks):
        expected=ast.parse('def expected_container():\n'+block).body[0].body[0]
        assert ast.dump(expected,include_attributes=False)==ast.dump(generated_helpers[name],include_attributes=False),name
    help_result=subprocess.run([sys.executable,str(script),'--help'],cwd=ROOT,capture_output=True,text=True,check=True)
    (out/'offline_help.log').write_text(help_result.stdout+help_result.stderr)
    state=dict(complete=True,scope='Offline owner preparation only, no board access',
        source=str(base),source_sha256=sha(base),generator_sha256=sha(Path(__file__)),
        generated_sha256=sha(script),hardware_helpers_verbatim=[n for n in names if n not in adaptations],
        helper_adaptations=adaptations,board_execution_requested=args.execute)
    (out/'state.json').write_text(json.dumps(state,indent=2)+'\n');print('DUAL_STEREO_DW118_PREPARED',out,flush=True)
    if args.execute or args.preflight_only:
        command=[sys.executable,str(script),'--execute','--seconds',str(args.seconds),
            '--threads',str(args.threads),'--spin-count',str(args.spin_count),'--restored-record',str(args.restored_record.resolve())]
        command+=['--pl-poll-spin-us',str(args.pl_poll_spin_us)]
        command+=['--source-mode',args.source_mode]
        if args.uvc1080_module:command+=['--uvc1080-module',str(args.uvc1080_module.resolve())]
        command+=['--uvc-hwtimestamps',str(args.uvc_hwtimestamps)]
        if args.portable_hardware:command+=['--portable-hardware',str(args.portable_hardware.resolve())]
        if args.dw128_v2:command+=['--dw128-v2']
        if args.record_thread_cpu:command+=['--record-thread-cpu']
        if args.pipeline_pairs:command+=['--pipeline-pairs',str(args.pipeline_pairs)]
        command+=['--pair-limit-us',str(args.pair_limit_us)]
        if args.ioctl_cpu_profile:command+=['--ioctl-cpu-profile']
        if args.stop_after_pairs:command+=['--stop-after-pairs',str(args.stop_after_pairs)]
        if args.strict_image_reference:command+=['--strict-image-reference']
        if args.bounded_clock_check:command+=['--bounded-clock-check']
        command+=['--clock-capture-samples',str(args.clock_capture_samples)]
        if args.preflight_only:command+=['--preflight-only']
        if args.candidate_host:command+=['--candidate-host',str(args.candidate_host.resolve())]
        if args.candidate_host_sha256:command+=['--candidate-host-sha256',args.candidate_host_sha256]
        raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
