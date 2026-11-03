"""Generate an isolated ABI9 owner from reviewed clock/ownership/rollback helpers."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
from retest_original_after_region import U02, ROOT, sha

HEAD = r'''
def main():
    ap=argparse.ArgumentParser(description='DW128 /9 full-network ABI7 vs ABI9 pipeline A/B with original /11 rollback')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--restored-record',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,default=U02/'milestones/shared_model_pipeline_prepared_20261018')
    ap.add_argument('--reuse-baseline',type=Path,help='Frozen, restored same-boot ROI-entry failure; reuse its completed baseline only')
    args=ap.parse_args()
    from prepare_dw128_v2_board_ab import verify_frozen
    from dw128_v2_stereo_hardware import accepted_candidate, BIT_SHA256
    from pipeline_output_bundle import download_pipeline_outputs
    import hashlib
    prepared=args.prepared.resolve();assert prepared.is_relative_to((U02/'milestones').resolve())
    verify_frozen(prepared)
    contract=json.loads((prepared/'summary.json').read_text())
    assert contract['cpu_whole_network_verified'] and not contract['pl_board_verified'] and not contract['formal_promoted']
    formal=json.loads((U02/'inference/stream_ingress_parallel_affine_runtime.json').read_text())
    prior=load_restored_board_session(args.restored_record,formal['bit_sha256'])
    portable=load_prepared(U02/'milestones/portable_hardware_prepared_20261015/runtime_tree')
    candidate=accepted_candidate();assert sha(candidate['bit'])==BIT_SHA256
    old=U02/'milestones/matched_cpu1_and_1080_entry_verified_20261016';verify_frozen(old)
    ref=U02/'milestones/stream_range_cache_pl_verified_20261015';verify_archive(ref);reference=ref/'matched_pl'
    files={'pair_host':old/'cpu_current/tested_pose_pair_network',
           'pipeline_host':prepared/'binaries/pose_pipeline_network',
           'u02_stream_dual.ko':U02/'milestones/dual_context_transport_prepared_20261015/abi7_build/u02_stream_dual.ko',
           'u02_stream_pipeline.ko':prepared/'binaries/u02_stream_pipeline.ko',
           'dma_probe':U02/'inference/build_fast_nearest/pipeline_host/probe_pipeline_dma'}
    assert sha(files['pair_host'])=='94806cbd5c4c8d15db26f3392e33e28073d2c75852d082cadcb551b126a3f65c'
    assert sha(files['pipeline_host'])==contract['host_sha256']
    assert sha(files['u02_stream_pipeline.ko'])==contract['module_sha256']
    assert sha(files['u02_stream_dual.ko'])=='f67ec7145e579b65491199af534de02ea986af80c1615b42479607bbb1a85fa8'
    elf={n:verify_pair_host_elf(files[n]) for n in ('pair_host','pipeline_host')}
    tools_path=Path('D:/Develop/Xilink/SDK/2018.3/gnu/aarch64/nt/aarch64-linux/bin')
    probe_elf=subprocess.check_output([str(tools_path/'aarch64-linux-gnu-readelf.exe'),'-h','-l',str(files['dma_probe'])]).decode()
    assert 'AArch64' in probe_elf and 'INTERP' not in probe_elf
    assets=U02/'milestones/fresh_formal_stage_verified_20261015/assets'
    metas=[json.loads((ROOT/f'board_light_pose/outputs/samples/{i}.json').read_text()) for i in range(10)]
    assert all((m['source_size'],m['roi_xywh'])==(metas[0]['source_size'],metas[0]['roi_xywh']) for m in metas)
    geometry=' '.join(map(str,metas[0]['source_size']+metas[0]['roi_xywh']))
    out=U02/'reports'/('pipeline_board_'+str(time.time_ns()));out.mkdir()
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    (out/'helpers').mkdir()
    for name in ('bounded_pl_clock_check.py','observe_pl_frequency.py','verify_core_idle_readonly.py','pair_network_output_bundle.py','pipeline_output_bundle.py'):
        shutil.copyfile(U02/'inference'/name,out/'helpers'/name)
    for name,path in files.items():shutil.copyfile(path,out/('tested_'+name))
    hardware={'original':dict(project=portable['original']['project'],reports=portable['original']['reports']),
              'candidate':dict(project=candidate['project'],reports=candidate['reports'])}
    remote=prior['remote'];tools='pair_network_'+uuid.uuid4().hex
    directories={label:'pair_network_'+uuid.uuid4().hex for label in ('baseline','candidate','smoke','baseline_after')}
    state=dict(complete=False,terminal=False,phase='preflight',remote=remote,boot_id=prior['boot_id'],
        formal_bit_sha256=formal['bit_sha256'],candidate_bit_sha256=BIT_SHA256,candidate_kind='shared_model_abi9_whole_network',
        script_sha256=sha(out/'executed_script.py'),files_sha256={n:sha(p) for n,p in files.items()},elf=elf,
        restored_session=prior,driver_state='formal',bit_identity='original',dma_ambiguous=False,command_uncertain=False,
        board_connected=False,programming_pending=False,driver_changed=False,bitstream_programming_attempted=False,
        formal_restored=False,governor_restored=False,clock_changed=False,startup_files_modified=False,reboot=False,
        tools=tools,directories=directories,runs=[],summary={},formal_promoted=False,
        measured_clock_divisor=9,dma_allocation_bytes=8388608,
        scope='Same DW128 bit /9, one PL core, fixed normalized full networks; excludes cameras/JPEG/network',
        parameters=dict(threads_per_network=1,GOMP_SPINCOUNT=300000,poll_spin_us=1000,cached_dma=True,
                        warmup_pairs_per_input_pattern=6,measured_pairs_per_input_pattern=10),
        cpu_proof_state_sha256=sha(prepared/'cpu_verified/state.json'))
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('PIPELINE_BOARD_RECORD',out,flush=True)
    if args.reuse_baseline:
        previous=args.reuse_baseline.resolve();verify_frozen(previous.parent)
        old_state=json.loads((previous/'state.json').read_text())
        assert not old_state['complete'] and all(old_state[k] for k in ('terminal','formal_restored','governor_restored'))
        assert not any(old_state[k] for k in ('dma_ambiguous','command_uncertain','programming_pending','clock_changed','reboot','startup_files_modified'))
        assert 'Invalid source/ROI' in old_state['error'] and old_state['boot_id']==prior['boot_id']
        assert old_state['candidate_bit_sha256']==BIT_SHA256 and old_state['files_sha256']['pair_host']==sha(files['pair_host'])
        assert old_state['files_sha256']['u02_stream_dual.ko']==sha(files['u02_stream_dual.ko'])
        assert old_state['cpu_before']['governor']==old_state['cpu_after']['governor']
        assert sha(previous/'baseline/outputs.tar.gz')==old_state['baseline_outputs_sha256']
        assert len(old_state['runs'])==30 and all(r['stage']=='baseline' for r in old_state['runs'])
        shutil.copytree(previous/'baseline',out/'baseline')
        state['baseline_outputs_sha256']=old_state['baseline_outputs_sha256']
        state['summary'].update(old_state['summary'])
        state['reused_baseline']=dict(record=str(previous),state_sha256=sha(previous/'state.json'),
            archive_manifest_sha256=sha(previous.parent/'manifest.json'),same_boot=True,
            original_governor=old_state['cpu_before']['governor'],
            reason='Candidate-only padded-ROI entry fix; avoid repeating already completed identical baseline/CPU controls')
        save()
    if not args.execute:state.update(terminal=True,offline_preparation_only=True);save();return
    c=None;owned=False;governor_changed=False;before=None;log=(out/'uart.log').open('wb')
    modules={'formal':'u02_stream_resident','dual':'u02_stream_dual','pipeline':'u02_stream_pipeline'}
'''

TAIL = r'''
    def compare_known(actual,mode,sample):
        expected=reference/(('cpu2_parallel_' if mode=='cpu' else 'pl2_parallel_')+str(sample)+'.left.json')
        assert Path(str(actual)+'.heatmap').read_bytes()==Path(str(expected)+'.heatmap').read_bytes(),str(actual)
        assert json.loads(actual.read_text())['keypoints']==json.loads(expected.read_text())['keypoints'],str(actual)
    def legacy(stage,include_cpu):
        cases=[('pl1_parallel',1,'pl-stream')]
        if include_cpu:cases += [('cpu1_parallel',1,'cpu'),('cpu2_parallel',2,'cpu')]
        names=[]
        for index in range(10):
            right=(index+5)%10
            for label,threads,mode in (cases if index%2==0 else list(reversed(cases))):
                param='original_cpu.param' if mode=='cpu' else 'model.stream.param'
                run(f'cd {remote} && OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=300000 POSE_THREADS={threads} POSE_WARMUP=6 POSE_REPEAT=10 '
                    f'POSE_QUANT_VECTOR=1 POSE_PAIR_EXECUTION=parallel ./{tools}/pair_host {mode} {param} model.opt.bin '
                    f'samples/{index}.f32 samples/{right}.f32 {directories[stage]}/{label}_{index} {geometry}',mode=='pl-stream')
                state['runs'].append(dict(stage=stage,label=label,index=index,backend=mode,threads=threads));save()
                names += [f'{label}_{index}.json']+[f'{label}_{index}.{side}.json{suffix}' for side in ('left','right') for suffix in ('','.heatmap')]
            print('PIPELINE_ABI7_SAMPLE_DONE',stage,index,flush=True)
        dest=out/stage;dest.mkdir()
        state[stage+'_outputs_sha256']=download_pair_outputs(c,remote+'/'+directories[stage],names,dest);save()
        for label,threads,mode in cases:
            rows=[];elapsed_ns=0
            for index in range(10):
                p=dest/f'{label}_{index}';data=json.loads(Path(str(p)+'.json').read_text())
                assert data['dma_abi']==(7 if mode=='pl-stream' else 0) and data['threads_per_network']==threads
                assert data['warmup_pairs']==6 and data['measured_pairs']==10 and len(data['rows'])==10
                assert data['measured_pl_calls']==(2200 if mode=='pl-stream' else 0)
                assert data['all_pl_calls']==(3520 if mode=='pl-stream' else 0)
                for side,sample in (('left',index),('right',(index+5)%10)):
                    compare_known(Path(str(p)+'.'+side+'.json'),mode,sample)
                rows+=data['rows'];elapsed_ns+=sum(row['pair_end_ns']-row['pair_start_ns'] for row in data['rows'])
            state['summary'][stage+'_'+label]=summarize(rows,elapsed_ns,False)
        save()
    def pipeline_names(depth,index):
        prefix=f'pipeline_{depth}_{index}'
        return [prefix+'.json']+[f'{prefix}.{kind}.{i}.json{suffix}' for kind in ('actual','reference') for i in range(4) for suffix in ('','.heatmap')]
    def pipeline_case(stage,depth,index,warmup,repeats):
        indices=[index,(index+5)%10,(index+1)%10,(index+6)%10]
        inputs=' '.join(f'samples/{i}.f32' for i in indices)
        run(f'cd {remote} && OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=300000 POSE_WARMUP={warmup} POSE_REPEAT={repeats} '
            f'POSE_QUANT_VECTOR=1 POSE_INFLIGHT_PAIRS={depth} ./{tools}/pipeline_host pl-stream model.stream.param model.opt.bin '
            f'{inputs} {directories[stage]}/pipeline_{depth}_{index} {geometry}',True)
        state['runs'].append(dict(stage=stage,depth=depth,index=index,backend='pl-stream',warmup=warmup,repeats=repeats));save()
    def check_pipeline(dest,depth,index,warmup,repeats):
        p=dest/f'pipeline_{depth}_{index}';d=json.loads(Path(str(p)+'.json').read_text())
        assert d['backend']=='pl-stream' and d['dma_abi']==9 and d['contexts']==depth*2 and d['inflight_pairs']==depth
        assert d['threads_per_network']==1 and d['dma_arena_bytes']==8388608 and d['dma_used_bytes']==5767168
        assert d['shared_model_packets']==110 and d['shared_model_payload_bytes']==1445696 and d['pl_cores']==1
        assert d['warmup_pairs']==warmup and d['measured_pairs']==repeats and len(d['rows'])==repeats
        assert d['all_pl_calls']==(warmup+repeats)*220 and d['measured_pl_calls']==repeats*220
        assert d['all_full_heatmaps_exact'] and d['throughput_uses_elapsed_window_not_inverse_pair_latency']
        assert d['measured_end_ns']>d['measured_start_ns']
        for i,sample in enumerate([index,(index+5)%10,(index+1)%10,(index+6)%10]):
            compare_known(Path(str(p)+f'.actual.{i}.json'),'pl-stream',sample)
            assert Path(str(p)+f'.actual.{i}.json.heatmap').read_bytes()==Path(str(p)+f'.reference.{i}.json.heatmap').read_bytes()
        return d
    def summarize(rows,elapsed_ns,pipelined):
        assert len(rows)==100 and elapsed_ns>0
        latencies=[];profiles=[]
        for row in rows:
            assert row['pair_end_ns']>row['pair_start_ns']
            latencies.append((row['pair_end_ns']-row['pair_start_ns'])*1e-6)
            for side in ('left','right'):profiles.append(row[side])
        pl=any(p['pl_calls'] for p in profiles)
        assert all(p['pl_calls']==(110 if pl else 0) and p['pl_macs']==(72928704 if pl else 0) for p in profiles)
        def stat(values):return dict(median=statistics.median(values),p95=sorted(values)[math.ceil(.95*len(values))-1])
        return dict(pairs=100,elapsed_ns=elapsed_ns,processing_pairs_per_s=100*1e9/elapsed_ns,
            pair_latency_ms=stat(latencies),effective_ms_per_pair=elapsed_ns/1e6/100,
            sum_of_pair_latencies_is_not_throughput_denominator=pipelined,
            profile_median_ms={k:statistics.median(p[k] for p in profiles) for k in
                ('driver_wait_ms','submission_queue_ms','quantize_ms','dequantize_ms')})
    try:
        connect();before=snapshot_cpu_state(c);state['cpu_before']=before;save()
        if args.reuse_baseline:assert before['governor']==state['reused_baseline']['original_governor']
        assert before['online']=='0-3' and 'clk_ignore_unused' in before['kernel_cmdline'] and 'cpuidle.off=1' in before['kernel_cmdline']
        no_fds();module_idle();clock_readonly()
        state['initial_parameters']=checked('cat /sys/module/u02_stream_resident/parameters/cached_dma; cat /sys/module/u02_stream_resident/parameters/poll_spin_us')
        assert re.search(r'(?m)^Y\r?$',state['initial_parameters']) and re.search(r'(?m)^1000\r?$',state['initial_parameters'])
        expected={'original_pose':formal['host_sha256'],'original_resident.ko':formal['driver_sha256'],
                  'original_cpu.param':formal['cpu_param_sha256'],'model.stream.param':formal['stream_param_sha256'],'model.opt.bin':formal['model_weights_sha256']}
        expected.update({f'samples/{i}.f32':sha(assets/f'samples/{i}.f32') for i in range(10)})
        text=checked(f'cd {remote} && sha256sum original_pose original_resident.ko original_cpu.param model.stream.param model.opt.bin samples/[0-9].f32')
        assert dict((n,d) for d,n in re.findall(r'(?m)^([a-f0-9]{64})\s+(\S+)\r?$',text))==expected
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w:gz') as archive:
            for name,path in files.items():archive.add(path,arcname=name)
        raw=payload.getvalue();state['upload_bytes']=len(raw);save()
        ready='__PIPE_RX_'+uuid.uuid4().hex;done='__PIPE_RX_DONE_'+uuid.uuid4().hex
        c.send((f'mkdir {remote}/{tools} && timeout 30 nc -l -p 18082 > {remote}/{tools}/tools.tar.gz < /dev/null & rx=$!; sleep 1; '
            f'printf "\\n{ready}\\n"; wait $rx; printf "\\n{done}:%s\\n" "$?"\n').encode())
        c.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
        with socket.create_connection(('192.168.50.2',18082),timeout=10,source_address=('192.168.50.1',0)) as sock:
            sock.settimeout(30);sock.sendall(raw);sock.shutdown(socket.SHUT_WR)
        _,match=c.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',40);assert int(match[1])==0
        assert hashlib.sha256(raw).hexdigest() in checked(f'sha256sum {remote}/{tools}/tools.tar.gz')
        checked(f'cd {remote}/{tools} && tar -xzf tools.tar.gz && chmod 700 pair_host pipeline_host dma_probe')
        for name,digest in state['files_sha256'].items():assert digest in checked(f'sha256sum {remote}/{tools}/{name}')
        checked('mkdir '+' '.join(remote+'/'+d for d in directories.values()))
        assert re.fullmatch('[a-z]+',before['governor']);governor_changed=True
        checked('printf %s performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
        state['registers_baseline']=registers();assert state['registers_baseline']['ff5e00c0']==0x01010b02
        assert state['registers_baseline']['ff5e0030']==0x14000 and state['registers_baseline']['ff5e0040']&2;save()
        # Fresh independent JTAG idle/reset/AXI proof before the first MMIO job.
        owned=True;unload('initial_original_idle');program('verify_original_before','original',False);load('formal');original('before')
        unload('idle_before_candidate');program('program_candidate_at11','candidate');switch_clock(9)
        program('verify_candidate_at9','candidate',False)
        if not args.reuse_baseline:
            load('dual');legacy('baseline',True);unload('idle_before_abi9')
        load('pipeline')
        run(f'{remote}/{tools}/dma_probe {remote}/{tools}/dma_probe.json',False)
        c.download(remote+'/'+tools+'/dma_probe.json',out/'dma_probe.json')
        probe=json.loads((out/'dma_probe.json').read_text())
        assert probe['complete'] and probe['abi']==9 and probe['arena_bytes']==8388608 and probe['checked_regions']==24
        assert probe['cpu_mapping_disjoint'] and probe['original_bytes_restored'] and probe['pl_submissions']==0
        state['mapping_probe_verified']=True;state['memory_with_abi9']=checked('cat /proc/meminfo');save()
        pipeline_case('smoke',2,0,2,2)
        smoke=out/'smoke';smoke.mkdir()
        state['smoke_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories['smoke'],pipeline_names(2,0),smoke)
        check_pipeline(smoke,2,0,2,2);state['first_four_context_whole_network_exact']=True;save()
        names=[]
        for index in range(0,10,2):
            for depth in ((1,2) if index%4==0 else (2,1)):
                pipeline_case('candidate',depth,index,12,20);names+=pipeline_names(depth,index)
            print('PIPELINE_ABI9_QUARTET_DONE',index,flush=True)
        dest=out/'candidate';dest.mkdir()
        state['candidate_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories['candidate'],names,dest)
        for depth in (1,2):
            rows=[];elapsed=0
            for index in range(0,10,2):
                d=check_pipeline(dest,depth,index,12,20);rows+=d['rows'];elapsed+=d['measured_end_ns']-d['measured_start_ns']
            state['summary']['pipeline_'+str(depth)]=summarize(rows,elapsed,depth==2)
        save();unload('idle_after_abi9');load('dual');legacy('baseline_after',False)
        unload('idle_after_final_baseline');rollback()
        state.update(complete=True,phase='finished',all_full_heatmaps_and_keypoints_exact=True)
    except Exception as error:state['error']=repr(error);raise
    finally:
        try:
            if owned and not state['formal_restored']:rollback()
            if c and not state['command_uncertain'] and not state['dma_ambiguous']:
                if governor_changed:checked('printf %s '+before['governor']+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
                state['cpu_after']=snapshot_cpu_state(c)
                assert state['cpu_after']['boot_id']==prior['boot_id'] and state['cpu_after']['online']=='0-3'
                assert state['cpu_after']['governor']==before['governor'];clock_readonly();state['governor_restored']=True
            if owned and not state['formal_restored']:state['complete']=False
        except Exception as error:state['complete']=False;state['restore_error']=repr(error)
        if c:c.port.close()
        log.close();state['terminal']=True;save();print('PIPELINE_BOARD_TERMINAL',out,flush=True)

if __name__=='__main__':main()
'''


def generate():
    source=U02/'inference/validate_dual_fp32_board.py';text=source.read_text();tree=ast.parse(text)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes={n.name:n for n in main.body if isinstance(n,ast.FunctionDef)}
    names=('connect','checked','no_fds','module_idle','clock_readonly','registers','switch_clock','idle','unload','module_command','program','load','run','original','rollback')
    lines=text.splitlines(keepends=True);blocks=[];adaptations={}
    for name in names:
        node=nodes[name];block=''.join(lines[node.lineno-1:node.end_lineno])
        if name=='switch_clock':
            old='assert args.integer_dw118 and divisor in (9,11)';new='assert divisor in (9,11)'
            assert block.count(old)==1;block=block.replace(old,new);adaptations[name]=[old,new]
        if name=='rollback':
            old="        assert (out/'formal_before.json.heatmap').read_bytes()==(out/'formal_restored.json.heatmap').read_bytes()"
            new="        if (out/'formal_before.json.heatmap').exists():\n    "+old
            assert block.count(old)==1;block=block.replace(old,new);adaptations[name]=[old,new]
        blocks.append(block)
    prefix=text[:text.index('def main():')]
    # Keep the original imports, but identify the generated owner accurately.
    doc=tree.body[0]
    prefix='"""Isolated ABI9/DW128 whole-network owner; original /11 rollback retained."""\n'+''.join(prefix.splitlines(keepends=True)[doc.end_lineno:])
    prefix=prefix.replace('from retest_original_after_region import',
        'import sys\nsys.path.insert(0,'+repr(str(U02/'inference'))+')\nfrom retest_original_after_region import',1)
    generated=prefix+HEAD+'\n'+'\n'.join(blocks)+'\n'+TAIL
    parsed=ast.parse(generated);body=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    actual={n.name:n for n in body.body if isinstance(n,ast.FunctionDef)}
    for name,block in zip(names,blocks):
        expected=ast.parse('def container():\n'+block).body[0].body[0]
        assert ast.dump(expected,include_attributes=False)==ast.dump(actual[name],include_attributes=False),name
    return generated,dict(source=str(source),source_sha256=sha(source),unchanged_helpers=[n for n in names if n not in adaptations],helper_adaptations=adaptations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--restored-record',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=U02/'milestones/shared_model_pipeline_prepared_20261018')
    parser.add_argument('--reuse-baseline',type=Path)
    args=parser.parse_args();text,evidence=generate()
    out=U02/'reports'/('pipeline_board_prepared_'+str(time.time_ns()));out.mkdir()
    script=out/'validate_pipeline_board_owned.py';script.write_text(text)
    result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,check=True)
    (out/'offline_help.log').write_bytes(result.stdout+result.stderr)
    evidence.update(complete=True,board_accessed=False,generator_sha256=sha(Path(__file__)),generated_sha256=sha(script))
    (out/'state.json').write_text(json.dumps(evidence,indent=2)+'\n');print('PIPELINE_BOARD_PREPARED',out,flush=True)
    if args.execute or args.preflight:
        command=[sys.executable,str(script),'--restored-record',str(args.restored_record.resolve())]
        command+=['--prepared',str(args.prepared.resolve())]
        if args.reuse_baseline:command+=['--reuse-baseline',str(args.reuse_baseline.resolve())]
        if args.execute:command+=['--execute']
        raise SystemExit(subprocess.call(command,cwd=ROOT))


if __name__=='__main__':main()
