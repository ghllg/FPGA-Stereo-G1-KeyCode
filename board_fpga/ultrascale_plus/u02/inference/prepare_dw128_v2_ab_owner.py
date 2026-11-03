"""Prepare a three-stage full-network A/B owner; default is offline generation.

Generated execution requires an explicit current-boot restored record. It
retains the established clock/AXI/DMA helpers and never changes startup media.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from prepare_dw128_v2_board_ab import U02, sha, verify_frozen


def hardware_tree():
    root = U02 / "milestones/dw128_v2_hardware_prepared_20261017"
    verify_frozen(root)
    descriptor = json.loads((root / "descriptor.json").read_text())
    assert descriptor["schema"] == "u02_dw128_v2_hardware_tree_v1"
    assert descriptor["offline_prepared"] and not descriptor["board_accessed"]
    result = {}
    for label in ("original", "dw118", "candidate"):
        item = descriptor["hardware"][label]
        project, reports = root / item["project"], root / item["reports"]
        bit = project / "u02_stream.runs/impl_1/compute_wrapper.bit"
        assert sha(bit) == item["bit_sha256"]
        assert sha(reports / "compute.ltx") == item["ltx_sha256"]
        result[label] = dict(project=project, reports=reports, bit=bit)
    assert len({sha(item["bit"]) for item in result.values()}) == 3
    return result


def prepared_children():
    root = U02 / "milestones/dw128_v2_pair_child_prepared_20261017"
    verify_frozen(root)
    return {label: root / (label + "_pair_child.py") for label in ("dw118", "candidate")}


def compare_children(records):
    """Recheck full saved outputs, not just successful process exit/summary flags."""
    states = {}
    for label, directory in records.items():
        state = json.loads((directory / "state.json").read_text())
        assert all(state[key] for key in ("complete", "terminal", "driver_restored",
                                         "governor_restored", "all_heatmaps_keypoints_confidences_exact"))
        assert not state["dma_ambiguous"] and state["command_returned"]
        assert state["pl_clock_divisor"] == 9
        assert state["host_sha256"] == "94806cbd5c4c8d15db26f3392e33e28073d2c75852d082cadcb551b126a3f65c"
        states[label] = state
    base = states["baseline_before"]
    for state in states.values():
        assert state["assets_before"] == base["assets_before"]
        for key in ("boot_id", "online", "kernel_cmdline", "min_frequency_khz", "max_frequency_khz"):
            assert state["cpu_before"][key] == base["cpu_before"][key], key
    for case, calls in (("pl1_parallel", 110), ("cpu1_parallel", 0)):
        for index in range(10):
            for label, directory in records.items():
                value = json.loads((directory / f"{case}_{index}.json").read_text())
                assert value["threads_per_network"] == 1 and value["execution"] == "parallel"
                assert value["warmup_pairs"] == 3 and value["measured_pairs"] == 10
                assert len(value["rows"]) == 10 and value["measured_pl_calls"] == calls * 2 * 10
                assert all(row[side]["pl_calls"] == calls for row in value["rows"] for side in ("left", "right"))
                for side in ("left", "right"):
                    name = f"{case}_{index}.{side}.json"
                    reference = records["baseline_before"] / name
                    output = directory / name
                    heatmap = output.with_suffix(".json.heatmap").read_bytes()
                    assert len(heatmap) == 52224
                    assert heatmap == reference.with_suffix(".json.heatmap").read_bytes()
                    assert json.loads(output.read_text())["keypoints"] == json.loads(reference.read_text())["keypoints"]
    medians = {label: state["summary"]["pl1_parallel"]["pair_wall_median_ms"] for label, state in states.items()}
    candidate_ms = medians["candidate"]
    return {
        "all_saved_outputs_exact": True, "metrics": {label: state["summary"] for label, state in states.items()},
        "baseline_before_over_candidate": medians["baseline_before"] / candidate_ms,
        "baseline_after_over_candidate": medians["baseline_after"] / candidate_ms,
        "candidate_faster_than_both_baselines": candidate_ms < min(medians["baseline_before"], medians["baseline_after"]),
        "baseline_after_over_before": medians["baseline_after"] / medians["baseline_before"],
        "timing_scope": "Prepared full-network pairs; excludes cameras/JPEG/network",
        "formal_promoted": False,
    }


HEAD = r'''def main():
    parser=argparse.ArgumentParser(description="Owned DW118 / candidate / DW118 complete-network A/B")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--restored-record',type=Path,required=True)
    args=parser.parse_args()
    args.bounded_clock_check=True;args.clock_capture_samples=13
    from prepare_dw128_v2_ab_owner import hardware_tree,prepared_children,compare_children
    from restored_board_session import load_restored_board_session,load_restored_baseline_metadata
    from bounded_pl_clock_check import instrument_capture,verify_capture
    from verify_core_idle_readonly import verify_core_idle_readonly
    for name,digest in REQUIRED_SOURCES.items():assert sha(Path(name))==digest,name
    hardware=hardware_tree();children=prepared_children()
    formal=json.loads((U02/'inference/stream_ingress_parallel_affine_runtime.json').read_text())
    assets=U02/'milestones/fresh_formal_stage_verified_20261015/assets'
    formal.update(host=str(assets/'original_pose'),driver=str(assets/'original_resident.ko'))
    for key in ('host','driver'):assert sha(Path(formal[key]))==formal[key+'_sha256']
    prior=load_restored_board_session(args.restored_record,formal['bit_sha256'])
    baseline_meta=load_restored_baseline_metadata(args.restored_record,formal['bit_sha256'])
    remote=prior['remote'];results='dw_ab_'+uuid.uuid4().hex
    original_project=hardware['original']['project'];original_reports=hardware['original']['reports']
    out=U02/'reports'/('dw128_v2_board_ab_'+str(time.time_ns()));out.mkdir()
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    state=dict(schema='u02_dw128_v2_ab_owner_v1',complete=False,terminal=False,phase='prepared',
        owner_pid=os.getpid(),remote=remote,results=results,boot_id=prior['boot_id'],
        restored_record_sha256=sha(args.restored_record/'state.json'),
        script_sha256=sha(out/'executed_script.py'),formal_bit_sha256=formal['bit_sha256'],
        host_sha256=formal['host_sha256'],driver_sha256=formal['driver_sha256'],
        driver_loaded=True,clock_changed=False,formal_restored=False,governor_restored=False,
        dma_ambiguous=False,command_uncertain=False,gate_pending=False,
        startup_files_modified=False,reboot=False,formal_promoted=False,children={})
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('DW128_V2_BOARD_AB_RECORD',out,flush=True)
    if args.prepare_only:
        state.update(terminal=True,offline_preparation_only=True,board_accessed=False)
        save();return
    c=None;baseline=None;hardware_owned=False;log=(out/'uart.log').open('wb')
'''

TAIL = r'''
    def no_fds_and_refs():
        opened=checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/u02*|/dev/video*) printf "OPEN:%s\\n" "$t";; esac; done; true')
        assert not re.search(r'(?m)^OPEN:',opened)
        assert re.search(r'(?m)^u02_stream_resident\s+\d+\s+0\s',checked('cat /proc/modules'))
    try:
        connect();state['board_accessed']=True;state['cpu_before']=snapshot_cpu_state(c)
        assert state['cpu_before']['online']=='0-3'
        assert all(flag in state['cpu_before']['kernel_cmdline'].split() for flag in ('clk_ignore_unused','cpuidle.off=1'))
        no_fds_and_refs()
        settings=checked('cat /sys/module/u02_stream_resident/parameters/cached_dma; cat /sys/module/u02_stream_resident/parameters/poll_spin_us')
        assert re.search(r'(?m)^Y\r?$',settings) and re.search(r'(?m)^1000\r?$',settings)
        for name,digest in (('original_pose',formal['host_sha256']),('original_resident.ko',formal['driver_sha256']),
                            ('model.stream.param',formal['stream_param_sha256']),('model.opt.bin',formal['model_weights_sha256'])):
            assert digest in checked(f'sha256sum {remote}/{name}')
        state['original_governor']=state['cpu_before']['governor']
        assert re.fullmatch('[a-z]+',state['original_governor'])
        state['governor_changed']=True;save()
        checked('printf %s performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor');time.sleep(.1)
        baseline=registers()
        assert baseline['ff5e00c0']==0x01010b02 and baseline['ff5e0030']==0x14000 and baseline['ff5e0040']&2
        state['registers_baseline']=baseline;save();checked(f'mkdir {remote}/{results}')
        hardware_owned=True;unload()
        gate('original_before',original_project,original_reports,True,11);load()
        measure('original',0,'pl-stream')
        records={}
        for label,variant in (('baseline_before','dw118'),('candidate','candidate'),('baseline_after','dw118')):
            state.update(phase='switching',formal_restored=False);save()
            no_fds_and_refs();unload();clock(11)
            item=hardware[variant]
            gate(label+'_same_clock',item['project'],item['reports'],True,11)
            clock(9);gate(label+'_high_clock',item['project'],item['reports'],False,9);load()
            child_source=children[variant]
            state.update(phase='child_running',active_bit_sha256=sha(item['bit']),
                active_pl_clock_divisor=9,child_script_sha256=sha(child_source),dma_ambiguous=True)
            save();c.port.close();c=None
            with (out/(label+'.log')).open('wb') as stream:
                proc=subprocess.Popen([sys.executable,str(child_source),'--execute','--restored-record',str(args.restored_record.resolve()),
                    '--owner-record',str(out)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                state['comparison_pid']=proc.pid;save();code=proc.wait()
            text=(out/(label+'.log')).read_text(errors='replace')
            match=re.search(r'PAIR_DUAL_RECORD (.+)',text);assert match,text
            child_record=Path(match.group(1).strip());child=json.loads((child_record/'state.json').read_text())
            state['children'][label]=dict(record=str(child_record),exit_code=code,state_sha256=sha(child_record/'state.json'))
            save()
            assert child['terminal'] and child['driver_restored'] and child['governor_restored']
            assert child['command_returned'] and not child['dma_ambiguous']
            assert child['hardware_bit_sha256']==sha(item['bit']) and child['pl_clock_divisor']==9
            state['gate_pending']=True;save()
            verify_core_idle_readonly(item['project'],item['reports'],out/(label+'_idle_after'),clock_divisor=9)
            state['gate_pending']=False;save();connect();no_fds_and_refs()
            state['dma_ambiguous']=False;state['phase']='child_finished';save()
            assert code==0 and child['complete'],child
            records[label]=child_record
        state['comparison']=compare_children(records);save()
        restore();assert state['formal_restored']
        measure('restored',0,'pl-stream');assert registers()==baseline
        state['rollback_outputs_sha256']={}
        for label in ('original','restored'):
            for suffix in ('.json','.json.heatmap'):
                name=label+'_0'+suffix;c.download(remote+'/'+results+'/'+name,out/name)
                state['rollback_outputs_sha256'][name]=sha(out/name)
        assert (out/'original_0.json.heatmap').read_bytes()==(out/'restored_0.json.heatmap').read_bytes()
        state['complete']=True;state['phase']='finished'
    except Exception as error:
        state['error']=repr(error);raise
    finally:
        try:
            if hardware_owned and not state['formal_restored']:restore()
            if c and not state['dma_ambiguous'] and not state['command_uncertain'] and not state['gate_pending']:
                if state.get('governor_changed'):
                    checked('printf %s '+state['original_governor']+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
                state['cpu_after']=snapshot_cpu_state(c)
                assert state['cpu_after']['boot_id']==prior['boot_id'] and state['cpu_after']['online']=='0-3'
                assert state['cpu_after']['governor']==state['original_governor']
                state['governor_restored']=True
                if state['formal_restored']:assert registers()['ff5e00c0']==0x01010b02
            if not state['formal_restored'] or not state['governor_restored']:state['complete']=False
        except Exception as error:
            state['complete']=False;state['restore_error']=repr(error)
        if c:c.port.close()
        log.close()
        state['terminal']=not (state['gate_pending'] or (state['phase']=='child_running' and state['dma_ambiguous']))
        if not state['terminal']:
            state['observer_detached_child_may_be_running']=True
            state['complete']=False
        save();print('DW128_V2_BOARD_AB_OWNER_EXIT',out,flush=True)

if __name__=='__main__':main()
'''


def generate():
    base = U02 / "inference/validate_dw_stem_single.py"
    generator = U02 / "inference/prepare_dual_stereo_dw118.py"
    source = base.read_text()
    main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    helpers = {n.name: n for n in main.body if isinstance(n, ast.FunctionDef)}
    parent = next(n for n in ast.parse(generator.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assignment = next(n for n in parent.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "adaptations" for t in n.targets))
    adaptations = ast.literal_eval(assignment.value)
    lines = source.splitlines(keepends=True)
    blocks = {}
    for name in ("connect", "checked", "registers", "unload", "load", "clock", "gate", "measure", "restore"):
        node = helpers[name]
        text = "".join(lines[node.lineno - 1:node.end_lineno])
        for before, after in adaptations.get(name, []):
            assert text.count(before) == 1, (name, before)
            text = text.replace(before, after)
        blocks[name] = text
    dependencies = [base, generator, Path(__file__),
                    U02 / "inference/prepare_dw128_v2_pair_child.py",
                    U02 / "inference/prepare_dw128_v2_board_ab.py",
                    U02 / "inference/restored_board_session.py",
                    U02 / "inference/verify_core_idle_readonly.py",
                    U02 / "inference/bounded_pl_clock_check.py",
                    U02 / "inference/observe_pl_frequency.py"]
    prefix = source[:source.index("def main():")]
    prefix = prefix.replace("from retest_original_after_region import",
        "sys.path.insert(0," + repr(str(U02 / "inference")) + ")\nfrom retest_original_after_region import", 1)
    required = {str(path.resolve()): sha(path) for path in dependencies}
    generated = prefix + "\nREQUIRED_SOURCES=" + repr(required) + "\n\n" + HEAD + "\n" + "\n".join(blocks.values()) + "\n" + TAIL
    compile(generated, "<dw128-v2-owner>", "exec")
    parsed = next(n for n in ast.parse(generated).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    actual = {n.name: n for n in parsed.body if isinstance(n, ast.FunctionDef)}
    for name, block in blocks.items():
        expected = ast.parse("def container():\n" + block).body[0].body[0]
        assert ast.dump(expected, include_attributes=False) == ast.dump(actual[name], include_attributes=False), name
    assert "taskkill" not in generated
    return generated, required


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    hardware_tree();prepared_children()
    source, required = generate()
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    script = output / "run_owned_ab.py";script.write_text(source)
    help_result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=True)
    (output / "help.log").write_text(help_result.stdout + help_result.stderr)
    (output / "preparation.json").write_text(json.dumps({
        "offline_prepared": True, "board_accessed": False, "hardware_executed": False,
        "owner_script_sha256": sha(script), "helper_sources": required,
        "sequence": ["DW118 baseline_before", "DW128 candidate", "DW118 baseline_after", "original rollback"],
        "requires": "Explicit execute with current-boot restored record; board use must be available",
    }, indent=2) + "\n")
    print("DW128_V2_AB_OWNER_PREPARED", output)


if __name__ == "__main__":
    main()
