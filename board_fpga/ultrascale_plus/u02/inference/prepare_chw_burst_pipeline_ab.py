"""Matched hardware-only ABI9 pipeline A/B using accepted ownership helpers."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_dw128_v2_board_ab import U02, sha, verify_frozen
from prepare_chw_burst_board_ab import hardware_tree, compare_children
from prepare_pipeline_board import generate as pipeline_source
from prepare_pipeline_irq_board import EXTRA, replace_once

PREPARED = U02 / 'milestones/chw_burst_pipeline_ab_prepared_20261018'


def accepted_hardware():
    archive = U02 / 'milestones/chw_burst_board_ab_verified_20261018'
    verify_frozen(archive)
    state = json.loads((archive / 'hardware_owner/state.json').read_text())
    assert all(state[k] for k in ('complete', 'terminal', 'formal_restored', 'governor_restored'))
    assert state['candidate_kind'] == 'chw_burst'
    assert not any(state[k] for k in ('dma_ambiguous', 'gate_pending', 'command_uncertain', 'formal_promoted'))
    records = {label: archive / label for label in ('baseline_before', 'candidate', 'baseline_after')}
    comparison = compare_children(records)
    assert comparison == state['comparison'] and comparison['candidate_faster_than_both_baselines']
    tree = hardware_tree()
    for label, role in (('candidate', 'candidate'), ('baseline_before', 'baseline'), ('baseline_after', 'baseline')):
        assert json.loads((records[label] / 'state.json').read_text())['hardware_bit_sha256'] == sha(tree[role]['bit'])
    return tree


RUN = r'''
        load('dual');legacy('baseline',True);unload('idle_before_pipeline_baseline')
        load('pipeline');mapping_probe('baseline');cohort('baseline')
        unload('idle_before_chw');switch_clock(11)
        hardware['candidate']=candidate
        state['active_experimental_bit_sha256']=sha(candidate['bit']);save()
        program('program_chw_at11','candidate');switch_clock(9)
        program('verify_chw_at9','candidate',False)
        load('pipeline');mapping_probe('candidate')
        pipeline_case('smoke',2,0,2,2)
        smoke=out/'smoke';smoke.mkdir()
        state['smoke_outputs_sha256']=download_pipeline_outputs(c,remote+'/'+directories['smoke'],pipeline_names(2,0),smoke)
        check_pipeline(smoke,2,0,2,2);state['first_four_context_whole_network_exact']=True;save()
        cohort('candidate')
        unload('idle_before_baseline_after');switch_clock(11)
        hardware['candidate']=baseline
        state['active_experimental_bit_sha256']=sha(baseline['bit']);save()
        program('program_dw128_after_at11','candidate');switch_clock(9)
        program('verify_dw128_after_at9','candidate',False)
        load('pipeline');mapping_probe('baseline_after');cohort('baseline_after')
        unload('idle_after_final_baseline');rollback()
        state.update(complete=True,phase='finished',all_full_heatmaps_and_keypoints_exact=True)
'''


def generate():
    text, evidence = pipeline_source()
    text = replace_once(text, 'from dw128_v2_stereo_hardware import accepted_candidate, BIT_SHA256',
        'from prepare_chw_burst_pipeline_ab import accepted_hardware')
    text = replace_once(text, 'candidate=accepted_candidate();assert sha(candidate[\'bit\'])==BIT_SHA256',
        "hardware_set=accepted_hardware();candidate=hardware_set['candidate'];baseline=hardware_set['baseline'];BIT_SHA256=sha(candidate['bit'])")
    text = replace_once(text, "args=ap.parse_args()", "args=ap.parse_args()\n    assert not args.reuse_baseline,'Fresh hardware controls required'\n    for p,d in REQUIRED_SOURCES.items():assert sha(Path(p))==d,p")
    text = replace_once(text, "default=U02/'milestones/shared_model_pipeline_prepared_20261018'",
        "default=U02/'milestones/shared_model_pipeline_roi_prepared_20261018'")
    text = replace_once(text, "('pipeline_board_'+str(time.time_ns()))", "('chw_pipeline_board_'+str(time.time_ns()))")
    text = replace_once(text, "candidate_kind='shared_model_abi9_whole_network'", "candidate_kind='chw_burst_abi9_hardware_ab'")
    text = replace_once(text, "'candidate':dict(project=candidate['project'],reports=candidate['reports'])", "'candidate':baseline")
    text = replace_once(text, "scope='Same DW128 bit /9, one PL core, fixed normalized full networks; excludes cameras/JPEG/network'",
        "scope='DW128/CHW/DW128 same polling ABI9 Host, depth2 full networks; excludes cameras/JPEG/network',active_experimental_bit_sha256=sha(baseline['bit'])")
    text = replace_once(text, 'warmup_pairs_per_input_pattern=6,measured_pairs_per_input_pattern=10',
        'warmup_pairs_per_input_pattern=12,measured_pairs_per_input_pattern=20,inflight_pairs=2')
    text = replace_once(text, '    elf={n:verify_pair_host_elf',
        "    files['recover_pipeline']=U02/'milestones/stream_irq_prepared_20261018/binaries/recover_pipeline_transport'\n    elf={n:verify_pair_host_elf")
    text = replace_once(text, 'chmod 700 pair_host pipeline_host dma_probe', 'chmod 700 pair_host pipeline_host dma_probe recover_pipeline')
    text = replace_once(text, "        if not c:connect()\n        if state['driver_state']!='none':unload",
        "        if not c:connect()\n        if state['driver_state']=='pipeline' and state['dma_ambiguous']:recover_observed_pipeline_failure()\n        if state['driver_state']!='none':unload")
    helpers = EXTRA[:EXTRA.index('    def irq_stats')] + EXTRA[EXTRA.index('    def mapping_probe'):]
    helpers = helpers.replace("print('IRQ_AB_QUARTET_DONE'", "print('CHW_PIPELINE_AB_QUARTET_DONE'")
    # Baseline's ABI7 files already occupy out/baseline; give pipeline cohorts separate folders.
    helpers = replace_once(helpers, 'dest=out/stage;dest.mkdir()', "dest=out/('pipeline_'+stage);dest.mkdir()")
    text = replace_once(text, '    try:\n        connect();before=snapshot_cpu_state(c)', helpers + '\n    try:\n        connect();before=snapshot_cpu_state(c)')
    start = text.index("        if not args.reuse_baseline:\n            load('dual')")
    end = text.index('    except Exception as error:', start)
    text = text[:start] + RUN + text[end:]
    text = text.replace('DW128 /9 full-network ABI7 vs ABI9 pipeline A/B with original /11 rollback',
        'DW128/CHW/DW128 same-Host ABI9 full-network A/B and original rollback')
    paths = [Path(__file__), U02 / 'inference/prepare_pipeline_board.py',
        U02 / 'inference/prepare_pipeline_irq_board.py', U02 / 'inference/prepare_chw_burst_board_ab.py',
        U02 / 'inference/validate_dual_fp32_board.py', U02 / 'inference/verify_core_idle_readonly.py']
    required = {str(p.resolve()): sha(p) for p in paths}
    text = replace_once(text, '\ndef main():', '\nREQUIRED_SOURCES=' + repr(required) + '\n\ndef main():')
    compile(text, '<chw-pipeline-owner>', 'exec')
    def functions(source):
        main = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        return {n.name: ast.dump(n, include_attributes=False) for n in main.body if isinstance(n, ast.FunctionDef)}
    original, _ = pipeline_source()
    before, after = functions(original), functions(text)
    assert all(before[n] == after[n] for n in before if n != 'rollback')
    evidence.update(complete=True, board_accessed=False, hardware_executed=False,
        required_sources=required, unchanged_helpers=[n for n in before if n != 'rollback'],
        rollback_adaptation='Observed-exit independent-idle kernel RECOVER before unloading pinned pipeline DMA',
        sequence=['DW128 ABI9', 'CHW ABI9', 'DW128 ABI9', 'original /11 rollback'],
        transport='Same Host and polling driver; only active experimental bit/LTX changed while unloaded at /11')
    return text, evidence


def prepare(restored_record):
    assert not PREPARED.exists()
    accepted_hardware()
    text, evidence = generate()
    PREPARED.mkdir()
    script = PREPARED / 'run_owned_pipeline_ab.py'
    script.write_text(text, encoding='utf-8')
    result = subprocess.run([sys.executable, str(script), '--restored-record', str(restored_record.resolve())],
        capture_output=True, text=True, check=True)
    (PREPARED / 'offline_preflight.log').write_text(result.stdout + result.stderr)
    evidence['script_sha256'] = sha(script)
    (PREPARED / 'preparation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    shutil.copyfile(Path(__file__), PREPARED / 'executed_prepare.py')
    (PREPARED / '.gitattributes').write_bytes(b'* -text whitespace=cr-at-eol,-blank-at-eof\n')
    (PREPARED / '.gitignore').write_bytes(b'!*\n!*/\n')
    files = {p.name: sha(p) for p in PREPARED.iterdir() if p.is_file()}
    (PREPARED / 'manifest.json').write_text(json.dumps(dict(files_sha256=files), indent=2) + '\n')
    verify_frozen(PREPARED)
    print('CHW_PIPELINE_HARDWARE_AB_PREPARED', PREPARED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored-record', type=Path, required=True)
    prepare(parser.parse_args().restored_record)
