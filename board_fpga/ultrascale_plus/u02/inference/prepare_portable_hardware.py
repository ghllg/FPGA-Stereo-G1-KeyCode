"""Prepare runtime-only bit/LTX trees from frozen Git evidence, without board access.

This verifies deployment identity and archived acceptance, not a new synthesis,
route, RTL run or cold boot. It does not replace the existing build audit.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

U02=Path(__file__).resolve().parents[1]
ORIGINAL_BIT='1de1bf6d4c13a9bf2be443554f142baabccb69e8464b608a9c14cece836861fb'
DW_BIT='f9422aca895e2c7f336907b80fd3712108ef2f9ddc562635f117fe3cc9a1f4d6'
RTL_STATE='66789760b5836ac76b6b5d2b9fa2c1305f5501a52653b896c13fb23c67a25d97'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify_archive(directory):
    directory=directory.resolve();manifest=json.loads((directory/'manifest.json').read_text())
    entries=manifest.get('files_sha256',manifest.get('files'))
    assert entries
    for name,digest in entries.items():
        name=name.replace('\\','/')
        assert ':' not in name and not name.startswith('/')
        path=(directory/name).resolve();assert path.is_relative_to(directory)
        assert path.is_file() and sha(path)==digest,name
    return dict(directory=str(directory),manifest_sha256=sha(directory/'manifest.json'),verified_files=len(entries))


def load_prepared(directory):
    """Resolve a prepared tree, verifying its bytes and frozen proof links."""
    directory=Path(directory).resolve();verify_archive(directory)
    descriptor=json.loads((directory/'descriptor.json').read_text())
    assert descriptor['schema']=='u02_portable_hardware_tree_v1'
    assert descriptor['complete'] and not descriptor['board_accessed']
    sources=('formal_stereo_8p46_20261015','formal_dw_routed_118m_20261015',
             'formal_dw_full_rtl_pass_20261015','dual_stereo_dw118_5min_verified_20261015')
    assert len(descriptor['source_evidence'])==len(sources)
    for name,evidence in zip(sources,descriptor['source_evidence']):
        assert sha(U02/'milestones'/name/'manifest.json')==evidence['manifest_sha256']
    dw=U02/'milestones/formal_dw_routed_118m_20261015'
    rtl=U02/'milestones/formal_dw_full_rtl_pass_20261015'
    state=json.loads((dw/'state.json').read_text())
    assert state['complete'] and state['terminal'] and state['route_checks_passed'] and state['full_rtl_verified']
    assert state['rtl_proof_sha256']==sha(rtl/'state.json')==RTL_STATE
    result={}
    source_ltx={'original':U02/'milestones/formal_stereo_8p46_20261015/runtime/compute.ltx',
                'dw118':dw/'implementation/compute.ltx'}
    for label,expected in [('original',ORIGINAL_BIT),('dw118',DW_BIT)]:
        item=descriptor['hardware'][label];project=(directory/item['project']).resolve();reports=(directory/item['reports']).resolve()
        assert project.is_relative_to(directory) and reports.is_relative_to(directory)
        bit=project/'u02_stream.runs/impl_1/compute_wrapper.bit'
        assert sha(bit)==item['bit_sha256']==expected
        assert sha(reports/'compute.ltx')==item['ltx_sha256']==sha(source_ltx[label])
        result[label]=dict(project=project,reports=reports,bit=bit)
    result['descriptor_sha256']=sha(directory/'descriptor.json')
    return result


def prepare(destination):
    destination=destination.resolve()
    assert destination.is_relative_to(U02/'reports') and not destination.exists()
    original=U02/'milestones/formal_stereo_8p46_20261015'
    dw=U02/'milestones/formal_dw_routed_118m_20261015'
    rtl=U02/'milestones/formal_dw_full_rtl_pass_20261015'
    accepted=U02/'milestones/dual_stereo_dw118_5min_verified_20261015'
    evidence=[verify_archive(p) for p in (original,dw,rtl,accepted)]
    state=json.loads((dw/'state.json').read_text());route=json.loads((dw/'route_inspection.json').read_text())
    for key in ('complete','terminal','full_rtl_verified','route_checks_passed','original_bit_unchanged'):assert state[key] is True
    assert state['bit_sha256']==route['bit_sha256']==sha(dw/'compute_wrapper.bit')==DW_BIT
    assert state['original_bit_sha256']==ORIGINAL_BIT
    assert state['rtl_proof_sha256']==sha(rtl/'state.json')==RTL_STATE
    assert route['route_checks_passed'] and route['route_errors']==0
    assert abs(route['timing']['pl_clock_report_rounded_period_ns']-8.437)<.002
    assert min(route['timing'][k] for k in ('wns_ns','whs_ns','wpws_ns'))>=0
    assert route['timing']['unconstrained_or_clock_check_violations']==0
    for name,digest in route['report_sha256'].items():
        folder=dw/('implementation' if name in ('timing.rpt','utilization.rpt','drc.rpt') else 'route_audit')
        assert sha(folder/name)==digest,name
    proof=json.loads((rtl/'state.json').read_text());groups=json.loads((rtl/'all_groups.json').read_text())
    assert proof['complete'] and proof['terminal'] and not proof['board_accessed']
    assert proof['transaction_count']==243 and proof['real_groups']==110 and proof['values']==871872
    assert groups['transactions']==list(range(243)) and groups['unknown_words']==0
    assert len(groups['rows'])==groups['group_count']==110 and sum(g['values'] for g in groups['rows'])==871872
    assert all(g['mismatches']==g['unknown_values']==0 for g in groups['rows'])
    log=(rtl/'rtl_replay.log').read_text(errors='replace')
    assert re.search(r'RTL Simulation : 243 / 243',log) and '$finish called' in log
    assert not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:)',log)
    post=(rtl/'postcheck.log').read_text(errors='replace')
    for marker in ('U02_STREAM_ALL_REAL_GROUPS_PASSED groups=110 values=871872',
                   'U02_DW_ADDRESS_EXHAUSTIVE_PASSED checks=4926884 shapes=3072',
                   'U02_STREAM_REQUANT_CACHE_CONTRACT_PASSED cases=2 rejected=12 recovery=2',
                   'U02_STREAM_BANK_AND_WEIGHT_TAILS_PASSED cases=4',
                   'U02_STREAM_UNALIGNED_INPUT_BOUNDARIES_PASSED cases=3'):assert marker in post,marker
    assert all(proof['phases'][key]['exit_code']==0 for key in ('rtl_replay','postcheck','independent_compare'))
    raw_expected=[digest for path,digest in proof['files_sha256'].items()
                  if path.replace('\\','/').endswith('/rtldatafile/rtl.u02_stream.autotvout_results.dat')]
    assert len(raw_expected)==1
    digest=hashlib.sha256()
    with gzip.open(rtl/'rtl_results.dat.gz','rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==raw_expected[0]
    runtime=json.loads((U02/'inference/dual_context_stereo_dw118_5min_verified_runtime_20261015.json').read_text())
    child_path=accepted/'verified/state.json';owner_path=accepted/'hardware_owner/state.json'
    assert sha(child_path)==runtime['record_sha256'] and sha(owner_path)==runtime['hardware_owner_record_sha256']
    child=json.loads(child_path.read_text());owner=json.loads(owner_path.read_text())
    assert child['complete'] and child['terminal'] and child['final_both_views_replay_exact']
    assert child['hardware_bit_sha256']==owner['candidate_bit_sha256']==DW_BIT
    assert owner['child_state_sha256']==sha(child_path)
    assert owner['complete'] and owner['terminal'] and owner['formal_restored'] and owner['governor_restored']
    assert not owner['dma_ambiguous'] and not owner['clock_changed'] and owner['automatic_controller_completed']
    assert sha(accepted/'hardware_owner/executed_script.py')==owner['script_sha256']
    assert child['camera_elapsed_s']>=299 and child['network_received_pairs']>3000
    for side in ('left','right'):
        for suffix in ('.input.f32','.heatmap'):
            assert (accepted/f'verified/stereo.json.{side}.json{suffix}').read_bytes()==(accepted/f'verified/replay_{side}.json{suffix}').read_bytes()
    result=dict(schema='u02_portable_hardware_tree_v1',complete=True,board_accessed=False,
        source_evidence=evidence,scope='Archived deployment identity and acceptance; no new build/RTL/board/cold-boot claim',hardware={})
    destination.mkdir()
    for label,bit,ltx,expected in [('original',original/'runtime/compute_wrapper.bit',original/'runtime/compute.ltx',ORIGINAL_BIT),
                                   ('dw118',dw/'compute_wrapper.bit',dw/'implementation/compute.ltx',DW_BIT)]:
        assert sha(bit)==expected
        project=destination/label/'project';reports=destination/label/'reports'
        target=project/'u02_stream.runs/impl_1/compute_wrapper.bit';target.parent.mkdir(parents=True)
        reports.mkdir();shutil.copy2(bit,target);shutil.copy2(ltx,reports/'compute.ltx')
        assert sha(target)==expected and sha(reports/'compute.ltx')==sha(ltx)
        result['hardware'][label]=dict(project=project.relative_to(destination).as_posix(),reports=reports.relative_to(destination).as_posix(),
            bit_sha256=expected,ltx_sha256=sha(ltx))
        for kind,filename in [('program','program_stream.tcl'),('verify','verify_resident_interface.tcl'),('idle_readonly','verify_resident_interface.tcl')]:
            text=(U02/'inference'/filename).read_text()
            old='source [file join [file dirname [file normalize [info script]]] stream_artifacts.tcl]'
            assert text.count(old)==1
            paths=[project,reports,U02/'board'];assert all(not any(c in str(p) for c in '{}\r\n') for p in paths)
            text=text.replace(old,'set u02_project {'+project.as_posix()+'}\nset u02_implementation {'+reports.as_posix()+'}')
            old='set root [file normalize [file join [file dirname [file normalize [info script]]] ../board]]'
            assert text.count(old)==1;text=text.replace(old,'set root {'+(U02/'board').as_posix()+'}')
            if kind=='idle_readonly':
                text=text[:text.index('create_hw_axi_txn u02_write_count')]
                text+='close_hw_target\ndisconnect_hw_server\nclose_hw\nputs U02_READONLY_CORE_IDLE_VERIFIED\n'
                assert '-type WRITE' not in text and 'program_hw_devices' not in text
            assert 'D:/u02_' not in text and 'D:\\u02_' not in text
            (destination/label/(kind+'.tcl')).write_text(text)
    (destination/'descriptor.json').write_text(json.dumps(result,indent=2)+'\n')
    (destination/'.gitattributes').write_text('* -text\n')
    (destination/'.gitignore').write_text('!**/\n!**/*\n')
    manifest={p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination/'manifest.json').write_text(json.dumps(dict(files_sha256=manifest),indent=2)+'\n')
    return destination


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    destination=args.output or U02/'reports'/('portable_hardware_'+str(time.time_ns()))
    print('PORTABLE_HARDWARE_PREPARED',prepare(destination),flush=True)


if __name__=='__main__':main()
