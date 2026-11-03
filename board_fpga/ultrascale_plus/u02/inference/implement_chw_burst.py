"""One isolated implementation, enabled only by accepted full-network CHW RTL evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from build_chw_burst import U02,TOP,sha
from review_chw_burst_hls import frozen
from inspect_stream_route import inspect

VIVADO=Path('D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat')

def templates(accepted,record,project):
    oldrecord=Path(accepted['reports']).parent
    archive=U02/'milestones/dw128_write_v2_routed_118m_20261017'
    original=archive/'implement.tcl';audit=archive/'audit.tcl'
    for p in (original,audit):assert sha(p)==accepted['input_sha256'][str(oldrecord/p.name)]
    text=original.read_text()
    def once(before,after):
        nonlocal text
        assert text.count(before)==1,before;text=text.replace(before,after)
    once((oldrecord/'tool_work').as_posix(),(record/'tool_work').as_posix())
    once('u02:inference:u02_stream_128write_v2:1.0','u02:inference:'+TOP+':1.0')
    oldclock_original=Path(accepted['recovery_from'])/'target_clock.xdc'
    oldclock=archive/'target_clock.xdc'
    assert sha(oldclock)==accepted['input_sha256'][str(oldclock_original)]
    once(oldclock_original.as_posix(),(record/'target_clock.xdc').as_posix())
    assert 'check_dw128_generated_axi.py' in text and 'axi_generated_rtl_widths.json' in text
    text=text.replace('DW128_V2','CHW_BURST')
    at=audit.read_text();assert at.count(Path(accepted['project']).as_posix())==1
    assert at.count((oldrecord/'route_audit').as_posix())==1
    at=at.replace(Path(accepted['project']).as_posix(),project.as_posix()).replace(
        (oldrecord/'route_audit').as_posix(),(record/'route_audit').as_posix())
    return text,at,oldclock,original,audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--project',type=Path,required=True);a=ap.parse_args()
    record=a.record.resolve();project=a.project.resolve()
    proof=U02/'milestones/chw_burst_full_rtl_verified_20261018';frozen(proof)
    full=json.loads((proof/'state.json').read_text());summary=json.loads((proof/'summary.json').read_text())
    analysis=json.loads((proof/'analysis.json').read_text())
    assert full['complete'] and full['terminal'] and full['rtl_cosimulation_complete']
    assert full['auxiliary_outputs_and_returns_independently_accepted']
    assert summary['eligible_for_implementation'] and all(summary['implementation_gates'].values())
    assert analysis['all_246_outputs_and_returns_equal_accepted_reference']
    assert analysis['before_cycles']==1855263 and analysis['after_cycles']<=1855263*.99
    assert all(r['after_cycles']<=r['before_cycles']*1.01 for r in analysis['groups'])
    hls=Path(full['project']);package=hls/'zu5ev/impl/ip'
    for section in ('source_sha256','rtl_sha256'):
        assert all(sha(Path(p))==d for p,d in full[section].items())
    baseline=U02/'milestones/dw128_write_v2_routed_118m_20261017'
    old_manifest=json.loads((baseline/'manifest.json').read_text())
    assert all(sha(baseline/name)==digest for name,digest in old_manifest.items())
    accepted=json.loads((baseline/'state.json').read_text());assert accepted['complete'] and accepted['route_checks_passed']
    assert not record.exists() and not project.exists() and not package.exists(),'Preserve previous/live builds'
    assert record.drive.upper()=='E:' and project.drive.upper()=='D:'
    assert shutil.disk_usage(project.parent).free>15*1024**3
    assert shutil.disk_usage(record.parent).free>10*1024**3
    impl_text,audit_text,oldclock,oldimpl,oldaudit=templates(accepted,record,project)
    inputs=[Path(__file__),U02/'inference/prepare_chw_burst_ip.py',U02/'inference/prepare_stream_ip.py',
        U02/'inference/hls/export_chw_burst.tcl',U02/'inference/inspect_stream_route.py',
        U02/'inference/check_dw128_generated_axi.py',U02/'inference/vivado_tcl_bootstrap.tcl',
        U02/'board/ps_config_2018.tcl',U02/'board/ps_required_baseline.tsv',U02/'board/por_reset.v',
        U02/'board/add_clock_probe.tcl',proof/'manifest.json',baseline/'manifest.json',oldimpl,oldaudit,oldclock]
    # Carry forward only proven PS/tool dependencies; source paths and DUT intentionally differ.
    for p in inputs:
        if str(p) in accepted['input_sha256']:assert sha(p)==accepted['input_sha256'][str(p)]
    originals=dict(accepted['original_bit_sha256'])
    originals[str(baseline/'compute_wrapper.bit')]=accepted['bit_sha256']
    assert all(sha(Path(p))==d for p,d in originals.items())
    record.mkdir();(record/'tool_work').mkdir();(record/'implementation').mkdir();(record/'route_audit').mkdir()
    (record/'inputs').mkdir()
    for p in inputs:
        dest=record/'inputs'/p.name
        # Distinct manifests/templates can share a basename; preserve every original without collision.
        if dest.exists():dest=dest.with_name(str(inputs.index(p))+'_'+p.name)
        shutil.copyfile(p,dest)
    (record/'implement.tcl').write_text(impl_text)
    (record/'audit.tcl').write_text(audit_text)
    shutil.copyfile(oldclock,record/'target_clock.xdc')
    fingerprints={str(p):sha(p) for p in inputs}
    fingerprints.update(full['source_sha256']);fingerprints.update(full['rtl_sha256'])
    for n in ('implement.tcl','audit.tcl','target_clock.xdc'):fingerprints[str(record/n)]=sha(record/n)
    state=dict(complete=False,terminal=False,board_accessed=False,hardware_verified=False,phase='prepared',
        project=str(project),hls_project=str(hls),top=TOP,reports=str(record/'implementation'),
        input_sha256=fingerprints,original_bit_sha256=originals,target_period_ns=8.437,
        rtl_manifest_sha256=sha(proof/'manifest.json'),automatic_kill_deadline=None,build_attempt_limit=1,phases={})
    def save():(record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    def unchanged():
        assert all(sha(Path(p))==d for p,d in fingerprints.items()),'Frozen input changed'
        assert all(sha(Path(p))==d for p,d in originals.items()),'Protected bitstream changed'
    env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
        TCL_LIBRARY='D:/Develop/Xilink/Vivado/2018.3/tps/tcl/tcl8.5',
        U02_STREAM_HLS_PROJECT=str(hls),U02_STREAM_PROJECT_DIR=str(project),U02_STREAM_REPORT_DIR=str(record/'implementation'))
    env.pop('U02_STREAM_RESUME_EMPTY',None)
    active=None
    def command(script,hls_tool=False):
        if hls_tool:return [str(VIVADO.with_name('vivado_hls.bat')),'-f',str(script)]
        return [str(VIVADO),'-mode','batch','-nojournal','-nolog','-source',str(script)]
    def run(name,cmd,cwd):
        nonlocal active
        unchanged();state['phase']=name;save();print('CHW_BURST_BUILD_PHASE',name,flush=True)
        with (record/(name+'.log')).open('wb') as log:
            active=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            phase=dict(pid=active.pid,started_unix_ns=time.time_ns(),command=cmd);state['phases'][name]=phase;save()
            phase['exit_code']=active.wait()  # Never terminate on an observation timeout.
        phase.update(ended_unix_ns=time.time_ns(),log_sha256=sha(record/(name+'.log')));save();unchanged()
        return phase['exit_code'],(record/(name+'.log')).read_text(errors='replace')
    save();print('CHW_BURST_IMPLEMENTATION_RECORD',record,flush=True)
    try:
        code,log=run('export',command(U02/'inference/hls/export_chw_burst.tcl',True),record/'tool_work')
        errors=re.findall(r'(?m)^ERROR:.*$',log)
        if code or errors:
            assert 'set_property core_revision $Revision $core' in log and errors
            assert all('IMPL 213-28' in line for line in errors),errors
            state['known_core_revision_export_failure']=True
        assert (package/'run_ippack.tcl').exists()
        prep=record/'ip_preparation.json'
        code,log=run('prepare_ip',[sys.executable,str(U02/'inference/prepare_chw_burst_ip.py'),'--project',str(hls),'--manifest',str(prep)],record/'tool_work')
        assert code==0;prepared=json.loads(prep.read_text())
        assert {r['file']:r['source_sha256'] for r in prepared['files']}=={Path(p).name:d for p,d in full['rtl_sha256'].items()}
        code,log=run('package_ip',command(package/'run_ippack.tcl'),package)
        assert code==0 and not re.search(r'(?m)^ERROR:',log) and (package/'component.xml').is_file()
        assert {p.name:sha(p) for p in (package/'hdl/verilog').glob('*.v')}=={r['file']:r['package_sha256'] for r in prepared['files']}
        code,log=run('implementation',command(record/'implement.tcl'),record/'tool_work')
        assert code==0 and 'CHW_BURST_IMPLEMENTATION_COMPLETE_NOT_DEPLOYED' in log and not re.search(r'(?m)^ERROR:',log)
        assert 'CHW_BURST_AXI_WIDTHS_VERIFIED' in log
        code,log=run('audit',command(record/'audit.tcl'),record/'tool_work')
        assert code==0 and 'U02_STREAM_ROUTED_NETLIST_AUDIT_RECORDED' in log
        bit=project/'u02_stream.runs/impl_1/compute_wrapper.bit'
        result=inspect('chw_burst',prep,artifacts=dict(project=project,hls=hls,implementation=record/'implementation',
            prefix='chw_burst',bit=bit),audit_directory=record/'route_audit',expected_pl_period_ns=8.437,expected_top=TOP)
        (record/'route_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
        state.update(complete=True,route_checks_passed=True,bit_sha256=sha(bit))
    except Exception as exc:
        state['error']=repr(exc);raise
    finally:
        state['terminal']=active is None or active.poll() is not None
        state['ended_unix_ns']=time.time_ns() if state['terminal'] else None;save()
    print('CHW_BURST_IMPLEMENTATION_PASSED_REQUIRES_BOARD_AB',flush=True)

if __name__=='__main__':main()
