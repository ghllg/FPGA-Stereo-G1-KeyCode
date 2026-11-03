"""One isolated implementation of the fully verified writer; no board access or kill deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from inspect_stream_route import inspect

U02=Path(__file__).resolve().parents[1]
VIVADO=Path('D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat')
TOP='u02_stream_128write_v2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(path):
    manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['complete']
    for name,digest in manifest['files_sha256'].items():assert sha(path/name)==digest,name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--project',type=Path,required=True)
    args=parser.parse_args();out=args.record.resolve();project=args.project.resolve()
    proof=U02/'milestones/dw128_write_v2_full_rtl_verified_20261017';verify_archive(proof)
    full=json.loads((proof/'state.json').read_text())
    numeric=json.loads((proof/'independent_comparison.json').read_text())
    assert full['complete'] and full['terminal'] and full['rtl_cosimulation_complete']
    assert numeric['complete'] and numeric['numeric']['all_groups_exact']
    assert numeric['numeric']['unknown_words']==0 and numeric['after_cycles']==1855263
    assert numeric['before_cycles']==2052369 and all(g['saved']>0 for g in numeric['groups'])
    hls=Path(full['project']);package=hls/'zu5ev/impl/ip'
    for section in ('source_sha256','rtl_sha256'):
        assert all(sha(Path(p))==d for p,d in full[section].items())
    assert not out.exists() and not project.exists() and not package.exists(), 'Preserve prior attempts'
    assert shutil.disk_usage(project.parent).free>15*1024**3, 'Insufficient build space plus reserve'
    out.mkdir(parents=True);reports=out/'implementation';reports.mkdir();audit=out/'route_audit';audit.mkdir()
    inputs=[Path(__file__),U02/'inference/prepare_dw128_write_v2_ip.py',U02/'inference/prepare_stream_ip.py',
            U02/'inference/inspect_stream_route.py',U02/'inference/create_stream_project.tcl',
            U02/'inference/audit_stream_route.tcl',U02/'inference/hls/export_dw128_write_v2.tcl',
            U02/'board/ps_config_2018.tcl',U02/'board/ps_required_baseline.tsv',
            U02/'board/por_reset.v',U02/'board/add_clock_probe.tcl']
    fingerprints={str(p):sha(p) for p in inputs}
    fingerprints.update(full['source_sha256']);fingerprints.update(full['rtl_sha256'])
    originals=[U02/'milestones/formal_dw_routed_118m_20261015/compute_wrapper.bit']
    formal=json.loads((U02/'inference/stream_ingress_parallel_affine_runtime.json').read_text())
    originals.append(U02/formal['bit']);assert sha(originals[-1])==formal['bit_sha256']
    original_hashes={str(p):sha(p) for p in originals}
    (out/'inputs').mkdir()
    for p in inputs:shutil.copy2(p,out/'inputs'/p.name)
    clock=out/'target_clock.xdc'
    clock.write_text('''set c [get_clocks -quiet clk_pl_0]
if {[llength $c] != 1 || [get_property IS_GENERATED $c]} {error "Unexpected PL clock"}
set pins [get_property SOURCE_PINS $c]
if {[llength $pins] != 1} {error "Ambiguous PL clock source"}
create_clock -name clk_pl_0 -period 8.437 -waveform {0 4.218} [get_pins $pins]
''')
    template=(U02/'inference/create_stream_project.tcl').read_text()

    def replace_once(before,after):
        nonlocal template
        assert template.count(before)==1,before
        template=template.replace(before,after)

    replace_once('set inference [file dirname [file normalize [info script]]]',
                 'set inference {'+(U02/'inference').as_posix()+'}')
    replace_once('u02:inference:u02_stream:1.0','u02:inference:'+TOP+':1.0')
    replace_once('validate_bd_design\nsave_bd_design','''validate_bd_design
set width_report [open [file join $reports axi_widths.tsv] w]
foreach {pin expected} {compute_ip/m_axi_weights 64 compute_ip/m_axi_features 64 compute_ip/m_axi_results 128 ddr/S02_AXI 128 ddr/M00_AXI 128 ps/S_AXI_HP0_FPD 128} {
    set actual [get_property CONFIG.DATA_WIDTH [get_bd_intf_pins $pin]]
    puts $width_report "$pin\t$actual\t$expected"
    if {$actual ne $expected} {close $width_report; error "AXI width mismatch: $pin=$actual expected=$expected"}
}
close $width_report
puts DW128_V2_AXI_WIDTHS_VERIFIED
save_bd_design''')
    replace_once('launch_runs synth_1 -jobs 1','''set target_clock {%s}
add_files -fileset constrs_1 -norecurse $target_clock
set_property USED_IN_SYNTHESIS false [get_files $target_clock]
set_property PROCESSING_ORDER LATE [get_files $target_clock]
launch_runs synth_1 -jobs 1'''%clock.as_posix())
    marker='launch_runs impl_1 -to_step write_bitstream -jobs 1';assert template.count(marker)==1
    template=template[:template.index(marker)]+'''launch_runs impl_1 -to_step route_design -jobs 1
wait_on_run impl_1
set status [get_property STATUS [get_runs impl_1]]
if {![string match "*Complete*" $status]} {error "Implementation did not complete: $status"}
open_run impl_1
if {abs([get_property PERIOD [get_clocks clk_pl_0]]-8.437)>0.001} {error "Target clock constraint not active"}
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $reports timing_before_bit.rpt]
report_utilization -file [file join $reports utilization_before_bit.rpt]
report_drc -file [file join $reports drc_before_bit.rpt]
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup] || ![llength $hold]} {error "Timing paths missing"}
puts "DW128_V2_SETUP_SLACK_NS [get_property SLACK $setup]"
puts "DW128_V2_HOLD_SLACK_NS [get_property SLACK $hold]"
if {[get_property SLACK $setup]<0 || [get_property SLACK $hold]<0} {error "Timing rejected; no bitstream"}
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]]} {error "DRC rejected; no bitstream"}
write_bitstream [file join $project_dir u02_stream.runs impl_1 compute_wrapper.bit]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $reports timing.rpt]
report_utilization -file [file join $reports utilization.rpt]
report_drc -file [file join $reports drc.rpt]
write_debug_probes [file join $reports compute.ltx]
puts DW128_V2_IMPLEMENTATION_COMPLETE_NOT_DEPLOYED
close_project
'''
    impl=out/'implement.tcl';impl.write_text(template)
    text=(U02/'inference/audit_stream_route.tcl').read_text()
    before='source [file join [file dirname [file normalize [info script]]] stream_artifacts.tcl]'
    assert text.count(before)==1
    text=text.replace(before,'set u02_project {'+project.as_posix()+'}')
    before='set report_dir [file join $root reports ${u02_prefix}_route_audit]';assert text.count(before)==1
    audit_script=out/'audit.tcl';audit_script.write_text(text.replace(before,'set report_dir {'+audit.as_posix()+'}'))
    for p in (clock,impl,audit_script):fingerprints[str(p)]=sha(p)
    state=dict(complete=False,terminal=False,board_accessed=False,phase='prepared',project=str(project),
               hls_project=str(hls),reports=str(reports),input_sha256=fingerprints,
               original_bit_sha256=original_hashes,target_period_ns=8.437,build_attempt_limit=1,
               rtl_manifest_sha256=sha(proof/'manifest.json'),automatic_kill_deadline=None,phases={})

    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')

    def unchanged():
        assert all(sha(Path(p))==d for p,d in fingerprints.items()), 'Frozen build input changed'
        assert all(sha(Path(p))==d for p,d in original_hashes.items()), 'Original bitstream changed'

    env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
        U02_STREAM_HLS_PROJECT=str(hls),U02_STREAM_PROJECT_DIR=str(project),U02_STREAM_REPORT_DIR=str(reports))
    active=None

    def command(script,hls_tool=False):
        if hls_tool:return [str(VIVADO.with_name('vivado_hls.bat')),'-f',str(script)]
        return [str(VIVADO),'-mode','batch','-nojournal','-nolog','-source',str(script)]

    def run(name,cmd,cwd=U02):
        nonlocal active
        unchanged();state['phase']=name;save();print('DW128_V2_BUILD_PHASE',name,flush=True)
        with (out/(name+'.log')).open('wb') as log:
            active=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            phase=dict(pid=active.pid,command=cmd,started_unix_ns=time.time_ns(),automatic_kill_deadline=None)
            state['phases'][name]=phase;save()
            phase['exit_code']=active.wait()
        phase.update(log_sha256=sha(out/(name+'.log')),ended_unix_ns=time.time_ns());save();unchanged()
        return phase['exit_code'],(out/(name+'.log')).read_text(errors='replace')

    save();print('DW128_V2_IMPLEMENTATION_RECORD',out,flush=True)
    try:
        code,log=run('export',command(U02/'inference/hls/export_dw128_write_v2.tcl',True))
        errors=re.findall(r'(?m)^ERROR:.*$',log)
        if code or errors:
            assert 'set_property core_revision $Revision $core' in log and errors
            assert all('IMPL 213-28' in line for line in errors),errors
            state['known_core_revision_export_failure']=True
        assert (package/'run_ippack.tcl').exists()
        prep=out/'ip_preparation.json'
        code,log=run('prepare_ip',[sys.executable,str(U02/'inference/prepare_dw128_write_v2_ip.py'),
                                 '--project',str(hls),'--manifest',str(prep)])
        assert code==0
        prepared=json.loads(prep.read_text())
        assert {p['file']:p['source_sha256'] for p in prepared['files']}=={Path(p).name:d for p,d in full['rtl_sha256'].items()}
        code,log=run('package_ip',command(package/'run_ippack.tcl'),package)
        assert code==0 and not re.search(r'(?m)^ERROR:',log) and (package/'component.xml').is_file()
        assert {p.name:sha(p) for p in (package/'hdl/verilog').glob('*.v')}=={p['file']:p['package_sha256'] for p in prepared['files']}
        code,log=run('implementation',command(impl))
        assert code==0 and 'DW128_V2_IMPLEMENTATION_COMPLETE_NOT_DEPLOYED' in log and not re.search(r'(?m)^ERROR:',log)
        assert 'DW128_V2_AXI_WIDTHS_VERIFIED' in log
        code,log=run('audit',command(audit_script))
        assert code==0 and 'U02_STREAM_ROUTED_NETLIST_AUDIT_RECORDED' in log
        bit=project/'u02_stream.runs/impl_1/compute_wrapper.bit'
        result=inspect('dw128_write_v2',prep,artifacts=dict(project=project,hls=hls,implementation=reports,
            prefix='dw128_write_v2',bit=bit),audit_directory=audit,expected_pl_period_ns=8.437,expected_top=TOP)
        (out/'route_inspection.json').write_text(json.dumps(result,indent=2)+'\n')
        state.update(complete=True,route_checks_passed=True,hardware_verified=False,bit_sha256=sha(bit))
        print('DW128_V2_IMPLEMENTATION_PASSED_REQUIRES_BOARD_AB',flush=True)
    except Exception as exc:
        state['error']=repr(exc)
        raise
    finally:
        terminal=active is None or active.poll() is not None
        state.update(terminal=terminal,ended_unix_ns=time.time_ns() if terminal else None)
        if not terminal:state['observer_detached_child_still_running']=True
        save()


if __name__=='__main__':main()
