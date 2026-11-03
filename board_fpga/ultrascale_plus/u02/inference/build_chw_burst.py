"""One isolated complete-network CHW writer C simulation and HLS synthesis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from analyze_chw_burst_budget import analyze
from cosim_dw128_write_v2 import MARKERS,FIXTURE_SHA

U02=Path(__file__).resolve().parents[1]
TOP='u02_stream_chw_burst'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def sources():
    root=U02/'inference/hls'
    pending=['stream_chw_burst.cpp','test_stream_chw_burst.cpp','test_store_final_128_chw_burst.cpp']
    result={}
    while pending:
        name=pending.pop()
        if name in result:continue
        path=root/name;result[name]=path
        pending.extend(re.findall(r'^#include "([^"]+)"',path.read_text(),re.M))
    return result

def verify_fork():
    root=U02/'inference/hls'
    assert sha(root/'stream_dw128_write_v2.cpp')=='845c95826b8261aa573e8a85737a97c32e8e10a074415e1ea997519c39df910b'
    assert sha(root/'store_final_128_v2.h')=='71fd73b67168ac524ce4b1ebc66dd617ce5afc79f7d063bc7784e7c1b29b8fce'
    assert (root/'stream_chw_burst.cpp').read_text()==(root/'stream_dw128_write_v2.cpp').read_text().replace(
        '"store_final_128_v2.h"','"store_final_128_chw_burst.h"').replace('u02_stream_128write_v2',TOP)
    assert (root/'test_stream_chw_burst.cpp').read_text()==(root/'test_stream_dw128_write_v2.cpp').read_text().replace(
        '"dw128_v2_test_bridge.h"','"chw_burst_test_bridge.h"')
    assert (root/'chw_burst_test_bridge.h').read_text()==(root/'dw128_v2_test_bridge.h').read_text().replace('u02_stream_128write_v2',TOP)
    # Old partial-word and tiled fallback must remain literally intact.
    old=(root/'store_final_128_v2.h').read_text();new=(root/'store_final_128_chw_burst.h').read_text()
    assert old[:old.index('    const int words=count/2;')] in new
    assert old[old.index('    const int words=count/2;'):old.rfind('\n}')] in new

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
    record=args.record.resolve();project=args.project.resolve()
    assert not record.exists() and not project.exists(),'Preserve existing runs; no restart on observation timeout'
    assert project.drive.upper()=='D:' and record.drive.upper()=='E:'
    assert shutil.disk_usage(project.parent).free>11*1024**3
    assert shutil.disk_usage(record.parent).free>10*1024**3
    verify_fork();budget=analyze();fixture=U02/'milestones/formal_dw_full_rtl_pass_20261015/all_real_groups.fixture'
    assert sha(fixture)==FIXTURE_SHA
    record.mkdir();snapshot=record/'source';snapshot.mkdir()
    for name,path in sources().items():shutil.copyfile(path,snapshot/name)
    shutil.copyfile(Path(__file__),record/'executed_build.py')
    shutil.copyfile(U02/'inference/analyze_chw_burst_budget.py',record/'executed_budget.py')
    (record/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    script=record/'build.tcl'
    script.write_text('cd {%s}\nopen_project {%s}\nset_top %s\n'
        'add_files {%s} -cflags "-std=c++11 -ffp-contract=off"\n'
        'add_files -tb {%s} -cflags "-std=c++11 -ffp-contract=off"\n'
        'open_solution zu5ev\nset_part {xczu5ev-sfvc784-2-i}\n'
        'create_clock -period 10.312 -name default\nconfig_interface -m_axi_addr64\n'
        'csim_design -O\ncsynth_design\nexit\n' %
        (project.parent.as_posix(),project.name,TOP,(snapshot/'stream_chw_burst.cpp').as_posix(),
         (snapshot/'test_stream_chw_burst.cpp').as_posix()))
    state=dict(complete=False,terminal=False,board_accessed=False,hardware_verified=False,
        project=str(project),record=str(record),top=TOP,automatic_kill_deadline=None,
        fixture_sha256=sha(fixture),source_sha256={p.name:sha(p) for p in snapshot.iterdir()},
        runner_sha256=sha(Path(__file__)),script_sha256=sha(script),phase='native_writer_test')
    def save():
        (record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    env=os.environ.copy();env.update(U02_STREAM_FIXTURE=str(fixture),TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'))
    process=None
    def run(command,logfile):
        nonlocal process
        with (record/logfile).open('wb') as log:
            process=subprocess.Popen(command,cwd=record,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(pid=process.pid,started_unix_ns=time.time_ns());save()
            rc=process.wait()  # Intentionally no wall-clock kill.
        state['last_exit_code']=rc;save();assert rc==0,(command,rc)
    print('CHW_BURST_BUILD_RECORD',record,flush=True);save()
    try:
        linux=lambda p:'/mnt/'+p.drive[0].lower()+p.as_posix()[2:]
        native=record/'test_writer'
        run(['wsl.exe','--exec','env','TMPDIR='+linux(U02/'toolchain/tmp'),'/usr/bin/g++','-std=c++11','-O2',
             linux(snapshot/'test_store_final_128_chw_burst.cpp'),'-o',linux(native)],'native_compile.log')
        run(['wsl.exe','--exec',linux(native)],'native_writer.log')
        native_result=json.loads((record/'native_writer.log').read_text())
        assert native_result['complete'] and native_result['native_packing_only']
        state.update(native_writer=native_result,phase='csim_and_synthesis');save()
        run(['D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat','-f',str(script)],'build.log')
        text=(record/'build.log').read_text(errors='replace')
        assert not re.search(r'(?mi)^\s*(?://\s*)?(?:ERROR|FATAL)\b',text)
        csim=project/'zu5ev/csim/report'/f'{TOP}_csim.log'
        ct=csim.read_text();assert all(m in ct for m in MARKERS) and 'CSim done with 0 errors.' in ct
        assert 'Finished generating all RTL models' in text and 'Exiting vivado_hls' in text
        report=project/'zu5ev/syn/report'/f'{TOP}_csynth.xml';xml=ET.parse(report).getroot()
        assert xml.findtext('UserAssignments/TopModelName')==TOP
        assert xml.findtext('UserAssignments/Part')=='xczu5ev-sfvc784-2-i'
        resources={e.tag:int(e.text) for e in xml.find('AreaEstimates/Resources')}
        state.update(complete=True,phase='await_synthesis_review',csim_markers_passed=list(MARKERS),
            resources_estimated=resources,estimated_clock_ns=float(xml.findtext('PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod')),
            synthesis_report_sha256=sha(report),csim_log_sha256=sha(csim))
        assert all(sha(snapshot/name)==digest for name,digest in state['source_sha256'].items())
        verify_fork()
    except Exception as exc:
        state['error']=repr(exc);raise
    finally:
        state['terminal']=process is None or process.poll() is not None
        state['ended_unix_ns']=time.time_ns() if state['terminal'] else None
        save()
    print('CHW_BURST_COMPLETE_NETWORK_CSIM_AND_HLS_FINISHED',json.dumps(state['resources_estimated']),flush=True)

if __name__=='__main__':main()
