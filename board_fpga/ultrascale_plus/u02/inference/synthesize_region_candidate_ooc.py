"""Map an exact audited candidate, retaining diagnostic failures and actual RAM counts."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from synthesize_region_dual_ooc import signed_leaves
from audit_region_dual_ooc import sha, resource_rows, drc_rows

U02=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hls-record',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--project-dir',type=Path,required=True)
    args=parser.parse_args();proof_path=args.hls_record/'result.json';proof=json.loads(proof_path.read_text())
    audit=json.loads(args.audit.read_text());project=args.project_dir.resolve()
    if project.exists() or project.drive.upper() not in ('D:','E:'):raise ValueError('Fresh D/E mapping project required')
    if not proof.get('complete') or not proof.get('synthesized') or not audit.get('single_compute_instance_verified'):
        raise ValueError('Completed candidate HLS and single-engine structure audit required')
    for path,digest in proof['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('HLS source changed: '+path)
    for name,digest in proof['generated_sha256'].items():
        if sha(args.hls_record/name)!=digest:raise ValueError('Generated candidate changed')
    rtl=args.hls_record/'hls_project/zu5ev/syn/verilog'
    for name,digest in audit['rtl_sha256'].items():
        if sha(rtl/name)!=digest:raise ValueError('Audited candidate RTL changed')
    files=[p for p in rtl.iterdir() if p.suffix in ('.v','.dat') or p.name.endswith('_ip.tcl')]
    if len([p for p in files if p.name.endswith('_ip.tcl')])!=13:raise ValueError('Unexpected floating dependency count')
    leaves=signed_leaves(rtl)
    record=U02/('reports/region_candidate_ooc_'+str(time.time_ns()));record.mkdir()
    inputs=record/'inputs';inputs.mkdir();staged={}
    for path in files:
        raw=path.read_bytes()
        if path.name in leaves:
            needle=('module '+leaves[path.name]+'(').encode()
            if raw.count(needle)!=1:raise ValueError('Ambiguous signed DSP mapping attribute')
            raw=raw.replace(needle,b'(* use_dsp = "yes" *)\n'+needle)
        target=inputs/path.name;target.write_bytes(raw)
        staged[path.name]=dict(source_sha256=sha(path),staged_sha256=sha(target),dsp_leaf=leaves.get(path.name))
    for path in inputs.glob('*.v'):
        for name in re.findall(r'\$readmemh\("(?:\./)?([^"/]+)"',path.read_text()):
            if not (inputs/name).is_file():raise ValueError('Missing initializer: '+name)
    script=Path(__file__).with_name('region_candidate_ooc.tcl')
    frozen={str(p):sha(p) for p in (Path(__file__),script,proof_path,args.audit,
            Path(__file__).with_name('synthesize_region_dual_ooc.py'),Path(__file__).with_name('audit_region_dual_ooc.py'),*files)}
    frozen.update({str(inputs/name):entry['staged_sha256'] for name,entry in staged.items()})
    state=dict(complete=False,terminal=False,board_accessed=False,routed=False,deployment_accepted=False,
               diagnostic_only=True,hls_record=str(args.hls_record.resolve()),project=str(project),
               input_sha256=frozen,staged_files=staged,
               scope='Actual candidate OOC mapping with signed integer DSP attributes, clock and optimization; no arithmetic or RAM-style rewrite. Numeric RTL and full-design route remain separate gates.')
    def save():(record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('U02_REGION_CANDIDATE_OOC_RECORD',record,flush=True)
    try:
        env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
            U02_REGION_CANDIDATE_INPUTS=inputs.as_posix(),U02_REGION_CANDIDATE_PROJECT=project.as_posix())
        with (record/'mapping.log').open('wb') as output:
            process=subprocess.Popen(['D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat','-mode','batch','-source',str(script),
                '-log',str(record/'vivado.log'),'-journal',str(record/'vivado.jou')],cwd=record,env=env,
                stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();state['exit_code']=process.wait()
        log=(record/'mapping.log').read_text(errors='replace')
        if state['exit_code'] or 'U02_REGION_CANDIDATE_MAPPING_FINISHED' not in log or re.search(r'(?m)^\s*ERROR:',log):
            raise ValueError('Candidate mapping failed')
        if any(sha(Path(p))!=digest for p,digest in frozen.items()):raise ValueError('Frozen mapping input changed')
        resources=resource_rows((project/'utilization.rpt').read_text());drc=drc_rows((project/'drc.rpt').read_text())
        timing=(project/'timing.rpt').read_text()
        clock=bool(re.search(r'(?m)^ap_clk\s+.*\b10\.312\b',timing)) and 'There are 0 register/latch pins with no clock.' in timing
        state.update(complete=True,resources=resources,drc=drc,clock_constraint_verified=clock,
                     resource_capacity_pass=all(row['within_capacity'] for row in resources.values()),
                     drc_error_count=sum(row['count'] for row in drc if row['severity']=='Error'),
                     artifacts_sha256={name:sha(project/name) for name in
                         ('region_candidate_ooc.dcp','utilization.rpt','utilization_hierarchical.rpt','drc.rpt','timing.rpt')})
    except Exception as error:state['error']=str(error);raise
    finally:
        state.update(terminal=True,log_sha256=sha(record/'mapping.log'));save()
        print('U02_REGION_CANDIDATE_OOC_DONE',record,flush=True)


if __name__=='__main__':main()
