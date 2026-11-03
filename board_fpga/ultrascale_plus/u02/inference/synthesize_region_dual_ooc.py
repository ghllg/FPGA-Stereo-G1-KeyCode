"""Map the complete dual top in isolation; pending RTL cannot authorize deployment."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from lower_stream_region import U02,sha


def signed_leaves(rtl):
    leaves={}
    for filename,widths,output_width,expected in (
        ('pw_mac.v',((8,8),),16,512),('dw_mac.v',((8,8),),16,32),
        ('requant_words.v',((32,24),(24,32)),56,16)):
        instances=re.findall(r'\b(u02_region_dual_\w+)\s*#',(rtl/filename).read_text());matched=0
        for module in set(instances):
            path=rtl/(module+'.v');text=path.read_text();ports={}
            for name in ('a','b'):
                found=re.search(r'\binput\s*\[\s*(\d+)\s*-\s*1\s*:\s*0\s*\]\s*'+name+r'\s*;',text)
                if found:ports[name]=int(found.group(1))
            result=re.search(r'\boutput\s*\[\s*(\d+)\s*-\s*1\s*:\s*0\s*\]\s*p\s*;',text)
            if tuple(ports.get(name) for name in ('a','b')) not in widths or not result or int(result.group(1))!=output_width:continue
            for name,width in ports.items():
                if not re.search(r'reg signed\s*\[\s*'+str(width)+r'\s*-\s*1\s*:\s*0\s*\]\s*'+name+r'_reg0\s*;',text):
                    raise ValueError('Unsigned multiplier leaf')
            names=re.findall(r'module\s+(\w+_MulnS_\d+)\(clk, ce, a, b, p\);',text)
            if len(names)!=1 or 'assign tmp_product = a_reg0 * b_reg0;' not in text:
                raise ValueError('Unexpected multiplier leaf')
            leaves[path.name]=names[0];matched+=instances.count(module)
        if matched!=expected:raise ValueError('Wrong multiplier instance count '+filename+': '+str(matched))
    return leaves


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hls-record',type=Path,required=True)
    parser.add_argument('--project-dir',type=Path,help='Fresh short D/E project path for Windows tools')
    args=parser.parse_args();proof_path=args.hls_record/'result.json';proof=json.loads(proof_path.read_text())
    audit_path=U02/'milestones/stream_region_dispatch/dual_synthesis_audit.json';audit=json.loads(audit_path.read_text())
    if not proof['complete'] or not proof['synthesized'] or not audit['single_compute_instance_verified']:
        raise ValueError('Successful dual synthesis and structure audit required')
    for path,digest in proof['input_sha256'].items():
        if sha(Path(path))!=digest:raise ValueError('HLS input changed '+path)
    rtl=args.hls_record/'hls_project/zu5ev/syn/verilog'
    for name,digest in audit['rtl_sha256'].items():
        if sha(rtl/name)!=digest:raise ValueError('RTL changed '+name)
    files=[p for p in rtl.iterdir() if p.suffix in ('.v','.dat') or p.name.endswith('_ip.tcl')]
    if len([p for p in files if p.name.endswith('_ip.tcl')])!=13:raise ValueError('Floating dependencies changed')
    leaves=signed_leaves(rtl)
    record=U02/('reports/region_dual_ooc_'+str(time.time_ns()));record.mkdir()
    inputs=record/'inputs';inputs.mkdir();project=args.project_dir.resolve() if args.project_dir else record/'project';staged={}
    if project.exists():raise ValueError('Preserve any existing resource project')
    if project.drive.upper() not in ('D:','E:'):raise ValueError('Large resource project must stay on D/E')
    for path in files:
        data=path.read_bytes()
        if path.name in leaves:
            needle=('module '+leaves[path.name]+'(').encode()
            if data.count(needle)!=1:raise ValueError('Ambiguous signed DSP attribute')
            data=data.replace(needle,b'(* use_dsp = "yes" *)\n'+needle)
        target=inputs/path.name;target.write_bytes(data)
        staged[path.name]=dict(source_sha256=sha(path),staged_sha256=sha(target),dsp_leaf=leaves.get(path.name))
    for path in inputs.glob('*.v'):
        for name in re.findall(r'\$readmemh\("(?:\./)?([^"/]+)"',path.read_text()):
            if not (inputs/name).is_file():raise ValueError('Missing memory initialization '+name)
    script=U02/'inference/region_dual_ooc.tcl'
    frozen={str(p):sha(p) for p in (Path(__file__),script,proof_path,audit_path,*files)}
    frozen.update({str(inputs/name):entry['staged_sha256'] for name,entry in staged.items()})
    state=dict(complete=False,terminal=False,board_accessed=False,diagnostic_only=True,rtl_cosim_verified=False,
               routed=False,deployment_accepted=False,hls_record=str(args.hls_record),project=str(project),
               staged_files=staged,input_sha256=frozen,
               scope='Actual complete dual-top Vivado OOC mapping with signed DSP attributes; no arithmetic rewrite or board access.')
    def save():(record/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('U02_REGION_DUAL_OOC_RECORD',record,flush=True)
    try:
        env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
            U02_REGION_OOC_INPUTS=inputs.as_posix(),U02_REGION_OOC_PROJECT=project.as_posix())
        with (record/'synthesis.log').open('wb') as output:
            process=subprocess.Popen(['D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat','-mode','batch','-source',str(script),
                '-log',str(record/'vivado.log'),'-journal',str(record/'vivado.jou')],cwd=record,env=env,
                stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();state['exit_code']=process.wait()
        output=(record/'synthesis.log').read_text(errors='replace')
        if state['exit_code'] or re.search(r'(?m)^\s*ERROR:',output) or 'U02_REGION_DUAL_DIAGNOSTIC_OOC_FINISHED' not in output:
            raise ValueError('Resource mapping failed; preserve this record before any repair')
        if any(sha(Path(path))!=digest for path,digest in frozen.items()):raise ValueError('Input changed')
        state['artifacts_sha256']={name:sha(project/name) for name in ('utilization.rpt','timing.rpt','drc.rpt','region_dual_ooc.dcp')}
        state['complete']=True
    except Exception as error:
        state['error']=str(error);raise
    finally:
        state.update(terminal=True,log_sha256=sha(record/'synthesis.log'));save()
        print('U02_REGION_DUAL_OOC_DONE',record,flush=True)


if __name__=='__main__':main()
