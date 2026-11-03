"""Run full real-group C replay and HLS synthesis for validated boundary pipeline."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from prepare_region_boundary_pipeline import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,required=True)
    args=parser.parse_args()
    numeric_path=args.validation/'result.json';numeric=json.loads(numeric_path.read_text())
    if not numeric.get('complete') or not numeric.get('terminal') or numeric.get('real_boundaries')!=70 or numeric.get('nonfinite_rejections')!=18:
        raise ValueError('Completed real boundary and rejection proof required')
    frozen=dict(numeric['input_sha256'])
    for path,digest in frozen.items():
        if sha(Path(path))!=digest:raise ValueError('Numeric input changed: '+path)
    for name,digest in numeric['log_sha256'].items():
        if sha(args.validation/name)!=digest:raise ValueError('Numeric evidence changed: '+name)
    prepared=Path(numeric['preparation']);preparation=prepared/'preparation.json';plan=json.loads(preparation.read_text())
    baseline_path=Path(plan['baseline_record'])/'result.json';baseline=json.loads(baseline_path.read_text())
    if sha(baseline_path)!=plan['baseline_proof_sha256']:raise ValueError('Baseline proof changed')
    frozen.update({str(p):sha(p) for p in (Path(__file__),numeric_path,preparation)})
    fixtures=U02/'reports/fp32_output_fixtures_1791794443783245700'
    for name,digest in baseline['fixture_sha256'].items():
        if sha(fixtures/name)!=digest:raise ValueError('Real model archive changed')
        frozen[str(fixtures/name)]=digest
    record=U02/('reports/region_boundary_pipeline_synthesis_'+str(time.time_ns()));record.mkdir()
    for name,entry in plan['files'].items():
        source=prepared/name
        if sha(source)!=entry['candidate_sha256']:raise ValueError('Prepared header changed')
        (record/name).write_bytes(source.read_bytes())
    state=dict(complete=False,terminal=False,synthesized=False,board_accessed=False,
               hardware_verified=False,rtl_cosim_verified=False,full_network_verified=False,
               input_sha256=frozen,generated_sha256={name:sha(record/name) for name in plan['files']},
               fixture_sha256=baseline['fixture_sha256'],native_boundary_validation=str(args.validation.resolve()),
               source_compute_call_sites=1,
               scope='Boundary pipeline candidate: all 1100 real legacy groups plus synthetic region/switch-back C simulation and HLS schedule; RTL, mapping and full-network board gates remain independent.')
    def save():(record/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('U02_REGION_BOUNDARY_PIPELINE_SYNTHESIS_RECORD',record,flush=True)
    try:
        env=os.environ.copy();milestone=U02/'milestones/stream_region_dispatch'
        env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
                   U02_REGION_HLS_PROJECT=(record/'hls_project').as_posix(),U02_REGION_GENERATED_DIR=record.as_posix(),
                   U02_REGION_METADATA=(milestone/'first_region.bin').as_posix(),U02_REGION_MODEL=(milestone/'first_region.model').as_posix(),
                   U02_FP32_FIXTURE_DIR=fixtures.as_posix(),U02_REGION_SYNTHESIS='1')
        with (record/'hls.log').open('wb') as output:
            process=subprocess.Popen(['D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat','-f',
                                      str(U02/'inference/hls/build_region_dual.tcl')],cwd=record,env=env,
                                     stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();state['exit_code']=process.wait()
        log=(record/'hls.log').read_text(errors='replace')
        marker='U02_REGION_DUAL_CSIM_PASS legacy_groups=1100 fp32_values=8718720 rejected=1100 synthetic_region=1 rejected_switch=1 successful_switch_back=1 board_accessed=0'
        if state['exit_code'] or marker not in log or re.search(r'(?m)^\s*ERROR:',log) or 'CSim done with 0 errors.' not in log:
            raise ValueError('Boundary pipeline HLS validation failed')
        if any(sha(Path(p))!=digest for p,digest in frozen.items()):raise ValueError('Input changed during synthesis')
        if any(sha(record/name)!=digest for name,digest in state['generated_sha256'].items()):raise ValueError('Generated source changed')
        report=record/'hls_project/zu5ev/syn/report/u02_region_dual_csynth.xml';root=ET.parse(report).getroot()
        if root.findtext('UserAssignments/TopModelName')!='u02_region_dual':raise ValueError('Wrong synthesized top')
        state.update(complete=True,synthesized=True,csim_passed=True,legacy_groups=1100,fp32_values=8718720,
                     rejected_legacy_packets=1100,synthetic_region_cases=1,successful_switch_back=1,
                     synthesis_report_sha256=sha(report),hls_resources={n.tag:int(n.text) for n in root.find('AreaEstimates/Resources')})
    except Exception as error:state['error']=str(error);raise
    finally:
        state.update(terminal=True,log_sha256={p.name:sha(p) for p in record.glob('*.log')});save()
        print('U02_REGION_BOUNDARY_PIPELINE_SYNTHESIS_DONE',record,flush=True)


if __name__=='__main__':main()
