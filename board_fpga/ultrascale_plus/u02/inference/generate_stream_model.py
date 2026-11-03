"""Emit a fused NCNN graph and matching compiled descriptors from a checked plan."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from plan_streaming import parse_param, ROOT, U02

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,default=U02/'reports/streaming_plan.json')
    p.add_argument('--output',type=Path,default=U02/'inference/build')
    p.add_argument('--layout-trace',type=Path,help='Verified board trace for exact channel-shuffle folding')
    a=p.parse_args(); plan=json.loads(a.plan.read_text())
    source=ROOT/'board_light_pose/models/model.pl.param'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_param_sha256']
    nodes,_,_=parse_param(source)
    groups=plan['groups']; starts={g['source_layer_indices'][0]:g for g in groups}
    absorbed={i for g in groups for i in g['source_layer_indices']}
    rows=[]; blobs=set(); available=set()
    for n in nodes:
        if n['index'] in starts:
            g=starts[n['index']]; inputs=[g['input_blob']]; outputs=[g['output_blob']]
            line=f"U02Stream stream_{g['id']} 1 1 {inputs[0]} {outputs[0]} 0={g['id']}"
        elif n['index'] in absorbed: continue
        else:
            inputs=n['inputs']; outputs=n['outputs']
            line=' '.join([n['type'],n['name'],str(len(inputs)),str(len(outputs)),*inputs,*outputs,
                           *(f'{k}={v}' for k,v in n['params'].items())])
        assert all(v in available for v in inputs)
        assert not set(outputs)&blobs
        available.update(outputs);blobs.update(outputs);rows.append(line)
    header=['#pragma once','#include "ultrascale_plus/u02/inference/hls/stream_contract.h"',
            'struct StreamModelOp {StreamOp op; int bias_term, scale_offset, scale_count;};',
            'struct StreamModelGroup {int first, count;};']
    scales=[]; specs=[]; group_specs=[]
    for g in groups:
        group_specs.append((len(specs),len(g['operators'])))
        weight_offset=requant_offset=0
        for op in g['operators']:
            _,ci,h,w=op['input_shape']; _,co,oh,ow=op['output_shape']
            values=[int(op['kind']=='depthwise'),ci,co,h,w,oh,ow,op['stride'],op['pad'],int(op['relu']),weight_offset,requant_offset]
            specs.append('{ {'+','.join(map(str,values))+'},'+','.join(map(str,[op['bias_term'],len(scales),len(op['candidate_input_scales'])]))+' }')
            scales.extend(op['candidate_input_scales']);weight_offset+=op['weight_count'];requant_offset+=co
    header.append('static const float STREAM_MODEL_SCALES[]={'+','.join(format(s,'.9e')+'f' for s in scales)+'};')
    header.append('static const StreamModelOp STREAM_MODEL_OPS[]={\n'+',\n'.join(specs)+'\n};')
    header.append('static const StreamModelGroup STREAM_MODEL_GROUPS[]={'+','.join('{%d,%d}'%s for s in group_specs)+'};')
    header.append('static const int STREAM_MODEL_GROUP_COUNT='+str(len(groups))+';')
    header.append('static const char STREAM_MODEL_PLAN_SHA256[]="'+hashlib.sha256(a.plan.read_bytes()).hexdigest()+'";')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'stream_model_generated.h').write_text('\n'.join(header)+'\n')
    target=a.output/'model.stream.param'
    target.write_text('7767517\n'+str(len(rows))+' '+str(len(blobs))+'\n'+'\n'.join(rows)+'\n')
    parse_param(target)
    if a.layout_trace:
        original=a.output/'model.stream.unoptimized.param'
        shutil.copyfile(target,original)
        subprocess.run([sys.executable,str(Path(__file__).with_name('optimize_stream_layout.py')),
                        '--param',str(original),'--trace',str(a.layout_trace),'--output',str(a.output)],check=True)
    final_nodes,_,_=parse_param(target)
    print('U02_FUSED_MODEL_GENERATED',len(groups),'groups;',len(final_nodes),'NCNN layers;',len(scales),'candidate scales')

if __name__=='__main__':main()
