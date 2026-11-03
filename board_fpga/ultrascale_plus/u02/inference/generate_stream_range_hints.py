"""Generate model-bound scalar range references from the verified graph audit."""
import argparse
import hashlib
import json
from pathlib import Path

U02=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    frozen=U02/'milestones/channel_range_reuse_audit_20261015'
    from prepare_portable_hardware import verify_archive
    verify_archive(frozen)
    audit=json.loads((frozen/'verified/analysis.json').read_text());graph=json.loads((frozen/'graph.json').read_text())
    param=U02/'milestones/dw118_stem_stereo_verified_20261015/assets/model.stream.param'
    assert hashlib.sha256(param.read_bytes()).hexdigest()==graph['param_sha256']
    tensors={t['blob']:t for t in graph['tensors']}
    groups={int(n['params']['0']):n for n in graph['schedule'] if n['type']=='U02Stream'}
    rows=sorted(audit['consumer_rows'],key=lambda x:x['group']);assert [r['group'] for r in rows]==list(range(110))
    origins=sorted({tuple(x) for r in rows for x in r['input_channel_origins'] if x is not None})
    slots={origin:i for i,origin in enumerate(origins)};assert len(slots)==2942
    inputs=[];outputs=[];specs=[]
    for row in rows:
        gid=row['group'];node=groups[gid];ci,h,w=tensors[node['inputs'][0]]['shape_chw'];co,oh,ow=tensors[node['outputs'][0]]['shape_chw']
        assert row['input_chw']==[ci,h,w]
        specs.append((ci,h,w,co,oh,ow,len(inputs),len(outputs)))
        for origin in row['input_channel_origins']:
            if origin is not None:assert origin[0]<gid
            inputs.append(-1 if origin is None else slots[tuple(origin)])
        outputs.extend(slots.get((gid,ch),-1) for ch in range(co))
    lines=['#pragma once','// Generated exact-layout range references; never DMA addresses.',
        'struct StreamRangeSpec {int ci,h,w,co,oh,ow,input_offset,output_offset;};',
        'static const char STREAM_RANGE_PARAM[] = '+json.dumps(param.read_bytes().decode())+';',
        f'static const unsigned STREAM_RANGE_SLOT_COUNT={len(slots)};',
        'static const StreamRangeSpec STREAM_RANGE_GROUPS[]={']
    lines+=['{'+','.join(map(str,s))+'},' for s in specs]
    lines+=['};','static const int STREAM_RANGE_INPUTS[]={'+','.join(map(str,inputs))+'};',
        'static const int STREAM_RANGE_OUTPUTS[]={'+','.join(map(str,outputs))+'};']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')
    print('STREAM_RANGE_HINTS_GENERATED',args.output,'slots',len(slots),'groups',len(specs))


if __name__=='__main__':main()
