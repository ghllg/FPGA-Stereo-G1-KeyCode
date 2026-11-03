"""Reserve actual concat/shuffle destinations early; preserve the original FP32 graph."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from plan_stream_buffers import compile_arena, verify_arena

U02=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    graph_path=U02/'reports/stream_boundary_schedule.json'
    graph=json.loads(graph_path.read_text());nodes=graph['schedule']
    tensors={t['blob']:t for t in graph['tensors']}
    reserve={};pairs=[]
    for cat,sh in zip(nodes,nodes[1:]):
        if cat['type']!='Concat' or sh['type']!='ShuffleChannel': continue
        assert len(cat['inputs'])==2 and len(cat['outputs'])==len(sh['outputs'])==1 and sh['inputs']==cat['outputs']
        assert cat['params'].get('0','0')=='0' and sh['params'].get('0')=='2' and sh['params'].get('1','0')=='0'
        assert tensors[cat['outputs'][0]]['consumers']==[sh['index']]
        a,b=[tensors[name]['shape_chw'] for name in cat['inputs']];dst=tensors[sh['outputs'][0]]['shape_chw']
        assert a==b and dst==[2*a[0],a[1],a[2]]
        reserve[sh['outputs'][0]]=cat['index'];pairs.append(dict(concat=cat['index'],shuffle=sh['index'],destination=sh['outputs'][0],bytes=tensors[sh['outputs'][0]]['fp32_bytes']))
    assert len(pairs)==49
    plan=compile_arena(graph,4,reserve);plan['verification']=verify_arena(graph,plan)
    roots=plan['root_allocations'];views=plan['tensor_views']
    for pair in pairs:
        view=views[pair['destination']];lo=view['offset'];hi=lo+view['bytes'];step=pair['concat']
        for name,root in roots.items():
            if name==view['root'] or not root['born']<=step<=root['last_use']:continue
            assert hi<=root['offset'] or root['offset']+root['reserved_bytes']<=lo,('Early destination overlap',pair,name)
    broken=copy.deepcopy(plan);first=pairs[0];dest=broken['root_allocations'][first['destination']]
    source=views[nodes[first['concat']]['inputs'][0]];dest['offset']=source['offset']
    try: verify_arena(graph,broken)
    except AssertionError: plan['early_write_overlap_fault_rejected']=True
    else: raise AssertionError('Missed early output overlap')
    out=U02/'reports/stream_direct_layout_buffer_plan.json';assert not out.exists()
    out.write_text(json.dumps(dict(schema='u02_direct_layout_fp32_arena_v1',graph_sha256=sha(graph_path),
         model_param_sha256=graph['param_sha256'],plans={'fp32':plan},early_reservations=reserve,pairs=pairs,
         board_validated=False,quantization_changed=False,scope='Planning and ownership verification only; hardware and numerical acceptance pending.'),indent=2)+'\n')
    target=U02/'inference/build_fast_nearest/direct_layout_headers';target.mkdir(exist_ok=False)
    subprocess.run([sys.executable,str(U02/'inference/generate_stream_schedule.py'),'--buffers',str(out),
                    '--output',str(target/'stream_schedule_generated.h')],check=True)
    print('DIRECT_LAYOUT_LIFETIME_VERIFIED',len(pairs),plan['arena_bytes'])

if __name__=='__main__': main()
