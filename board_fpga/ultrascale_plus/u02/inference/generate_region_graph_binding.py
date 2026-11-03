"""Bind region boundaries to exact full-graph blob IDs and original publication nodes."""
import argparse
import json
from pathlib import Path
from lower_stream_region import U02, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--param',type=Path,default=U02/'inference/build/model.stream.param')
    args=parser.parse_args()
    program_path=U02/'milestones/stream_region_dispatch/first_region.json'
    program=json.loads(program_path.read_text())
    if any(sha(Path(p))!=h for p,h in program['source_sha256'].items()):raise ValueError('Region sources changed')
    graph_path=next(Path(p) for p in program['source_sha256'] if Path(p).name=='schedule.json')
    graph=json.loads(graph_path.read_text());nodes=graph['schedule']
    # Match the actual NCNN blob ordering from its exact parameter file, rather
    # than tensor first-use order or the region-local view IDs.
    param_path=args.param
    if sha(param_path)!=graph['param_sha256']:raise ValueError('Graph parameter SHA mismatch')
    blobs=[]
    for line in param_path.read_text().splitlines()[2:]:
        fields=line.split();bottoms=int(fields[2]);tops=int(fields[3]);names=fields[4:4+bottoms+tops]
        for name in names:
            if name not in blobs:blobs.append(name)
    if len(blobs)!=graph['blob_count']:raise ValueError('Blob ordering is incomplete')
    tensors={t['blob']:t for t in graph['tensors']}
    manifest_path=U02/'milestones/stream_region_dispatch/first_region.manifest.json'
    metadata=manifest_path.with_name('first_region.bin');manifest=json.loads(manifest_path.read_text())
    if sha(metadata)!=manifest['sha256']:raise ValueError('Metadata changed')
    entries=[]
    for flags,key in ((1,'ingress'),(2,'egress')):
        for name in program[key]:
            t=tensors[name];c,h,w=t['shape_chw']
            birth=t['producer']
            if flags==1 and birth>=program['entry_node']:raise ValueError('Ingress not ready at entry')
            if flags==2 and birth not in program['replaced_nodes']:raise ValueError('Egress birth not replaced')
            entries.append(dict(name=name,tensor=manifest['tensor_names'].index(name),blob=blobs.index(name),
                birth=birth,c=c,h=h,w=w,flags=flags))
    output=['// Generated from exact region and full model graph. No device access.','#pragma once',
        'struct RegionGraphBoundary {unsigned tensor,blob,birth,c,h,w,flags;};',
        'static const unsigned REGION_GRAPH_ENTRY='+str(program['entry_node'])+';',
        'static const unsigned REGION_GRAPH_LAYERS='+str(graph['node_count'])+';',
        'static const unsigned REGION_GRAPH_BLOBS='+str(len(blobs))+';',
        'static const char REGION_GRAPH_PARAM[]='+json.dumps(param_path.read_bytes().decode())+';',
        'static const unsigned REGION_GRAPH_REPLACED[]={'+','.join(map(str,program['replaced_nodes']))+'};',
        'static const RegionGraphBoundary REGION_GRAPH_BOUNDARIES[]={']
    output += ['    {'+','.join(str(e[k]) for k in ('tensor','blob','birth','c','h','w','flags'))+'}, // '+e['name'] for e in entries]
    output += ['};','static const unsigned char REGION_GRAPH_METADATA[]={']
    raw=metadata.read_bytes()
    output += ['    '+','.join(str(b) for b in raw[i:i+32])+',' for i in range(0,len(raw),32)]
    output += ['};']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(output)+'\n')
    proof=dict(entry=program['entry_node'],replaced_nodes=program['replaced_nodes'],boundaries=entries,
        egress_publication='At each original producer node, after early region computation; not all at entry.',
        formal_legacy_groups_remaining=110-len(program['group_ids']),full_network_hardware_integrated=False,
        source_sha256={str(p):sha(p) for p in (program_path,graph_path,param_path,manifest_path,metadata,Path(__file__))},
        output_sha256=sha(args.output))
    Path(str(args.output)+'.manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('U02_REGION_GRAPH_BINDING',len(entries),'boundaries',len(program['replaced_nodes']),'nodes')


if __name__=='__main__':main()
