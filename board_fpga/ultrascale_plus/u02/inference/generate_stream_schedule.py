"""Emit a graph-bound FP32 arena schedule for the AArch64 full-network Host."""
import argparse
import hashlib
import json
from pathlib import Path
from plan_stream_buffers import verify_arena

U02 = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', type=Path, default=U02/'reports/stream_boundary_schedule.json')
    p.add_argument('--buffers', type=Path, default=U02/'reports/stream_buffer_plan.json')
    p.add_argument('--param', type=Path, default=U02/'inference/build/model.stream.param')
    p.add_argument('--output', type=Path, default=U02/'inference/build/stream_schedule_generated.h')
    a = p.parse_args()
    graph = json.loads(a.graph.read_text()); buffers = json.loads(a.buffers.read_text())
    assert buffers['graph_sha256'] == hashlib.sha256(a.graph.read_bytes()).hexdigest()
    assert buffers['model_param_sha256'] == graph['param_sha256'] == hashlib.sha256(a.param.read_bytes()).hexdigest()
    plan = buffers['plans']['fp32']; verify_arena(graph, plan)
    blobs = list(graph['tensors']); indices = {t['blob']: i for i, t in enumerate(blobs)}
    lines = ['#pragma once', '// Generated from the verified full-network graph and lifetime plan.',
             'struct StreamBlobSpec {const char* name; int c,h,w; size_t offset;};',
             'struct StreamLayerSpec {const char* name; const char* type; std::vector<int> inputs,outputs;};',
             'static const char STREAM_SCHEDULE_PARAM[] = '+json.dumps(a.param.read_bytes().decode('utf-8'))+';',
             'static const size_t STREAM_SCHEDULE_ARENA_BYTES = '+str(plan['arena_bytes'])+';',
             'static const StreamBlobSpec STREAM_SCHEDULE_BLOBS[] = {']
    for t in blobs:
        c,h,w = t['shape_chw']; view = plan['tensor_views'][t['blob']]
        assert (h*w*4) % 16 == 0, 'External NCNN Mat requires padded channel stride'
        lines.append('{'+f'{json.dumps(t["blob"])},{c},{h},{w},{view["offset"]}'+'},')
    lines += ['};', 'static const StreamLayerSpec STREAM_SCHEDULE_LAYERS[] = {']
    for n in graph['schedule']:
        ids = lambda names: '{'+','.join(str(indices[b]) for b in names)+'}'
        lines.append('{'+','.join([json.dumps(n['name']),json.dumps(n['type']),ids(n['inputs']),ids(n['outputs'])])+'},')
    lines += ['};', f'static const int STREAM_SCHEDULE_INPUT = {indices["image"]};',
              f'static const int STREAM_SCHEDULE_OUTPUT = {indices["conv2d_441.tmp_1"]};']
    a.output.write_text('\n'.join(lines)+'\n')
    print('U02_STREAM_SCHEDULE_GENERATED',len(blobs),len(graph['schedule']),plan['arena_bytes'])


if __name__ == '__main__':
    main()
