"""Prepare bounded boundary-loop pipelining without changing arithmetic."""
import argparse
import json
from pathlib import Path
import time
from prepare_region_dual_uram import U02, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hls-record', type=Path, required=True)
    args = parser.parse_args()
    proof_path = args.hls_record / 'result.json'
    proof = json.loads(proof_path.read_text())
    if not proof.get('complete') or not proof.get('synthesized'):
        raise ValueError('Validated original dual-mode top required')
    for name, digest in proof['input_sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError('Baseline source changed: ' + name)
    record = U02 / ('reports/region_boundary_pipeline_preparation_' + str(time.time_ns()))
    record.mkdir()
    changes = []
    files = {}
    for name, digest in proof['generated_sha256'].items():
        source = args.hls_record / name
        if sha(source) != digest:
            raise ValueError('Generated baseline changed: ' + name)
        text = source.read_text()
        if name == 'region_boundary_words.h':
            replacements = [
                ('inline float region_word_float(const uint64_t* data,size_t at) {\n',
                 'inline float region_word_float(const uint64_t* data,size_t at) {\n#pragma HLS INLINE\n'),
                ('    for(int c=0;c<first.ci;++c) {\n        float maximum=0;',
                 '    bool invalid_input=false;\n    for(int c=0;c<first.ci;++c) {\n        float maximum=0;'),
                ('        for(size_t p=0;p<plane;++p) {\n',
                 '        for(size_t p=0;p<plane;++p) {\n#pragma HLS PIPELINE II=1\n'),
                ('            if(!std::isfinite(value))return -4;',
                 '            if(!std::isfinite(value))invalid_input=true;'),
                ('    for(int i=0;i<(first.kind?first.ci:1);++i) {',
                 '    if(invalid_input)return -4;\n    for(int i=0;i<(first.kind?first.ci:1);++i) {'),
                ('    for(int c=0;c<first.ci;++c)for(size_t p=0;p<plane;++p) {\n',
                 '    for(int c=0;c<first.ci;++c)for(size_t p=0;p<plane;++p) {\n#pragma HLS PIPELINE II=1\n')]
            for before, after in replacements:
                if text.count(before) != 1:
                    raise ValueError('Boundary transform no longer unambiguous: ' + before)
                text = text.replace(before, after)
                changes.append(dict(before=before, after=after))
        target = record / name
        target.write_text(text)
        files[name] = dict(baseline_sha256=digest, candidate_sha256=sha(target))
    state = dict(prepared=True, complete=False, board_accessed=False, synthesized=False,
                 numerics_verified=False, deployment_accepted=False,
                 baseline_record=str(args.hls_record.resolve()), baseline_proof_sha256=sha(proof_path),
                 preparation_script_sha256=sha(Path(__file__)), files=files, changes=changes,
                 scope='Inline packed FP32 read and request II=1 for scan/quantize. Nonfinite rejection deferred until scan ends, before output mutation. Actual II/resource/RTL verification required; RAM bindings unchanged.')
    (record / 'preparation.json').write_text(json.dumps(state, indent=2) + '\n')
    print('U02_REGION_BOUNDARY_PIPELINE_PREPARATION', record)


if __name__ == '__main__':
    main()
