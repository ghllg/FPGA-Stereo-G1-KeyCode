"""Prepare a separate weight-staging URAM candidate; preserve validated baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import time

U02 = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hls-record', type=Path, required=True)
    args = parser.parse_args()
    proof_path = args.hls_record / 'result.json'
    proof = json.loads(proof_path.read_text())
    if not proof.get('complete') or not proof.get('synthesized'):
        raise ValueError('Validated complete dual-mode baseline required')
    for path, digest in proof['input_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Baseline source changed: ' + path)
    source = args.hls_record / 'region_dual_compute.h'
    raw = source.read_bytes()
    needle = b'#pragma HLS RESOURCE variable=weight_words core=RAM_2P_BRAM'
    if raw.count(needle) != 1 or b'static uint64_t weight_words[12800];' not in raw:
        raise ValueError('Unexpected baseline weight staging')
    record = U02 / ('reports/region_dual_uram_preparation_' + str(time.time_ns()))
    record.mkdir()
    files = {}
    for name, digest in proof['generated_sha256'].items():
        path = args.hls_record / name
        if sha(path) != digest:
            raise ValueError('Generated baseline changed: ' + name)
        data = path.read_bytes()
        if name == source.name:
            data = data.replace(needle, b'#pragma HLS RESOURCE variable=weight_words core=RAM_2P_URAM')
        target = record / name
        target.write_bytes(data)
        files[name] = dict(baseline_sha256=digest, candidate_sha256=sha(target))
    result = dict(prepared=True, complete=False, board_accessed=False,
                  numerics_verified=False, synthesized=False, deployment_accepted=False,
                  baseline_record=str(args.hls_record.resolve()), baseline_proof_sha256=sha(proof_path),
                  preparation_script_sha256=sha(Path(__file__)), files=files,
                  scope='Only weight_words RAM resource pragma changed; dimensions, access pattern, arithmetic and ABI unchanged. Candidate HLS, RTL and full-design acceptance still required.')
    (record / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
    print('U02_REGION_DUAL_URAM_PREPARATION', record)


if __name__ == '__main__':
    main()
