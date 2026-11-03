"""Bounded repair round 2: remap only PW/DW local partial-sum storage."""
import json
from pathlib import Path
from synthesize_fp32_ooc import U02, sha


def main():
    old = U02 / 'inference/hls/stream_overlap_fp32_wide_coalesced.cpp'
    new = U02 / 'inference/hls/stream_overlap_fp32_wide_partial_uram.cpp'
    runner = U02 / 'inference/validate_fp32_wide_partial_uram_csim.py'
    project = U02 / 'inference/stream_hls_23_fp32_wide_partial_uram'
    assert not new.exists() and not runner.exists() and not project.exists()
    proof = json.loads((U02 / 'milestones/fp32_wide_coalesced_post_opt_resource_failure/manifest.json').read_text())
    assert proof['remaining_error'] == 'UTLZ-1' and proof['dsp_cascade_error_count'] == 0
    rtl = json.loads((U02 / 'reports/fp32_wide_coalesced_cosim_1791866727589024900/state.json').read_text())
    assert rtl['complete']
    assert rtl['input_sha256'][str(old)] == sha(old)
    text = old.read_text()
    before = '#pragma HLS RESOURCE variable=partial core=RAM_2P_LUTRAM'
    after = '#pragma HLS RESOURCE variable=partial core=RAM_2P_URAM'
    assert text.count(before) == 2
    updated = text.replace(before, after)
    assert updated.replace(after, before) == text
    new.write_text(updated)
    template = (U02 / 'inference/validate_fp32_wide_coalesced_csim.py').read_text()
    prepared = template.replace('stream_hls_22_fp32_wide_coalesced', 'stream_hls_23_fp32_wide_partial_uram')
    prepared = prepared.replace('fp32_wide_coalesced', 'fp32_wide_partial_uram')
    prepared = prepared.replace('FP32_WIDE_COALESCED_CSIM_RECORD', 'FP32_WIDE_PARTIAL_URAM_CSIM_RECORD')
    prepared = prepared.replace('and eight UltraRAM weight staging banks',
        'and eight UltraRAM weight staging banks plus PW/DW partial-sum UltraRAM mapping (repair round 2)')
    runner.write_text(prepared)
    evidence = dict(repair_round=2, source_sha256=sha(old), candidate_sha256=sha(new),
        changed_pragmas=2, arithmetic_changed=False, external_abi_changed=False,
        candidate_project=str(project), board_accessed=False, resource_accepted=False,
        numeric_rtl_verified=False,
        rationale='Post-opt LUT 119121/117120; 37328 LUTs are memory. PW and DW each explicitly declare 16x512x32 LUTRAM partial sums. Remap these existing banks to unused UltraRAM. Actual mapping/latency/RTL/route/whole-network performance must be verified.',
        expected_uram_if_one_per_bank=40, available_uram=64,
        scope='Mapping-only candidate; predicted bank count is not a measured resource result.')
    output = U02 / 'milestones/fp32_wide_partial_uram_prepared'
    output.mkdir()
    (output / 'manifest.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
