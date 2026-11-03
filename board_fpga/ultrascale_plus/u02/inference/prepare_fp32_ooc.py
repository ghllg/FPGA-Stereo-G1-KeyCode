"""Stage independent Vivado inputs; do not touch the live HLS simulation."""
import hashlib
import json
from pathlib import Path
import re
from audit_fp32_synthesis import signed_leaves
from prepare_stream_ip import verify_compute_sharing, verify_register_map

U02 = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest_path = U02 / 'reports/stream_fp32_output_synthesis_manifest.json'
manifest = json.loads(manifest_path.read_text())
source = Path(manifest['project']) / 'zu5ev/syn/verilog'
target = U02 / 'inference/fp32_output_ooc_inputs'
assert not target.exists(), 'Do not overwrite previous preparation'
leaves = signed_leaves(source)
assert len(manifest['rtl_sha256']) == 43
assert len(manifest['float_ip_tcl_sha256']) == 4
records = {}
target.mkdir()
for group in ('rtl_sha256', 'float_ip_tcl_sha256'):
    for filename, digest in manifest[group].items():
        original = source / filename
        assert sha(original) == digest
        raw = original.read_bytes()
        if filename in leaves:
            text = raw.decode('ascii')
            needle = 'module ' + leaves[filename] + '('
            assert text.count(needle) == 1
            text = text.replace(needle, '(* use_dsp = "yes" *)\n' + needle)
            raw = text.encode('ascii')
        (target / filename).write_bytes(raw)
        records[filename] = dict(source_sha256=digest, staged_sha256=sha(target / filename),
                                 dsp_leaf=leaves.get(filename))
assert verify_compute_sharing(target, 'requant_words') == manifest['compute_sharing']
assert verify_register_map(target) == manifest['register_map']
assert signed_leaves(target) == leaves
for filename in manifest['float_ip_tcl_sha256']:
    text = (target / filename).read_text()
    name = re.search(r'create_ip .*?-module_name (\w+)', text).group(1)
    assert name + '_ip.tcl' == filename
    assert 'generate_synth_checkpoint false' in text
    assert 'generate_target {synthesis simulation}' in text
result = dict(prepared=True, synthesis_launched=False, hardware_verified=False,
              source_rtl_modified=False, floating_dependencies_preserved=True,
              manifest_sha256=sha(manifest_path), files=records,
              scope='Only DSP mapping attributes on three signed integer leaves; floating IP scripts byte-identical. Separate staged inputs, no access to board or running HLS project.')
(target / 'preparation.json').write_text(json.dumps(result, indent=2) + '\n')
print('FP32_OOC_INPUTS_PREPARED', len(records), target)
