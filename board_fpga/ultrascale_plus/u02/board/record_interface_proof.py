"""Bind observed JTAG readiness to the current bitstream and Linux boot."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

U02 = Path(__file__).resolve().parents[1]
ROOT = U02.parents[1]
sys.path.insert(0, str(ROOT / 'bringup/axu5evb_e_boot'))
from uart_console import Console


def native_boot_id(native, digest):
    """New recovery records must prove their boot identity in captured UART output."""
    assert native['complete'] and native['native_fpga_loaded']
    assert native['sd_file_staged_originals_verified']
    assert native['plan']['bit_sha256'] == digest
    boot_id = native['boot_id']
    assert re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', boot_id)
    after = native['after_boot']
    assert after['code'] == 0 and 'clk_ignore_unused' in after['output']
    assert re.search(r'(?m)^' + re.escape(boot_id) + r'\r*$', after['output'])
    return boot_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-record', type=Path)
    parser.add_argument('--jtag-log', type=Path)
    args = parser.parse_args()
    reports = U02 / 'reports'
    manifest = json.loads((reports / 'reset_fixed/program_manifest.json').read_text())
    digest = hashlib.sha256((reports / 'compute_reset_fixed.bit').read_bytes()).hexdigest()
    assert digest == manifest['bit_sha256']
    native_path = args.native_record or reports / 'native_loader_boot_1791649391389732400/result.json'
    native = json.loads(native_path.read_text())
    if args.native_record:
        expected_boot = native_boot_id(native, digest)
        assert args.jtag_log, 'A new recovery requires its own JTAG verification log'
    else:
        assert native['complete'] and native['native_fpga_loaded']
        assert native['sd_file_staged_originals_verified']
        expected_boot = 'd6cb790b-d8cd-4ab1-8daf-677c18055b46'
    jtag_path = args.jtag_log or reports / 'reset_fixed_program_console.log'
    if args.native_record:
        assert jtag_path.stat().st_mtime_ns >= native_path.stat().st_mtime_ns, 'JTAG log predates recovery'
    log = jtag_path.read_text()
    assert 'U02_RESET_AND_AXI_INDEPENDENT_VERIFICATION_PASSED' in log
    observations = {}
    for sample, name, value in re.findall(r'U02_VIO_SAMPLE (\d+) (\S+) ([0-9a-fA-F]+)', log):
        observations.setdefault(name, []).append(int(value, 16))
    assert len(observations) == 3, observations
    resets = [values for name, values in observations.items() if name.endswith('clock_probe_resets')]
    assert resets == [[3, 3, 3]], observations
    counters = [values for name, values in observations.items() if not name.endswith('clock_probe_resets')]
    assert all(len(values) == 3 and len(set(values)) > 1 for values in counters), observations
    uart_path = reports / ('reset_fixed/proof_' + expected_boot + '_uart.log')
    with uart_path.open('wb') as uart:
        console = Console('COM9', uart)
        try:
            console.sync()
            code, text = console.run('cat /proc/sys/kernel/random/boot_id; cat /proc/cmdline; cat /proc/modules')
            assert code == 0
            boot = re.search(r'(?m)^([0-9a-f-]{36})\r*$', text)
            assert boot and boot.group(1) == expected_boot
            assert 'clk_ignore_unused' in text
            assert not re.search(r'(?m)^(?:u02_pose|u02_stream|u02_ddr_test)\s+\d+\s', text)
        finally:
            console.close()
    proof = {'bit_sha256': digest, 'boot_id': boot.group(1),
             'clock_and_reset_observed': True, 'ps_pl_initialization_verified': True,
             'axi_control_response_verified_without_arm': True,
             'observations': observations, 'jtag_log': str(jtag_path.resolve().relative_to(reports)),
             'jtag_log_sha256': hashlib.sha256(jtag_path.read_bytes()).hexdigest(),
             'native_loader_record': str(native_path.resolve().relative_to(reports)),
             'native_loader_sha256': hashlib.sha256(native_path.read_bytes()).hexdigest(),
             'persistent_startup_fixed': False}
    (reports / 'pspl_interface_proof.json').write_text(json.dumps(proof, indent=2))
    print('U02_CURRENT_BOOT_INTERFACE_PROOF_RECORDED', digest, flush=True)


if __name__ == '__main__':
    main()
