"""One temporary U-Boot/PCAP validation, preserving all original startup files.

Default prints the exact plan. --stage-only adds the verified test bitstream
without restarting. --execute requires explicit user approval for software
reboot, adding the test bitstream to SD, and temporary boot arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
U02 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bringup/axu5evb_e_boot'))
from uart_console import Console

ORIGINALS = {
    'BOOT.BIN': 'c4da5363d8e3dc19a15792a33bacbc0d284336b76eb8211d389ffb584c0614fd',
    'image.ub': '1d68d58dd9556e23b2bb4df1853da769ab8cc4effc1e6a7e43f04ce4c3d036a0',
    'boot.scr': '54fb01989f23bc40f98cf2ba293904505d9119b9e100a38e3f8f2b67ef81c023',
    'system.dtb': '898d31092c608d0b60bec989d32f0843fe01731294dc4bda603a3d7e145f12f9',
}
ORIGINAL_ARGS = 'console=ttyPS0,115200 earlycon=cdns,mmio,0xff000000 rdinit=/init loglevel=7'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('remote')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stage-only', action='store_true')
    parser.add_argument('--disable-cpuidle', action='store_true', help='Temporary JTAG low-power diagnostic; never save boot arguments')
    args = parser.parse_args()
    if not re.fullmatch(r'/tmp/u02_[a-f0-9]{32}', args.remote):
        raise ValueError('Expected verified temporary transfer directory')
    if args.execute and args.stage_only:
        raise ValueError('Choose staging or an explicitly approved reboot')
    bit = U02 / 'reports/compute_reset_fixed.bit'
    digest = hashlib.sha256(bit.read_bytes()).hexdigest()
    expected = json.loads((U02 / 'reports/reset_fixed/program_manifest.json').read_text())['bit_sha256']
    assert digest == expected
    sd_directory = 'u02_native_' + digest[:16]
    plan = {'bit_sha256': digest, 'test_file': sd_directory + '/compute.bit', 'bytes': bit.stat().st_size,
            'original_files_unchanged': ORIGINALS, 'software_reboot': not args.stage_only,
            'temporary_bootargs': ORIGINAL_ARGS + ' clk_ignore_unused' + (' cpuidle.off=1' if args.disable_cpuidle else ''), 'saveenv': False,
            'loader': 'existing U-Boot fpga loadb through existing PMU/PCAP firmware',
            'firmware_or_kernel_rebuild': False}
    print(json.dumps(plan, indent=2), flush=True)
    if not (args.execute or args.stage_only): return
    report = U02 / 'reports' / ('native_loader_boot_' + str(time.time_ns()))
    report.mkdir()
    result = {'plan': plan, 'complete': False}
    in_uboot = False
    with (report / 'uart.log').open('wb') as log:
        console = Console('COM9', log)
        try:
            console.sync()
            code, text = console.run('mkdir -p /tmp/u02_sd_readonly && (grep -q " /tmp/u02_sd_readonly " /proc/mounts || mount -t vfat -o ro /dev/mmcblk0p1 /tmp/u02_sd_readonly)')
            if code: raise RuntimeError('Original SD could not be inspected read-only')
            code, text = console.run('cd /tmp/u02_sd_readonly && sha256sum BOOT.BIN image.ub boot.scr system.dtb')
            if code or any(value not in text for value in ORIGINALS.values()): raise RuntimeError('Startup file hashes differ')
            code, text = console.run(f'test -f /tmp/u02_sd_readonly/{sd_directory}/compute.bit')
            if code:
                code, text = console.run(f'sha256sum {args.remote}/compute_wrapper.bit')
                if code or digest not in text: raise RuntimeError('Transferred bitstream mismatch')
                code, text = console.run(f'test ! -e /tmp/u02_sd_readonly/{sd_directory} && mount -o remount,rw /tmp/u02_sd_readonly && mkdir /tmp/u02_sd_readonly/{sd_directory}')
                if code: raise RuntimeError('Refuse to overwrite an existing test directory')
                code, text = console.run(f'cp {args.remote}/compute_wrapper.bit /tmp/u02_sd_readonly/{sd_directory}/compute.bit && sync')
                if code: raise RuntimeError('SD test-file staging failed')
            code, text = console.run(f'sha256sum /tmp/u02_sd_readonly/{sd_directory}/compute.bit')
            if code or digest not in text: raise RuntimeError('SD test-file verification failed; never overwrite it')
            code, text = console.run('cd /tmp/u02_sd_readonly && sha256sum BOOT.BIN image.ub boot.scr system.dtb')
            if code or any(value not in text for value in ORIGINALS.values()): raise RuntimeError('Original files changed unexpectedly')
            code, text = console.run('cd / && sync && umount /tmp/u02_sd_readonly')
            if code: raise RuntimeError('SD must be cleanly unmounted before reboot')
            result['sd_file_staged_originals_verified'] = True
            if args.stage_only:
                result['staged_only'] = True
                result['complete'] = True
                return
            # PID1 in the recovery rootfs is a shell, not an init daemon.
            # BusyBox -f invokes the kernel reboot syscall instead of signalling PID1.
            console.send(b'reboot -f\n')
            console.wait(rb'Hit any key to stop autoboot', 50)
            console.send(b' ')
            console.wait(rb'ZynqMP> ', 10)
            in_uboot = True

            def command(value, timeout=30):
                console.send((value + '\n').encode())
                raw, _ = console.wait(rb'ZynqMP> ', timeout)
                output = raw.decode('utf-8', 'replace')
                result.setdefault('uboot_commands', []).append({'command': value, 'output': output})
                print(output, flush=True)
                return output

            command('help fpga')
            device = None
            for candidate in (1, 0):
                output = command(f'fatls mmc {candidate}:1')
                if all(name.lower() in output.lower() for name in ORIGINALS):
                    device = candidate
                    break
            if device is None: raise RuntimeError('Existing SD startup files not found')
            output = command(f'fatload mmc {device}:1 0x30000000 {sd_directory}/compute.bit')
            if f'{bit.stat().st_size} bytes read' not in output: raise RuntimeError('U-Boot bitstream size mismatch')
            output = command('fpga loadb 0 0x30000000 ${filesize}', timeout=50)
            status = command('echo __U02_FPGA_STATUS:$?')
            loaded = bool(re.search(r'(?:^|\n)__U02_FPGA_STATUS:0\r?\n', status))
            result['native_fpga_loaded'] = loaded
            # On rejected load, boot the exact original system arguments.
            arguments = ORIGINAL_ARGS + (' clk_ignore_unused' + (' cpuidle.off=1' if args.disable_cpuidle else '') if loaded else '')
            command(f"setenv bootargs '{arguments}'")
            output = command(f'fatload mmc {device}:1 0x10000000 image.ub')
            if '41629284 bytes read' not in output: raise RuntimeError('Kernel image load mismatch')
            output = command(f'fatload mmc {device}:1 0x00100000 system.dtb')
            if '25563 bytes read' not in output: raise RuntimeError('Device tree load mismatch')
            console.send(b'bootm 0x10000000:kernel 0x10000000:ramdisk 0x00100000\n')
            console.wait(rb'~ # ', 50)
            in_uboot = False
            # Early kernel messages can split the synchronization marker after
            # the shell prompt. Retry synchronization, never reboot on timeout.
            for attempt in range(3):
                try:
                    console.sync()
                    break
                except TimeoutError as error:
                    result.setdefault('post_boot_sync_retries', []).append(str(error))
                    if attempt == 2:
                        raise
            code, output = console.run('uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/cmdline; cat /proc/uptime; devmem 0xff5e0030 32; devmem 0xff5e0040 32; cat /proc/modules')
            result['after_boot'] = {'code': code, 'output': output}
            if code: raise RuntimeError('System did not recover correctly')
            boot = re.search(r'(?m)^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\r*$', output)
            if not boot: raise RuntimeError('Recovered boot identity missing')
            result['boot_id'] = boot.group(1)
            code, output = console.run('ifconfig eth0 192.168.50.2 netmask 255.255.255.0 up')
            result['network_setup'] = {'code': code, 'output': output}
            result['complete'] = loaded and 'clk_ignore_unused' in result['after_boot']['output']
        except Exception as error:
            result['error'] = str(error)
            if in_uboot:
                # Original bootcmd re-reads the unchanged original boot.scr.
                # Do not leave a failed diagnostic boot with modified arguments.
                try:
                    console.send(b'\x03run bootcmd\n')
                    console.wait(rb'~ # ', 50)
                    in_uboot = False
                    result['original_system_fallback_recovered'] = True
                except Exception as recovery_error:
                    result['fallback_error'] = str(recovery_error)
            raise
        finally:
            console.close()
            (report / 'result.json').write_text(json.dumps(result, indent=2))
            print('U02_NATIVE_LOADER_RECORD', report, flush=True)


if __name__ == '__main__':
    main()
