"""One bounded JTAG system reset, then the existing temporary native boot.

Never writes startup files, saves U-Boot environment, initializes PS by hand,
changes voltage, or retries a failed reset. User authorized reversible recovery.
"""
import hashlib
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

U02 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(U02.parents[1] / 'bringup/axu5evb_e_boot'))
from uart_console import Console
from native_loader_boot import ORIGINALS, ORIGINAL_ARGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-reset-catch', action='store_true')
    parser.add_argument('--disable-cpuidle', action='store_true', help='Temporary kernel argument only, for the JTAG/CPU low-power conflict diagnostic')
    args = parser.parse_args()
    report = U02 / 'reports' / ('uart_jtag_recovery_' + str(time.time_ns()))
    report.mkdir()
    result = dict(complete=False, startup_files_modified=False, saveenv=False,
                  reset_attempts=0, pl_computation=False)
    result['temporary_cpuidle_off'] = args.disable_cpuidle
    native = json.loads((U02 / 'reports/native_loader_boot_1791696210645006200/result.json').read_text())
    assert native['complete'] and native['plan']['original_files_unchanged'] == ORIGINALS
    bit = U02 / 'reports/compute_reset_fixed.bit'
    assert hashlib.sha256(bit.read_bytes()).hexdigest() == native['plan']['bit_sha256']
    tcl = report / 'reset.tcl'
    reset_commands = '' if args.resume_reset_catch else (
        'targets -set -filter {name == "PSU"}\n'
        'puts "U02_SINGLE_SYSTEM_RESET_BEGIN"\nrst -system\n'
        'puts "U02_SINGLE_SYSTEM_RESET_RETURNED"\n')
    tcl.write_text('connect -url tcp:127.0.0.1:3121\n'
                   'configparams processor-debug-claim 0\n' + reset_commands +
                   'targets -set -filter {name == "Cortex-A53 #0"}\n'
                   'con\nputs "U02_RESET_CATCH_CORE_ZERO_RESUMED"\n'
                   'configparams processor-debug-claim 2\n'
                   'puts "U02_MPSOC_PROCESSOR_DEBUG_EXCLUDED"\ndisconnect\nexit\n')
    with (report / 'uart.log').open('wb') as log:
        console = Console('COM9', log)
        child = None
        in_uboot = False
        try:
            with (report / 'reset.log').open('wb') as reset_log:
                child = subprocess.Popen([
                    'D:/Develop/Xilink/SDK/2018.3/bin/xsdb.bat', str(tcl)],
                    stdout=reset_log, stderr=subprocess.STDOUT)
                result['reset_attempts'] = 0 if args.resume_reset_catch else 1
                result['resume_reset_catch'] = args.resume_reset_catch
                result['reset_pid'] = child.pid
                console.wait(rb'Hit any key to stop autoboot', 50)
                console.send(b' ')
                console.wait(rb'ZynqMP> ', 10)
                in_uboot = True
                result['uboot_reached'] = True
                child.wait(timeout=15)
                result['reset_exit_code'] = child.returncode
                if child.returncode:
                    raise RuntimeError('Reset tool did not complete normally')

            def command(value, timeout=30):
                console.send((value + '\n').encode())
                raw, _ = console.wait(rb'ZynqMP> ', timeout)
                out = raw.decode('utf8', 'replace')
                result.setdefault('uboot_commands', []).append(dict(command=value, output=out))
                print(out, flush=True)
                return out

            listing = command('fatls mmc 0:1')
            assert all(name.lower() in listing.lower() for name in ORIGINALS)
            out = command('fatload mmc 0:1 0x30000000 ' + native['plan']['test_file'])
            assert f'{bit.stat().st_size} bytes read' in out
            command('fpga loadb 0 0x30000000 ${filesize}', 50)
            status = command('echo __U02_FPGA_STATUS:$?')
            assert re.search(r'(?:^|\n)__U02_FPGA_STATUS:0\r?\n', status)
            arguments = ORIGINAL_ARGS + ' clk_ignore_unused' + (' cpuidle.off=1' if args.disable_cpuidle else '')
            result['temporary_bootargs'] = arguments
            command("setenv bootargs '" + arguments + "'")
            assert '41629284 bytes read' in command('fatload mmc 0:1 0x10000000 image.ub')
            assert '25563 bytes read' in command('fatload mmc 0:1 0x00100000 system.dtb')
            console.send(b'bootm 0x10000000:kernel 0x10000000:ramdisk 0x00100000\n')
            console.wait(rb'~ # ', 50)
            in_uboot = False
            time.sleep(3)
            for attempt in range(3):
                try:
                    console.sync()
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
            code, out = console.run('uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/cmdline; cat /proc/modules')
            assert code == 0 and 'clk_ignore_unused' in out and '6.6.40' in out
            result['after_boot'] = out
            code, out = console.run('cat /sys/devices/system/cpu/online; cat /sys/devices/system/cpu/present')
            result['cpu_online_check'] = dict(code=code, output=out)
            if code or len(re.findall(r'(?m)^0-3\r*$',out)) != 2:
                raise RuntimeError('Console recovered but four-core recovery is incomplete')
            code, out = console.run('mkdir -p /tmp/u02_sd_readonly && mount -t vfat -o ro /dev/mmcblk0p1 /tmp/u02_sd_readonly && cd /tmp/u02_sd_readonly && sha256sum BOOT.BIN image.ub boot.scr system.dtb')
            assert code == 0 and all(sha in out for sha in ORIGINALS.values())
            result['originals_verified_after_boot'] = True
            code, out = console.run('cd / && umount /tmp/u02_sd_readonly && ifconfig eth0 192.168.50.2 netmask 255.255.255.0 up')
            assert code == 0
            result['complete'] = True
        except Exception as error:
            result['error'] = str(error)
            if in_uboot:
                try:
                    console.send(b'\x03run bootcmd\n')
                    console.wait(rb'~ # ', 50)
                    result['original_boot_fallback_reached_shell'] = True
                except Exception as fallback_error:
                    result['fallback_error'] = str(fallback_error)
            raise
        finally:
            console.close()
            if child and child.poll() is not None:
                result['reset_exit_code'] = child.returncode
            (report / 'result.json').write_text(json.dumps(result, indent=2))
            print('U02_UART_JTAG_RECOVERY_RECORD', report, flush=True)


if __name__ == '__main__':
    main()
