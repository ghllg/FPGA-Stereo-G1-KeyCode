"""Reboot an idle verified system; use only volatile U-Boot arguments.

Default prints a plan. Execution requires a completed restored test record,
or an explicit observed cold boot UUID with no U02 modules/device owners.
Original boot files and the existing verified SD diagnostic bit are read-only.
The XSDB sidecar resumes A53#0 only if it reports Reset Catch; never resets PS.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
from uart_console import Console
from native_loader_boot import ORIGINALS,ORIGINAL_ARGS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('completed_record',type=Path,nargs='?')
    parser.add_argument('--cold-boot-id',help='Explicit observed fresh boot UUID; requires no U02 modules or device owners')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--native-assets',type=Path,help='Frozen native-loader result and diagnostic bit; avoids historical local report paths')
    args=parser.parse_args()
    if bool(args.completed_record)==bool(args.cold_boot_id):
        parser.error('Choose a completed restored record or --cold-boot-id, never both')
    if args.cold_boot_id and not re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',args.cold_boot_id):
        parser.error('Invalid cold boot UUID')
    if args.native_assets:
        sys.path.insert(0,str(U02/'inference'))
        from prepare_portable_hardware import verify_archive
        native_proof=verify_archive(args.native_assets)
        native_path=args.native_assets/'result.json';bit=args.native_assets/'compute_reset_fixed.bit'
    else:
        native_proof=None;native_path=U02/'reports/native_loader_boot_1791696210645006200/result.json'
        bit=U02/'reports/compute_reset_fixed.bit'
    native=json.loads(native_path.read_text())
    assert native['complete'] and native['plan']['original_files_unchanged']==ORIGINALS
    assert hashlib.sha256(bit.read_bytes()).hexdigest()==native['plan']['bit_sha256']
    sd_bit=native['plan']['test_file']
    assert re.fullmatch(r'u02_native_[a-f0-9]{16}/compute.bit',sd_bit)
    bootargs=ORIGINAL_ARGS+' clk_ignore_unused cpuidle.off=1'
    plan=dict(startup_files_modified=False,sd_writes=False,saveenv=False,jtag_system_reset_commands=False,
              kernel_reboot=True,temporary_bootargs=bootargs,cached_sd_bit=sd_bit,
              reset_catch_resume_only=True,requires_four_cores=True)
    plan.update(native_result_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),native_bit_sha256=native['plan']['bit_sha256'])
    if native_proof:plan['native_assets']=native_proof
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:return
    source=args.completed_record.resolve() if args.completed_record else None
    if source:
        prior=json.loads((source/'state.json').read_text())
        assert prior['complete'] and prior['terminal'] and prior['formal_restored'] and prior['governor_restored']
        assert not prior['dma_ambiguous'] and time.time()-(source/'state.json').stat().st_mtime<3600
    else:
        prior={'boot_id':args.cold_boot_id}
    out=U02/'reports'/('validated_linux_reboot_'+str(time.time_ns()));out.mkdir()
    state=dict(complete=False,terminal=False,source_record=str(source) if source else None,plan=plan,
               entry_mode='restored_record' if source else 'observed_cold_boot_no_u02_owners',
               reboot_sent=False,uboot_reached=False,old_boot_id=prior['boot_id'])
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    state['script_sha256']=hashlib.sha256((out/'executed_script.py').read_bytes()).hexdigest()
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('VALIDATED_LINUX_REBOOT_RECORD',out,flush=True)
    # Uses the installed SDK's state command: do not issue con to an unknown
    # stopped state, and do not leave processor debug claims attached to Linux.
    tcl=out/'resume_reset_catch.tcl'
    tcl.write_text('''after 2000
connect -url tcp:127.0.0.1:3121
configparams processor-debug-claim 0
set rc [catch {
    targets -set -filter {name == "Cortex-A53 #0"}
    set current [state]
    puts "U02_REBOOT_CPU_STATE $current"
    if {[string match "Running*" $current]} {
        puts U02_REBOOT_CPU_ALREADY_RUNNING
    } elseif {[string match "Stopped*" $current] && [regexp -nocase {Reset.?Catch} $current]} {
        con
        puts U02_REBOOT_RESET_CATCH_RESUMED_ONCE
    } else {
        error "Refuse to resume an unknown processor state: $current"
    }
} reason]
if {!$rc} {
    set marker {__DYNAMIC_KERNEL_MARKER__}
    set deadline [expr {[clock milliseconds] + 90000}]
    while {![file exists $marker]} {
        if {[clock milliseconds] > $deadline} {set rc 1;set reason "Kernel marker timeout";break}
        after 100
    }
}
if {!$rc} {
    set rc [catch {
        after 500
        set resumed {}
        for {set pass 0} {$pass < 40} {incr pass} {
            foreach core {0 1 2 3} {
                targets -set -filter [format {name == "Cortex-A53 #%d"} $core]
                set current [state]
                if {[string match "Stopped*" $current]} {
                    if {![regexp -nocase {Reset.?Catch} $current]} {error "Unknown stop core $core: $current"}
                    if {[lsearch -exact $resumed $core] >= 0} {error "Repeated reset catch core $core"}
                    con
                    lappend resumed $core
                    puts "U02_SECONDARY_CORE_${core}_RESET_CATCH_RESUMED"
                }
            }
            after 250
        }
        puts "U02_SECONDARY_RESUMED_LIST $resumed"
    } reason]
}
configparams processor-debug-claim 2
puts U02_REBOOT_PROCESSOR_DEBUG_EXCLUDED
disconnect
if {$rc} {puts stderr $reason;exit 1}
exit 0
'''.replace('__DYNAMIC_KERNEL_MARKER__',str(out/'kernel_boot_requested').replace('\\','/')))
    c=None;child=None;in_uboot=False;mount='/tmp/u02_bootcheck_'+uuid.uuid4().hex
    with (out/'uart.log').open('wb') as uart:
        try:
            c=Console('COM9',uart);c.sync()
            def checked(command,timeout=30):
                code,text=c.run(command,timeout)
                if code:raise RuntimeError(command+'\n'+text)
                return text
            def originals():
                text=checked('cd '+mount+' && sha256sum BOOT.BIN image.ub boot.scr system.dtb')
                assert all(digest in text for digest in ORIGINALS.values())
                assert native['plan']['bit_sha256'] in checked('sha256sum '+mount+'/'+sd_bit)
            status=checked('cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/online; cat /proc/modules')
            assert prior['boot_id'] in status and re.search(r'(?m)^0-3\r*$',status)
            opened=checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/video*|/dev/u02*) printf "OPEN:%s\\n" "$t";; esac; done; true')
            assert not re.search(r'(?m)^OPEN:',opened)
            modules=checked('cat /proc/modules')
            if args.cold_boot_id:
                assert not re.search(r'(?m)^u02_\S+\s',modules), 'Cold boot entry requires no U02 modules'
            assert not re.search(r'(?m)^u02_(?!stream_resident\s)\S+\s',modules)
            if re.search(r'(?m)^u02_stream_resident\s',modules):
                assert re.search(r'(?m)^u02_stream_resident\s+\d+\s+0\s',modules)
                checked('rmmod u02_stream_resident')
            state['driver_unloaded_before_reboot']=True;save()
            checked('mkdir '+mount+' && mount -t vfat -o ro /dev/mmcblk0p1 '+mount)
            originals();state['originals_verified_before']=True;save()
            checked('cd / && sync && umount '+mount)
            env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'))
            with (out/'resume_reset_catch.log').open('wb') as log:
                child=subprocess.Popen(['D:/Develop/Xilink/SDK/2018.3/bin/xsdb.bat',str(tcl)],
                    cwd=U02,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                state['resume_pid']=child.pid;state['reboot_sent']=True;save()
                c.send(b'reboot -f\n')
                c.wait(rb'Hit any key to stop autoboot',60);c.send(b' ')
                c.wait(rb'ZynqMP> ',10);in_uboot=True;state['uboot_reached']=True;save()
                state['resume_monitor_active_at_uboot']=True;save()
            def command(value,timeout=30):
                c.send((value+'\n').encode());raw,_=c.wait(rb'ZynqMP> ',timeout)
                text=raw.decode('utf-8','replace');state.setdefault('uboot_commands',[]).append(dict(command=value,output=text));save()
                return text
            listing=command('fatls mmc 0:1');assert all(n.lower() in listing.lower() for n in ORIGINALS)
            assert str(bit.stat().st_size)+' bytes read' in command('fatload mmc 0:1 0x30000000 '+sd_bit)
            command('fpga loadb 0 0x30000000 ${filesize}',50)
            assert re.search(r'(?:^|\n)__U02_FPGA_STATUS:0\r*\n',command('echo __U02_FPGA_STATUS:$?'))
            command("setenv bootargs '"+bootargs+"'")
            assert '41629284 bytes read' in command('fatload mmc 0:1 0x10000000 image.ub')
            assert '25563 bytes read' in command('fatload mmc 0:1 0x00100000 system.dtb')
            (out/'kernel_boot_requested').write_text('ready')
            c.send(b'bootm 0x10000000:kernel 0x10000000:ramdisk 0x00100000\n')
            c.wait(rb'~ # ',60);in_uboot=False
            code=child.wait(timeout=30);state['resume_exit_code']=code;assert code==0
            time.sleep(2)
            for attempt in range(3):
                try:c.sync();break
                except TimeoutError:
                    if attempt==2:raise
            after=checked('uname -r; cat /proc/sys/kernel/random/boot_id; cat /sys/devices/system/cpu/online; cat /proc/cmdline; cat /proc/modules')
            boot=re.search(r'(?m)^([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})\r*$',after)
            assert boot and boot.group(1)!=prior['boot_id'] and re.search(r'(?m)^0-3\r*$',after)
            assert '6.6.40' in after and 'clk_ignore_unused' in after and 'cpuidle.off=1' in after
            state.update(new_boot_id=boot.group(1),after_boot=after,four_cores_verified=True)
            checked('mkdir '+mount+' && mount -t vfat -o ro /dev/mmcblk0p1 '+mount)
            originals();checked('cd / && umount '+mount)
            state['originals_verified_after']=True
            checked('ifconfig eth0 192.168.50.2 netmask 255.255.255.0 up')
            state.update(complete=True,network_configured=True,pl_computation_validated=False)
        except Exception as error:
            state['error']=repr(error)
            if in_uboot:
                try:
                    c.send(b'\x03run bootcmd\n');c.wait(rb'~ # ',60)
                    state['original_boot_fallback_reached_shell']=True
                except Exception as fallback:state['fallback_error']=repr(fallback)
            raise
        finally:
            if c:c.port.close()
            if child and child.poll() is not None:state['resume_exit_code']=child.returncode
            state['terminal']=True;save()
            print('VALIDATED_LINUX_REBOOT_TERMINAL',out,flush=True)


if __name__=='__main__':main()
