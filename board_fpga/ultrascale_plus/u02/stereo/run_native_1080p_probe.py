"""Owned 1080p capture: native UVC or an explicit isolated 1080p-only candidate.

Temporarily adds the known point-to-point address only if needed, preserving
the board's existing addresses. A completed process and closed camera FDs
are independently verified before normal module/binding cleanup. No PL or
boot changes; an observation timeout is not process exit.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import sys
import tarfile
import time
import uuid

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02/'inference'))
from retest_original_after_region import Console,sha,snapshot_cpu_state
from prepare_portable_hardware import verify_archive
from pair_host_elf import verify_pair_host_elf


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true',required=True)
    ap.add_argument('--candidate-module',type=Path,help='Explicit completed build directory for the isolated 1080p30-only module')
    args=ap.parse_args()
    archive=U02/'milestones/stereo_1080p_prepared_20261016';verify_archive(archive)
    binary=U02/'inference/build_fast_nearest/stereo_1080p_host/test_dual_capture_1080p'
    elf=verify_pair_host_elf(binary)
    module=None;module_proof=None
    if args.candidate_module:
        module_proof=json.loads((args.candidate_module/'state.json').read_text())
        assert module_proof['complete'] and module_proof['terminal'] and module_proof['module_name']=='u02_uvc1080'
        module=args.candidate_module/'u02_uvc1080.ko';assert sha(module)==module_proof['module_sha256']
    out=U02/'reports'/('native_1080p_probe_'+str(time.time_ns()));out.mkdir()
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    (out/'tested_probe').write_bytes(binary.read_bytes())
    remote='/tmp/u02_'+uuid.uuid4().hex
    state=dict(complete=False,terminal=False,remote=remote,board_programmed=False,
        modules_changed=False,startup_files_modified=False,inference_executed=False,
        capture_process_uncertain=False,temporary_address_added=False,
        binary_sha256=sha(binary),frames_requested_per_camera=90,source_size=[1920,1080])
    state.update(module_command_uncertain=False,candidate_module_sha256=sha(module) if module else None,
                 candidate_module_loaded=False,native_camera_bindings_restored=False,elf=elf)
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('NATIVE_1080P_PROBE_RECORD',out,flush=True)
    c=None;log=(out/'uart.log').open('wb')
    def checked(cmd,timeout=30):
        assert len(cmd.encode())<700
        rc,text=c.run(cmd,timeout);assert rc==0,(cmd,text);return text
    def no_fds():
        text=checked('for f in /proc/[0-9]*/fd/*; do t=$(readlink "$f"); case "$t" in /dev/u02*|/dev/video*) printf "OPEN:%s\\n" "$f:$t";; esac; done; true')
        assert not re.search(r'(?m)^OPEN:',text)
    def module_command(command):
        state['module_command_uncertain']=True;save()
        rc,result=c.run(command,90)
        state['module_command_uncertain']=False;save()
        assert rc==0,(command,result)
        return result
    def receive(name,limit):
        digest=re.search(r'(?m)^([a-f0-9]{64})\s',checked('sha256sum '+remote+'/'+name)).group(1)
        ready='__P1080_READY_'+uuid.uuid4().hex;done='__P1080_DONE_'+uuid.uuid4().hex
        c.send((f'timeout 30 nc -l -p 18083 < {remote}/{name} > /dev/null & tx=$!; sleep 1; '
            f'printf "\\n{ready}\\n"; wait $tx; printf "\\n{done}:%s\\n" "$?"\n').encode())
        c.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
        raw=bytearray()
        with socket.create_connection(('192.168.50.2',18083),timeout=10,source_address=('192.168.50.1',0)) as sock:
            sock.settimeout(30);sock.shutdown(socket.SHUT_WR)
            while True:
                block=sock.recv(65536)
                if not block:break
                raw.extend(block);assert len(raw)<=limit
        _,match=c.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',40)
        assert int(match.group(1))==0 and hashlib.sha256(raw).hexdigest()==digest
        (out/name).write_bytes(raw);return raw
    try:
        c=Console('COM9',log);c.sync();state['cpu_before']=snapshot_cpu_state(c)
        assert '6.6.40' in checked('uname -r') and 'aarch64' in checked('uname -m')
        no_fds();state['modules_before']=checked('cat /proc/modules')
        assert not re.search(r'(?m)^u02_\S+\s',state['modules_before']), 'Existing experiment runtime: do not take over'
        for dev in ('1-1.4:1.0','1-1.2:1.0'):
            assert '/drivers/uvcvideo' in checked('readlink /sys/bus/usb/devices/'+dev+'/driver')
        state['addresses_before']=checked('ip -4 addr show dev eth0');save()
        if '192.168.50.2/24' not in state['addresses_before']:
            checked('ip addr add 192.168.50.2/24 dev eth0');state['temporary_address_added']=True;save()
        assert '192.168.50.2/24' in checked('ip -4 addr show dev eth0')
        payload=io.BytesIO()
        with tarfile.open(fileobj=payload,mode='w:gz') as t:
            t.add(binary,arcname='capture')
            if module:t.add(module,arcname='u02_uvc1080.ko')
        raw=payload.getvalue();ready='__P1080_RX_'+uuid.uuid4().hex;done='__P1080_RX_DONE_'+uuid.uuid4().hex
        c.send((f'mkdir {remote} && timeout 30 nc -l -p 18082 > {remote}/tools.tar.gz < /dev/null & rx=$!; sleep 1; '
            f'printf "\\n{ready}\\n"; wait $rx; printf "\\n{done}:%s\\n" "$?"\n').encode())
        c.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
        with socket.create_connection(('192.168.50.2',18082),timeout=10,source_address=('192.168.50.1',0)) as sock:
            sock.settimeout(30);sock.sendall(raw);sock.shutdown(socket.SHUT_WR)
        _,match=c.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',40)
        assert int(match.group(1))==0
        assert hashlib.sha256(raw).hexdigest() in checked('sha256sum '+remote+'/tools.tar.gz')
        checked(f'cd {remote} && tar -xzf tools.tar.gz && chmod 700 capture')
        assert sha(binary) in checked('sha256sum '+remote+'/capture')
        checked('dmesg > '+remote+'/kernel_before.log')
        if module:
            assert sha(module) in checked('sha256sum '+remote+'/u02_uvc1080.ko')
            module_command('insmod '+remote+'/u02_uvc1080.ko payload_limit=2048')
            state.update(candidate_module_loaded=True,modules_changed=True);save()
            assert re.search(r'(?m)^2048\r?$',checked('cat /sys/module/u02_uvc1080/parameters/payload_limit'))
            for dev in ('1-1.4:1.0','1-1.2:1.0'):
                module_command(f"printf '%s' {dev} > /sys/bus/usb/drivers/uvcvideo/unbind")
                module_command(f"printf '%s' {dev} > /sys/bus/usb/drivers/u02_uvc1080/bind")
                assert '/drivers/u02_uvc1080' in checked('readlink /sys/bus/usb/devices/'+dev+'/driver')
        state['capture_process_uncertain']=True;save()
        checked(f'(cd {remote}; ./capture 90 1920 1080 --record-all > capture.log 2>&1 < /dev/null & app=$!; '
            'printf "%s\\n" "$app" > capture.pid; wait "$app"; printf "%s\\n" "$?" > capture.exit) > '+remote+'/launcher.log 2>&1 < /dev/null & sleep 1')
        pid_text=checked('cat '+remote+'/capture.pid');pid=re.search(r'(?m)^(\d+)\r?$',pid_text).group(1)
        state['capture_pid']=int(pid);save();deadline=time.monotonic()+120
        state['stream_altsettings']=checked('for d in 1-1.4:1.1 1-1.2:1.1; do printf "%s:" "$d"; cat /sys/bus/usb/devices/$d/bAlternateSetting; done; true')
        while True:
            if time.monotonic()>=deadline:raise TimeoutError('Capture observation deadline; preserve owned runtime, no forced kill')
            # A completion-query timeout is recoverable only via a new read.
            try:rc,text=c.run('test -s '+remote+'/capture.exit',10)
            except TimeoutError:
                c.sync();continue
            if rc==0:break
            if time.monotonic()>=deadline:raise TimeoutError('Capture still owned; preserve runtime, no forced kill')
            time.sleep(2)
        exit_text=checked('cat '+remote+'/capture.exit')
        state['capture_exit_code']=int(re.search(r'(?m)^(\d+)\r?$',exit_text).group(1))
        checked('test ! -d /proc/'+pid);no_fds();state['capture_process_uncertain']=False;save()
        state['capture_log']=checked('cat '+remote+'/capture.log')
        checked('dmesg > '+remote+'/kernel_after.log')
        checked(f'cd {remote} && tar -czf diagnostics.tar.gz capture.log capture.exit capture.pid launcher.log kernel_before.log kernel_after.log')
        receive('diagnostics.tar.gz',4*1024**2)
        if state['capture_exit_code']==0:
            checked(f'cd {remote} && tar -czf images.tar.gz left.csv right.csv left_*.jpg right_*.jpg')
            raw=receive('images.tar.gz',132*1024**2)
            with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
                names={side+'.csv' for side in ('left','right')}|{f'{side}_{i}.jpg' for side in ('left','right') for i in range(90)}
                assert {m.name for m in t.getmembers()}==names
                for m in t.getmembers():assert m.isfile();(out/m.name).write_bytes(t.extractfile(m).read())
            state['complete']=True
        state['cpu_after']=snapshot_cpu_state(c)
        assert state['cpu_after']['boot_id']==state['cpu_before']['boot_id']
    except Exception as e:state['error']=repr(e);raise
    finally:
        try:
            if c and not state['capture_process_uncertain'] and not state['module_command_uncertain']:
                no_fds()
                if state['candidate_module_loaded']:
                    for dev in ('1-1.4:1.0','1-1.2:1.0'):
                        rc,driver=c.run('readlink /sys/bus/usb/devices/'+dev+'/driver',30)
                        if rc==0 and '/drivers/u02_uvc1080' in driver:
                            module_command(f"printf '%s' {dev} > /sys/bus/usb/drivers/u02_uvc1080/unbind")
                        if rc!=0 or '/drivers/u02_uvc1080' in driver:
                            module_command(f"printf '%s' {dev} > /sys/bus/usb/drivers/uvcvideo/bind")
                        assert '/drivers/uvcvideo' in checked('readlink /sys/bus/usb/devices/'+dev+'/driver')
                    module_command('rmmod u02_uvc1080');state['candidate_module_removed']=True
                state['native_camera_bindings_restored']=True
                state['modules_after']=checked('cat /proc/modules')
                assert not re.search(r'(?m)^u02_\S+\s',state['modules_after'])
                if state['temporary_address_added']:
                    checked('ip addr del 192.168.50.2/24 dev eth0');state['temporary_address_removed']=True
                state['addresses_after']=checked('ip -4 addr show dev eth0')
        except Exception as e:state['cleanup_error']=repr(e);state['complete']=False
        if c:c.port.close()
        log.close();state['terminal']=True;save();print('NATIVE_1080P_PROBE_TERMINAL',out,flush=True)


if __name__=='__main__':main()
