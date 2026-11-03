"""Fetch only numbered inference outputs from board RAM, verifying archive hashes."""
import argparse
import base64
import hashlib
import io
from pathlib import Path
import re
import socket
import sys
import tarfile
import uuid
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'bringup/axu5evb_e_boot'))
from uart_console import Console


def _connect(timeout=15):
    deadline=time.monotonic()+timeout
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Output connection deadline expired')
        try:
            connection=socket.create_connection(('192.168.50.2',18083),timeout=min(3,remaining),
                                                source_address=('192.168.50.1',0))
            connection.settimeout(timeout)
            return connection
        except OSError:
            if time.monotonic()>=deadline:raise
            time.sleep(.2)


def _stop_sender(console,pid,path):
    # Check both argv and stdin ownership before stopping this one sender.
    # An exited sender has no stdin fd and needs no signal.
    command=(f'if test -e /proc/{pid}/fd/0; then '
             f'cmd=$(tr "\\000" " " < /proc/{pid}/cmdline); '
             f'input=$(readlink /proc/{pid}/fd/0); '
             f'if test -e /proc/{pid}/fd/0; then '
             f'test "$cmd" = "nc -l -p 18083 " && test "$input" = "{path}" || exit 44; '
             f'kill {pid} 2>/dev/null || true; fi; fi')
    code,text=console.run(command)
    if code:raise RuntimeError('Output sender ownership/cleanup failed: '+text)


def _serial_download(console,path,expected):
    code,text=console.run('wc -c < '+path)
    sizes=re.findall(r'(?m)^\s*(\d+)\s*\r?$',text)
    if code or len(sizes)!=1:raise RuntimeError('Output size unavailable')
    size=int(sizes[0])
    if not 0<size<=8*1024*1024:raise RuntimeError('Output exceeds bounded UART fallback')
    token=uuid.uuid4().hex
    begin='__U02_SERIAL_BEGIN_'+token;end='__U02_SERIAL_END_'+token
    timeout=max(30,min(1300,int(size*4/3/9000)+30))
    code,text=console.run(f"printf '\\n{begin}\\n'; base64 {path}; printf '\\n{end}\\n'",timeout=timeout)
    if code:raise RuntimeError('UART output read failed')
    lines=text.replace('\r','').splitlines()
    first,last=lines.index(begin),lines.index(end)
    raw=base64.b64decode(''.join(lines[first+1:last]),validate=True)
    if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=expected:
        raise RuntimeError('UART output size/checksum mismatch')
    return raw


def fetch(console, remote, prefix, count, destination, fixtures=False,transfer_record=None):
    if not re.fullmatch(r'/tmp/u02_[a-f0-9]{32}', remote):
        raise ValueError('Unexpected board RAM directory')
    if not re.fullmatch(r'[a-z][a-z0-9_]*', prefix) or not 1 <= count <= 100:
        raise ValueError('Unexpected output prefix/count')
    name = prefix + '_outputs.tar.gz'
    fixture_glob=f' {prefix}_*.json.fixture' if fixtures else ''
    code, text = console.run(f'cd {remote} && tar -czf {name} {prefix}_*.json {prefix}_*.json.heatmap{fixture_glob} && sha256sum {name}')
    match = re.search(r'(?m)^([0-9a-f]{64})\s+' + re.escape(name), text)
    if code or not match: raise RuntimeError('Output archive unavailable')
    expected = match.group(1)
    record=transfer_record if transfer_record is not None else {}
    record.update(complete=False,method='tcp',expected_sha256=expected)
    path=remote+'/'+name
    code,text=console.run(f'nc -l -p 18083 < {path} > /dev/null & printf "U02_SENDER_PID=%s\\n" "$!"; sleep 1')
    match=re.search(r'(?m)^U02_SENDER_PID=(\d+)\r?$',text)
    if code or not match:raise RuntimeError('Output sender did not start')
    pid=int(match.group(1))
    if pid<=1:raise RuntimeError('Invalid output sender PID')
    record['sender_pid']=pid
    raw=bytearray()
    try:
        with _connect() as connection:
            connection.shutdown(socket.SHUT_WR)
            while True:
                chunk=connection.recv(65536)
                if not chunk:break
                raw.extend(chunk)
                if len(raw)>16*1024*1024:raise RuntimeError('Unexpected archive size')
    except OSError as error:
        record['tcp_error']=repr(error)
        _stop_sender(console,pid,path)
        record['sender_cleaned']=True
        raw=_serial_download(console,path,expected)
        record['method']='uart'
    finally:
        if not record.get('sender_cleaned'):
            _stop_sender(console,pid,path)
            record['sender_cleaned']=True
    if hashlib.sha256(raw).hexdigest() != expected: raise RuntimeError('Archive checksum mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / name).write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        suffixes=('', '.heatmap', '.fixture') if fixtures else ('', '.heatmap')
        allowed = {f'{prefix}_{i}.json{s}' for i in range(count) for s in suffixes}
        if set(archive.getnames()) != allowed: raise RuntimeError('Unexpected output set')
        for member_name in allowed:
            member = archive.getmember(member_name)
            limit=16*1024*1024 if member_name.endswith('.fixture') else 1000000
            if not member.isfile() or member.size > limit: raise RuntimeError('Unexpected archive member')
            (destination / member_name).write_bytes(archive.extractfile(member).read())
    record.update(complete=True,archive_bytes=len(raw))
    return expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('remote')
    parser.add_argument('prefix')
    parser.add_argument('count', type=int)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with (args.destination / 'fetch_uart.log').open('wb') as log:
        console = Console('COM9', log)
        try:
            console.sync()
            print('U02_OUTPUT_ARCHIVE_VERIFIED', fetch(console, args.remote, args.prefix, args.count, args.destination))
        finally:
            console.close()
