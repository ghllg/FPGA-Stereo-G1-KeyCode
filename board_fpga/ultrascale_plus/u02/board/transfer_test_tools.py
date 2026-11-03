"""Transfer verified U02 tools into board /tmp using the existing UART and LAN."""
import hashlib
import argparse
import io
import json
from pathlib import Path
import re
import socket
import sys
import tarfile
import uuid
import time
import base64

ROOT = Path(__file__).resolve().parents[3]
U02 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bringup/axu5evb_e_boot"))
from uart_console import Console


def uart_archive(console, raw, remote, result):
    """Transfer only a private, hash-checked tool archive, without PL access."""
    if len(raw) > 32*1024*1024:
        raise RuntimeError('UART tool archive exceeds the explicit 32 MiB limit')
    token = uuid.uuid4().hex
    ready, done = '__U02_UART_ARCHIVE_'+token, '__U02_UART_ARCHIVE_DONE_'+token
    console.send((f'u02_uart_saved=$(stty -g); stty -echo; '
                  f"printf '\\n{ready}\\n'; base64 -d > {remote}/tools.tar.gz; "
                  f'u02_uart_code=$?; stty "$u02_uart_saved"; '
                  f"printf '\\n{done}:%s\\n' \"$u02_uart_code\"\n").encode())
    console.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
    encoded = base64.encodebytes(raw)
    next_update = time.monotonic()
    try:
        for offset in range(0,len(encoded),2048):
            console.send(encoded[offset:offset+2048])
            if time.monotonic() >= next_update:
                print('U02_UART_ARCHIVE_PROGRESS', min(offset+2048,len(encoded)), len(encoded), flush=True)
                next_update = time.monotonic()+5
        console.port.flush()
        console.send(b'\x04')
        _,match = console.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',60)
        if int(match.group(1)):
            raise RuntimeError('UART base64 receiver failed')
    except Exception:
        console.send(b'\x03\nstty echo\n')
        raise
    result['transport'] = 'uart_base64'
    result['uart_encoded_bytes'] = len(encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--clock-restore', action='store_true')
    parser.add_argument('--diagnostic-bit', action='store_true')
    parser.add_argument('--pose-backend', action='store_true')
    parser.add_argument('--stream-model', action='store_true')
    parser.add_argument('--samples-dir', type=Path)
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--host', type=Path)
    parser.add_argument('--stream-param', type=Path)
    parser.add_argument('--cpu-param', type=Path)
    parser.add_argument('--stream-driver', type=Path)
    parser.add_argument('--stream-fixture', type=Path)
    parser.add_argument('--quant-test', action='store_true')
    parser.add_argument('--quant-test-host', type=Path)
    args = parser.parse_args()
    if args.quant_test and not args.inference:parser.error('--quant-test requires --inference')
    if args.quant_test_host and (not args.quant_test or not re.fullmatch(r'test_stream_[a-z_]+',args.quant_test_host.name)):
        parser.error('--quant-test-host requires --quant-test and a test_stream_* executable')
    if args.pose_backend and not args.inference:
        parser.error('--pose-backend requires --inference')
    if args.stream_model and not args.inference:
        parser.error('--stream-model requires --inference')
    if not 1 <= args.sample_count <= 100 or (args.samples_dir and not args.inference):
        parser.error('Sample overrides require inference and a count from 1 to 100')
    if args.host and (not args.inference or args.host.name != 'light_pose'):
        parser.error('--host requires inference and a file named light_pose')
    if args.stream_param and (not args.stream_model or args.stream_param.name != 'model.stream.param'):
        parser.error('--stream-param requires --stream-model and model.stream.param')
    if args.cpu_param and (not args.inference or args.cpu_param.name != 'model.opt.param'):
        parser.error('--cpu-param requires --inference and model.opt.param')
    if args.stream_driver and (not args.stream_model or not args.pose_backend or args.stream_driver.name != 'u02_stream.ko'):
        parser.error('--stream-driver requires --stream-model --pose-backend and u02_stream.ko')
    if args.stream_fixture and (not args.stream_model or not args.pose_backend or args.stream_fixture.name != 'fixture_0.json.fixture'):
        parser.error('--stream-fixture requires --stream-model --pose-backend and fixture_0.json.fixture')
    token = uuid.uuid4().hex
    directory = U02 / "reports" / ("transfer_" + token)
    directory.mkdir()
    files = [U02 / "reports" / name for name in ("u02_ddr_test.ko", "u02_ddr_test_host")]
    if args.clock_restore:
        files = [U02 / 'reports/u02_rpll_restore.ko']
    if args.diagnostic_bit:
        files = [U02 / 'board/debug_project/u02_compute.runs/impl_1/compute_wrapper.bit']
    if args.inference:
        files = [args.host or U02 / 'inference/build/light_pose']
        files += [ROOT / 'board_light_pose/models' / name for name in
                  ('model.pl.param', 'model.pairs.param', 'model.opt.bin')]
        files += [args.cpu_param or ROOT/'board_light_pose/models/model.opt.param']
        samples = args.samples_dir or ROOT / 'board_light_pose/outputs/samples'
        files += [samples / (str(i) + suffix)
                  for i in range(args.sample_count) for suffix in ('.f32', '.json')]
    if args.pose_backend:
        files += [U02 / 'reports/u02_pose.ko', U02 / 'inference/build/test_backend', U02 / 'inference/build/test_transport']
    if args.stream_model:
        files += [args.stream_param or U02 / 'inference/build/model.stream.param']
        if args.pose_backend:
            files += [args.stream_driver or U02 / 'reports/u02_stream.ko', U02 / 'inference/build/test_stream_packets',
                      U02 / 'inference/build/test_stream_transport']
        if args.stream_fixture:
            files.append(args.stream_fixture)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.quant_test:
        tool=args.quant_test_host or U02/'inference/build/test_stream_quantize';files.append(tool)
        hashes[tool.name]=hashlib.sha256(tool.read_bytes()).hexdigest()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for file in files:
            archive.add(file, arcname=file.name)
    raw = payload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    remote = "/tmp/u02_" + token
    ready = "__U02_NET_READY_" + token
    done = "__U02_NET_DONE_" + token
    result = {"directory": remote, "files": hashes, "archive_sha256": digest}
    with (directory / "uart.log").open("wb") as log:
        console = Console("COM9", log)
        try:
            console.sync()
            code, output = console.run("uname -r; zcat /proc/config.gz | sha256sum; cat /proc/uptime; cat /sys/class/net/eth0/carrier")
            result["before"] = output
            if code or not re.search(r"(?:^|\n)6\.6\.40\r?\n", output):
                raise RuntimeError("Unexpected running kernel")
            console.send((f"mkdir -p {remote}; nc -l -p 18082 > {remote}/tools.tar.gz < /dev/null & u02_rx_pid=$!; "
                          f"sleep 1; printf '\\n{ready}:%s\\n' \"$u02_rx_pid\"; wait $u02_rx_pid; "
                          f"printf '\\n{done}:%s\\n' \"$?\"\n").encode())
            _, receiver_match = console.wait(rb"(?:^|\r?\n)" + ready.encode() + rb":([0-9]+)\r?\n", 10)
            receiver_pid = int(receiver_match.group(1))
            result['receiver_pid'] = receiver_pid
            # Retry only connection establishment: no bytes or PL work have
            # been submitted. Never retry a partially sent archive.
            connection = None
            result['connection_attempts'] = []
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    connection = socket.create_connection(("192.168.50.2", 18082),
                        timeout=min(3, max(.1, deadline-time.monotonic())),
                        source_address=('192.168.50.1', 0))
                    result['connection_attempts'].append({'connected':True})
                    break
                except OSError as error:
                    result['connection_attempts'].append({'connected':False,'error':str(error)})
                    time.sleep(.2)
            if connection is None:
                result['tcp_connection_fallback'] = True
                # Interrupt the shell's wait only. Identify the owned receiver
                # by both exact argv and its private archive before killing it.
                console.send(b'\x03\n')
                console.sync()
                command = (f'p=/proc/{receiver_pid}; '
                           'if test -d "$p"; then '
                           'a=$(tr "\\000" " " < "$p/cmdline"); '
                           f'test "$a" = "nc -l -p 18082 " && '
                           f'test "$(readlink "$p/fd/1")" = "{remote}/tools.tar.gz" && '
                           f'kill {receiver_pid}; else true; fi')
                code, output = console.run(command)
                if code:
                    raise RuntimeError('Receiver ownership uncertain; refuse UART fallback')
                result['receiver_cleanup'] = output
                uart_archive(console, raw, remote, result)
            else:
                connection.settimeout(30)
                with connection:
                    connection.sendall(raw)
                    connection.shutdown(socket.SHUT_WR)
                _, match = console.wait(rb"(?:^|\r?\n)" + done.encode() + rb":([0-9]+)\r?\n", 30)
                if int(match.group(1)):
                    raise RuntimeError("Receive failed")
                result['transport'] = 'tcp'
            code, output = console.run(f"sha256sum {remote}/tools.tar.gz")
            if code or digest not in output:
                raise RuntimeError("Transfer archive hash mismatch")
            executable = 'light_pose' if args.inference else 'u02_ddr_test_host'
            if args.pose_backend: executable += ' test_backend test_transport'
            if args.pose_backend and args.stream_model: executable += ' test_stream_packets test_stream_transport'
            if args.quant_test: executable += ' '+tool.name
            chmod = '' if args.clock_restore or args.diagnostic_bit else f' && chmod 700 {executable}'
            # Keep the BusyBox UART command below its canonical input line limit.
            code, output = console.run(f"cd {remote} && tar -xzf tools.tar.gz{chmod}")
            if code: raise RuntimeError('Archive extraction failed')
            names = list(hashes)
            for start in range(0, len(names), 16):
                batch = names[start:start+16]
                code, output = console.run(f"cd {remote} && sha256sum " + ' '.join(batch))
                if code or any(not re.search(r'(?m)^'+hashes[name]+r'\s+\*?'+re.escape(name)+r'\r?$', output) for name in batch):
                    raise RuntimeError("Transferred file hashes mismatch")
            result["verified"] = True
            result["module_loaded"] = False
            result["host_run"] = False
            print("U02_TOOLS_TRANSFERRED", remote, flush=True)
        except Exception as error:
            result["error"] = str(error)
            raise
        finally:
            console.close()
            (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
