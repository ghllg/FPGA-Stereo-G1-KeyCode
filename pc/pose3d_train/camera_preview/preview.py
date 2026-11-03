"""Live, unsaved side-by-side preview of the two ZU5EV-connected cameras."""
import argparse
import hashlib
import io
from pathlib import Path
import re
import socket
import struct
import sys
import tarfile
import threading
import time
import uuid

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DEMO = Path(r'E:\FPGAchina\move test one\demo_20261013_205000\demo_20261013_205000')
HEADER = struct.Struct('!4sIIIQ')
WINDOW = 'Dual camera LIVE preview - Q / Esc to exit, G for guides'


def receive_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError('开发板已关闭预览连接。')
        data.extend(part)
    return bytes(data)


def receive_frame(sock):
    magic, camera, sequence, size, timestamp = HEADER.unpack(receive_exact(sock, HEADER.size))
    if magic != b'DVP1' or camera not in (0, 1) or not 4 <= size <= 16 * 1024**2:
        raise ValueError('预览数据头异常。')
    data = receive_exact(sock, size)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (720, 1280, 3):
        raise ValueError('画面解码失败，或分辨率不是 1280 x 720。')
    return camera, sequence, timestamp, frame


class Receiver(threading.Thread):
    def __init__(self, sock):
        super().__init__(daemon=True)
        self.sock = sock
        self.latest = [None, None]
        self.error = None
        self.stopping = threading.Event()
        self.lock = threading.Lock()

    def run(self):
        counts, started = [0, 0], time.monotonic()
        try:
            while not self.stopping.is_set():
                camera, sequence, timestamp, frame = receive_frame(self.sock)
                now = time.monotonic()
                counts[camera] += 1
                with self.lock:
                    self.latest[camera] = (frame, sequence, timestamp, now,
                                           counts[camera] / max(now - started, 0.1))
        except Exception as exc:
            if not self.stopping.is_set():
                self.error = exc


def draw_preview(latest, guides=True):
    canvas = np.full((456, 1280, 3), 24, np.uint8)
    now = time.monotonic()
    for camera, item in enumerate(latest):
        x = camera * 640
        label = ('LEFT  / USB 1.4', 'RIGHT / USB 1.2')[camera]
        cv2.putText(canvas, label, (x + 14, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        if item is None:
            cv2.putText(canvas, 'Waiting for live camera...', (x + 40, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 210, 255), 1)
            continue
        frame, sequence, timestamp, received, fps = item
        panel = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        if guides:
            # Existing inference ROI: x=370, y=0, width=540, height=720 in source pixels.
            cv2.rectangle(panel, (185, 1), (454, 358), (0, 230, 255), 2)
            cv2.line(panel, (320, 0), (320, 359), (255, 220, 0), 1)
            cv2.line(panel, (0, 180), (639, 180), (255, 220, 0), 1)
            cv2.putText(panel, 'MODEL ROI', (193, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 230, 255), 1)
        canvas[40:400, x:x+640] = panel
        stale = now - received > 1
        status = 'STALE - NO NEW FRAME' if stale else f'LIVE  seq {sequence}  received {fps:.1f} fps'
        cv2.putText(canvas, status, (x + 14, 423), cv2.FONT_HERSHEY_SIMPLEX, .53,
                    (0, 80, 255) if stale else (100, 240, 130), 1)
    cv2.putText(canvas, 'Q / Esc: exit    G: guides    Full image, no mirror    Framing only; no recording',
                (14, 448), cv2.FONT_HERSHEY_SIMPLEX, .48, (215, 215, 215), 1)
    return canvas


def show_preview(sock):
    receiver = Receiver(sock)
    receiver.start()
    guides = True
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1280, 456)
        while True:
            with receiver.lock:
                latest = list(receiver.latest)
            cv2.imshow(WINDOW, draw_preview(latest, guides))
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('g'), ord('G')):
                guides = not guides
            if receiver.error:
                raise receiver.error
    finally:
        receiver.stopping.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        receiver.join(timeout=4)
        cv2.destroyAllWindows()


def check_mapping(console, demo):
    for node, port in (('video0', '1-1.4'), ('video2', '1-1.2')):
        text = demo.checked(console, 'readlink /sys/class/video4linux/' + node)
        if f'/{port}/{port}:1.0/video4linux/{node}' not in text:
            raise RuntimeError('摄像头节点顺序发生变化，请先修复左右 USB 对应关系；未启动取图。')


def upload_server(args, console, demo, remote):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as archive:
        archive.add(HERE / 'board_preview', arcname='board_preview')
        for file in sorted((HERE / 'runtime').iterdir()):
            archive.add(file, arcname='runtime/' + file.name)
    data = payload.getvalue()
    demo.checked(console, 'mkdir ' + remote)
    token = uuid.uuid4().hex
    ready, done = '__PREVIEW_RX_' + token, '__PREVIEW_DONE_' + token
    command = (f'timeout 30 nc -l -p 18088 > {remote}/preview.tar.gz < /dev/null & rx=$!; '
               f'sleep 1; printf "\\n{ready}\\n"; wait $rx; printf "\\n{done}:%s\\n" "$?"\n')
    console.send(command.encode())
    console.wait(rb'(?:^|\r?\n)' + ready.encode() + rb'\r?\n', 10)
    with demo.connection(args, 18088) as stream:
        stream.sendall(data)
        stream.shutdown(socket.SHUT_WR)
    _, match = console.wait(rb'(?:^|\r?\n)' + done.encode() + rb':([0-9]+)\r?\n', 35)
    if int(match.group(1)):
        raise RuntimeError('预览程序传输失败。')
    text = demo.checked(console, 'sha256sum ' + remote + '/preview.tar.gz')
    if not re.search(r'(?m)^' + hashlib.sha256(data).hexdigest() + r'\s', text):
        raise RuntimeError('预览程序传输校验失败。')
    demo.checked(console, f'cd {remote} && tar -xzf preview.tar.gz && chmod 700 board_preview runtime/ld-linux-aarch64.so.1')


def server_command(remote):
    return f'{remote}/runtime/ld-linux-aarch64.so.1 --library-path {remote}/runtime {remote}/board_preview'


def stop_server(console, demo, remote):
    # Only stop this invocation's executable; never kill a different camera process.
    command = (f'if test -f {remote}/pid; then p=$(cat {remote}/pid); '
               'case "$p" in ""|*[!0-9]*) exit 1;; esac; '
               f'if test "$(readlink /proc/$p/exe)" = {remote}/runtime/ld-linux-aarch64.so.1; then '
               'kill -TERM "$p"; i=0; '
               f'while test "$(readlink /proc/$p/exe)" = {remote}/runtime/ld-linux-aarch64.so.1; do '
               'i=$((i+1)); test "$i" -le 50 || exit 1; sleep 0.1; done; fi; fi')
    demo.checked(console, command, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, default=DEFAULT_DEMO)
    parser.add_argument('--serial', default='COM22')
    parser.add_argument('--board', default='192.168.137.2')
    args = parser.parse_args()
    sys.path.insert(0, str(args.demo / 'pc'))
    import demo
    from uart_console import Console

    demo.verify()
    output = HERE.parent / 'runs' / ('camera_preview_' + time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    print('检查开发板和左右摄像头；稍后打开实时画面。按 Q / Esc 或关闭画面窗口退出。', flush=True)
    print('仅预览，不保存照片或视频。诊断日志：', output, flush=True)
    remote = '/tmp/u02_preview_' + uuid.uuid4().hex
    bound = False
    launched = False
    with (output / 'uart.log').open('wb') as log:
        console = Console(args.serial, log)
        try:
            console.sync()
            demo.preflight(console)
            staged = demo.staged(console)
            demo.restore(console)  # Includes the no-open-camera guard before any rebind.
            check_mapping(console, demo)
            upload_server(args, console, demo, remote)
            bound = True
            demo.bind_cameras(console, staged)
            check_mapping(console, demo)
            launched = True
            demo.checked(console, f'{server_command(remote)} > {remote}/server.log 2>&1 < /dev/null & echo $! > {remote}/pid')
            print('正在启动双摄预览。黄框是当前模型的裁剪范围，青线是画面中线。', flush=True)
            with demo.connection(args, 18085) as stream:
                stream.settimeout(3)
                show_preview(stream)
        finally:
            try:
                if launched:
                    stop_server(console, demo, remote)
                    _, text = console.run('cat ' + remote + '/server.log')
                    (output / 'board_server.log').write_text(text, encoding='utf-8')
                if bound:
                    demo.restore(console)
                    print('预览已停止，两路摄像头已关闭，原摄像头驱动已恢复。', flush=True)
            finally:
                console.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('已退出预览。', flush=True)
        sys.exit(130)
    except Exception as exc:
        print(f'预览未能正常运行：{exc}', file=sys.stderr, flush=True)
        print('请把本窗口报错发回来；先不要重复部署或重启开发板。', file=sys.stderr, flush=True)
        sys.exit(1)
