from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QPixmap, QFontDatabase, QPainter, QColor, QFont
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from capture import Capture, Config, Device, Recording, list_devices, format_time_ns

ROOT = Path(__file__).resolve().parent


class Scan(QThread):
    result = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.result.emit(list_devices())
        except Exception as exc:
            self.failed.emit(str(exc))


class Export(QThread):
    result = Signal(str)
    failed = Signal(str)

    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    def run(self):
        try:
            shutil.copytree(self.source, self.destination)
            self.result.emit(str(self.destination))
        except Exception as exc:
            self.failed.emit('导出未完成，原始录像仍在采集目录。\n' + str(exc))


class Preview(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName('preview')
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setObjectName('cameraTitle')
        self.picture = QLabel('连接相机后显示画面')
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(320, 240)
        self.picture.setStyleSheet('color: #67768c; background: #0a1019; border-radius: 8px;')
        self.status = QLabel('等待相机')
        self.status.setObjectName('muted')
        layout.addWidget(heading)
        layout.addWidget(self.picture, 1)
        layout.addWidget(self.status)

    def show_image(self, image, info=None, origin_ns=None):
        if image:
            pixmap = QPixmap.fromImage(image).scaled(
                self.picture.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            if info:
                painter = QPainter(pixmap)
                painter.fillRect(0, 0, pixmap.width(), 66, QColor(0, 0, 0, 185))
                painter.setPen(QColor('#ffffff'))
                painter.setFont(QFont('Microsoft YaHei', 10))
                frame = f'录像帧 #{info.recording_index}（从 0 开始）' if info.recording_index is not None else f'预览帧 #{info.capture_index}'
                origin = info.recording_origin_ns if info.recording_origin_ns is not None else origin_ns
                relative = format_time_ns(info.timestamp_ns - origin) if origin is not None else '等待共同时间起点'
                painter.drawText(10, 20, f'{frame}  |  t={relative}')
                painter.drawText(10, 44, f'源时间戳：{info.timestamp_ns} ns')
                painter.end()
            self.picture.setPixmap(pixmap)


class Window(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.capture = None
        self.recording = None
        self.last_folder = None
        self.scan = None
        self.exporter = None
        self.closing = False
        self.failure = False
        self.started_at = None
        self.record_started_at = None
        self.last_disk_check = 0
        self.smoke_started = False
        self.setWindowTitle('双视角动捕 · 数据采集')
        self.resize(1220, 830)
        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(28, 24, 28, 24)
        page.setSpacing(16)
        title_row = QHBoxLayout()
        title = QLabel('双视角动捕采集')
        title.setObjectName('title')
        self.badge = QLabel('模拟模式' if args.demo else '未连接')
        self.badge.setObjectName('badge')
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.badge)
        page.addLayout(title_row)
        subtitle = QLabel('H200S / USB · 独立视频 · 逐帧时间戳 · 软件时间对齐')
        subtitle.setObjectName('muted')
        page.addWidget(subtitle)

        setup = QFrame()
        setup.setObjectName('panel')
        grid = QGridLayout(setup)
        self.left = QComboBox()
        self.right = QComboBox()
        self.left.setMinimumWidth(220)
        self.right.setMinimumWidth(220)
        self.detect = QPushButton('检测设备')
        self.detect.clicked.connect(self.scan_devices)
        self.mode = QComboBox()
        self.mode.addItem('1920 × 1080 · 60 fps · MJPG', (1920, 1080, 60))
        self.mode.addItem('1920 × 1080 · 30 fps · MJPG', (1920, 1080, 30))
        self.mode.addItem('1280 × 720 · 60 fps · MJPG', (1280, 720, 60))
        self.mode.addItem('1280 × 720 · 30 fps · MJPG', (1280, 720, 30))
        self.mode.addItem('640 × 480 · 30 fps · MJPG', (640, 480, 30))
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(.1, 100)
        self.tolerance.setValue(10)
        self.tolerance.setSuffix(' ms')
        self.tolerance.setToolTip('只影响保存后的帧配对表，不会改变相机曝光时刻。')
        self.timestamp_mode = QComboBox()
        self.timestamp_mode.addItem('驱动采样时间（优先）', 'driver')
        self.timestamp_mode.addItem('回调系统时间（兼容）', 'callback')
        self.timestamp_mode.setToolTip('均非已确认的传感器曝光时间；请用共同闪灯/拍手验证实际同步。')
        self.connect = QPushButton('连接并预览')
        self.connect.clicked.connect(self.toggle_capture)
        grid.addWidget(QLabel('视角 A（left）'), 0, 0)
        grid.addWidget(QLabel('视角 B（right）'), 0, 1)
        grid.addWidget(QLabel('录像模式'), 0, 2)
        grid.addWidget(self.left, 1, 0)
        grid.addWidget(self.right, 1, 1)
        grid.addWidget(self.mode, 1, 2)
        grid.addWidget(self.detect, 1, 3)
        grid.addWidget(QLabel('配对容差'), 2, 0)
        grid.addWidget(self.tolerance, 2, 1)
        grid.addWidget(self.timestamp_mode, 2, 2)
        grid.addWidget(self.connect, 2, 3)
        page.addWidget(setup)
        previews = QHBoxLayout()
        self.previews = [Preview('A  /  正面视角'), Preview('B  /  侧前方视角')]
        for preview in self.previews:
            previews.addWidget(preview, 1)
        page.addLayout(previews, 1)

        metrics = QHBoxLayout()
        self.duration = QLabel('00:00.0')
        self.duration.setObjectName('timer')
        self.quality = QLabel('等待双路画面')
        self.quality.setObjectName('muted')
        self.size = QLabel('已写入 0 MB')
        metrics.addWidget(self.duration)
        metrics.addSpacing(24)
        metrics.addWidget(self.quality, 1)
        metrics.addWidget(self.size)
        page.addLayout(metrics)

        destination = QHBoxLayout()
        self.task_name = QLineEdit('全身动作')
        self.task_name.setMaximumWidth(170)
        self.output = QLineEdit(str(Path(args.output).resolve() if args.output else ROOT / 'recordings'))
        browse = QPushButton('选择目录')
        browse.clicked.connect(self.choose_output)
        self.browse = browse
        destination.addWidget(QLabel('任务'))
        destination.addWidget(self.task_name)
        destination.addWidget(QLabel('采集目录'))
        destination.addWidget(self.output, 1)
        destination.addWidget(browse)
        page.addLayout(destination)
        controls = QHBoxLayout()
        self.start = QPushButton('●  开始录像')
        self.start.setObjectName('start')
        self.stop = QPushButton('■  停止并保存')
        self.save = QPushButton('另存到…')
        self.open_folder = QPushButton('打开录像目录')
        self.start.clicked.connect(self.start_recording)
        self.stop.clicked.connect(self.stop_recording)
        self.save.clicked.connect(self.export_recording)
        self.open_folder.clicked.connect(self.open_recording)
        for button in (self.start, self.stop, self.save, self.open_folder):
            button.setMinimumHeight(44)
            controls.addWidget(button)
        page.addLayout(controls)
        self.message = QLabel('请连接两台相机。建议两路完整覆盖身体与脚部，录制前拍手或闪灯，供后续检查同步。')
        self.message.setWordWrap(True)
        page.addWidget(self.message)
        notice = QLabel('同步指标是采样时间差，不能代表真实曝光误差。原始录像保留全部帧，不烧录时间戳、不补帧。')
        notice.setObjectName('muted')
        notice.setWordWrap(True)
        page.addWidget(notice)
        self.apply_style()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)
        self.update_buttons()
        if args.demo:
            self.left.addItem('模拟相机 A', Device('DEMO A', 'demo-left'))
            self.right.addItem('模拟相机 B', Device('DEMO B', 'demo-right'))
            self.mode.setCurrentIndex(4 if args.smoke else 3)
            if args.smoke:
                QTimer.singleShot(0, self.toggle_capture)
        else:
            QTimer.singleShot(0, self.scan_devices)

    def apply_style(self):
        self.setStyleSheet('''
            QMainWindow, QWidget { background: #101722; color: #e8edf4; font-family: "Microsoft YaHei UI"; font-size: 13px; }
            QLabel#title { font-size: 27px; font-weight: 700; }
            QLabel#muted { color: #8e9cb1; }
            QLabel#badge { color: #72d4cb; background: #193537; padding: 7px 16px; border-radius: 14px; }
            QLabel#cameraTitle { font-size: 15px; font-weight: 600; padding: 5px; }
            QLabel#timer { font-size: 28px; font-family: Consolas; color: #72d4cb; }
            QFrame#panel, QFrame#preview { background: #172130; border: 1px solid #29374b; border-radius: 10px; }
            QFrame#panel QLabel, QFrame#preview QLabel { background: transparent; }
            QComboBox, QLineEdit, QDoubleSpinBox { background: #0d1520; border: 1px solid #354459; border-radius: 5px; padding: 7px; }
            QPushButton { background: #243247; border: 1px solid #35455e; border-radius: 6px; padding: 9px 16px; }
            QPushButton:hover { background: #334864; }
            QPushButton#start { background: #147a72; border: 1px solid #249b90; font-weight: 700; }
            QPushButton#start:hover { background: #20998e; }
            QPushButton:disabled { background: #1b2431; color: #5b687a; border-color: #263143; }
        ''')

    def update_buttons(self):
        live = self.capture is not None
        recording = self.recording is not None
        scanning = self.scan is not None and self.scan.isRunning()
        exporting = self.exporter is not None and self.exporter.isRunning()
        for widget in (self.left, self.right, self.mode, self.tolerance, self.timestamp_mode):
            widget.setEnabled(not live and not scanning)
        for widget in (self.output, self.browse, self.task_name):
            widget.setEnabled(not recording)
        self.detect.setEnabled(not live and not scanning and not self.args.demo)
        self.connect.setEnabled(not recording and not scanning and not (live and self.capture.stopping.is_set())
                                and self.left.count() > 0 and self.right.count() > 0)
        self.connect.setText('断开预览' if live else '连接并预览')
        self.start.setEnabled(live and self.capture.stream_ready and not self.capture.stopping.is_set()
                              and not recording and not self.closing)
        self.stop.setEnabled(recording and self.recording.accepting)
        self.save.setEnabled(self.last_folder is not None and not recording and not exporting)
        self.open_folder.setEnabled(self.last_folder is not None)

    def scan_devices(self):
        self.message.setText('正在检测 USB 视频设备…')
        self.scan = Scan()
        self.scan.result.connect(self.scanned)
        self.scan.failed.connect(self.report_error)
        self.scan.finished.connect(self.update_buttons)
        self.scan.start()
        self.update_buttons()

    def scanned(self, devices):
        self.left.clear()
        self.right.clear()
        self.right.addItem('暂不启用（单相机）', None)
        for i, device in enumerate(devices):
            label = f'{i + 1}. {device.name}'
            self.left.addItem(label, device)
            self.right.addItem(label, device)
            self.left.setItemData(i, device.identifier, Qt.ItemDataRole.ToolTipRole)
            self.right.setItemData(i + 1, device.identifier, Qt.ItemDataRole.ToolTipRole)
        external = [i for i, device in enumerate(devices)
                    if 'usb#vid_' in device.identifier.lower() and 'integrated' not in device.name.lower()]
        if external:
            self.left.setCurrentIndex(external[0])
        if len(external) >= 2:
            self.right.setCurrentIndex(external[1] + 1)
        self.message.setText(f'检测到 {len(devices)} 个视频设备，请确认 A、B 是两只 H200S。相同名称通过设备路径区分。')

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, '选择采集目录', self.output.text())
        if path:
            self.output.setText(path)

    def toggle_capture(self):
        if self.capture:
            self.capture.stop()
            self.connect.setEnabled(False)
            return
        left, right = self.left.currentData(), self.right.currentData()
        if left is None or (right is not None and left.identifier == right.identifier):
            self.report_error('请选择相机 A；若启用 B，请选择不同的设备。')
            return
        width, height, fps = self.mode.currentData()
        config = Config((left, right) if right is not None else (left,), width, height, fps,
                        self.tolerance.value(), self.args.demo, self.timestamp_mode.currentData())
        self.previews[1].picture.clear()
        self.previews[1].picture.setText('连接相机后显示画面' if right is not None else 'B 未启用 · 单相机模式')
        self.previews[1].status.setText('等待相机' if right is not None else '未进行双路同步')
        self.capture = Capture(config)
        self.capture.failed.connect(self.capture_failed)
        self.capture.initialized.connect(self.capture_ready)
        self.capture.finished.connect(self.capture_finished)
        self.started_at = time.monotonic()
        self.failure = False
        self.capture.start()
        self.message.setText(f'正在打开 {len(config.devices)} 路 MJPG 视频，请稍候…')
        self.badge.setText('连接中')
        self.update_buttons()

    def capture_ready(self):
        self.badge.setText('模拟预览' if self.args.demo else ('双路预览' if self.capture.camera_count == 2 else '单路预览'))
        self.message.setText('画面已就绪。确认全身可见后开始录像；实际帧率请看画面下方。')
        self.update_buttons()
        if self.args.smoke and not self.smoke_started:
            self.smoke_started = True
            self.start_recording()
            QTimer.singleShot(2500, self.stop_recording)
            if self.args.screenshot:
                QTimer.singleShot(1500, lambda: self.grab().save(self.args.screenshot))

    def start_recording(self):
        if self.recording or not self.capture or not self.capture.stream_ready:
            return
        try:
            root = Path(self.output.text()).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < 1024 ** 3:
                raise RuntimeError('采集目录剩余空间不足 1 GB，请选择其他磁盘。')
            name = ''.join('_' if char in '<>:"/\\|?*' or ord(char) < 32 else char for char in self.task_name.text()).strip(' .')[:60] or '动作'
            folder = root / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{name}'
            _, timestamps, received, _, _ = self.capture.snapshot()
            # Start in source-clock time; discard packets buffered before button press.
            now = time.perf_counter_ns()
            origin = max(stamp + now - arrival for stamp, arrival in zip(timestamps, received))
            self.recording = Recording(folder, self.capture.config, origin)
            self.capture.attach_recording(self.recording)
            self.record_started_at = time.monotonic()
            self.last_disk_check = 0
            self.badge.setText('● 正在录像')
            self.message.setText(f'正在写入：{folder}\n停止后自动保存两路视频和时间戳。')
            self.update_buttons()
        except Exception as exc:
            self.report_error(str(exc))

    def stop_recording(self):
        if self.recording:
            self.recording.stop()
            if self.capture:
                self.capture.attach_recording(None)
            self.badge.setText('保存中')
            self.message.setText('正在写完已采集帧、关闭视频并生成帧配对表…')
            self.update_buttons()

    def report_error(self, message):
        self.failure = True
        self.message.setText(message)
        logging.error('%s', message)
        if not self.args.smoke:
            QMessageBox.warning(self, '采集提示', message)
        else:
            self.closing = True

    def capture_failed(self, message):
        if self.recording:
            self.recording.stop(reason=message)
            self.stop_recording()
        if self.capture:
            self.capture.stop()
        self.report_error(message)

    def capture_finished(self):
        if self.recording and self.recording.accepting:
            self.recording.stop(reason='预览采集中断')
        self.capture = None
        self.badge.setText('采集中断' if self.failure else '未连接')
        self.update_buttons()

    def refresh(self):
        now = time.monotonic()
        if self.capture:
            images, timestamps, received, recent, counts = self.capture.snapshot()
            images, preview_info, preview_origin = self.capture.preview_snapshot()
            for camera, preview in enumerate(self.previews):
                if camera >= len(images):
                    continue
                preview.show_image(images[camera], preview_info[camera], preview_origin)
                values = recent[camera]
                fps = (len(values) - 1) * 1e9 / (values[-1] - values[0]) if len(values) > 1 else 0
                preview.status.setText(f'实测 {fps:.1f} fps  ·  已接收 {counts[camera]} 帧')
            if len(recent) == 1:
                self.quality.setText('单相机模式 · 未进行双路同步')
            elif all(values for values in recent):
                differences = [min(abs(stamp - other) for other in recent[1]) / 1e6 for stamp in recent[0][-30:] if recent[1][0] <= stamp <= recent[1][-1]]
                delta = sorted(differences)[len(differences) // 2] if differences else None
                low = any(len(values) > 20 and (len(values) - 1) * 1e9 / (values[-1] - values[0]) < self.capture.config.fps * .85 for values in recent)
                text = f'近期最近帧采样时间差（中位数）：{delta:.2f} ms' if delta is not None else '等待时间重叠'
                self.quality.setText(text + ('  ·  帧率低于目标' if low else ''))
                if all(info is not None for info in preview_info):
                    display_delta = (preview_info[1].timestamp_ns - preview_info[0].timestamp_ns) / 1e6
                    self.quality.setText(self.quality.text() + f'\n当前显示帧 B−A：{display_delta:+.2f} ms（预览抽帧，非训练配对）')
            if not self.capture.stopping.is_set():
                stale = self.capture.stream_ready and any(time.perf_counter_ns() - value > 3_000_000_000 for value in received if value is not None)
                timeout = not self.capture.stream_ready and now - self.started_at > 20
                if stale or timeout:
                    self.capture_failed('相机连续 3 秒无帧。' if stale else '连接超过 20 秒未就绪，请检查相机、占用程序和所选 MJPG 模式。')
        if self.recording:
            record = self.recording
            elapsed = now - self.record_started_at
            self.duration.setText(f'{int(elapsed // 60):02d}:{elapsed % 60:04.1f}')
            frame_text = ' / '.join(f'{"AB"[i]} {count}' for i, count in enumerate(record.frames))
            self.size.setText(f'已写入 {record.bytes_written / 1024 ** 2:.0f} MB · {frame_text} 帧')
            if record.accepting and now - self.last_disk_check > 2:
                self.last_disk_check = now
                try:
                    if shutil.disk_usage(record.folder).free < 512 * 1024 ** 2:
                        record.stop(reason='剩余磁盘空间不足 512 MB')
                except OSError as exc:
                    record.stop(reason=str(exc))
            if not record.accepting and self.capture:
                self.capture.attach_recording(None)
            if record.done.is_set():
                self.last_folder = record.folder
                self.recording = None
                self.badge.setText('已保存' if not record.error and not record.reason else '已保存 · 异常')
                stats = record.metadata.get('synchronization', {})
                frame_text = '，'.join(f'{"AB"[i]} {count} 帧' for i, count in enumerate(record.frames))
                paired = f'；有效时间配对 {stats.get("paired_frames", 0)} 对。' if record.camera_count == 2 else '；单相机录像。'
                if record.camera_count == 2:
                    ratios = stats.get('paired_ratio_per_camera', [0, 0])
                    delta = stats.get('mean_abs_sample_delta_ms')
                    paired += f' 配对覆盖率 A {ratios[0]:.1%} / B {ratios[1]:.1%}'
                    if delta is not None:
                        paired += f'，平均时间差 {delta:.2f} ms。'
                self.message.setText(f'已保存到 {record.folder}\n{frame_text}{paired}')
                if record.error or record.reason:
                    self.report_error(record.error or record.reason)
                self.update_buttons()
                if self.args.smoke:
                    self.closing = True
        if self.closing and not self.recording:
            if self.capture:
                self.capture.stop()
            elif not (self.scan and self.scan.isRunning()) and not (self.exporter and self.exporter.isRunning()):
                QTimer.singleShot(0, self.close)
        self.update_buttons()

    def export_recording(self):
        if not self.last_folder:
            return
        root = QFileDialog.getExistingDirectory(self, '另存录像到其他目录（复制，保留原件）', str(self.last_folder.parent))
        if not root:
            return
        destination = Path(root) / self.last_folder.name
        if destination.exists():
            self.report_error('目标目录已存在，请选择其他目录，避免覆盖录像。')
            return
        self.exporter = Export(self.last_folder, destination)
        self.exporter.result.connect(lambda path: self.message.setText('另存完成：' + path))
        self.exporter.failed.connect(self.report_error)
        self.exporter.finished.connect(self.update_buttons)
        self.exporter.start()
        self.message.setText('正在复制录像，请稍候…')
        self.update_buttons()

    def open_recording(self):
        if self.last_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_folder)))

    def closeEvent(self, event):
        busy = self.capture or self.recording or (self.scan and self.scan.isRunning()) or (self.exporter and self.exporter.isRunning())
        if busy:
            event.ignore()
            self.closing = True
            self.stop_recording()
            if self.capture:
                self.capture.stop()
            self.message.setText('正在保存并关闭…')
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='双视角动捕数据采集')
    parser.add_argument('--demo', action='store_true', help='不连接相机，使用模拟双路视频')
    parser.add_argument('--smoke', action='store_true', help='自动录制模拟视频后退出')
    parser.add_argument('--output', help='采集目录')
    parser.add_argument('--screenshot', help='自动验收时保存界面截图')
    args = parser.parse_args()
    if args.smoke:
        args.demo = True
    log_folder = ROOT / 'logs'
    log_folder.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_folder / f'{datetime.now():%Y%m%d_%H%M%S}.log', encoding='utf-8')])
    app = QApplication(sys.argv)
    # Explicitly load the system CJK font, including with Qt's offscreen platform.
    for filename in ('msyh.ttc', 'msyhbd.ttc'):
        font = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
    window = Window(args)
    window.show()
    if args.smoke:
        # Watchdog for failed startup: never leave unattended verification hanging.
        QTimer.singleShot(30000, lambda: (setattr(window, 'failure', True), window.close()))
    result = app.exec()
    return 1 if window.failure else result


if __name__ == '__main__':
    sys.exit(main())
