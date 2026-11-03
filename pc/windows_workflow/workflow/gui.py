import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QLabel, QPushButton, QComboBox, QLineEdit, QFileDialog,
                              QCheckBox, QMessageBox, QSpinBox, QDoubleSpinBox)
from windows_camera.capture import Config, Device, list_devices
from mocap.calibration import load_calibration
from mocap.lafan import PARENTS
from . import ROOT
from .engine import Engine


class Skeleton(QWidget):
    def __init__(self):
        super().__init__(); self.pose = None; self.setMinimumSize(330, 330)

    def paintEvent(self, event):
        painter = QPainter(self); painter.fillRect(self.rect(), QColor('#18202a'))
        painter.setPen(QColor('white')); painter.drawText(15, 25, '实时 LAFAN1 · 22 关节 · Y 朝上')
        if not self.pose or not self.pose.get('valid'):
            painter.drawText(15, 60, '等待有效全身骨架（缺失时不补造动作）'); return
        p = np.array(self.pose['fk_positions_cm'])
        # Oblique projection of Y-up BVH centimetres; centred for inspection.
        xy = np.column_stack([p[:, 0]+.35*p[:, 2], -p[:, 1]+.15*p[:, 2]])
        xy -= (xy.max(axis=0)+xy.min(axis=0))/2
        scale = min((self.width()-60)/max(100, np.ptp(xy[:, 0])),
                    (self.height()-75)/max(160, np.ptp(xy[:, 1])))
        xy = xy*scale+[self.width()/2, self.height()/2+20]
        painter.setPen(QPen(QColor('#40ddbc'), 3))
        for j, parent in enumerate(PARENTS):
            if parent >= 0: painter.drawLine(int(xy[parent, 0]), int(xy[parent, 1]), int(xy[j, 0]), int(xy[j, 1]))
        for x, y in xy: painter.drawEllipse(int(x)-3, int(y)-3, 6, 6)


class Window(QWidget):
    def __init__(self, demo=False):
        super().__init__()
        font = Path('C:/Windows/Fonts/msyh.ttc')
        if font.exists(): QFontDatabase.addApplicationFont(str(font))
        QApplication.instance().setFont(QFont('Microsoft YaHei', 10))
        self.demo = demo; self.engine = None; self.devices = []; self.output = None
        self.setWindowTitle('Windows 端 · 双摄入口同步 → 实时三维骨架 → LAFAN1'); self.resize(1360, 850)
        layout = QVBoxLayout(self); form = QGridLayout(); layout.addLayout(form)
        self.left, self.right = QComboBox(), QComboBox()
        self.scan = QPushButton('刷新 USB 相机'); self.scan.clicked.connect(self.scan_devices)
        form.addWidget(QLabel('cam0 / 左眼'), 0, 0); form.addWidget(self.left, 0, 1)
        form.addWidget(QLabel('cam1 / 右眼'), 0, 2); form.addWidget(self.right, 0, 3); form.addWidget(self.scan, 0, 4)
        self.cal = QLineEdit(str(ROOT/'windows_vision/outputs/calibration_world_2.json'))
        browse = QPushButton('选择世界标定 JSON'); browse.clicked.connect(self.browse_cal)
        form.addWidget(QLabel('世界坐标标定'), 1, 0); form.addWidget(self.cal, 1, 1, 1, 3); form.addWidget(browse, 1, 4)
        self.capture_fps = QComboBox(); self.capture_fps.addItems(['30', '60'])
        self.pose_fps = QSpinBox(); self.pose_fps.setRange(1, 60); self.pose_fps.setValue(30)
        self.clock = QComboBox(); self.clock.addItem('驱动采样时间', 'driver'); self.clock.addItem('回调时钟', 'callback')
        self.compute = QComboBox(); self.compute.addItems(['auto', 'cpu', '0'])
        self.infer_size = QComboBox(); self.infer_size.addItems(['640', '960', '512', '416', '320'])
        self.model_name = QComboBox(); self.model_name.addItems(['yolo11n-pose.pt', 'yolo11m-pose.pt'])
        self.refine_person = QCheckBox('人体区域精细识别（较慢）')
        form.addWidget(QLabel('采集 FPS'), 2, 0); form.addWidget(self.capture_fps, 2, 1)
        form.addWidget(QLabel('骨架目标 FPS'), 2, 2); form.addWidget(self.pose_fps, 2, 3)
        form.addWidget(self.clock, 2, 4); form.addWidget(self.compute, 2, 5)
        form.addWidget(QLabel('模型输入边长'), 4, 0); form.addWidget(self.infer_size, 4, 1)
        form.addWidget(QLabel('降低此值可提速，但远处人体识别可能变差；原相机分辨率与标定保持一致。'), 4, 2, 1, 4)
        form.addWidget(QLabel('人体模型'), 5, 0); form.addWidget(self.model_name, 5, 1)
        form.addWidget(self.refine_person, 5, 2, 1, 3)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setRange(1, 30); self.tolerance.setValue(10); self.tolerance.setSuffix(' ms')
        form.addWidget(QLabel('配对最大时间差'), 3, 0); form.addWidget(self.tolerance, 3, 1)
        self.confirm = QCheckBox('已核对两只物理相机的顺序，机位、朝向、镜头和焦点与该标定一致')
        form.addWidget(self.confirm, 3, 2, 1, 4)
        hint = QLabel('流程：启动预览 → 入口闪光校时 → 全身站稳等待骨长初始化 → 开始记录 → 停止并保存。\n'
                      '闪光校时：白纸放在两路画面中央，同时照亮至少 3 次，间隔长短不同。避免整片反光、自动曝光跳变。\n'
                      '分辨率严格沿用标定；改变机位后旧标定不能继续用。延迟统计不包含相机到主机收到帧之前的耗时。')
        hint.setWordWrap(True); layout.addWidget(hint)
        buttons = QHBoxLayout(); layout.addLayout(buttons)
        self.start_button = QPushButton('① 启动双路预览'); self.start_button.clicked.connect(self.start)
        self.sync_button = QPushButton('② 入口闪光校时（10 秒）'); self.sync_button.clicked.connect(lambda: self.command('sync'))
        self.record_button = QPushButton('③ 开始记录'); self.record_button.clicked.connect(self.record)
        self.save_button = QPushButton('④ 停止记录并保存 BVH'); self.save_button.clicked.connect(lambda: self.command('save'))
        self.stop_button = QPushButton('停止流程'); self.stop_button.clicked.connect(self.stop)
        open_button = QPushButton('打开输出目录'); open_button.clicked.connect(self.open_output)
        for b in (self.start_button, self.sync_button, self.record_button, self.save_button, self.stop_button, open_button): buttons.addWidget(b)
        views = QHBoxLayout(); layout.addLayout(views, stretch=1)
        self.previews = []
        for title in ('cam0 / 左眼', 'cam1 / 右眼'):
            column = QVBoxLayout(); label = QLabel(title); column.addWidget(label)
            image = QLabel('等待输入'); image.setAlignment(Qt.AlignmentFlag.AlignCenter); image.setMinimumSize(300, 250)
            image.setStyleSheet('background:#18202a;color:white'); column.addWidget(image, stretch=1)
            self.previews.append(image); views.addLayout(column, stretch=1)
        self.skeleton = Skeleton(); views.addWidget(self.skeleton, stretch=1)
        self.stats = QLabel('尚未运行'); self.stats.setWordWrap(True); layout.addWidget(self.stats)
        self.status = QLabel('Windows 端实时流程。原有录像和离线工具位于 windows_camera、windows_vision。')
        self.status.setWordWrap(True); layout.addWidget(self.status)
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh); self.timer.start(60)
        if demo:
            self.devices = [Device('模拟左眼', 'demo-left'), Device('模拟右眼', 'demo-right')]
            self.populate(); self.confirm.setChecked(True); self.cal.setText('模拟内存标定，不读取真实标定')
            self.scan.setEnabled(False)

    def populate(self):
        self.left.clear(); self.right.clear()
        for device in self.devices:
            text = device.name+' · '+device.identifier[-45:]
            self.left.addItem(text); self.right.addItem(text)
        if len(self.devices) > 1: self.right.setCurrentIndex(1)
        if not self.demo:
            try:
                cal = load_calibration(self.cal.text())
                for combo, identifier in zip((self.left, self.right), cal['camera_ids']):
                    for i, d in enumerate(self.devices):
                        if d.identifier == identifier: combo.setCurrentIndex(i)
            except Exception: pass

    def scan_devices(self):
        try: self.devices = list_devices(); self.populate(); self.status.setText(f'发现 {len(self.devices)} 路视频设备；请核对物理左右眼。')
        except Exception as exc: QMessageBox.warning(self, '枚举失败', str(exc))

    def browse_cal(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择世界标定', str(ROOT/'windows_vision/outputs'), 'JSON (*.json)')
        if path: self.cal.setText(path); self.populate()

    def start(self):
        try:
            if self.engine and self.engine.thread.is_alive(): raise ValueError('流程已经运行。')
            if len(self.devices) < 2: raise ValueError('先刷新相机并选择两路。')
            if not self.confirm.isChecked(): raise ValueError('请先核对标定对应的物理机位和左右顺序。')
            a, b = self.left.currentIndex(), self.right.currentIndex()
            if a == b: raise ValueError('左右不能选择同一只相机。')
            size = [640, 360] if self.demo else load_calibration(self.cal.text())['image_size']
            config = Config((self.devices[a], self.devices[b]), width=size[0], height=size[1],
                            fps=int(self.capture_fps.currentText()), tolerance_ms=self.tolerance.value(),
                            timestamp_mode=self.clock.currentData(), demo=self.demo)
            self.engine = Engine(config, self.cal.text(), self.compute.currentText(), target_fps=self.pose_fps.value(),
                                 demo=self.demo, inference_size=int(self.infer_size.currentText()),
                                 model=str(ROOT/'windows_vision/models'/self.model_name.currentText()),
                                 refine_person=self.refine_person.isChecked())
            self.engine.start()
        except Exception as exc: QMessageBox.warning(self, '无法启动', str(exc))

    def command(self, name):
        if self.engine and self.engine.thread.is_alive(): self.engine.request(name)

    def record(self):
        folder = ROOT/'windows_workflow/outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_live')
        if self.engine and self.engine.thread.is_alive(): self.engine.request('record', str(folder))

    def stop(self):
        if self.engine: self.engine.stop()

    def open_output(self):
        import os
        folder = self.output or str(ROOT/'windows_workflow/outputs')
        Path(folder).mkdir(parents=True, exist_ok=True); os.startfile(folder)

    def refresh(self):
        if not self.engine: return
        snapshot = self.engine.read(); running = self.engine.thread.is_alive()
        self.start_button.setEnabled(not running)
        for control in (self.left, self.right, self.cal, self.scan, self.capture_fps, self.pose_fps, self.clock, self.compute,
                        self.confirm, self.tolerance, self.infer_size, self.model_name, self.refine_person):
            control.setEnabled(not running and not (self.demo and control is self.scan))
        self.sync_button.setEnabled(running and not snapshot.get('recording', False))
        self.record_button.setEnabled(running and snapshot.get('initialized', False) and not snapshot.get('recording', False))
        self.save_button.setEnabled(running and snapshot.get('recording', False))
        for label, im in zip(self.previews, snapshot.get('images', [])):
            if im is None: continue
            data = np.ascontiguousarray(im)
            image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGB888).copy()
            label.setPixmap(QPixmap.fromImage(image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        self.skeleton.pose = snapshot.get('lafan'); self.skeleton.update()
        timing = snapshot.get('timing', {}); sync = snapshot.get('synchronization', {})
        delta_text = f'{timing["pair_delta_ms"]:.2f} ms' if timing else '尚未配对'
        offset_text = f'{sync["right_offset_ns"]/1e6:.1f} ms' if sync else '尚未校时'
        self.stats.setText(f'采集累计帧：{snapshot.get("capture_counts", [0,0])} ｜ 入口跳过：{snapshot.get("drops", [0,0])} ｜ '
                           f'骨架实测：{snapshot.get("pose_fps", 0):.1f} FPS ｜ 有效点：{snapshot.get("valid_joints", 0)}/17\n'
                           f'原始时间戳 ns：{snapshot.get("source_timestamps_ns", [])} ｜ 配对差：{delta_text} ｜ '
                           f'右路偏移：{offset_text}\n'
                           f'本对处理：{timing.get("processing_ms", 0):.1f} ms ｜ 主机收到帧到骨架：{timing.get("host_receive_to_pose_ms", 0):.1f} ms\n'
                           f'解码 {timing.get("decode_ms", 0):.1f} / 推理 {timing.get("inference_ms", 0):.1f} / '
                           f'三维 {timing.get("reconstruction_ms", 0):.1f} / LAFAN {timing.get("lafan_ms", 0):.1f} ms')
        self.status.setText(snapshot.get('status', ''))
        if snapshot.get('output'): self.output = snapshot['output']

    def closeEvent(self, event):
        if self.engine and self.engine.thread.is_alive():
            self.engine.stop(); self.status.setText('正在停止并保存，请稍候…'); event.ignore()
            QTimer.singleShot(500, self.close)
        else: event.accept()


def launch(demo=False):
    app = QApplication.instance() or QApplication([])
    window = Window(demo); window.show(); return app.exec()
