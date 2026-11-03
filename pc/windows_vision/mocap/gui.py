"""Windows Chinese workflow UI; heavy processing runs in a separate process."""
from __future__ import annotations
import os
import sys
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox,
    QComboBox, QPlainTextEdit, QMessageBox, QCheckBox, QScrollArea)

HERE = Path(__file__).resolve().parents[1]


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Windows 端 · 双摄视频 → LAFAN1 骨架')
        self.resize(980, 800)
        layout = QVBoxLayout(self)
        title = QLabel('Windows 视频处理：标定 → 人体识别 → 三维重建 → LAFAN1 BVH')
        title.setStyleSheet('font-size: 19px; font-weight: bold; padding: 10px;')
        layout.addWidget(title)
        note = QLabel('输入为 windows_camera 保存的双路会话目录。标定后相机位置、焦点和分辨率保持不变。')
        note.setWordWrap(True); layout.addWidget(note)
        tabs = QTabWidget(); layout.addWidget(tabs)
        cal_tab = QWidget(); cal = QFormLayout(cal_tab)
        self.board_type = QComboBox()
        self.board_type.addItem('普通棋盘（填写内角点数）', 'checkerboard')
        self.board_type.addItem('ChArUco（填写格子数）', 'charuco')
        cal.addRow('标定板类型', self.board_type)
        self.cols = self.spin(3, 30, 8); self.rows = self.spin(3, 30, 5)
        self.square = QDoubleSpinBox(); self.square.setRange(1, 200); self.square.setValue(25); self.square.setSuffix(' mm')
        self.cols_label = QLabel('棋盘内角点列数'); self.rows_label = QLabel('棋盘内角点行数')
        cal.addRow(self.cols_label, self.cols); cal.addRow(self.rows_label, self.rows)
        cal.addRow('实测格子边长', self.square)
        self.marker = QDoubleSpinBox(); self.marker.setRange(.1, 199); self.marker.setValue(24); self.marker.setSuffix(' mm')
        cal.addRow('ChArUco 标记边长', self.marker)
        self.dictionary = QComboBox(); self.dictionary.addItems(['DICT_5X5_100', 'DICT_4X4_50', 'DICT_4X4_100', 'DICT_5X5_250', 'DICT_6X6_250'])
        cal.addRow('ChArUco 字典', self.dictionary)
        self.first_id = self.spin(0, 999, 0); cal.addRow('连续标记起始 ID', self.first_id)
        self.legacy = QCheckBox('使用 OpenCV 4.6 之前的旧排列（以商家图案 / 源码为准）'); cal.addRow(self.legacy)
        self.white_first = QCheckBox('左上角为带标记白格（已验证桌面实物照片）'); cal.addRow(self.white_first)
        presets = QHBoxLayout()
        shop = QPushButton('实物预设：12列×9行 / 30mm格 / 24mm码'); shop.clicked.connect(lambda: self.preset(12,9,30,24,True)); presets.addWidget(shop)
        larger = QPushButton('大格预设：7列×5行 / 50mm格 / 40mm码'); larger.clicked.connect(lambda: self.preset(7,5,50,40)); presets.addWidget(larger)
        cal.addRow(presets)
        self.board_type.currentIndexChanged.connect(self.board_changed)
        self.board_changed()
        button = QPushButton('生成可打印标定板 SVG'); button.clicked.connect(self.board); cal.addRow(button)
        self.cal_session = self.path(cal, '双路标定录像目录', directory=True)
        self.cal_output = self.path(cal, '相机标定输出 JSON', save=True, default=str(HERE/'outputs'/'calibration.json'))
        self.prior_intrinsics = self.path(cal, '复用内参 JSON（可选：仅机位改变，镜头/焦点/分辨率不变）', default='')
        button = QPushButton('计算双摄内参 / 外参'); button.clicked.connect(self.calibrate); cal.addRow(button)
        instructions = QLabel('拍摄至少 12 个不同位置与倾角，建议 20–40 个。每个姿态停稳约 2 秒，板子须同时完整出现在两路画面中。程序检查各路相隔 6 帧的角点位移，剔除仍在移动的画面；日志会显示移动帧剔除数量。停稳筛选不能代替硬件同步或最终标定验收。')
        instructions.setWordWrap(True); cal.addRow(instructions)
        cal_scroll = QScrollArea(); cal_scroll.setWidgetResizable(True); cal_scroll.setWidget(cal_tab)
        tabs.addTab(cal_scroll, '① 双摄标定')
        world_tab = QWidget(); world = QFormLayout(world_tab)
        self.world_cal = self.path(world, '已有相机标定 JSON', default=self.cal_output.text())
        self.world_session = self.path(world, '世界坐标标定板双路录像', directory=True)
        self.world_placement = QComboBox()
        self.world_placement.addItem('水平板：正面朝上', 'horizontal')
        self.world_placement.addItem('竖直板：普通棋盘 / ChArUco，列水平、行向下', 'vertical')
        world.addRow('板子摆放方式', self.world_placement)
        self.board_height = QDoubleSpinBox(); self.board_height.setRange(0,3000); self.board_height.setDecimals(1)
        self.board_height.setSuffix(' mm'); world.addRow('板子印刷面离地高度（放地面填 0）',self.board_height)
        self.vertical_height = QDoubleSpinBox(); self.vertical_height.setRange(0,3000); self.vertical_height.setDecimals(1)
        self.vertical_height.setSuffix(' mm'); self.vertical_height.setEnabled(False)
        world.addRow('竖直板：格子图案上边缘离地高度（不含白边）', self.vertical_height)
        self.world_placement.currentIndexChanged.connect(lambda _: (self.board_height.setEnabled(self.world_placement.currentData() == 'horizontal'), self.vertical_height.setEnabled(self.world_placement.currentData() == 'vertical')))
        self.world_output = self.path(world, '世界标定输出 JSON', save=True, default=str(HERE/'outputs'/'calibration_world.json'))
        self.world_pair = self.spin(-1, 1000000, -1); self.world_pair.setSpecialValueText('自动找第一个可见标定板')
        world.addRow('5 ms 阈值后的配对序号（非视频帧号）', self.world_pair)
        button = QPushButton('建立地面世界坐标'); button.clicked.connect(self.world); world.addRow(button)
        text = QLabel('使用第一页相同的板型和行列参数，相机保持不动。水平板须朝上并用水平尺检查。竖直模式支持普通棋盘和 ChArUco：用垂线/水平尺固定板面竖直，按参数的列数横向摆放，列水平、行向下，不可旋转90°。填写黑白格图案上边缘离地高度，不含白边。普通棋盘第一内角点低一格，程序自动扣除，无需手工扣；原点在第一内角点正下方地面。ChArUco原点在图案左上角正下方地面。相机画面须正立，程序不能自动验证板是否竖直。')
        text.setWordWrap(True); world.addRow(text)
        tabs.addTab(world_tab, '② 地面坐标')
        run_tab = QWidget(); run = QFormLayout(run_tab)
        self.session = self.path(run, '真人双路录像目录', directory=True)
        self.calibration = self.path(run, '带世界坐标的标定 JSON', default=self.world_output.text())
        self.output = self.path(run, '结果父目录（自动新建时间目录）', directory=True, default=str(HERE/'outputs'))
        precise_model = HERE/'models/yolo11m-pose.pt'
        self.model = self.path(run, '人体模型文件（空白自动用 YOLO11n-pose）', default=str(precise_model) if precise_model.exists() else '')
        self.device = QComboBox(); self.device.addItem('自动选择 GPU / CPU', 'auto'); self.device.addItem('NVIDIA GPU 0', '0'); self.device.addItem('CPU', 'cpu')
        run.addRow('推理设备', self.device)
        self.every = self.spin(1, 60, 2); run.addRow('每多少个配对帧识别一次', self.every)
        self.inference_size = self.spin(320, 1536, 960); run.addRow('识别输入尺寸（越大越慢）', self.inference_size)
        self.refine_person = QCheckBox('人体区域二次识别（可选，需比较质量）')
        run.addRow(self.refine_person)
        self.align_content = QCheckBox('离线校正两路录像的内容时间差（需有明显动作）')
        run.addRow(self.align_content)
        self.recover_articulated = QCheckBox('离线骨长/时序约束恢复（保留严格版，恢复结果为估计）')
        self.recover_articulated.toggled.connect(lambda checked: self.align_content.setChecked(True) if checked else None)
        run.addRow(self.recover_articulated)
        self.ensemble_person = QCheckBox('录像补救：双候选识别＋内容同步＋约束恢复（较慢，结果为估计）')
        self.ensemble_person.toggled.connect(lambda checked: self.recover_articulated.setChecked(True) if checked else None)
        run.addRow(self.ensemble_person)
        self.fps = self.spin(1, 120, 30); run.addRow('BVH 输出帧率', self.fps)
        self.delta = QDoubleSpinBox(); self.delta.setRange(.1, 100); self.delta.setValue(10); self.delta.setSuffix(' ms')
        run.addRow('双路采样时间差上限', self.delta)
        button = QPushButton('一键：视频 → 二维点 → 三维骨架 → LAFAN1 BVH'); button.clicked.connect(self.run_pipeline); run.addRow(button)
        text = QLabel('输出 .bvh、.npz、原始二维点、三维骨架和质量报告。长时间遮挡拆分片段；脊柱、脚趾与不可观测扭转使用标注的估计规则。此流程不执行机器人映射或模型训练。')
        text.setWordWrap(True); run.addRow(text)
        tabs.addTab(run_tab, '③ 视频到 LAFAN1')
        tools_tab = QWidget(); tools = QFormLayout(tools_tab)
        self.npz = self.path(tools, '导出的 motion_001.npz')
        button = QPushButton('回放三维骨架'); button.clicked.connect(lambda: self.start(['view', '--npz', self.npz.text()])); tools.addRow(button)
        button = QPushButton('生成“人工抬手”接口与导出样例'); button.clicked.connect(self.sample); tools.addRow(button)
        label = QLabel('人工样例仅用于协议 / 导出自检，不代表相机识别效果。'); tools.addRow(label)
        tabs.addTab(tools_tab, '④ 骨架回放 / 自检')
        self.status = QLabel('就绪 · Windows 端'); layout.addWidget(self.status)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(1500); layout.addWidget(self.log)
        bottom = QHBoxLayout(); layout.addLayout(bottom)
        stop = QPushButton('停止当前处理'); stop.clicked.connect(self.stop); bottom.addWidget(stop)
        open_button = QPushButton('打开结果目录'); open_button.clicked.connect(self.open_output); bottom.addWidget(open_button)
        self.process = QProcess(self); self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(lambda err: self.log.appendPlainText('进程错误：'+self.process.errorString()))
        self.last_output = HERE/'outputs'; self.stopping = False
        self.setStyleSheet('QWidget { background: #182230; color: #edf4fa; font-size: 13px; } '
            'QLineEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#101822; padding:5px; } '
            'QPushButton { background:#27576c; padding:8px; border-radius:4px; } '
            'QTabBar::tab { padding:10px; } QTabBar::tab:selected { background:#27576c; }')

    @staticmethod
    def spin(low, high, default):
        s = QSpinBox(); s.setRange(low, high); s.setValue(default); return s

    def path(self, form, title, directory=False, save=False, default=''):
        row = QWidget(); box = QHBoxLayout(row); box.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit(default); box.addWidget(field); button = QPushButton('选择'); box.addWidget(button)
        def choose():
            if directory: result = QFileDialog.getExistingDirectory(self, title, field.text() or str(HERE))
            elif save: result = QFileDialog.getSaveFileName(self, title, field.text(), 'JSON (*.json)')[0]
            else: result = QFileDialog.getOpenFileName(self, title, field.text(), '支持的文件 (*.json *.pt *.onnx *.npz);;全部文件 (*)')[0]
            if result: field.setText(result)
        button.clicked.connect(choose); form.addRow(title, row); return field

    def board_args(self):
        args = ['--board-type', self.board_type.currentData(), '--cols', str(self.cols.value()),
                '--rows', str(self.rows.value()), '--square-mm', str(self.square.value())]
        if self.board_type.currentData() == 'charuco':
            args += ['--marker-mm', str(self.marker.value()), '--dictionary', self.dictionary.currentText(), '--first-marker-id', str(self.first_id.value())]
            if self.legacy.isChecked(): args.append('--legacy-pattern')
            if self.white_first.isChecked(): args.append('--top-left-white')
        return args

    def board_changed(self):
        charuco = self.board_type.currentData() == 'charuco'
        self.cols_label.setText('ChArUco 格子列数（不是内角点）' if charuco else '棋盘内角点列数')
        self.rows_label.setText('ChArUco 格子行数（不是内角点）' if charuco else '棋盘内角点行数')
        for widget in (self.marker, self.dictionary, self.first_id, self.legacy, self.white_first): widget.setEnabled(charuco)

    def preset(self, cols, rows, square, marker, white_first=False):
        self.board_type.setCurrentIndex(1); self.cols.setValue(cols); self.rows.setValue(rows)
        self.square.setValue(square); self.marker.setValue(marker); self.dictionary.setCurrentText('DICT_5X5_100')
        self.first_id.setValue(0); self.legacy.setChecked(False)
        self.white_first.setChecked(white_first)

    def board(self):
        path = QFileDialog.getSaveFileName(self, '保存标定板', str(HERE/'outputs'/'checkerboard.svg'), 'SVG (*.svg)')[0]
        if path: self.start(['board', '--output', path]+self.board_args())

    def calibrate(self):
        args = ['calibrate', '--session', self.cal_session.text(), '--output', self.cal_output.text()]+self.board_args()
        if self.prior_intrinsics.text(): args += ['--intrinsics-from', self.prior_intrinsics.text()]
        self.start(args)

    def world(self):
        args = ['world', '--calibration', self.world_cal.text(), '--session', self.world_session.text(),
                    '--output', self.world_output.text(), '--pair', str(self.world_pair.value()),
                    '--placement', self.world_placement.currentData()]
        if self.world_placement.currentData() == 'vertical':
            args += ['--vertical-origin-height-mm', str(self.vertical_height.value())]
        else:
            args += ['--board-height-mm', str(self.board_height.value())]
        self.start(args+self.board_args())

    def run_pipeline(self):
        output = Path(self.output.text())/datetime.now().strftime('%Y%m%d_%H%M%S_%f_video_to_lafan1')
        args = ['run', '--session', self.session.text(), '--calibration', self.calibration.text(), '--output', str(output),
                '--device', self.device.currentData(), '--every', str(self.every.value()), '--fps', str(self.fps.value()),
                '--max-delta-ms', str(self.delta.value()), '--image-size', str(self.inference_size.value())]
        if self.align_content.isChecked(): args += ['--align-content']
        if self.refine_person.isChecked(): args += ['--refine-person']
        if self.recover_articulated.isChecked(): args += ['--recover-articulated']
        if self.ensemble_person.isChecked(): args += ['--ensemble-person']
        if self.model.text(): args += ['--model', self.model.text()]
        self.last_output = output; self.start(args)

    def sample(self):
        path = HERE/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_SYNTHETIC')
        self.last_output = path; self.start(['sample', '--output', str(path)])

    def start(self, args):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, '正在处理', '请等待当前处理完成，或点击停止。'); return
        if any(arg == '' for arg in args):
            QMessageBox.information(self, '缺少输入', '请填写本操作需要的目录和文件。'); return
        self.stopping = False; self.status.setText('处理中 · Windows 端'); self.log.appendPlainText('开始：'+' '.join(args))
        env = QProcessEnvironment.systemEnvironment(); env.insert('PYTHONUTF8', '1'); env.insert('PYTHONIOENCODING', 'utf-8')
        self.process.setProcessEnvironment(env); self.process.setWorkingDirectory(str(HERE))
        self.process.start(sys.executable, ['-u', '-m', 'mocap']+args)

    def read_output(self):
        self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace').rstrip())

    def finished(self, code, status):
        self.read_output()
        self.status.setText('已停止 · 输出可能不完整' if self.stopping else ('完成 · Windows 端' if code == 0 else '处理失败 · 请查看日志'))
        if code == 0 and self.last_output.is_dir():
            import json
            report_path = self.last_output/'lafan1/export_report.json'
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                if 'coverage_ratio' in report:
                    summary = f'导出 {report["exported_frames"]} 帧 / {report["exported_duration_s"]:.2f} 秒 · 时间覆盖率 {report["coverage_ratio"]:.1%}'
                    estimated = report.get('source_reconstruction_method') == 'offline_articulated_soft_l1'
                    prefix = '完成但动作不完整 · ' if report['coverage_ratio'] < .9 else ('完成 · 离线估计恢复 · ' if estimated else '完成 · ')
                    self.status.setText(prefix + summary)
                    if any(segment.get('fit_warning') for segment in report.get('segments', [])):
                        self.log.appendPlainText('回放提示：固定骨长后的部分关节与重建目标偏差较大，请核对动作形状；覆盖率不是精度。')
            files = sorted(self.last_output.glob('lafan1/motion_*.npz'))
            if files: self.npz.setText(str(files[0]))

    def stop(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.stopping = True; self.process.terminate()

    def open_output(self):
        path = self.last_output if self.last_output.is_dir() else self.last_output.parent
        if path.exists(): os.startfile(str(path))

    def closeEvent(self, event):
        self.stop(); self.process.waitForFinished(3000); event.accept()


def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'msyh.ttc'
    if font.exists(): QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont('Microsoft YaHei', 10))
    window = Window(); window.show(); return app.exec()
