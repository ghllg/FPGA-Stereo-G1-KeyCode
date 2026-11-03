from __future__ import annotations
import argparse
import bisect
import json
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QFrame,QVBoxLayout,QHBoxLayout,
    QPushButton,QComboBox,QSlider,QListWidget,QListWidgetItem,QFileDialog,QDialog,QFormLayout,
    QLineEdit,QSpinBox,QDialogButtonBox,QPlainTextEdit,QSplitter,QScrollArea,QMessageBox)
from .core import HOME,ROOT,RUNS,Session,Timeline,StageTimingAverage,stage_metrics,milliseconds,read_json,write_json,refined_display_cameras
from .services import load_config,BoardClient,new_session,snapshot_assets,motion_config
from .widgets import STYLE,label,Card,ImageCanvas,SceneCanvas,ComparisonPlot
from .workers import Task,MediaReader,OverviewCamera
from .robot_view import RobotView
from .pipeline import G1Runner,simulate_recording
from .import_media import import_videos
from .theme import apply_theme


ROUND1_LIVE_GROUPS = {
    'R1-01': (60,'时钟与共同事件','0—10秒自然静止；10—30秒重复缓慢抬手—停止，保证两路都能看清起动和停止；30—60秒交替慢动作和静止。全身和双脚始终在两路画面内。'),
    'R1-02': (30,'普通动作第1次','0—7秒自然静止；7—12秒双臂缓慢抬起；12—14秒保持；14—17秒放下；17—20秒保持；20—25秒一次舒适上举/左右换向；25—30秒静止。全身和双脚始终在两路画面内。'),
    'R1-03': (30,'普通动作第2次','0—7秒自然静止；7—12秒双臂缓慢抬起；12—14秒保持；14—17秒放下；17—20秒保持；20—25秒一次舒适上举/左右换向；25—30秒静止。全身和双脚始终在两路画面内。'),
    'R1-04': (30,'普通动作第3次','0—7秒自然静止；7—12秒双臂缓慢抬起；12—14秒保持；14—17秒放下；17—20秒保持；20—25秒一次舒适上举/左右换向；25—30秒静止。全身和双脚始终在两路画面内。'),
    'R1-05': (40,'左臂小动作','0—5秒静止；5—15秒左臂小幅慢伸屈；15—20秒保持；20—30秒左臂较大舒适伸屈/换向；30—40秒静止。全身和双脚始终在两路画面内。'),
    'R1-06': (40,'右臂小动作','0—5秒静止；5—15秒右臂小幅慢伸屈；15—20秒保持；20—30秒右臂较大舒适伸屈/换向；30—40秒静止。全身和双脚始终在两路画面内。'),
    'R1-07': (30,'慢伸展','0—5秒静止；5—13秒双臂逐渐伸展；13—18秒保持；18—25秒缓慢回落；25—30秒静止。全身和双脚始终在两路画面内。'),
    'R1-08': (40,'遮挡与恢复','0—8秒静止；随后做两次各约2秒胸前交叉，每次后打开双臂恢复可见；再做一次约2秒手到身后，然后恢复。其余时间保持两路可见，最后静止。全身和双脚始终在两路画面内。'),
    'R1-09': (30,'左脚低抬诊断','0—5秒双脚站立；5—8秒舒适地向右脚转移重心；8—10秒低抬左脚；10—13秒保持；13—15秒落脚；15—30秒双脚站立。不勉强平衡，不适立即落脚停止；全身和双脚始终在两路画面内。'),
    'R1-10': (30,'右脚低抬诊断','0—5秒双脚站立；5—8秒舒适地向左脚转移重心；8—10秒低抬右脚；10—13秒保持；13—15秒落脚；15—30秒双脚站立。不勉强平衡，不适立即落脚停止；全身和双脚始终在两路画面内。'),
}


class Settings(QDialog):
    def __init__(self,config,parent):
        super().__init__(parent);self.setWindowTitle('设备与会话配置');self.resize(700,710);self.config=dict(config);self.fields={}
        outer=QVBoxLayout(self);scroll=QScrollArea();scroll.setWidgetResizable(True);body=QWidget();form=QFormLayout(body)
        names={'host':'板端 IP','ssh_port':'SSH 端口','user':'SSH 用户','key_file':'SSH 私钥文件','board_port':'实时骨架端口',
               'remote_root':'板端工作目录','model_dir':'板端二维模型目录','seconds':'录制秒数','max_mib':'录制上限 MiB',
               'left_bus':'左相机 USB bus','right_bus':'右相机 USB bus','pair_limit_us':'配对容差 μs',
               'left_roi':'左 ROI [x,y,w,h]','right_roi':'右 ROI [x,y,w,h]','calibration':'标定 JSON',
               'checkpoint':'三维模型 checkpoint','binding':'相机绑定 JSON','board_runtime_record':'本轮板端资格记录 JSON','g1_dir':'G1 便携运行目录','g1_url':'G1 服务 WebSocket','overview_camera':'第三台相机编号'}
        for key,title in names.items():
            value=config[key];entry=QLineEdit(json.dumps(value) if isinstance(value,list) else str(value));self.fields[key]=entry
            if key in ('key_file','calibration','checkpoint','binding','board_runtime_record','g1_dir'):
                row=QHBoxLayout();row.addWidget(entry);button=QPushButton('浏览');button.clicked.connect(lambda checked=False,k=key:self.browse(k));row.addWidget(button);form.addRow(title,row)
            else:form.addRow(title,entry)
        hint=label('机位变化后请重新标定，并选择与标定匹配的三维模型。\n板端应已加载匹配的 ABI9 位流和驱动；SSH 使用已有密钥。',12,'muted');form.addRow(hint)
        scroll.setWidget(body);outer.addWidget(scroll);buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save);buttons.rejected.connect(self.reject);outer.addWidget(buttons)
    def browse(self,key):
        path=QFileDialog.getExistingDirectory(self,'选择 G1 目录',str(HOME)) if key=='g1_dir' else QFileDialog.getOpenFileName(self,'选择文件',str(ROOT))[0]
        if path:self.fields[key].setText(path)
    def save(self):
        try:
            for key,entry in self.fields.items():
                value=entry.text().strip();old=self.config[key]
                if isinstance(old,int):value=int(value)
                elif isinstance(old,list):
                    value=json.loads(value)
                    if len(value)!=4 or any(type(x)!=int or x<0 for x in value) or min(value[2:])<=0:raise ValueError('ROI 格式错误')
                self.config[key]=value
            if not 1<=self.config['seconds']<=300:raise ValueError('录制时长为 1–300 秒')
            if not 1<=self.config['pair_limit_us']<=20000:raise ValueError('配对容差应为 1–20000 μs')
            for key in ('ssh_port','board_port'):
                if not 1<=self.config[key]<=65535:raise ValueError('端口范围错误')
            write_json(HOME/'config.json',self.config);self.accept()
        except Exception as e:QMessageBox.warning(self,'配置未保存',str(e))


class Station(QMainWindow):
    live_snapshot=Signal(object)
    def __init__(self):
        super().__init__();self.config=load_config();self.session=None;self.timeline=Timeline();self.task=None;self.overview=None;self.live_group=None
        self.timing_average=StageTimingAverage();self.active_motion_mode='full_body';self.active_standing_small_motion_check=False
        self.preparation_pose=None;self.preparation_displayed=threading.Event();self.preparation_stage='正在检查本组资源'
        self.input_displayed=threading.Event();self.first_input_display=None;self.waiting_for_input=False
        self.full_body_check=False;self.hybrid_refinement=False;self.active_hybrid_refinement=False;self.active_full_body_check=False;self.physics_displayed=threading.Event();self.first_physics_display=None;self.last_physics_display=None;self.live_run_started_ns=0;self.live_run_path=None;self.live_robot_received_ns=0;self.input_gate_evidence=None
        self.live=False;self.live_frames={};self.live_seen={};self.live_started=None;self.live_seconds=0;self.live_tracking={};self.display_log=None;self.display_lock=threading.Lock();self.live_log_error=None;self.g1=None;self.generation=0;self._last_media=None;self._closing=False
        self.setWindowTitle('Motion Studio · FPGA 双目动捕演示');self.resize(1580,940);self.setMinimumSize(1180,820)
        root=QWidget();self.setCentralWidget(root);layout=QHBoxLayout(root);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        rail=QFrame();rail.setObjectName('sidebar');rail.setFixedWidth(215);side=QVBoxLayout(rail);side.setContentsMargins(18,25,18,18);side.setSpacing(12)
        side.addWidget(label('Motion\nStudio',26,'text',True));side.addWidget(label('双目动捕 / FPGA / G1',11,'muted'));side.addSpacing(18)
        self.mode=QComboBox();self.mode.addItems(['历史会话回放','实时动捕']);self.mode.currentIndexChanged.connect(self.change_mode);side.addWidget(self.mode)
        self.motion_mode=QComboBox();self.motion_mode.addItem('全身跟随','full_body');self.motion_mode.addItem('站立上肢 · 下肢固定','standing_upper_body');side.addWidget(self.motion_mode)
        self.motion_mode.setToolTip('站立上肢仅使用可靠手臂观测；根部和下肢采用固定站姿参考，不代表全身跟踪通过。')
        side.addWidget(label('会话记录',13,'text',True));self.library=QListWidget();self.library.setWordWrap(True);self.library.itemActivated.connect(self.select_item);self.library.itemClicked.connect(self.select_item);side.addWidget(self.library,1)
        self.import_btn=self.button('打开会话文件夹',self.open_folder);side.addWidget(self.import_btn)
        self.video_btn=self.button('导入双目视频',self.import_video);side.addWidget(self.video_btn)
        self.settings_btn=self.button('设备与标定设置',self.settings);side.addWidget(self.settings_btn)
        self.preflight_btn=self.button('本机准备检查',self.local_check);side.addWidget(self.preflight_btn)
        self.install_btn=self.button('部署板端录制工具',self.install);side.addWidget(self.install_btn)
        self.simulate_btn=self.button('生成三维与仿真结果',self.simulate);side.addWidget(self.simulate_btn)
        side.addWidget(label('实验候选版本\n数据缺失时保留空缺',10,'muted'));layout.addWidget(rail)
        main=QVBoxLayout();main.setContentsMargins(24,21,24,16);main.setSpacing(14);layout.addLayout(main,1)
        header=QHBoxLayout();titles=QVBoxLayout();titles.addWidget(label('从人体动作到机器人',26,'text',True));self.subtitle=label('选择历史会话，或连接板端开始实时演示',12,'muted');titles.addWidget(self.subtitle);header.addLayout(titles,1)
        self.full_btn=self.button('全屏演示  F11',self.fullscreen);header.addWidget(self.full_btn);main.addLayout(header)
        self.theme_choice=QComboBox();self.theme_choice.addItems(['跟随系统','浅色','深色']);self.theme_choice.setCurrentIndex(['system','light','dark'].index(self.config.get('theme','system')));header.insertWidget(1,self.theme_choice)
        self.theme_choice.currentIndexChanged.connect(self.theme_changed)
        toolbar=QHBoxLayout();self.status=label('● 离线回放',12,'cyan',True);self.status.setWordWrap(True);toolbar.addWidget(self.status,1)
        self.camera_btn=self.button('开启现场画面',self.toggle_camera);toolbar.addWidget(self.camera_btn)
        self.live_btn=self.button('连接实时动捕',self.start_live);self.live_btn.setObjectName('primary');toolbar.addWidget(self.live_btn)
        self.record_btn=self.button('录制双目',self.capture);toolbar.addWidget(self.record_btn)
        self.compare_btn=self.button('同板 CPU / FPGA 对比',self.compare);toolbar.addWidget(self.compare_btn)
        self.cancel_btn=self.button('停止任务',self.cancel);self.cancel_btn.setObjectName('danger');self.cancel_btn.setEnabled(False);toolbar.addWidget(self.cancel_btn);main.addLayout(toolbar)
        scene=QHBoxLayout();scene.setSpacing(0)
        self.source=ImageCanvas();self.source.empty='现场画面使用第三台相机\n回放时显示双目原始图像'
        self.pose=ImageCanvas(stereo=True,overlay=True);self.pose.empty='等待二维关键点'
        self.human=SceneCanvas();self.robot=RobotView();self.robot.problem.connect(self.error)
        self.robot.presented.connect(self.on_robot_presented)
        self.source_card=Card('','原始画面','现场参考 / 回放原图',self.source)
        self.pose_card=Card('1','双目二维骨架','左 / 右相机 · COCO17',self.pose)
        self.human_card=Card('2','三维重建','空间坐标 · 米',self.human)
        self.robot_card=Card('3','Unitree G1','保存的仿真状态 · 可拖动回放',self.robot)
        self.robot_card.footer.setWordWrap(True)
        scene.addWidget(self.source_card,25);self.arrow_2d=self.arrow(scene,'双目识别');scene.addWidget(self.pose_card,28)
        self.arrow_3d=self.arrow(scene,'三维重建');scene.addWidget(self.human_card,28)
        self.arrow_g1=self.arrow(scene,'仿真处理');scene.addWidget(self.robot_card,36);main.addLayout(scene,1)
        self.infer_label=label('双目识别 —     板端采集→发布 —     端到端 —（缺少同步测量）',11,'muted');main.addWidget(self.infer_label)
        lower=QHBoxLayout();box=QFrame();box.setObjectName('metric');v=QVBoxLayout(box);v.setContentsMargins(16,12,16,12)
        row=QHBoxLayout();row.addWidget(label('同板计算对比',15,'text',True));row.addStretch();row.addWidget(label('同一录制输入 / 实测',10,'muted'));v.addLayout(row)
        self.plot=ComparisonPlot();v.addWidget(self.plot);lower.addWidget(box,3)
        notes=QFrame();notes.setObjectName('metric');v=QVBoxLayout(notes);v.setContentsMargins(16,12,16,12);v.addWidget(label('会话与运行状态',15,'text',True))
        self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(200);self.log.setMaximumHeight(105);v.addWidget(self.log);lower.addWidget(notes,2);main.addLayout(lower)
        transport=QFrame();transport.setObjectName('transport');v=QVBoxLayout(transport);v.setContentsMargins(16,12,16,12)
        self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,10000);self.slider.sliderPressed.connect(self.pause);self.slider.valueChanged.connect(self.seek_slider);v.addWidget(self.slider)
        row=QHBoxLayout();self.play=self.button('播放',self.toggle_play);self.play.setObjectName('primary');row.addWidget(self.play)
        row.addWidget(self.button('上一帧',lambda:self.step(-1)));row.addWidget(self.button('下一帧',lambda:self.step(1)))
        self.clock=label('00:00.000 / 00:00.000',12,'text');row.addWidget(self.clock);row.addStretch()
        row.addWidget(label('回放速度',11,'muted'));self.speed=QComboBox();self.speed.addItems(['0.25×','0.5×','1×','1.5×','2×']);self.speed.setCurrentIndex(2);self.speed.currentIndexChanged.connect(lambda i:setattr(self.timeline,'speed',[.25,.5,1,1.5,2][i]));row.addWidget(self.speed)
        self.backend=QComboBox();self.backend.addItems(['FPGA 结果','CPU 结果']);self.backend.currentIndexChanged.connect(self.refresh);row.addWidget(self.backend)
        row.addWidget(self.button('视角复位',self.reset_views));v.addLayout(row);main.addWidget(transport)
        self.media=MediaReader(self);self.media.ready.connect(self.media_ready);self.media.failed.connect(self.error);self.media.start()
        self.live_snapshot.connect(self.on_live_snapshot);self.timer=QTimer(self);self.timer.setTimerType(Qt.PreciseTimer);self.timer.timeout.connect(self.refresh);self.timer.start(33)
        for keys,callback in [('Space',self.toggle_play),('Left',lambda:self.step(-1)),('Right',lambda:self.step(1)),('F11',self.fullscreen),('Escape',self.showNormal)]:
            shortcut=QShortcut(QKeySequence(keys),self);shortcut.activated.connect(callback)
        for card in (self.source_card,self.pose_card,self.human_card,self.robot_card):
            card.expand.clicked.connect(lambda checked=False,c=card:self.expand(c))
        self.scan();self.change_mode(0)
        self.robot.start_renderer(self.config.get('g1_dir') or HOME/'data/g1/G1_Windows_AMD_CPU')
        QApplication.instance().styleHints().colorSchemeChanged.connect(self.system_theme_changed)

    def theme_changed(self,index):
        self.config['theme']=['system','light','dark'][index];write_json(HOME/'config.json',self.config);apply_theme(QApplication.instance(),self.config['theme'])
    def system_theme_changed(self,*args):
        if self.config.get('theme','system')=='system':apply_theme(QApplication.instance(),'system')

    def button(self,text,slot):
        b=QPushButton(text);b.clicked.connect(slot);return b
    def arrow(self,layout,title):
        widget=QWidget();widget.setFixedWidth(68);v=QVBoxLayout(widget);v.setContentsMargins(1,0,1,0);v.addStretch();v.addWidget(label(title,10,'muted'),0,Qt.AlignCenter)
        timing=label('—',11,'cyan',True);v.addWidget(timing,0,Qt.AlignCenter)
        v.addWidget(label('2秒均值',9,'muted'),0,Qt.AlignCenter)
        timing.setToolTip('平均耗时，每2秒更新一次；只统计界面采样到的不同有效记录。暂停保持，跳转后重新统计。')
        v.addWidget(label('⟶',28,'cyan'),0,Qt.AlignCenter);v.addStretch();layout.addWidget(widget);return timing
    def scan(self):
        self.library.clear();paths=[]
        local=HOME/'data/sessions'
        extra=self.config.get('session_root')
        if extra and Path(extra).resolve()!=local.resolve() and Path(extra).exists():paths+=sorted((p for p in Path(extra).iterdir() if (p/'session.json').is_file()),reverse=True)
        if local.exists():paths+=sorted(local.iterdir(),reverse=True)
        if RUNS.exists():paths+=sorted(RUNS.glob('live_*'))
        for path in paths:
            if not path.is_dir():continue
            name=read_json(path/'session.json').get('name',path.name) if (path/'session.json').exists() else '归档 '+path.name[5:13]
            item=QListWidgetItem(name+'\n'+path.name[:19]);item.setData(Qt.UserRole,str(path));item.setSizeHint(item.sizeHint().expandedTo(__import__('PySide6.QtCore',fromlist=['QSize']).QSize(160,54)));self.library.addItem(item)
    def select_item(self,item):
        if not self.live:self.load(item.data(Qt.UserRole))
    def open_folder(self):
        path=QFileDialog.getExistingDirectory(self,'打开含 session.json 或归档日志的会话',str(HOME/'data'))
        if path:self.load(path)
    def import_video(self):
        dialog=QDialog(self);dialog.setWindowTitle('导入带采集时间戳的双目视频');dialog.resize(670,320);outer=QVBoxLayout(dialog);form=QFormLayout();entries={}
        for key,title in [('left','左视频'),('right','右视频'),('left_timestamps','左时间戳 CSV'),('right_timestamps','右时间戳 CSV')]:
            row=QHBoxLayout();entry=QLineEdit();entries[key]=entry;row.addWidget(entry);button=QPushButton('浏览')
            def pick(checked=False,k=key):
                filename=QFileDialog.getOpenFileName(dialog,'选择文件',str(HOME))[0]
                if filename:entries[k].setText(filename)
            button.clicked.connect(pick);row.addWidget(button);form.addRow(title,row)
        outer.addLayout(form);outer.addWidget(label('视频须为原始 1920×1080；CSV 列为 frame_index,timestamp_us。\n左右时间戳必须来自同一采集时钟，不能用两个视频各自的起点代替。\n标定、模型、相机绑定将从“设备与标定设置”复制到新会话。',11,'muted'))
        buttons=QDialogButtonBox(QDialogButtonBox.Open|QDialogButtonBox.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);outer.addWidget(buttons)
        if dialog.exec():
            files={k:v.text().strip() for k,v in entries.items()}
            if not all(Path(p).is_file() for p in files.values()):self.error('请选择两路视频和两份时间戳 CSV');return
            self.job(lambda p,c:import_videos(files,self.config,p,c),self.load)
    def load(self,path):
        try:
            session=Session(path)
            if not session.duration:raise ValueError('该目录没有可回放的时间戳数据')
            self.live=False;self.preparation_pose=None;self.mode.setCurrentIndex(0);self.session=session;self.timeline.playing=False;self.timeline.duration=session.duration;self.timeline.seek(0);self._last_media=None;self.generation+=1
            self.subtitle.setText(session.name);self.plot.data=session.comparison;self.plot.update();self.status.setText('● 历史回放 · 按动作时间对齐')
            self.live_tracking={}
            self.active_motion_mode=session.manifest.get('motion_mode','full_body')
            self.active_standing_small_motion_check=session.manifest.get('standing_small_motion_check',False)
            if session.manifest.get('state')=='failed':
                tracking=session.path/'tracking_status.json'
                self.live_tracking=read_json(tracking) if tracking.exists() else {}
                self.live_tracking.update(state='failed',error=session.manifest.get('error','本组未产生仿真状态'))
                self.status.setText('● 历史回放 · '+self.g1_state_text(bool(session.robot.values)))
            self.source.images={};self.pose.images={};self.append(f'载入 {session.duration:.2f} 秒：二维 {len(session.paired.values)} / 三维 {len(session.skeleton.values)} / G1 {len(session.robot.values)}')
            if session.manifest.get('notes'):self.append(session.manifest['notes'])
            for issue in session.issues:self.append(str(issue))
            for name,ok,p in session.verify_assets():
                if not ok:self.append('关联文件校验失败：'+name)
            self.refresh()
        except Exception as e:self.error(str(e))
    def append(self,text):self.log.appendPlainText(time.strftime('%H:%M:%S')+'  '+str(text))
    def error(self,text):
        if self.live and not self.live_frames.get('robot'):
            self.live_tracking.update(state='failed',error=text)
            self.status.setText('● '+self.g1_state_text(False))
        else:self.status.setText('● 操作未完成')
        self.status.setToolTip(text);self.append(text)
    def settings(self):
        dialog=Settings(self.config,self)
        if dialog.exec():
            old_root=self.config.get('g1_dir');self.config=dialog.config;self.append('配置已保存至本项目')
            if self.config.get('g1_dir')!=old_root:self.robot.start_renderer(self.config.get('g1_dir') or HOME/'data/g1/G1_Windows_AMD_CPU')
    def change_mode(self,index):
        if self.live and index==0:self.cancel()
        if index==0 and self.overview and self.overview.isRunning():self.overview.stop.set();self.camera_btn.setText('开启现场画面')
        self.live_btn.setVisible(index==1);self.camera_btn.setVisible(index==1);self.slider.setEnabled(index==0);self.play.setEnabled(index==0)
        self.pose.overlay=index==0;self.refresh()
        self.arrow_2d.parentWidget().layout().itemAt(1).widget().setText('板端双目' if index else '双目识别')
        self.source.caption='第三台相机 / 现场参考' if index else '双目原始视频'
        self.source.empty='第三台相机独立采集现场画面\n识别使用板端双目输入' if index else '此会话未附原始视频'
    def busy(self,value):
        for b in (self.live_btn,self.record_btn,self.compare_btn,self.install_btn,self.simulate_btn,self.settings_btn,self.preflight_btn,self.import_btn,self.video_btn,self.library,self.mode,self.motion_mode):b.setEnabled(not value)
        self.cancel_btn.setEnabled(value)
    def job(self,fn,on_result):
        if self.task and self.task.isRunning():return
        self.busy(True);self.task=Task(fn,self);self.task.progress.connect(self.append);self.task.failed.connect(self.error);self.task.result.connect(on_result);self.task.finished.connect(self.job_finished);self.task.start()
    def job_finished(self):
        self.live=False;self.g1=None;self.busy(False);self.scan()
        if self._closing:self.close()
    def install(self):self.job(lambda p,c:BoardClient(self.config,p,c).install(),self.append)
    def local_check(self):
        def work(progress,stop):
            from .pipeline import local_preflight
            report=local_preflight(self.config);target=HOME/'output/local_preflight.json';write_json(target,report)
            progress('本机检查报告：'+str(target))
            return report
        def done(report):
            self.append('本机资源检查通过；仍需现场核对机位、板端身份和服务。' if report.get('ok') else '本机资源未就绪：'+str(report.get('errors',report)))
        self.job(work,done)
    def capture_request(self,kind):
        dialog=QDialog(self);dialog.setWindowTitle(kind+' · 本组动作确认');dialog.resize(650,440)
        preset=ROUND1_LIVE_GROUPS.get(self.live_group) if kind=='实时动捕' else None
        upper=kind=='实时动捕' and self.motion_mode.currentData()=='standing_upper_body'
        full_check=kind=='实时动捕' and self.full_body_check and self.motion_mode.currentData()=='full_body'
        hybrid=full_check and self.hybrid_refinement
        if upper:
            selected=motion_config(self.config,'standing_upper_body');roi=selected['left_roi']
            preset=(30,'站立上肢测试',f'本组上肢动作框 {roi[2]}×{roi[3]}。保持站立、不迈步。先静止3秒；左臂缓慢抬至约45度、保持、放下；右臂同样动作；最后静止。双肩、肘、手腕始终留在两路画面动作框内。根部和下肢采用固定站姿参考。')
        if full_check:
            selected_config={**self.config,'full_body_check':True}
            if hybrid:selected_config['full_body_check_binding']=str(HOME/'profiles/full_body_hybrid_roi810_20261028_binding.json')
            selected=motion_config(selected_config,'full_body');roi=selected['left_roi']
            preset=(30,'FPGA 引导 / 电脑精修短测' if hybrid else '全身站立短测',f'本组动作框 {roi[2]}×{roi[3]}。确保手腕、脚踝位置轮廓清楚，双脚稍分开，全身留在两路框内。初始化时双脚站稳；看到“现在开始本组动作”后：0—5秒静止，5—15秒双臂小幅交替抬放，15—25秒轻微屈膝、左右重心移动，25—30秒静止。不迈步、不抬脚；模型准备画面不代表人体跟踪已启动。')
        if full_check and self.config.get('standing_small_motion_check'):
            preset=(preset[0],'站立小动作协同候选',preset[2]+' 双脚保持在开始位置，屈膝时髋部只下降约5厘米，身体基本直立。若出现范围暂停，请回到原站姿等待恢复；下肢仍实时跟随。')
        outer=QVBoxLayout(dialog);form=QFormLayout();name=QLineEdit(('HF-01 ' if hybrid else 'FB-01 ' if full_check else 'UB-01 ' if upper else (self.live_group or '')+' ')+preset[1] if preset else '普通动作短测');seconds=QSpinBox();seconds.setRange(1,300);seconds.setValue(preset[0] if preset else int(self.config['seconds']))
        actions=QPlainTextEdit(preset[2] if preset else '先自然站立7秒；缓慢抬臂、保持、放下；最后回站保持。全身和双脚始终在两路画面内。')
        actions.setMaximumHeight(100);form.addRow('本组名称',name);form.addRow('采集秒数',seconds);form.addRow('本组动作（按计划修改）',actions);outer.addLayout(form)
        notice=('本组保存同帧双目原图、FPGA 骨架、电脑精修结果和处理日志。\nFPGA 负责初定位，电脑计算最终精细二维点。' if hybrid else '将保存本组原始测量、时间戳及处理日志。实时模式默认只收骨架，\n不等同于保存双目原图。')
        outer.addWidget(label(notice+'资源检查通过后倒数3秒。\n可随时点击“停止任务”；不会控制实体机器人。',12,'muted'))
        confirm=QLineEdit();confirm.setPlaceholderText('准备好后输入 START');outer.addWidget(confirm)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.button(QDialogButtonBox.Ok).setText('确认本组并准备')
        buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        def valid():buttons.button(QDialogButtonBox.Ok).setEnabled(confirm.text().strip()=='START' and bool(name.text().strip()) and bool(actions.toPlainText().strip()))
        confirm.textChanged.connect(valid);name.textChanged.connect(valid);actions.textChanged.connect(valid)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);outer.addWidget(buttons)
        if dialog.exec():return dict(name=name.text().strip(),actions=actions.toPlainText().strip(),seconds=seconds.value())
        return None
    @staticmethod
    def countdown(progress,stop):
        for remaining in range(3,0,-1):
            if stop.is_set():raise RuntimeError('采集开始前已取消')
            progress(f'采集倒数 {remaining} 秒')
            if stop.wait(1):raise RuntimeError('采集开始前已取消')
        progress('开始本组采集')
    def capture(self):
        protocol=self.capture_request('录制双目')
        if protocol is None:return
        config={**self.config,'seconds':protocol['seconds'],'capture_protocol':protocol}
        def work(progress,stop):
            client=BoardClient(config,progress,stop);client.probe();self.countdown(progress,stop);return client.capture()
        self.pause();self.job(work,self.load)
    def compare(self):
        if not self.session:self.error('先选择一个包含双目原图的录制会话');return
        if not (self.session.path/'pairs.json').exists():self.error('该会话缺少可比较的双目原图；先录制双目或导入带时间戳的视频');return
        path=self.session.path;self.pause();self.job(lambda p,c:BoardClient(self.config,p,c).compare(path),self.load)
    def simulate(self):
        if not self.session:self.error('先选择录制会话');return
        path=self.session.path
        streams=self.session.manifest.get('streams',{})
        if 'robot' in streams or 'robot_report' in streams:
            missing=[str(self.session.resolve(streams[key])) for key in ('robot','robot_report') if key in streams and not self.session.resolve(streams[key]).is_file()]
            self.error('已保存结果引用失效：'+'；'.join(missing) if missing else '已有仿真结果，不覆盖；可直接拖动回放');return
        if (path/'g1').exists() or self.session.stream_path('robot','g1/poses.ndjson').exists():self.error('已有仿真结果，不覆盖；可直接拖动回放');return
        self.pause();self.job(lambda p,c:simulate_recording(self.config,path,p,c),self.load)
    def cancel(self):
        if self.task:self.task.cancel.set();self.append('正在停止当前任务并保留已写入数据…')
    def start_live(self):
        protocol=self.capture_request('实时动捕')
        if protocol is None:return
        config={**self.config,'seconds':protocol['seconds'],'capture_protocol':protocol}
        self.active_motion_mode=self.motion_mode.currentData();config['motion_mode']=self.active_motion_mode
        self.active_standing_small_motion_check=bool(config.get('standing_small_motion_check'))
        config['full_body_check']=bool(self.full_body_check and self.active_motion_mode=='full_body');self.active_full_body_check=config['full_body_check']
        config['hybrid_refinement']=bool(self.hybrid_refinement and config['full_body_check']);self.active_hybrid_refinement=config['hybrid_refinement']
        if config['hybrid_refinement']:
            config['full_body_check_binding']=str(HOME/'profiles/full_body_hybrid_roi810_20261028_binding.json')
            protocol.update(hybrid_refinement=True)
        config=motion_config(config,self.active_motion_mode)
        protocol['motion_mode']=self.active_motion_mode
        if self.active_full_body_check:protocol['full_body_check']=True
        self.preparation_pose=None;self.preparation_displayed.clear();self.preparation_stage='正在检查本组资源'
        self.input_displayed.clear();self.first_input_display=None;self.waiting_for_input=False
        self.physics_displayed.clear();self.first_physics_display=self.last_physics_display=self.input_gate_evidence=None;self.live_run_started_ns=time.perf_counter_ns();self.live_run_path=None;self.live_robot_received_ns=0
        self.live=True;self.session=None;self.live_frames={};self.live_seen={};self.live_tracking={};self.live_started=None;self.live_log_error=None;self.live_seconds=protocol['seconds'];self.timeline.playing=False;self.timeline.seek(0);self.source.images={};self.pose.images={};self.plot.data=None;self.plot.update();self.generation+=1
        self.subtitle.setText(protocol['name']);self.status.setText('● 正在检查本组资源')
        def work(progress,stop):
            path=new_session(protocol['name'],root=config.get('session_root'));state=read_json(path/'session.json');state.update(capture_protocol=protocol,source='live_board',hardware_accepted=False,motion_mode=config['motion_mode'])
            if config['motion_mode']=='standing_upper_body':state['motion_scope']='真实上肢观测驱动；根部与下肢为固定站姿参考，未验收全身跟踪'
            if config['full_body_check']:state.update(full_body_check=True,motion_scope='原全身初始化与控制链路；补点仍标为估计，等待真人验收')
            if config.get('standing_small_motion_check'):state.update(standing_small_motion_check=True,motion_scope='双脚站立小动作候选；下肢继续跟随，超出范围保护暂停；尚未验收通过')
            state['hybrid_hip_flip_check']=bool(config.get('hybrid_hip_flip_check'))
            state['hybrid_latency_check']=bool(config.get('hybrid_latency_check'))
            state['hybrid_reprojection_limit_px']=config.get('hybrid_reprojection_limit_px',5.)
            if config['hybrid_refinement']:state.update(hybrid_refinement=True,motion_scope='FPGA 计算初始骨架并定位人体；电脑 RTMPose 计算最终二维点；原三维与 G1 保护保持，等待真人验收')
            g1=client=None
            try:
                state['assets']=snapshot_assets(path,config);write_json(path/'session.json',state)
                self.live_snapshot.emit(dict(preparation_stage='正在加载本机模型与控制服务，请先保持原位'))
                g1=G1Runner(config,path,progress);g1.start(stop);self.g1=g1
                self.live_snapshot.emit(dict(preparation_stage='本机服务已启动，正在核对板端；尚未开始动作'))
                client=BoardClient(config,progress,stop);preparation=client.prepare_live_service(path)
                if not preparation.get('ok'):raise RuntimeError('板端准备未通过：'+'；'.join(preparation.get('errors',[])))
                with self.display_lock:self.display_log=(path/'ui_display.ndjson').open('x',encoding='utf8',buffering=1)
                if config['motion_mode']=='standing_upper_body':self.wait_for_preparation(path,progress,stop)
                elif config['full_body_check']:self.wait_for_preparation(path,progress,stop,full_body_check=True)
                self.live_snapshot.emit(dict(preparation_stage='模型准备画面已显示，倒数 3 秒后启动相机；请继续站稳等待全身初始化' if config['full_body_check'] else '准备画面已显示，即将倒数 3 秒；随后开始本组动作' if config['motion_mode']=='standing_upper_body' else '本机资源已核对，倒计时后等待人体观测'))
                self.countdown(progress,stop)
                proof=client.start_live_service(path,protocol['seconds']);write_json(path/'board_preflight.json',proof)
                if not proof.get('ok'):raise RuntimeError('本组板端服务身份未通过检查')
                g1.start_input(binding=proof['binding'],preflight=proof)
                if config['motion_mode']=='standing_upper_body':self.wait_for_live_input(g1,progress,stop)
                elif config['full_body_check']:self.wait_for_live_input(g1,progress,stop,full_body_check=True)
                started=time.monotonic();state['action_start']=dict(monotonic_s=started,pc_ns=time.perf_counter_ns(),first_gui_submission=self.first_input_display)
                if config['full_body_check']:state['action_start'].update(first_physics_display=self.first_physics_display,input_gate=self.input_gate_evidence)
                write_json(path/'session.json',state);self.live_snapshot.emit(dict(started=started,state='running'))
                progress('现在开始本组动作；动作计时已启动。')
                next_poll=time.monotonic()
                while not stop.wait(max(.001,next_poll-time.monotonic())) and time.monotonic()-started<protocol['seconds']:
                    next_poll=time.monotonic()+.02
                    snapshot=g1.poll();self.live_snapshot.emit(snapshot)
                    if snapshot.get('error'):raise RuntimeError(snapshot['error'])
                    if snapshot.get('state')=='failed':raise RuntimeError('实时处理失败，检查本组日志')
                    if snapshot.get('state') in ('stopped','finished'):break
                progress('结束输入，正在保存并收完仿真尾段…');g1.finish();final=g1.poll();self.live_snapshot.emit(final)
                if final.get('error') or final.get('state')=='failed':raise RuntimeError(final.get('error') or '本组收尾失败')
                if self.live_log_error:raise RuntimeError('显示记录保存失败：'+self.live_log_error)
                state.update(state='stopped' if stop.is_set() else 'complete')
            except Exception as error:
                state.update(state='failed',error=str(error));raise
            finally:
                try:
                    try:
                        try:
                            if g1:g1.close()
                        finally:
                            with self.display_lock:
                                try:
                                    if self.display_log:self.display_log.close()
                                finally:self.display_log=None
                            if self.live_log_error:raise RuntimeError('显示记录保存失败：'+self.live_log_error)
                    finally:
                        if client:
                            cleanup=client.stop_live_service()
                            if cleanup.get('error') or cleanup.get('artifact_error') or (cleanup.get('remote_job') and not cleanup.get('stopped')):
                                raise RuntimeError('本组板端停止未确认，保留日志：'+str(cleanup))
                except Exception as error:
                    state.update(state='failed',cleanup_error=str(error));raise
                finally:
                    write_json(path/'session.json',state);progress('本组数据目录：'+str(path))
            return str(path)
        self.job(work,self.load)
    def wait_for_preparation(self,path,progress,stop,full_body_check=False):
        ready=read_json(path/('full_body_ready.json' if full_body_check else 'upper_body_ready.json'))
        pose=read_json(path/('full_body_preparation_pose.json' if full_body_check else 'g1/preparation_pose.json'))
        verified=(ready.get('model_verified') is True and pose.get('model_verified') is True
            and pose.get('preparation_kind')=='official_model_initial_pose' and pose.get('physics') is False
            and ready.get('preparation_steps')==0 and ready.get('session_output')==str(path.resolve())
            and pose.get('session_output')==str(path.resolve())) if full_body_check else ready.get('preparation_steps',0)>=25 and pose.get('physics') is True
        if not (ready.get('ready') is True and verified
                and ready.get('human_observation_count')==0 and ready.get('session')
                and pose.get('session')==ready['session'] and pose.get('preparation') is True
                and pose.get('human_observed') is False):
            raise RuntimeError('本次站立准备证据不完整；不能开始动作')
        progress('本次官方模型已核对；等待模型初始姿态显示，尚未完成人体初始化…' if full_body_check else '本机站立准备已完成；等待准备画面实际显示（尚无人体观测）…')
        self.live_snapshot.emit(dict(preparation_pose=pose,run_path=str(path.resolve()),preparation_stage='正在绘制官方模型初始姿态；尚无人体观测' if full_body_check else '正在绘制本次站立准备姿态；尚无人体观测'))
        deadline=time.monotonic()+20
        while not self.preparation_displayed.wait(.05):
            if stop.is_set():raise RuntimeError('准备画面显示前已取消')
            if self.live_log_error:raise RuntimeError('显示记录保存失败：'+self.live_log_error)
            if time.monotonic()>=deadline:raise TimeoutError('准备画面尚未实际显示；本组未启动，请检查仿真窗口')
        if stop.is_set():raise RuntimeError('准备完成后已取消')
        if self.live_log_error:raise RuntimeError('显示记录保存失败：'+self.live_log_error)
        progress('本次模型初始姿态已显示；倒计时后请站稳等待全身初始化。' if full_body_check else '本次站立准备画面已显示；现在进入动作倒计时。')
    def wait_for_live_input(self,g1,progress,stop,full_body_check=False):
        message='全身初始化中，请双脚站稳保持静止；尚未开始动作' if full_body_check else '相机启动中，请保持站立；尚未开始动作'
        progress(message);self.live_snapshot.emit(dict(waiting_for_input=True,preparation_stage=message))
        budget=20 if full_body_check else 10;deadline=time.monotonic()+budget
        while not stop.is_set():
            snapshot=g1.poll()
            if snapshot.get('error') or snapshot.get('state') in ('failed','finished','stopped'):
                raise RuntimeError(snapshot.get('error') or '输入准备期间处理已结束，尚未开始动作')
            self.live_snapshot.emit({**snapshot,'preparation_stage':message})
            full_ready=True
            if full_body_check:
                now=time.perf_counter_ns();robot=snapshot.get('robot') or {};last=self.last_physics_display or {}
                full_ready=(snapshot.get('tracking',{}).get('state')=='tracking' and robot.get('physics') is True
                    and robot.get('preparation') is not True and self.physics_displayed.is_set()
                    and last.get('robot',{}).get('session')==robot.get('session')
                    and 0<=now-self.live_robot_received_ns<=250_000_000
                    and 0<=now-last.get('pc_paint_ns',0)<=250_000_000)
                if full_ready:self.input_gate_evidence=dict(tracking_state='tracking',checked_pc_ns=now,
                    latest_robot_session=robot.get('session'),latest_robot_seq=robot.get('seq'),
                    pc_ui_receive_ns=self.live_robot_received_ns,last_physics_display=last)
            if self.input_displayed.is_set() and full_ready:
                if self.live_log_error:raise RuntimeError('显示记录保存失败：'+self.live_log_error)
                return
            if time.monotonic()>=deadline:
                detail=self.full_body_initialization_text() if full_body_check else ''
                raise TimeoutError(f'未在 {budget} 秒内显示本次二维、三维'+('及全身实际仿真画面' if full_body_check else '观测')+'；本组动作未启动'+('。'+detail if detail else ''))
            stop.wait(.02)
        raise RuntimeError('首条人体观测显示前已取消')
    @staticmethod
    def live_packet_identity(packet):
        if packet is None:return None
        row=packet.get('message',packet);seq=row.get('pair_id',row.get('seq'))
        if seq is None:return None
        return (row.get('session_id',row.get('session')),row.get('boot_id',row.get('clock_id')),seq)
    def on_live_snapshot(self,snapshot):
        if not self.live:return
        if 'run_path' in snapshot:self.live_run_path=snapshot['run_path']
        if 'preparation_stage' in snapshot:self.preparation_stage=snapshot['preparation_stage']
        if 'preparation_pose' in snapshot:self.preparation_pose=snapshot['preparation_pose'];self.refresh()
        if snapshot.get('waiting_for_input'):self.waiting_for_input=True
        if 'started' in snapshot:self.live_started=snapshot['started'];self.waiting_for_input=False
        now=time.monotonic()
        for key in ('paired','skeleton','robot','refined'):
            value=snapshot.get(key)
            old=self.live_frames.get(key);identity=self.live_packet_identity(value)
            if (value is not None and value is not old
                    and (identity is None or identity!=self.live_packet_identity(old))):
                self.live_frames[key]=value;self.live_seen[key]=now
                if key=='robot':self.live_robot_received_ns=time.perf_counter_ns()
        input_keys=('paired','skeleton','refined') if self.active_hybrid_refinement else ('paired','skeleton')
        if self.waiting_for_input and not self.input_displayed.is_set() and all(key in self.live_frames and now-self.live_seen[key]<.6 for key in input_keys):
            self.refresh()
            if self.pose.cameras is not None and self.human.packet is not None:
                self.first_input_display=dict(pc_gui_submit_ns=time.perf_counter_ns(),
                    paired_seq=self.live_frames['paired'].get('seq',self.live_frames['paired'].get('message',{}).get('seq')),
                    paired_pair_id=self.live_frames['paired'].get('message',{}).get('pair_id'),
                    skeleton_seq=self.live_frames['skeleton'].get('seq'),skeleton_timestamp_ns=self.live_frames['skeleton'].get('timestamp_ns'))
                self.on_robot_presented(dict(event='live_input_submitted',**self.first_input_display))
                if not self.live_log_error:self.input_displayed.set()
        tracking=snapshot.get('tracking')
        if tracking and tracking!=self.live_tracking:self.live_tracking=tracking
        if snapshot.get('error'):self.live_tracking={**self.live_tracking,'state':'failed','error':snapshot['error']}
        status_text=('● '+(self.full_body_initialization_text(compact=True) or self.preparation_stage) if self.active_full_body_check and self.waiting_for_input and self.live_tracking.get('state')!='failed'
                            else '● '+self.preparation_stage if 'preparation_stage' in snapshot
                            else '● 实时采集中 · '+self.g1_state_text(bool(self.live_frames.get('robot'))))
        if self.status.text()!=status_text:self.status.setText(status_text)
        tooltip=str(self.live_tracking.get('reason') or self.live_tracking.get('error') or '')
        if self.status.toolTip()!=tooltip:self.status.setToolTip(tooltip)
    def full_body_initialization_text(self,compact=False):
        info=self.live_tracking.get('initialization') or {}
        if not info:return ''
        elapsed=f" · 已等 {float(info.get('elapsed_s') or 0):.1f} 秒"
        stage=info.get('stage');names={'left_ankle':'左脚踝','right_ankle':'右脚踝','left_knee':'左膝','right_knee':'右膝',
            'left_hip':'左髋','right_hip':'右髋','left_shoulder':'左肩','right_shoulder':'右肩',
            'left_elbow':'左肘','right_elbow':'右肘','left_wrist':'左手腕','right_wrist':'右手腕'}
        if stage=='waiting_fresh_reference':
            fresh=info.get('startup_freshness') or {}
            text=f"人体初始化已通过，等待新鲜首帧：{fresh.get('aligned_age_ms',0):.0f} ms / 预算 {fresh.get('budget_ms',info.get('startup_freshness_budget_ms',80)):g} ms"
        elif stage=='waiting_standing_reference':
            preparation=info.get('standing_preparation') or {}
            text=preparation.get('reason','等待可靠站立基准')
            if not preparation.get('missing_joints'):
                text+=f"：{preparation.get('accepted_frames',0)} 帧，{preparation.get('span_s',0):.1f}/1.4 秒"
        elif self.live_tracking.get('state') in ('paused','recovering'):
            text='全身控制暂缓，等待可靠输入恢复'
        elif stage=='waiting_ankle_anchors':
            anchors=info.get('ankle_anchors',{})
            detail='，'.join(names[j]+f" {min(a.get('observed_reliable_count',0),a.get('required_reliable_count',3))}/{a.get('required_reliable_count','?')}"
                for j,a in anchors.items() if j in ('left_ankle','right_ankle'))
            text='等待脚踝可靠观测'+('：'+detail if detail else '')+('' if compact else '；让两侧脚踝轮廓清楚、双脚分开')
        elif stage=='waiting_body_quality':
            missing=list(dict.fromkeys(names.get(b.get('joint'),b.get('joint') or '身体映射') for b in info.get('blockers',[])))
            bone_error=any(b.get('reason') in ('body_bone_proportion','body_bone_length_invalid','observed_body_bone_length',
                'estimated_body_bone_length','estimated_leg_length') for b in info.get('blockers',[]))
            text=('骨长观测不可信，等待可靠全身骨架' if bone_error else '等待可靠全身骨架')+('：'+'、'.join(missing) if missing else '')+'；请保持全身可见'
        elif stage=='collecting_body_bones':
            warm=info.get('warmup',{})
            text=f"建立人体骨长：{warm.get('accepted_frames',0)}/{warm.get('required_frames','?')} 帧，{float(warm.get('span_s') or 0):.1f}/{warm.get('required_span_s','?')} 秒；请站稳"
        elif stage=='initializing_gmr':text='人体骨长已就绪，正在初始化全身映射；请站稳'
        elif stage=='tracking':text='全身映射已就绪，等待本次真实仿真画面'
        else:return ''
        return text+elapsed+'；尚未开始动作'
    def g1_state_text(self,has_output):
        state=self.live_tracking.get('state','preparing');reason=str(self.live_tracking.get('reason') or self.live_tracking.get('error') or '')
        if self.live and self.active_full_body_check and self.waiting_for_input and state!='failed':
            detail=self.full_body_initialization_text()
            if detail:return ('模型准备画面 · ' if not has_output else '')+detail
        if self.live and self.active_full_body_check and self.preparation_pose and not has_output and state!='failed':
            return '官方模型初始姿态 · 等待全身人体初始化；尚无真人仿真输出'
        if self.active_motion_mode=='standing_upper_body':
            if not self.live and has_output:return ('本组未通过 · ' if state=='failed' else '')+'站立上肢记录 · 根部/下肢为固定参考'
            if self.preparation_pose and not has_output and state!='failed':return '本机站立准备姿态 · 尚无人体观测 · 下肢固定参考'
            title={'preparing':'准备仿真','tracking':'上肢跟随','holding':'短时保持','holding_standing':'保持站姿','paused':'保持姿态','recovering':'恢复观测','ended':'本组已结束','failed':'本组未通过'}.get(state,'等待手臂观测')
            if (self.live and has_output and state in ('tracking','holding','holding_standing','paused','recovering')
                    and time.monotonic()-self.live_seen.get('robot',float('-inf'))<.25):title='站立控制继续'
            arms=self.live_tracking.get('upper_body',{}).get('arms',{})
            labels={'tracking':'跟随','holding':'短时保持','paused':'无可靠观测'}
            detail='；'.join(name+labels.get(arms.get(side,{}).get('state'),'等待观测')+
                ('（肩向短时保持）' if arms.get(side,{}).get('shoulder_basis_cached') else '')
                for side,name in (('left','左臂'),('right','右臂'))) if arms else reason
            return '站立上肢 · '+title+' · 下肢固定参考'+('：'+detail if detail else '')
        reason={
            'ankle_anchor_not_initialized':'等待双脚踝的可靠观测',
            'estimated_leg_without_recent_observation':'腿部缺少近期可靠观测',
            'missing_body_joint':'身体关键点不完整',
            'estimated_leg_speed':'腿部估计跳变过大',
            'estimated_leg_length':'腿部估计长度异常',
            'waiting for reliable body':'等待可靠的全身骨架',
            'following current admitted input':'正在跟随当前动作',
            'source ended; diagnostics complete':'本组输入与诊断记录已结束',
            'source ended while paused; references not drained':'本组结束时仍在暂停，仿真未完成',
            'source ended; G1 never initialized because body quality was insufficient':'骨架质量不足，预热未完成，本组没有仿真输出',
        }.get(reason,reason)
        if not has_output:
            ready=(self.live_tracking.get('initialization') or {}).get('human_calibration_ready',False)
            title='G1 未启动 · 本组失败' if state in ('failed','ended') else ('G1 尚未启动 · 等待有效动作输入' if ready else 'G1 尚未启动 · 预热未完成')
            return title+'：'+(reason or '等待可靠的全身骨架')
        if reason.startswith('standing scope:'):
            scope=self.live_tracking.get('standing_scope') or {};details=[]
            for key,name,scale,unit in (
                    ('hip_drop_m','髋部下降',100,' cm'),('hip_rise_m','髋部上升',100,' cm'),
                    ('hip_horizontal_m','髋部水平移动',100,' cm'),('ankle_vertical_m','脚踝升降',100,' cm'),
                    ('ankle_horizontal_m','脚踝水平移动',100,' cm'),('torso_tilt_deg','躯干倾斜',1,'°')):
                if 'standing_scope_'+key not in reason:continue
                detail=name+'超限'
                measured=scope.get('measurements',{}).get(key);limit=scope.get('limits',{}).get(key)
                if scope.get('state')=='outside' and measured is not None and limit is not None:
                    detail+=f'（触发时估计 {measured*scale:.1f}{unit}，限制 {limit*scale:.1f}{unit}）'
                details.append(detail)
            reason=('；'.join(details)+'；请站直，双脚回到原位，等待自动恢复' if details else
                    '当前可靠观测不足；请站直并保持双脚原位，等待自动恢复')
        elif state=='recovering' and reason=='bounded transition to fresh reliable input':
            scope=self.live_tracking.get('standing_scope') or {}
            reason=('已回到允许范围，' if scope.get('state')=='within' else '')+'正在平滑恢复跟踪；请保持站稳'
        title={'preparing':'G1 正在准备','tracking':'G1 正在跟踪','paused':'G1 已暂停','recovering':'G1 正在恢复','ended':'G1 本组输入已结束','failed':'G1 本组失败'}.get(state,'G1 '+state)
        if state=='ended' and self.live_tracking.get('reference_drained') is True and self.live_tracking.get('reason')=='source ended; diagnostics complete':
            title='G1 本组正常结束';reason='本轮动作已处理完成'
        prefix='站立小动作候选 · ' if self.active_standing_small_motion_check else ''
        return prefix+title+('：'+reason if reason else '')
    def on_robot_presented(self,event):
        error=None
        with self.display_lock:
            if self.live and self.display_log:
                try:self.display_log.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
                except (OSError,ValueError) as problem:self.live_log_error=str(problem);error=str(problem)
        robot=event.get('robot',{})
        if (not error and self.live and self.preparation_pose and event.get('event')=='robot_image_presented'
                and robot.get('preparation') is True and robot.get('session')==self.preparation_pose.get('session')):
            self.preparation_displayed.set()
        context=event.get('source_context',{});current=self.live_frames.get('robot') or {}
        if (not error and self.live and self.active_full_body_check and self.waiting_for_input
                and event.get('event')=='robot_image_presented'
                and robot.get('physics') is True and robot.get('preparation') is not True
                and robot.get('session') and robot.get('session')==current.get('session')
                and context.get('run_started_ns')==self.live_run_started_ns
                and context.get('session_output')==self.live_run_path and self.live_run_path
                and context.get('pc_ui_receive_ns',0)>self.live_run_started_ns):
            self.last_physics_display=dict(pc_paint_ns=event['pc_paint_ns'],robot=dict(robot),
                source_context=dict(context),source=str(Path(self.live_run_path)/'g1/poses.ndjson'))
            if self.first_physics_display is None:self.first_physics_display=self.last_physics_display
            self.physics_displayed.set()
        if error:self.error(error);self.cancel()
    def toggle_camera(self):
        if self.overview and self.overview.isRunning():self.overview.stop.set();self.camera_btn.setText('开启现场画面');return
        self.overview=OverviewCamera(int(self.config['overview_camera']),self);self.overview.frame.connect(self.camera_frame);self.overview.failed.connect(self.error);self.overview.start();self.camera_btn.setText('关闭现场画面')
    def camera_frame(self,image):
        if self.mode.currentIndex()==1:self.source.images={'overview':image};self.source.update()
    def pause(self):self.timeline.playing=False
    def toggle_play(self):
        if self.live or not self.session:return
        if self.timeline.position>=self.timeline.duration:self.timeline.seek(0)
        self.timeline.playing=not self.timeline.playing;self.timeline.last=time.monotonic()
    def seek_slider(self,value):
        if not self.live:self.timeline.seek(self.timeline.duration*value/10000);self.refresh()
    def step(self,direction):
        if self.live or not self.session:return
        self.pause();times=sorted(set(self.session.paired.times+self.session.skeleton.times+self.session.robot.times+self.session.images['left'].times))
        if not times:return
        i=bisect.bisect_right(times,self.timeline.position+1e-8) if direction>0 else bisect.bisect_left(times,self.timeline.position-1e-8)-1
        self.timeline.seek(times[max(0,min(len(times)-1,i))]);self.refresh()
    @staticmethod
    def timestamp(seconds):
        ms=round(seconds*1000);return f'{ms//60000:02}:{ms//1000%60:02}.{ms%1000:03}'
    def refresh(self):
        stamp=self.timeline.tick();frame={'paired':None,'skeleton':None,'robot':None,'refined':None,'images':{}}
        if self.live:
            now=time.monotonic()
            for key,max_age in (('paired',.6),('skeleton',.6),('robot',.25),('refined',.6)):
                if now-self.live_seen.get(key,float('-inf'))<max_age:frame[key]=self.live_frames.get(key)
            if self.live_started and frame['paired'] is None:self.status.setText('● 等待当前骨架 · '+self.g1_state_text(bool(self.live_frames.get('robot'))))
        elif self.session:frame=self.session.frame(stamp,'cpu' if self.backend.currentIndex() else 'pl-stream')
        row=frame['paired'];benchmark=frame.get('benchmark');self.pose.cameras=row['message']['cameras'] if row else (benchmark['cameras'] if benchmark else None)
        refined=frame.get('refined');hybrid=self.active_hybrid_refinement if self.live else bool(self.session and self.session.manifest.get('hybrid_refinement'))
        backend_label='协同精修结果' if hybrid else 'FPGA 结果'
        if self.backend.itemText(0)!=backend_label:self.backend.setItemText(0,backend_label)
        if hybrid:self.pose.cameras=refined_display_cameras(refined) if refined else None
        self.human.packet=frame['skeleton'];self.robot.packet=frame['robot']
        if self.live and not self.live_frames.get('robot') and self.preparation_pose:self.robot.packet=self.preparation_pose
        robot_age=time.monotonic()-self.live_seen['robot'] if self.live and 'robot' in self.live_seen else None
        context=(id(self.session),self.live,self.mode.currentIndex(),self.timeline.revision);source_context=None
        if self.live and self.active_full_body_check:
            context+=(self.live_run_started_ns,self.live_run_path)
            source_context=dict(run_started_ns=self.live_run_started_ns,session_output=self.live_run_path,pc_ui_receive_ns=self.live_robot_received_ns)
        self.robot.request_frame(context,hold_last=self.live,source_age_s=robot_age,source_context=source_context)
        for canvas in (self.pose,self.human,self.robot):canvas.update()
        metrics=stage_metrics(frame,time.perf_counter_ns() if self.live else None)
        average=self.timing_average.update(frame,metrics,(id(self.session),self.live,self.mode.currentIndex(),self.backend.currentIndex(),self.timeline.revision),self.live or self.timeline.playing)
        self.arrow_2d.setText(milliseconds(average['inference']));self.arrow_3d.setText(milliseconds(average['reconstruction']));self.arrow_g1.setText(milliseconds(average['mapping']))
        latency='端到端 —（缺少同步测量）'
        if metrics['e2e_upper'] is not None:
            latency=('采集时间戳→二维界面提交 ' if self.live else '历史采集时间戳→PC接收 ')+f"{metrics['e2e_lower']:.1f}–{metrics['e2e_upper']:.1f} ms"
        self.infer_label.setText('双目识别 '+milliseconds(metrics['inference'])+'     板端采集→发布 '+milliseconds(metrics['board_age'])+'     '+latency)
        if hybrid:self.infer_label.setText('FPGA 初定位 '+milliseconds(metrics['inference'])+'     电脑精修 '+milliseconds((frame.get('skeleton') or {}).get('refinement_ms'))+'     板端采集→发布 '+milliseconds(metrics['board_age']))
        self.pose_card.footer.setText(('原图叠加关键点' if self.pose.images else '仅显示关键点（未附原图）') if self.pose.overlay else '仅传骨架，节省传输带宽')
        if hybrid:self.pose_card.footer.setText('FPGA 定位人体 → 电脑精修二维点' if refined else '等待本次电脑精修结果')
        self.human_card.footer.setText('三维与仿真基于 FPGA 结果' if benchmark else '虚线 = 模型补点；实线 = 几何观测')
        has_robot=bool(self.session.robot.values) if self.session else bool(self.live_frames.get('robot'))
        robot_status=self.g1_state_text(has_robot) if self.live or self.live_tracking or self.active_motion_mode=='standing_upper_body' else '拖动旋转 / 滚轮缩放 · 保存的仿真结果'
        if self.live and self.active_standing_small_motion_check and self.live_tracking.get('state') in ('paused','recovering'):
            robot_status=('G1 已暂停' if self.live_tracking['state']=='paused' else 'G1 正在恢复，请保持站稳')+'；详情见上方状态栏'
        self.robot_card.footer.setText(robot_status)
        self.robot_card.footer.setToolTip(str(self.live_tracking.get('reason') or self.live_tracking.get('error') or ''))
        self.source_card.footer.setText('独立参考，不参与双目重建' if self.mode.currentIndex()==1 else '原图缺失时不会生成替代画面')
        if not self.live and self.session and self.mode.currentIndex()==0:
            key=str(frame['images'])
            if key!=self._last_media:
                self._last_media=key;self.generation+=1;self.source.images={};self.pose.images={};self.source.update();self.media.submit(self.generation,self.session,frame['images'])
        self.slider.blockSignals(True);self.slider.setValue(round(10000*stamp/max(self.timeline.duration,1e-9)));self.slider.blockSignals(False)
        elapsed=max(0,time.monotonic()-self.live_started) if self.live_started else 0
        self.clock.setText(self.timestamp(elapsed if self.live else stamp)+' / '+self.timestamp(self.live_seconds if self.live else self.timeline.duration));self.play.setText('暂停' if self.timeline.playing else '播放')
    def media_ready(self,generation,images):
        if generation!=self.generation or self.live:return
        self.source.images=images;self.pose.images={k:v for k,v in images.items() if k in ('left','right')};self.source.update();self.pose.update()
    def fullscreen(self):self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def reset_views(self):
        for canvas in (self.human,self.robot):canvas.yaw=.65;canvas.pitch=.24;canvas.zoom=1.;canvas.fixed_center=None;canvas.update()
    def expand(self,card):
        widget=card.layout().itemAt(2).widget();dialog=QDialog(self);dialog.setWindowTitle('画面预览');dialog.resize(960,740);v=QVBoxLayout(dialog);v.addWidget(widget)
        dialog.exec();card.layout().insertWidget(2,widget,1)
    def closeEvent(self,event):
        if self.task and self.task.isRunning():
            self._closing=True;self.cancel();event.ignore();return
        if self.overview and self.overview.isRunning():
            self.overview.stop.set()
            if not self.overview.wait(2000):self.error('等待相机释放后再关闭');event.ignore();return
        self.media.stop.set()
        if not self.media.wait(3000):self.error('等待媒体读取结束后再关闭');event.ignore();return
        self.robot.shutdown();event.accept()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session');parser.add_argument('--screenshot');parser.add_argument('--at',type=float,default=3);parser.add_argument('--theme',choices=['system','light','dark']);parser.add_argument('--live-group',choices=ROUND1_LIVE_GROUPS);parser.add_argument('--standing-upper-body',action='store_true');parser.add_argument('--full-body-check',action='store_true');parser.add_argument('--hybrid-refinement',action='store_true');parser.add_argument('--standing-small-motion-check',action='store_true');parser.add_argument('--hybrid-hip-flip-check',action='store_true');parser.add_argument('--hybrid-backend',choices=['cpu','dml']);parser.add_argument('--hybrid-latency-check',action='store_true');parser.add_argument('--hybrid-reprojection-limit-px',type=float,choices=[5.,6.],default=5.)
    parser.add_argument('--session-root',type=Path,help='本次实时采集的保存目录，不自动修改默认配置')
    parser.add_argument('--max-mib',type=int,help='本次采集的原图容量上限 MiB（32–8192）')
    args=parser.parse_args()
    if args.session_root is not None and not args.session_root.is_absolute():parser.error('--session-root 必须是绝对路径')
    if args.max_mib is not None and not 32<=args.max_mib<=8192:parser.error('--max-mib 必须在 32–8192 范围内')
    if args.hybrid_reprojection_limit_px != 5. and not (args.hybrid_refinement and args.standing_small_motion_check):
        parser.error('6 px requires --hybrid-refinement and --standing-small-motion-check')
    if args.hybrid_latency_check and not (args.hybrid_refinement and args.standing_small_motion_check):
        parser.error('--hybrid-latency-check requires --hybrid-refinement and --standing-small-motion-check')
    if (args.hybrid_hip_flip_check or args.hybrid_backend) and not args.hybrid_refinement:
        parser.error('增强精修和推理后端只用于 --hybrid-refinement 协同候选')
    if args.standing_small_motion_check:args.full_body_check=True
    if args.hybrid_refinement:
        if args.standing_upper_body or args.live_group or args.session or args.screenshot:parser.error('--hybrid-refinement 应单独启动协同候选短测')
        args.full_body_check=True
    if args.live_group and (args.session or args.screenshot):parser.error('--live-group 不能与 --session 或 --screenshot 同用')
    if args.standing_upper_body and (args.live_group or args.session or args.screenshot):parser.error('--standing-upper-body 应单独启动，不能与回放或全身测试组同用')
    if args.full_body_check and (args.standing_upper_body or args.live_group or args.session or args.screenshot):parser.error('--full-body-check 应单独启动，不能与上肢、回放或其他测试组同用')
    app=QApplication(sys.argv[:1]);app.setStyle('Fusion');apply_theme(app,args.theme or load_config().get('theme','system'));window=Station();window.show()
    if args.session_root is not None:window.config['session_root']=str(args.session_root.resolve());window.scan()
    if args.max_mib is not None:window.config['max_mib']=args.max_mib
    if args.theme:
        window.config['theme']=args.theme;window.theme_choice.blockSignals(True);window.theme_choice.setCurrentIndex(['system','light','dark'].index(args.theme));window.theme_choice.blockSignals(False)
    if args.full_body_check:
        window.config['standing_small_motion_check']=args.standing_small_motion_check
        window.config['hybrid_hip_flip_check']=args.hybrid_hip_flip_check
        window.config['hybrid_latency_check']=args.hybrid_latency_check
        window.config['hybrid_reprojection_limit_px']=args.hybrid_reprojection_limit_px
        if args.hybrid_backend:window.config['hybrid_backend']=args.hybrid_backend
        window.full_body_check=True;window.hybrid_refinement=args.hybrid_refinement;window.motion_mode.setCurrentIndex(0);window.mode.setCurrentIndex(1);QTimer.singleShot(0,window.start_live)
    elif args.standing_upper_body:
        window.motion_mode.setCurrentIndex(1);window.mode.setCurrentIndex(1);QTimer.singleShot(0,window.start_live)
    elif args.live_group:
        window.live_group=args.live_group;window.mode.setCurrentIndex(1);QTimer.singleShot(0,window.start_live)
    else:
        session=args.session or window.config.get('startup_session')
        if session:window.load(session);window.timeline.seek(args.at)
    if args.screenshot:
        def shot():
            target=Path(args.screenshot);target.parent.mkdir(parents=True,exist_ok=True);window.grab().save(str(target));window.close();app.quit()
        QTimer.singleShot(6000,shot)
    return app.exec()


if __name__=='__main__':raise SystemExit(main())
