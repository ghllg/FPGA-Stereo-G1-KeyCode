"""Independent file-playback entry, reusing Motion Studio's four-panel interface."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import cv2
import numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtWidgets import (QApplication,QComboBox,QDialog,QDialogButtonBox,
    QDoubleSpinBox,QFileDialog,QFormLayout,QLabel,QLineEdit,QListWidgetItem,QStackedWidget,QVBoxLayout)
from .app import Station
from .bvh_playback import SCHEMA,BODY_EDGES,PlaybackSession,pose_packet
from .bvh_native import NativePlayback
from .core import HOME,read_json,write_json
from .services import load_config
from .theme import apply_theme
from .widgets import SceneCanvas,label


SESSIONS = HOME/'data/bvh_sessions'


def build_session(source,start,end,name,g1_root,progress,stop):
    if stop.is_set():raise InterruptedError('已取消')
    key = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    destination = SESSIONS/key
    log = HOME/'output'/('bvh_build_'+key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    command = [str(Path(g1_root)/'runtime/python.exe'),'-u',str(HOME/'bvh_worker.py'),
               '--source',str(source),'--output',str(destination),'--g1-root',str(g1_root),
               '--start',str(start),'--end',str(end),'--name',name]
    with log.open('w',encoding='utf-8') as output:
        process = subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,
                                   env={**os.environ,'PYTHONUTF8':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
        with log.open(encoding='utf-8',errors='replace') as reader:
            try:
                while process.poll() is None:
                    if stop.wait(.1):raise InterruptedError('已取消；未完成会话不会加入列表')
                    for line in reader.readlines():
                        if line.startswith(('IK ','Physics ')):progress(line.strip())
                if process.returncode:raise RuntimeError(log.read_text(encoding='utf-8')[-1600:])
                if stop.is_set():raise InterruptedError('已取消')
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(3)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                if stop.is_set() and (destination/'session.json').exists():
                    manifest=read_json(destination/'session.json');manifest['state']='cancelled'
                    write_json(destination/'session.json',manifest)
    return destination


def set_card(card,title,subtitle):
    card.layout().itemAt(0).layout().itemAt(1).widget().setText(title)
    card.layout().itemAt(1).widget().setText(subtitle)


class BVHStation(Station):
    def __init__(self):
        self.file_ready=False;self.errors=[]
        super().__init__()
        self.setWindowTitle('Motion Studio · 测试版本 · 文件回放')
        for widget in (self.mode,self.motion_mode,self.settings_btn,self.preflight_btn,self.install_btn,
                       self.camera_btn,self.live_btn,self.record_btn,self.compare_btn):widget.hide()
        for arrow in (self.arrow_2d,self.arrow_3d,self.arrow_g1):arrow.parentWidget().hide()
        self.plot.parentWidget().hide()
        for text in self.centralWidget().findChildren(QLabel):
            if text.text()=='实验候选版本\n数据缺失时保留空缺':text.setText('测试版本 · 文件回放')
            if text.text()=='双目动捕 / FPGA / G1':text.setText('动作文件 / 骨架 / G1')
        self.import_btn.setText('打开回放会话')
        self.video_btn.setText('导入真人视频')
        self.simulate_btn.setText('导入 BVH 片段')
        self.source_stack=QStackedWidget()
        self.reference=SceneCanvas();self.reference.legend='Z ↑ · m / 原始动作参考'
        self.reference.default_zoom=self.human.default_zoom=.70
        self.source_card.layout().removeWidget(self.source)
        self.source_stack.addWidget(self.reference);self.source_stack.addWidget(self.source)
        self.source_card.layout().insertWidget(2,self.source_stack,1)
        self.source_choice=QComboBox();self.source_choice.addItems(['动作参考','真人视频'])
        self.source_choice.currentIndexChanged.connect(self.source_changed)
        self.source_card.layout().itemAt(0).layout().insertWidget(2,self.source_choice)
        self.source.caption='真人视频 · 文件回放'
        self.source.empty='此时刻不在真人视频范围内'
        self.pose.view_labels={'left':'虚拟视角 A','right':'虚拟视角 B'};self.pose.edges=BODY_EDGES
        self.human.legend='Z ↑ · m / 橙色虚线为时序补点'
        set_card(self.source_card,'动作参考','同一动作时间轴 · 可替换真人视频')
        set_card(self.pose_card,'双视角二维骨架','虚拟相机投影 · 短时关节遮挡')
        set_card(self.human_card,'三维骨架与补点','双目几何重建 · 时序插值')
        set_card(self.robot_card,'Unitree G1','物理仿真 / 参考动画')
        self.backend.blockSignals(True);self.backend.clear();self.backend.addItem('物理仿真','physics');self.backend.addItem('参考动画','reference');self.backend.blockSignals(False)
        self.backend.currentIndexChanged.connect(self.robot_mode_changed)
        self.offset=QDoubleSpinBox();self.offset.setDecimals(2);self.offset.setRange(-36000,36000);self.offset.setSingleStep(.1);self.offset.setSuffix(' s')
        self.offset.setToolTip('视频时间 = 动作时间 + 偏移；正数跳过视频开头，负数延后视频出现')
        self.offset.valueChanged.connect(self.offset_changed)
        sidebar=self.video_btn.parentWidget().layout()
        sidebar.insertWidget(sidebar.indexOf(self.video_btn)+1,label('真人视频起点偏移',10,'muted'))
        sidebar.insertWidget(sidebar.indexOf(self.video_btn)+2,self.offset)
        self.native=NativePlayback(self);self.native.problem.connect(self.error)
        self.native.running_changed.connect(lambda running:self.preflight_btn.setText('MuJoCo 窗口已联动' if running else '打开 MuJoCo 窗口'))
        self.preflight_btn.setText('打开 MuJoCo 窗口');self.preflight_btn.show()
        self.file_ready=True;self.scan();self.refresh()
        startup=self.config.get('bvh_startup_session')
        if startup and (Path(startup)/'session.json').is_file():self.load(startup)

    def change_mode(self,index):
        # File entry never enables capture or board actions.
        pass

    def local_check(self):
        if not self.session:self.error('请先选择动作会话');return
        self.native.start(self.config['g1_dir']);self.refresh()

    def scan(self):
        if not hasattr(self,'library'):return
        self.library.clear()
        if SESSIONS.exists():
            for path in sorted(SESSIONS.iterdir()):
                try:manifest=read_json(path/'session.json')
                except (OSError,ValueError):continue
                if manifest.get('schema')!=SCHEMA or manifest.get('state')!='complete':continue
                item=QListWidgetItem(manifest['name']);item.setData(Qt.UserRole,str(path));self.library.addItem(item)

    def open_folder(self):
        path=QFileDialog.getExistingDirectory(self,'打开文件回放会话',str(SESSIONS))
        if path:self.load(path)

    def load(self,path):
        try:session=PlaybackSession(path)
        except Exception as error:self.error(str(error));return
        self.pause();self.session=session;self.timeline.duration=session.duration;self.timeline.seek(0)
        self._last_media=None;self.generation+=1;self.source.images={}
        self.media.submit(self.generation,session,{})
        self.subtitle.setText(session.name)
        self.backend.blockSignals(True);self.backend.clear()
        for mode in session.robot_modes:self.backend.addItem('物理仿真' if mode=='physics' else '参考动画',mode)
        default=session.manifest.get('default_robot_mode','reference')
        self.backend.setCurrentIndex(max(0,self.backend.findData(default)));self.backend.blockSignals(False)
        video=session.manifest.get('media',{}).get('overview')
        self.offset.blockSignals(True);self.offset.setValue(video.get('offset_seconds',0.) if video else 0.);self.offset.blockSignals(False)
        self.source_choice.setCurrentIndex(1 if video else 0);self.source_changed()
        self.config['bvh_startup_session']=str(session.path.resolve());write_json(HOME/'config.json',self.config)
        self.append(f'已载入 {session.name} · {len(session.times)} 帧 · {session.duration:.2f} 秒')
        physics=session.manifest['physics']
        if physics.get('completed'):
            self.append('G1 物理轨迹完整，无跌倒。'+('脚步漂移较大，默认使用参考动画。' if not physics.get('tracking_accepted',True) else ''))
        else:self.append('物理轨迹未通过完整性检查；提供参考动画。')
        stats=session.manifest['completion']
        if stats['completion_mean_cm'] is not None:self.append(f"短缺点补全：平均误差 {stats['completion_mean_cm']:.1f} cm，最大 {stats['completion_max_cm']:.1f} cm；仅为虚拟遮挡测试。")
        self.robot_mode_changed();self.reset_views();self.refresh()

    def busy(self,value):
        super().busy(value)
        if self.file_ready:self.offset.setEnabled(not value)

    def simulate(self):
        source=QFileDialog.getOpenFileName(self,'选择原版 LAFAN1 BVH',str(HOME/'data/bvh'),'BVH (*.bvh)')[0]
        if not source:return
        try:
            text=Path(source).read_text(encoding='utf-8-sig');motion=text.split('MOTION',1)[1].splitlines()
            count=int(next(line.split(':')[1] for line in motion if line.strip().startswith('Frames:')))
            dt=float(next(line.split(':')[1] for line in motion if line.strip().startswith('Frame Time:')))
            if count<2 or not 0<dt<=1:raise ValueError('帧数或帧时间无效')
        except (OSError,ValueError,IndexError,StopIteration) as error:self.error('BVH 读取失败：'+str(error));return
        dialog=QDialog(self);dialog.setWindowTitle('选择动作片段');layout=QVBoxLayout(dialog);form=QFormLayout();layout.addLayout(form)
        name=QLineEdit(Path(source).stem);form.addRow('会话名称',name)
        start=QDoubleSpinBox();end=QDoubleSpinBox()
        for spin in (start,end):spin.setDecimals(3);spin.setRange(0,count*dt);spin.setSuffix(' s')
        end.setValue(min(6,count*dt));form.addRow('起点',start);form.addRow('终点（不含）',end)
        layout.addWidget(label('单次 1–60 秒。仅支持原版 LAFAN1：cm / Y 向上。\n先生成真实物理轨迹；未通过时保留参考动画。',11,'muted'))
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if not dialog.exec():return
        if not 1<=end.value()-start.value()<=60:self.error('请选择 1–60 秒的片段');return
        g1_root=Path(self.config.get('g1_dir',''))
        if not (g1_root/'runtime/python.exe').is_file():self.error('本地 G1 运行目录不可用，请检查本入口的 config.json');return
        self.pause();a,b,title=start.value(),end.value(),name.text().strip() or Path(source).stem
        self.job(lambda progress,stop:build_session(source,a,b,title,g1_root,progress,stop),self.load)

    def attach_video(self,path):
        if not self.session:raise ValueError('请先选择动作会话')
        cap=cv2.VideoCapture(str(path))
        try:
            fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=cap.read()
            if not ok or not np.isfinite(fps) or fps<=0 or frames<=0:raise ValueError('视频无法解码或帧率无效')
        finally:cap.release()
        self.session.manifest['media']={'overview':dict(path=str(Path(path).resolve()),fps=fps,frames=frames,offset_seconds=0.)}
        write_json(self.session.path/'session.json',self.session.manifest)
        self.offset.setValue(0.);self._last_media=None;self.source_choice.setCurrentIndex(1);self.source_changed();self.refresh()
        self.append('真人视频已关联（引用原文件）；可用起点偏移对齐动作。')

    def import_video(self):
        if not self.session:self.error('请先选择动作会话');return
        path=QFileDialog.getOpenFileName(self,'导入真人模仿视频',str(HOME),'视频 (*.mp4 *.avi *.mov *.mkv)')[0]
        if path:
            try:self.attach_video(path)
            except (ValueError,OSError) as error:self.error(str(error))

    def source_changed(self,*args):
        if not self.file_ready:return
        video=bool(self.session and self.session.manifest.get('media',{}).get('overview'))
        self.source_choice.model().item(1).setEnabled(video)
        index=self.source_choice.currentIndex() if video else 0
        self.source_stack.setCurrentIndex(index)
        self.offset.setEnabled(video and not (self.task and self.task.isRunning()))
        set_card(self.source_card,'动作参考' if index==0 else '真人模仿','同一动作时间轴 · 录制视频可调起点')
        self._last_media=None;self.refresh()
        if index==0 and self.session:
            self.generation+=1;self.media.submit(self.generation,self.session,{})

    def offset_changed(self,value):
        if not self.file_ready or not self.session:return
        video=self.session.manifest.get('media',{}).get('overview')
        if not video:return
        video['offset_seconds']=value;write_json(self.session.path/'session.json',self.session.manifest)
        self._last_media=None;self.generation+=1;self.source.images={};self.refresh()

    def robot_mode_changed(self,*args):
        if not self.file_ready:return
        self.robot.invalidate_frame()
        mode=self.backend.currentData()
        set_card(self.robot_card,'Unitree G1','物理仿真 · 保存的实际状态' if mode=='physics' else '参考动画 · 关节映射')
        self.refresh()

    def step(self,direction):
        if not self.session:return
        self.pause();times=self.session.times
        index=np.searchsorted(times,self.timeline.position+(1e-8 if direction>0 else -1e-8),side='right' if direction>0 else 'left')
        if direction<0:index-=1
        self.timeline.seek(float(times[int(np.clip(index,0,len(times)-1))]));self.refresh()

    def refresh(self,*args):
        if not self.file_ready:return
        stamp=self.timeline.tick()
        if self.session:
            session=self.session;index=session.index(stamp);data=session.tracks
            self.reference.packet=pose_packet(data['body'][index])
            self.human.packet=pose_packet(data['completed'][index],data['estimated'][index])
            cameras=[]
            for camera,side in enumerate(('left','right')):
                xy=np.nan_to_num(data['projections'][index,camera]);valid=data['visible'][index,camera]
                points=np.column_stack((xy,valid.astype(float),valid.astype(float))).tolist()
                cameras.append(dict(camera_id=side,source_size=[1280,960],keypoints=points))
            self.pose.cameras=cameras
            mode=self.backend.currentData()
            self.robot.packet=session.robot_packet(stamp,mode)
            self.robot.request_frame((str(session.path),self.timeline.revision,mode))
            self.native.publish(self.robot.packet,stamp,self.timeline.playing,self.timeline.revision)
            self.status.setText('● 文件回放 · '+('播放中' if self.timeline.playing else '已暂停'))
            self.source_card.footer.setText('真人视频 · 起点偏移 '+f'{self.offset.value():+.2f}s' if self.source_stack.currentIndex() else '原始动作参考 · 与生成结果独立对照')
            self.pose_card.footer.setText('缺失点留空 · 两路投影使用同一帧')
            count=int(data['estimated'][index].sum())
            self.human_card.footer.setText(f'橙色虚线 = 时序补点 · 当前 {count} 点')
            self.robot_card.footer.setText('MuJoCo + 控制策略 · 已保存物理状态' if mode=='physics' else 'G1 模型关节动画 · 不含动力学验证')
            self.infer_label.setText('测试版本 · 文件回放     共用动作时间轴     G1 使用完整动作参考')
            if self.source_stack.currentIndex()==1:
                frame=session.video_frame(stamp);key=(str(session.path),frame,self.offset.value())
                if key!=self._last_media:
                    self._last_media=key;self.generation+=1;self.source.images={};self.source.update()
                    if frame is not None:self.media.submit(self.generation,session,{'overview':dict(frame_index=frame)})
        else:
            self.status.setText('● 文件回放 · 请选择动作会话')
            self.infer_label.setText('测试版本 · 文件回放')
        for canvas in (self.reference,self.human,self.pose):canvas.update()
        self.slider.blockSignals(True);self.slider.setValue(round(10000*stamp/max(self.timeline.duration,1e-9)));self.slider.blockSignals(False)
        self.clock.setText(self.timestamp(stamp)+' / '+self.timestamp(self.timeline.duration));self.play.setText('暂停' if self.timeline.playing else '播放')

    def media_ready(self,generation,images):
        if generation!=self.generation:return
        self.source.images=images;self.source.update()

    def reset_views(self):
        super().reset_views()
        if self.file_ready:
            self.human.zoom=.70
            self.reference.yaw=self.human.yaw;self.reference.pitch=self.human.pitch;self.reference.zoom=.70;self.reference.update()

    def error(self,text):
        self.errors.append(str(text));super().error(text)

    def closeEvent(self,event):
        super().closeEvent(event)
        if event.isAccepted():self.native.shutdown()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session');parser.add_argument('--screenshot');parser.add_argument('--at',type=float,default=0.)
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    app=QApplication(sys.argv);apply_theme(app,load_config().get('theme','light'))
    window=BVHStation()
    if args.session:window.load(args.session)
    window.timeline.seek(args.at);window.refresh();window.show()
    if args.native:window.local_check()
    if args.screenshot:
        def capture():
            target=Path(args.screenshot);target.parent.mkdir(parents=True,exist_ok=True)
            window.grab().save(str(target));window.close()
        QTimer.singleShot(3000,capture)
    return app.exec()


if __name__=='__main__':sys.exit(main())
