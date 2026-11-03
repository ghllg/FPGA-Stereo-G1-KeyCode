import base64
import json
import time
from pathlib import Path
from PySide6.QtCore import QProcess, QRectF, Qt, Signal
from PySide6.QtGui import QImage,QPainter,QColor
from .widgets import SceneCanvas,COLORS
from .core import HOME


class RobotView(SceneCanvas):
    problem=Signal(str)
    presented=Signal(object)
    MAX_RENDER_AGE_S=.25
    def __init__(self):
        super().__init__(robot=True);self.process=None;self.pending=None;self.inflight=False;self.buffer=b'';self.image=QImage();self.last_key=None;self.generation=0;self.image_id=-1;self.reported=False
        self.epoch=0;self.context=None;self.active_request=None;self.image_valid_at=0.
        self.pending_metadata=None;self.active_metadata=None;self.image_metadata=None;self.presented_image=None
        self.hold_last=False;self.source_age_s=None;self.display_state_key=None
    def view_signature(self):
        return (self.yaw,self.pitch,self.zoom,COLORS['bg'])
    def invalidate_frame(self):
        # Keep the outstanding request until its own reply arrives; discard its image.
        self.epoch+=1;self.pending=None;self.last_key=None;self.image=QImage();self.image_id=-1
        self.pending_metadata=None;self.image_metadata=None
        self.update()
    def render_failure(self,message):
        self.image=QImage();self.image_id=-1
        self.image_metadata=None
        if not self.reported:self.problem.emit(message)
        self.reported=True;self.update()
    def start_renderer(self,root):
        root=Path(root)
        if self.process:
            self.process.readyReadStandardOutput.disconnect(self.receive);self.process.finished.disconnect(self.renderer_finished);self.process.errorOccurred.disconnect()
            self.shutdown();self.process.deleteLater();self.process=None
        self.invalidate_frame();self.context=None;self.inflight=False;self.active_request=None;self.active_metadata=None;self.buffer=b'';self.reported=False
        if not (root/'runtime/python.exe').is_file():
            self.reported=True;self.problem.emit('未找到 G1 运行环境，请检查设备与标定设置中的目录');self.update();return
        self.process=QProcess(self);self.process.setWorkingDirectory(str(root));self.process.readyReadStandardOutput.connect(self.receive)
        self.process.errorOccurred.connect(lambda _:self.render_failure('G1 模型渲染进程无法启动'))
        self.process.finished.connect(self.renderer_finished)
        self.process.setProgram(str(root/'runtime/python.exe'));self.process.setArguments(['-u',str(HOME/'render_robot.py'),str(root)]);self.process.start()
    def renderer_finished(self,code):
        self.render_failure('G1 渲染进程已退出：'+bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace')[-600:])
    def request_frame(self,context=None,*,hold_last=False,source_age_s=None,source_context=None):
        self.hold_last=hold_last;self.source_age_s=source_age_s
        context=(context,self.view_signature())
        if context!=self.context:
            self.invalidate_frame();self.context=context
        if not self.packet:
            if not hold_last and self.last_key is not None:self.invalidate_frame()
            return
        if not self.process:return
        key=(tuple(self.packet['qpos']),self.packet.get('session'),self.packet.get('preparation',False),self.yaw,self.pitch,self.zoom,COLORS['bg'])
        if key==self.last_key and self.image_id==self.generation and not self.image.isNull():
            self.image_valid_at=time.monotonic()
        if key!=self.last_key:
            self.last_key=key;self.generation+=1
            self.pending=dict(id=self.generation,qpos=self.packet['qpos'],body_pos=self.packet['body_pos'],yaw=self.yaw,pitch=self.pitch,zoom=self.zoom,dark=COLORS['bg']=='#202020')
            fields=('session','seq','source_time','simulation_time','client_time','pc_receive_ns','pc_complete_ns',
                    'source_pc_receive_ns','source_pc_origin_ns','server_completed_ns','preparation','human_observed','physics','preparation_kind')
            self.pending_metadata=dict(pc_request_ns=time.perf_counter_ns(),
                robot={name:self.packet[name] for name in fields if name in self.packet and type(self.packet[name]) in (str,int,float,bool,type(None))})
            if source_context is not None:self.pending_metadata['source_context']=dict(source_context)
        self.send()
    def send(self):
        if self.pending and not self.inflight and self.process.state()==QProcess.Running:
            self.active_request=(self.pending['id'],self.epoch,time.monotonic())
            self.active_metadata=self.pending_metadata;self.pending_metadata=None
            self.active_metadata['pc_render_send_ns']=time.perf_counter_ns()
            self.process.write((json.dumps(self.pending)+'\n').encode());self.pending=None;self.inflight=True
    def receive(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:
                response=json.loads(line)
                if self.active_request is None or response.get('id')!=self.active_request[0]:continue
                request_id,epoch,requested_at=self.active_request
                metadata=self.active_metadata;self.active_metadata=None
                self.active_request=None;self.inflight=False
                if epoch!=self.epoch:continue
                if response.get('error'):
                    self.render_failure('G1 模型渲染：'+response['error'])
                elif request_id==self.generation or time.monotonic()-requested_at<=self.MAX_RENDER_AGE_S:
                    rendered=QImage.fromData(base64.b64decode(response['image']))
                    if rendered.isNull():raise ValueError('无效的渲染图像')
                    self.image_metadata={**metadata,'pc_image_received_ns':time.perf_counter_ns()}
                    self.image=rendered;self.image_id=request_id;self.image_valid_at=requested_at;self.reported=False;self.update()
            except (ValueError,KeyError) as error:self.render_failure('G1 渲染响应无效：'+str(error))
        self.send()
    def has_display_frame(self):
        if (not self.packet and not self.hold_last) or self.image.isNull() or self.context is None or self.context[1]!=self.view_signature():return False
        # A settled paused pose stays valid; only a moving target needs a bounded hold.
        return self.hold_last or self.image_id==self.generation or time.monotonic()-self.image_valid_at<=self.MAX_RENDER_AGE_S
    def display_status(self):
        state='empty';notice='';age=self.source_age_s
        if self.has_display_frame():
            state='fresh'
            if (self.image_metadata or {}).get('robot',{}).get('preparation'):
                state='preparing';notice='模型初始姿态 · 尚无人体观测' if self.image_metadata['robot'].get('preparation_kind')=='official_model_initial_pose' else '本机准备姿态 · 尚无人体观测'
            elif self.hold_last:
                received=(self.image_metadata or {}).get('robot',{}).get('pc_receive_ns')
                if received is not None:age=max(0.,(time.perf_counter_ns()-received)/1e9)
                if not self.packet:state='held';notice='输出暂停 · 上一次仿真状态'
                elif self.image_id!=self.generation and time.monotonic()-self.image_valid_at>self.MAX_RENDER_AGE_S:
                    state='render_delayed';notice='显示更新延迟 · 上一次仿真状态'
                if notice and age is not None:notice+=f' · {age:.1f} 秒前'
        return dict(state=state,notice=notice,source_age_s=age)
    def paintEvent(self,event):
        display=self.display_status();state_key=(self.epoch,display['state'])
        if state_key!=self.display_state_key:
            self.display_state_key=state_key
            if self.hold_last:self.presented.emit(dict(event='robot_display_state',image_epoch=self.epoch,
                request_id=self.image_id,pc_paint_ns=time.perf_counter_ns(),robot=dict((self.image_metadata or {}).get('robot',{})),**display))
        if self.has_display_frame():
            p=QPainter(self);p.fillRect(self.rect(),QColor(COLORS['canvas']));p.setRenderHint(QPainter.SmoothPixmapTransform)
            size=self.image.size();scale=min(self.width()/size.width(),self.height()/size.height());w=size.width()*scale;h=size.height()*scale
            p.drawImage(QRectF((self.width()-w)/2,(self.height()-h)/2,w,h),self.image)
            if display['notice']:
                p.fillRect(QRectF(0,0,self.width(),32),QColor(COLORS['canvas']));p.setPen(QColor(COLORS['muted']))
                p.drawText(QRectF(8,0,self.width()-16,32),Qt.AlignCenter,display['notice'])
            p.setPen(QColor(COLORS['muted']));p.drawText(10,self.height()-10,'G1 · 保存的 qpos / MuJoCo 模型');p.end()
            identity=(self.epoch,self.image_id)
            if self.image_metadata is not None and identity!=self.presented_image:
                self.presented_image=identity
                self.presented.emit({**self.image_metadata,'robot':dict(self.image_metadata['robot']),
                    'event':'robot_image_presented','request_id':self.image_id,'image_generation':self.image_id,
                    'image_epoch':self.epoch,'pc_paint_ns':time.perf_counter_ns()})
            return
        if self.packet:
            p=QPainter(self);p.fillRect(self.rect(),QColor(COLORS['canvas']));p.setPen(QColor(COLORS['muted']))
            p.drawText(self.rect(),Qt.AlignCenter,'G1 模型暂不可用，请查看运行状态' if self.reported else '正在绘制保存的 G1 状态');return
        # Never invent a robot topology when no saved state exists.
        packet=self.packet;self.packet=None;super().paintEvent(event);self.packet=packet
    def shutdown(self):
        if self.process:
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(1500):self.process.kill();self.process.waitForFinished(1500)
