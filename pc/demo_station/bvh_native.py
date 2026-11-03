"""Owned native-viewer process, bounded to one request plus the latest pending pose."""
import json
from pathlib import Path
from PySide6.QtCore import QObject,QProcess,Signal
from .core import HOME


class NativePlayback(QObject):
    problem=Signal(str)
    acknowledged=Signal(object)
    running_changed=Signal(bool)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.process=None;self.pending=None;self.active=None;self.last_key=None
        self.buffer=b'';self.serial=0;self.last_ack=None;self.stopping=False

    def start(self,root):
        if self.process and self.process.state()!=QProcess.NotRunning:return
        if self.process:self.process.deleteLater()
        self.process=QProcess(self);self.pending=None;self.active=None;self.last_key=None
        self.last_ack=None;self.buffer=b'';self.stopping=False
        self.process.setWorkingDirectory(str(root))
        self.process.setProgram(str(Path(root)/'runtime/python.exe'))
        self.process.setArguments(['-u',str(HOME/'bvh_native_viewer.py'),str(root)])
        self.process.started.connect(self.send)
        self.process.readyReadStandardOutput.connect(self.receive)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(lambda _:self.problem.emit('MuJoCo 独立窗口未能运行，请查看 G1 运行目录'))
        self.process.start();self.running_changed.emit(True)

    def publish(self,packet,position,playing,revision):
        if not self.process or self.process.state()==QProcess.NotRunning:return
        key=(packet['session'],packet['seq'],packet['physics'],playing,revision)
        if key==self.last_key:return
        self.last_key=key;self.serial+=1
        self.pending={**packet,'id':self.serial,'position':float(position),'playing':bool(playing)}
        self.send()

    def send(self):
        if self.pending and self.active is None and self.process.state()==QProcess.Running:
            self.active=self.pending;self.pending=None
            self.process.write((json.dumps(self.active)+'\n').encode('utf-8'))

    def receive(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            try:
                response=json.loads(line)
                if response.get('error'):raise ValueError(response['error'])
                if self.active is None or response.get('id')!=self.active['id']:continue
                if response['qpos']!=self.active['qpos']:raise ValueError('独立窗口姿态与主界面不一致')
                self.last_ack=response;self.active=None;self.acknowledged.emit(response)
            except (ValueError,KeyError) as error:self.problem.emit('MuJoCo 联动：'+str(error))
        self.send()

    def finished(self,code):
        self.active=None;self.pending=None;self.last_key=None;self.running_changed.emit(False)
        if code and not self.stopping:
            self.problem.emit('MuJoCo 独立窗口退出：'+bytes(self.process.readAllStandardError()).decode('utf-8',errors='replace')[-600:])

    def shutdown(self):
        self.stopping=True
        if self.process:
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(2000):self.process.kill();self.process.waitForFinished(1000)
