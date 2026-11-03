from __future__ import annotations
import threading
import time
from pathlib import Path
import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def qt_image(frame):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    return QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()


class Task(QThread):
    progress=Signal(str)
    result=Signal(object)
    failed=Signal(str)
    def __init__(self,fn,parent=None):
        super().__init__(parent);self.fn=fn;self.cancel=threading.Event()
    def run(self):
        try:self.result.emit(self.fn(self.progress.emit,self.cancel))
        except Exception as e:self.failed.emit(f'{type(e).__name__}: {e}')


class MediaReader(QThread):
    ready=Signal(int,object)
    failed=Signal(str)
    def __init__(self,parent=None):
        super().__init__(parent);self.stop=threading.Event();self.lock=threading.Lock();self.request=None
    def submit(self,generation,session,entries):
        with self.lock:self.request=(generation,session,entries)
    def run(self):
        cache={};videos={}
        try:
            while not self.stop.wait(.015):
                with self.lock:request=self.request;self.request=None
                if request is None:continue
                generation,session,entries=request;images={}
                active=set()
                for side,entry in entries.items():
                    media=session.manifest.get('media',{}).get(side,{})
                    if entry is not None and 'path' not in entry and media.get('path'):
                        active.add(str(session.resolve(media['path'])))
                for path in list(videos):
                    if path not in active:videos.pop(path).release()
                for side,entry in entries.items():
                    if entry is None:continue
                    key=(str(session.path),side,str(entry),str(session.manifest.get('media',{}).get(side,{})))
                    if key not in cache:
                        try:
                            if 'path' in entry:
                                import numpy as np
                                p=session.resolve(entry['path']);frame=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
                            else:
                                media=session.manifest['media'][side];p=str(session.resolve(media['path']))
                                if p not in videos:videos[p]=cv2.VideoCapture(p)
                                cap=videos[p];cap.set(cv2.CAP_PROP_POS_FRAMES,int(entry['frame_index']));ok,frame=cap.read()
                                if not ok:frame=None
                            if frame is None:raise ValueError('图像解码失败')
                        except (OSError,ValueError,KeyError,cv2.error) as error:
                            frame=None;self.failed.emit(f'{side} 媒体读取失败：{error}')
                        cache[key]=qt_image(frame) if frame is not None else QImage()
                        if len(cache)>40:cache.pop(next(iter(cache)))
                    images[side]=cache[key]
                self.ready.emit(generation,images)
        finally:
            for cap in videos.values():cap.release()


class OverviewCamera(QThread):
    frame=Signal(object)
    failed=Signal(str)
    def __init__(self,index,parent=None):
        super().__init__(parent);self.index=index;self.stop=threading.Event()
    def run(self):
        cap=cv2.VideoCapture(self.index,cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():raise RuntimeError('无法打开第三台相机，请检查相机编号和占用情况')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
            while not self.stop.is_set():
                ok,frame=cap.read()
                if not ok:raise RuntimeError('第三台相机读取中断')
                self.frame.emit(qt_image(frame));self.stop.wait(.03)
        except Exception as e:self.failed.emit(str(e))
        finally:cap.release()
