from __future__ import annotations
import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QPainterPath, QImage, QPolygonF
from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from .core import COCO_EDGES

from .theme import COLORS,stylesheet,label_style
STYLE=stylesheet()


def label(text,size=12,color='text',bold=False):
    w=QLabel(text);w.setWordWrap(False)
    w.setProperty('themeRole',color);w.setProperty('typeSize',size);w.setProperty('typeWeight',600 if bold else 400);label_style(w)
    return w


class Card(QFrame):
    def __init__(self,number,title,subtitle,content):
        super().__init__();self.setObjectName('card');self.setMinimumWidth(190)
        v=QVBoxLayout(self);v.setContentsMargins(14,16,14,12);v.setSpacing(8)
        row=QHBoxLayout();row.addWidget(label(number,12,'cyan',True));row.addWidget(label(title,15,'text',True));row.addStretch()
        self.expand=QPushButton('放大');self.expand.setFixedSize(36,27);self.expand.setStyleSheet('padding:0;font-size:10px;');self.expand.setAccessibleName('放大'+title);self.expand.setToolTip('放大此画面');row.addWidget(self.expand)
        v.addLayout(row);v.addWidget(label(subtitle,10,'muted'));v.addWidget(content,1)
        self.footer=label('等待数据',10,'muted');v.addWidget(self.footer)


class FlowArrow(QWidget):
    def __init__(self):
        super().__init__();self.setFixedHeight(45);self.values=['—','—','—'];self.names=['二维识别','三维重建','映射 / IK']

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        for i in range(3):
            x=self.width()*(i+1)/4
            p.setPen(QColor(COLORS['muted']));p.setFont(QFont('Microsoft YaHei UI',9))
            p.drawText(QRectF(x-110,0,220,20),Qt.AlignCenter,self.names[i]+'  '+self.values[i])
            p.setPen(QPen(QColor(COLORS['cyan']),1.5));p.drawLine(QPointF(x-60,31),QPointF(x+60,31))
            p.drawLine(QPointF(x+60,31),QPointF(x+53,27));p.drawLine(QPointF(x+60,31),QPointF(x+53,35))


class ImageCanvas(QWidget):
    def __init__(self,stereo=False,overlay=False):
        super().__init__();self.setMinimumHeight(200);self.images={};self.cameras=None;self.stereo=stereo;self.overlay=overlay
        self.empty='本会话未附原始视频';self.caption='';self.frame_counter=0
        self.view_labels={};self.edges=COCO_EDGES

    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor(COLORS['canvas']))
        sides=['left','right'] if self.stereo else ['overview']
        if not self.stereo and 'overview' not in self.images and ('left' in self.images or 'right' in self.images):sides=['left','right']
        for idx,side in enumerate(sides):
            box=QRectF(0,idx*self.height()/len(sides),self.width(),self.height()/len(sides))
            inner=box.adjusted(5,24,-5,-8);image=self.images.get(side)
            source=[1920,1080];camera=None
            if self.cameras:
                camera=next((c for c in self.cameras if c['camera_id']==side),None)
                if camera:source=camera.get('source_size',source)
            aspect=source[0]/source[1]
            if image is not None and not image.isNull():aspect=image.width()/image.height()
            width=min(inner.width(),inner.height()*aspect);height=width/aspect
            target=QRectF(inner.center().x()-width/2,inner.center().y()-height/2,width,height)
            if image is not None and not image.isNull():p.drawImage(target,image)
            else:
                p.setPen(QPen(QColor(COLORS['grid']),1))
                for k in range(1,8):
                    x=inner.left()+k*inner.width()/8;p.drawLine(QPointF(x,inner.top()),QPointF(x,inner.bottom()))
                for k in range(1,6):
                    y=inner.top()+k*inner.height()/6;p.drawLine(QPointF(inner.left(),y),QPointF(inner.right(),y))
                if not camera:
                    p.setPen(QColor(COLORS['muted']));p.setFont(QFont('Microsoft YaHei UI',10))
                    p.drawText(inner,Qt.AlignCenter|Qt.TextWordWrap,self.empty)
            if camera and camera.get('roi_xywh'):
                rx,ry,rw,rh=camera['roi_xywh'];p.save()
                p.setBrush(Qt.NoBrush);p.setPen(QPen(QColor('#facc15'),2))
                p.drawRect(QRectF(target.left()+rx/source[0]*width,target.top()+ry/source[1]*height,
                                 rw/source[0]*width,rh/source[1]*height))
                p.setFont(QFont('Microsoft YaHei UI',8))
                p.drawText(QPointF(target.left()+rx/source[0]*width+4,target.top()+ry/source[1]*height+15),camera.get('roi_label','模型输入范围'))
                p.restore()
            if camera and (self.overlay or image is None):
                points=camera['keypoints'];xy=[]
                for point in points:xy.append(QPointF(target.left()+point[0]/source[0]*width,target.top()+point[1]/source[1]*height))
                p.setPen(QPen(QColor(COLORS['cyan']),2.3))
                for a,b in self.edges:
                    if points[a][3] and points[b][3]:p.drawLine(xy[a],xy[b])
                for k,point in enumerate(points):
                    if point[3]:
                        p.setPen(Qt.NoPen);p.setBrush(QColor(COLORS['orange'] if point[2]<.5 else COLORS['point']));p.drawEllipse(xy[k],3,3)
            p.setFont(QFont('Segoe UI',8));p.setPen(QColor(COLORS['muted']))
            p.drawText(box.adjusted(9,3,-9,-3),Qt.AlignTop|Qt.AlignLeft,self.view_labels.get(side,{'left':'左相机','right':'右相机','overview':self.caption or '独立现场画面'}[side]))


class SceneCanvas(QWidget):
    def __init__(self,robot=False):
        super().__init__();self.setMinimumHeight(230);self.robot=robot;self.packet=None
        self.yaw=.65;self.pitch=.24;self.zoom=1.;self.drag=None;self.follow=True;self.fixed_center=None
        self.legend=None;self.default_zoom=1.
        self.setToolTip('拖动旋转视角 · 滚轮缩放 · 双击复位')

    def mousePressEvent(self,e):self.drag=e.position()
    def mouseReleaseEvent(self,e):self.drag=None
    def mouseMoveEvent(self,e):
        if self.drag is not None:
            d=e.position()-self.drag;self.yaw+=d.x()*.009;self.pitch=max(-.2,min(.8,self.pitch+d.y()*.006));self.drag=e.position();self.update()
    def wheelEvent(self,e):self.zoom=max(.5,min(2.5,self.zoom*math.exp(e.angleDelta().y()/1500)));self.update()
    def mouseDoubleClickEvent(self,e):self.yaw=.65;self.pitch=.24;self.zoom=self.default_zoom;self.fixed_center=None;self.update()

    def project(self,xyz):
        x,y,z=np.asarray(xyz)-self.center
        h=-y*math.cos(self.yaw)+x*math.sin(self.yaw)
        d=x*math.cos(self.yaw)+y*math.sin(self.yaw)
        v=(z-.7)*math.cos(self.pitch)-d*math.sin(self.pitch)
        scale=min(self.width()/1.65,self.height()/1.95)*self.zoom
        return QPointF(self.width()/2+h*scale,self.height()*.51-v*scale),d

    def paintEvent(self,e):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(),QColor(COLORS['canvas']))
        packet=self.packet;points=None;valid=None;estimated=None
        self.center=np.array([0.,0.,0.])
        if packet:
            if self.robot and packet.get('body_pos'):
                points=np.array(packet['body_pos'],dtype=float);valid=np.isfinite(points).all(axis=1)
                if len(points)>1:self.center[:2]=points[1,:2]
            elif not self.robot and packet.get('joints'):
                valid=np.array([j['valid'] and j['xyz'] is not None for j in packet['joints']])
                points=np.array([j['xyz'] if valid[i] else [0,0,0] for i,j in enumerate(packet['joints'])],float)
                estimated=[j.get('estimated',False) for j in packet['joints']]
                if valid[11:13].all():self.center[:2]=points[11:13,:2].mean(axis=0)
                elif valid.any():self.center[:2]=points[valid,:2].mean(axis=0)
        if self.fixed_center is None and points is not None:self.fixed_center=self.center.copy()
        if not self.follow and self.fixed_center is not None:self.center=self.fixed_center
        p.setPen(QPen(QColor(COLORS['grid']),1))
        for n in range(-8,9):
            offset=n*.25
            for a,b in (([self.center[0]+offset,self.center[1]-2,0],[self.center[0]+offset,self.center[1]+2,0]),
                        ([self.center[0]-2,self.center[1]+offset,0],[self.center[0]+2,self.center[1]+offset,0])):
                p.drawLine(self.project(a)[0],self.project(b)[0])
        if points is None or not valid.any():
            p.setPen(QColor(COLORS['muted']));p.setFont(QFont('Microsoft YaHei UI',10))
            p.drawText(self.rect(),Qt.AlignCenter,'此时刻无仿真状态' if self.robot else '等待有效三维骨架')
        else:
            if self.robot:
                # Geometry-free renderer of the saved Unitree 29-DOF body's positions.
                edges=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(1,9),(9,10),(10,11),(11,12),(12,13),(13,14),
                       (1,17),(17,18),(18,20),(18,22),(22,23),(23,24),(24,25),(25,26),(26,28),
                       (18,30),(30,31),(31,32),(32,33),(33,34),(34,36)]
            else:edges=packet.get('edges',COCO_EDGES)
            segments=[]
            for a,b in edges:
                if max(a,b)>=len(points) or not valid[a] or not valid[b]:continue
                qa,da=self.project(points[a]);qb,db=self.project(points[b]);segments.append(((da+db)/2,a,b,qa,qb))
            for _,a,b,qa,qb in sorted(segments,reverse=True):
                if self.robot:
                    width=max(5,min(self.width(),self.height())*.035*self.zoom)
                    p.setPen(QPen(QColor('#26394b'),width+4,Qt.SolidLine,Qt.RoundCap));p.drawLine(qa,qb)
                    p.setPen(QPen(QColor('#cedae5'),width,Qt.SolidLine,Qt.RoundCap));p.drawLine(qa,qb)
                    p.setPen(QPen(QColor('#f4f9ff'),max(1,width*.2),Qt.SolidLine,Qt.RoundCap));p.drawLine(qa+QPointF(-2,0),qb+QPointF(-2,0))
                else:
                    is_est=estimated[a] or estimated[b]
                    p.setPen(QPen(QColor(COLORS['orange'] if is_est else COLORS['cyan']),3,Qt.DashLine if is_est else Qt.SolidLine,Qt.RoundCap));p.drawLine(qa,qb)
            if self.robot and len(points)>20:
                # Draw armor around actual body transforms; no synthetic motion.
                for idx,w,h,color in [(18,.20,.25,'#b8cbdc'),(1,.19,.12,'#7390a8'),(20,.13,.16,'#d9e6ef')]:
                    center,_=self.project(points[idx]);scale=min(self.width()/1.65,self.height()/1.95)*self.zoom
                    box=QRectF(center.x()-w*scale/2,center.y()-h*scale/2,w*scale,h*scale)
                    p.setPen(QPen(QColor('#526e86'),1.5));p.setBrush(QColor(color));p.drawRoundedRect(box,8,8)
                    if idx==20:
                        visor=box.adjusted(3,box.height()*.25,-3,-box.height()*.43)
                        p.setPen(Qt.NoPen);p.setBrush(QColor('#163743'));p.drawRoundedRect(visor,3,3)
                        p.setPen(QPen(QColor(COLORS['cyan']),2));p.drawLine(visor.topLeft()+QPointF(2,3),visor.topRight()+QPointF(-2,3))
            if not self.robot:
                for i,point in enumerate(points):
                    if valid[i]:
                        pt,_=self.project(point);p.setPen(QPen(QColor(COLORS['panel']),1));p.setBrush(QColor(COLORS['orange'] if estimated[i] else COLORS['point']));p.drawEllipse(pt,4,4)
        p.setPen(QColor(COLORS['muted']));p.setFont(QFont('Segoe UI',8));p.drawText(8,self.height()-10,self.legend or 'Z ↑  ·  m   /   '+('G1 状态简模' if self.robot else '实测点  /  橙色为估计'))


class ComparisonPlot(QWidget):
    def __init__(self):
        super().__init__();self.data=None;self.setMinimumHeight(100)

    def paintEvent(self,e):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        if not self.data:
            p.setPen(QColor(COLORS['muted']));p.drawText(self.rect(),Qt.AlignCenter,'选择已录制会话，运行同板 CPU / FPGA 对比\n仅展示真实测量；尚无可用对比结果');return
        cpu=self.data['cpu'];pl=self.data['fpga'];maximum=max(cpu['pair_latency_median_ms'],pl['pair_latency_median_ms'],1)
        for i,(title,value,color) in enumerate([('ARM CPU',cpu,COLORS['blue']),('ARM + FPGA',pl,COLORS['cyan'])]):
            y=22+i*39;p.setPen(QColor(COLORS['text']));p.drawText(5,y,title)
            w=max(3,(self.width()-230)*value['pair_latency_median_ms']/maximum)
            p.setPen(Qt.NoPen);p.setBrush(QColor(color));p.drawRoundedRect(QRectF(112,y-13,w,16),4,4)
            p.setPen(QColor(COLORS['text']));p.drawText(int(124+w),y,f"{value['pair_latency_median_ms']:.1f} ms")
        p.setPen(QColor(COLORS['muted']));p.drawText(5,self.height()-6,'单对处理耗时 · 中位数 · 越短越好')
