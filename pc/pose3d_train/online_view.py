"""View already-computed skeletons; this window never opens a camera."""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage,QPixmap,QFont
from PySide6.QtWidgets import QApplication,QLabel,QVBoxLayout,QWidget
from .online import draw_skeleton


def main(source):
    source=Path(source)
    app=QApplication.instance() or QApplication([])
    window=QWidget();window.setWindowTitle('双摄实时三维骨架')
    layout=QVBoxLayout(window)
    text=QLabel('等待开发板关键点；此窗口不控制采集。绿色为几何结果，橙色为模型补点。')
    text.setFont(QFont('Microsoft YaHei',11));layout.addWidget(text)
    picture=QLabel();layout.addWidget(picture)
    last=[None]
    origin=[None]
    def refresh():
        try:
            value=json.loads(source.read_text(encoding='utf-8'))
        except (FileNotFoundError,PermissionError,json.JSONDecodeError):return
        finished=(source.parent/'summary.json').exists()
        stale=time.perf_counter_ns()-value.get('pc_complete_ns',0)>1_000_000_000
        state='本段已结束，显示最后一帧' if finished else ('超过1秒未收到新数据' if stale else '实时接收中')
        identity=(value['seq'],state)
        if identity==last[0]:return
        last[0]=identity
        if origin[0] is None and all(j['valid'] for j in value['joints'][11:13]):
            origin[0]=np.mean([j['xyz'] for j in value['joints'][11:13]],axis=0)
        valid=sum(j['valid'] for j in value['joints'][5:])
        text.setText(f"{state} · 已处理 {value['seq']+1} 组 · 身体关节 {valid}/12 · 绿色：几何结果，橙色：模型补点")
        frame=draw_skeleton(value,origin=origin[0])
        image=QImage(frame.data,frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format.Format_BGR888).copy()
        picture.setPixmap(QPixmap.fromImage(image))
    timer=QTimer(window);timer.timeout.connect(refresh);timer.start(50)
    window.resize(1220,690);window.show();app.exec()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    main(parser.parse_args().source)
