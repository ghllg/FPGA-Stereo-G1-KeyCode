import numpy as np
import os
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget


def view_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    positions = data['global_positions_cm']; parents = data['parents']; fps = float(data['fps'])
    class Viewer(QWidget):
        def __init__(self):
            super().__init__(); self.frame = 0; self.pause = False; self.view = 0
            self.setWindowTitle('Windows · LAFAN1 骨架回放（空格暂停，V 切换视角）')
            self.resize(850, 700)
            self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(round(1000/fps))
            self.center = np.median(positions[:, 0], axis=0)
            self.scale = 500 / max(np.ptp(positions[:, :, 1]), 100)
        def tick(self):
            if not self.pause: self.frame = (self.frame+1)%len(positions)
            self.update()
        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_Space: self.pause = not self.pause
            if event.key() == Qt.Key.Key_V: self.view = (self.view+1)%3
            self.update()
        def paintEvent(self, event):
            p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), QColor('#141c28')); p.setPen(QColor('white'))
            p.drawText(20, 30, f'LAFAN1 · 帧 {self.frame}/{len(positions)-1} · {fps:g} fps · BVH Y 向上 / cm')
            points = positions[self.frame]-self.center
            h = [points[:, 0], points[:, 2], points[:, 0]*.8 + points[:, 2]*.6][self.view]
            xy = np.column_stack([self.width()/2-h*self.scale, self.height()/2-points[:, 1]*self.scale])
            for i, parent in enumerate(parents):
                p.setPen(QPen(QColor('#35d8b0' if 'Left' in str(data['joint_names'][i]) else '#f2ba55'), 3))
                if parent >= 0: p.drawLine(round(xy[parent, 0]), round(xy[parent, 1]), round(xy[i, 0]), round(xy[i, 1]))
                p.drawEllipse(round(xy[i, 0])-3, round(xy[i, 1])-3, 6, 6)
    app = QApplication.instance() or QApplication([])
    font = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'msyh.ttc'
    if font.exists(): QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont('Microsoft YaHei', 10))
    window = Viewer(); window.show(); return app.exec()
