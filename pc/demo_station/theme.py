"""Neutral desktop surfaces with system light/dark selection."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QLabel

LIGHT=dict(bg='#f3f3f3',panel='#ffffff',canvas='#fafafa',line='#d4d4d4',text='#242424',muted='#616161',
           cyan='#265ca3',blue='#64748b',orange='#9c5b13',red='#b3261e',grid='#e0e0e0',
           hover='#ebebeb',pressed='#dedede',selected='#e7edf5',disabled='#8a8a8a',control='#ffffff',on_accent='#ffffff',point='#265ca3')
DARK=dict(bg='#202020',panel='#292929',canvas='#252525',line='#484848',text='#f0f0f0',muted='#b6b6b6',
          cyan='#9bbced',blue='#acacac',orange='#e3b773',red='#f2a4a0',grid='#3b3b3b',
          hover='#383838',pressed='#454545',selected='#3b4350',disabled='#858585',control='#303030',on_accent='#202020',point='#b9cfed')
COLORS=dict(LIGHT)


def stylesheet():
    c=COLORS
    return f'''
QMainWindow,QDialog {{background:{c['bg']};color:{c['text']};}}
QWidget {{font-family:"Microsoft YaHei UI","Segoe UI";font-size:12px;color:{c['text']};}}
QLabel {{background:transparent;}}
QFrame#sidebar {{background:{c['bg']};border-right:1px solid {c['line']};}}
QFrame#card,QFrame#metric,QFrame#transport {{background:{c['panel']};border:1px solid {c['line']};border-radius:6px;}}
QPushButton {{background:{c['control']};border:1px solid {c['line']};border-radius:4px;padding:9px 12px;color:{c['text']};}}
QPushButton:hover {{background:{c['hover']};border-color:{c['muted']};}}
QPushButton:pressed {{background:{c['pressed']};}}
QPushButton:checked {{background:{c['selected']};color:{c['cyan']};border-color:{c['cyan']};}}
QPushButton:disabled {{color:{c['disabled']};border-color:{c['line']};background:{c['bg']};}}
QPushButton#primary {{background:{c['cyan']};color:{c['on_accent']};border-color:{c['cyan']};font-weight:600;}}
QPushButton#primary:hover {{background:{c['text']};color:{c['panel']};}}
QPushButton#danger {{color:{c['red']};}}
QPushButton#danger:disabled {{color:{c['disabled']};}}
QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox {{background:{c['control']};border:1px solid {c['line']};border-radius:4px;padding:7px;min-height:20px;}}
QComboBox QAbstractItemView {{background:{c['panel']};color:{c['text']};selection-background-color:{c['selected']};selection-color:{c['text']};}}
QToolTip {{background:{c['panel']};color:{c['text']};border:1px solid {c['line']};padding:6px;}}
QSlider::groove:horizontal {{height:4px;background:{c['line']};}}
QSlider::sub-page:horizontal {{background:{c['cyan']};}}
QSlider::handle:horizontal {{width:14px;margin:-5px 0;background:{c['cyan']};border-radius:7px;}}
QScrollArea {{border:none;background:transparent;}}
QTextEdit,QPlainTextEdit {{background:{c['canvas']};border:1px solid {c['line']};border-radius:3px;}}
QCheckBox {{spacing:7px;padding:4px;}}
QListWidget {{background:{c['bg']};border:none;outline:none;}}
QListWidget::item {{padding:8px 5px;border-bottom:1px solid {c['line']};}}
QListWidget::item:selected {{background:{c['selected']};color:{c['text']};border-left:3px solid {c['cyan']};}}
QListWidget::item:hover {{background:{c['hover']};}}
QPushButton:focus,QComboBox:focus,QLineEdit:focus {{border:1px solid {c['cyan']};}}
QScrollBar:vertical {{background:{c['bg']};width:8px;}}
QScrollBar::handle:vertical {{background:{c['line']};min-height:22px;}}
'''


def label_style(widget):
    role=widget.property('themeRole')
    if role:
        widget.setStyleSheet(f"color:{COLORS.get(role,role)};font-size:{widget.property('typeSize')}px;font-weight:{widget.property('typeWeight')};")


def apply_theme(app,preference='system'):
    dark=preference=='dark' or (preference=='system' and app.styleHints().colorScheme()==Qt.ColorScheme.Dark)
    COLORS.clear();COLORS.update(DARK if dark else LIGHT)
    app.setStyleSheet(stylesheet())
    for widget in app.allWidgets():
        if isinstance(widget,QLabel):label_style(widget)
        widget.update()
    return dark
