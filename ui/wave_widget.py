"""
A lightweight, native animated wave background — three translucent wave layers
drifting slowly at different speeds, drawn with QPainter/QPainterPath and
advanced by a QTimer. No browser engine, no HTML — pure Qt.
"""
import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QPainterPath, QColor, QLinearGradient


class WaveBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)  # ~25 fps, slow/gentle motion

        # (color, amplitude, wavelength_px, speed, y_fraction, opacity)
        self._layers = [
            (QColor("#8ec5fc"), 18, 260, 0.35, 0.72, 70),
            (QColor("#a7e3e0"), 14, 200, -0.22, 0.80, 60),
            (QColor("#c9b6f2"), 22, 340, 0.15, 0.88, 45),
        ]

    def _tick(self):
        self._phase += 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # soft background gradient behind the waves
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor("#f4f9ff"))
        grad.setColorAt(0.5, QColor("#eef5fb"))
        grad.setColorAt(1.0, QColor("#e8f2fb"))
        painter.fillRect(self.rect(), grad)

        for color, amp, wavelength, speed, y_frac, alpha in self._layers:
            base_y = h * y_frac
            path = QPainterPath()
            path.moveTo(0, h)
            path.lineTo(0, base_y)
            step = 4
            x = 0
            while x <= w:
                y = base_y + amp * math.sin(
                    (x / wavelength) * 2 * math.pi + self._phase * speed * 0.05
                )
                path.lineTo(x, y)
                x += step
            path.lineTo(w, h)
            path.closeSubpath()

            fill_color = QColor(color)
            fill_color.setAlpha(alpha)
            painter.fillPath(path, fill_color)

        painter.end()
