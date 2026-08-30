"""
A lightweight, native animated wave background — three translucent wave layers
drifting slowly at different speeds, drawn with QPainter/QPainterPath and
advanced by a QTimer. No browser engine, no HTML — pure Qt.
Supports light/dark theming via set_theme() / the global theme system.
"""
import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QPainterPath, QColor, QLinearGradient

from ui import theme as theme_module


class WaveBackground(QWidget):
    def __init__(self, parent=None, mode=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._phase = 0.0
        self._mode = mode or theme_module.get_current_mode()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)  # ~25 fps, slow/gentle motion
        self._build_layers()
        theme_module.add_listener(self._on_global_theme_changed)

    def _build_layers(self):
        colors, bg = theme_module.get_wave_colors(self._mode)
        self._bg_colors = bg
        # (color, amplitude, wavelength_px, speed, y_fraction, opacity)
        self._layers = [
            (QColor(colors[0]), 18, 260, 0.35, 0.72, 70),
            (QColor(colors[1]), 14, 200, -0.22, 0.80, 60),
            (QColor(colors[2]), 22, 340, 0.15, 0.88, 45),
        ]

    def set_theme(self, mode):
        self._mode = mode
        self._build_layers()
        self.update()

    def _on_global_theme_changed(self):
        self.set_theme(theme_module.get_current_mode())

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
        grad.setColorAt(0.0, QColor(self._bg_colors[0]))
        grad.setColorAt(0.5, QColor(self._bg_colors[1]))
        grad.setColorAt(1.0, QColor(self._bg_colors[2]))
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
