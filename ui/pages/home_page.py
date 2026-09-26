import math
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QFrame, QHBoxLayout,
    QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPainterPath, QColor, QCursor

from ui import theme as theme_module
from ui.widgets import FirstRunWalkthrough


class WaveAccentLine(QWidget):
    """Subtle animated wave banner painted at the top of each module card."""

    def __init__(self, color_hex="#6fb1ea", phase_offset=0.0, parent=None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self.color_hex = color_hex
        self.phase_offset = phase_offset
        self.phase = 0.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_phase(self, phase):
        self.phase = phase + self.phase_offset
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        color = QColor(self.color_hex)
        color.setAlpha(160)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)

        path = QPainterPath()
        path.moveTo(0, h)
        wavelength = max(80.0, w / 2.5)
        amp = 3.5

        for x in range(0, w + 4, 4):
            y = (h / 2.0) + amp * math.sin((2 * math.pi * x / wavelength) + self.phase)
            path.lineTo(x, y)

        path.lineTo(w, h)
        path.closeSubpath()
        painter.drawPath(path)


class WavyFloatingCard(QWidget):
    """Large interactive module card with animated wavy floating motion."""

    def __init__(self, title, badge, desc, primitive, icon, nav_name, nav_callback=None, card_index=0, parent=None):
        super().__init__(parent)
        self.nav_name = nav_name
        self.nav_callback = nav_callback
        self.card_index = card_index
        self.phase_offset = card_index * (2.0 * math.pi / 5.0)

        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Inner frame that floats up and down
        self.inner = QFrame(self)
        self.inner.setObjectName("ModuleFloatingCard")
        self.inner.setCursor(QCursor(Qt.PointingHandCursor))
        self.inner.setStyleSheet(
            "QFrame#ModuleFloatingCard {"
            "  background: rgba(255, 255, 255, 0.88);"
            "  border-radius: 18px;"
            "  border: 1.5px solid rgba(111, 177, 234, 0.32);"
            "}"
            "QFrame#ModuleFloatingCard:hover {"
            "  background: rgba(255, 255, 255, 0.98);"
            "  border: 1.8px solid #6fb1ea;"
            "}"
        )

        inner_layout = QVBoxLayout(self.inner)
        inner_layout.setContentsMargins(18, 12, 18, 16)
        inner_layout.setSpacing(10)

        # Wave accent line at top of card
        accent_colors = ["#6fb1ea", "#8fd3c7", "#c9b6f2", "#f1c40f", "#e74c3c"]
        accent_color = accent_colors[card_index % len(accent_colors)]
        self.wave_accent = WaveAccentLine(accent_color, phase_offset=self.phase_offset, parent=self.inner)
        inner_layout.addWidget(self.wave_accent)

        # Top row: Icon + Title + Badge
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet(
            f"font-size: 26px; background: rgba(111, 177, 234, 0.16); border-radius: 12px; "
            f"min-width: 46px; max-width: 46px; min-height: 46px; max-height: 46px;"
        )
        icon_label.setAlignment(Qt.AlignCenter)
        top_row.addWidget(icon_label)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 1.08rem; font-weight: 800; color: #1c2a38;")

        badge_lbl = QLabel(badge)
        badge_lbl.setStyleSheet(
            "background: rgba(111, 177, 234, 0.18); color: #2f6690; font-size: 0.72rem; "
            "font-weight: 700; border-radius: 9px; padding: 2px 8px; letter-spacing: 0.02em;"
        )
        badge_lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        title_box.addWidget(title_lbl)
        title_box.addWidget(badge_lbl)
        top_row.addLayout(title_box, 1)

        inner_layout.addLayout(top_row)

        # Description text
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("Caption")
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 0.86rem; line-height: 1.45; color: #4a5d73;")
        inner_layout.addWidget(desc_lbl, 1)

        # Bottom row: DSP primitive pill + Launch button
        bot_row = QHBoxLayout()
        pill = QLabel(primitive)
        pill.setStyleSheet(
            "background: rgba(143, 211, 199, 0.22); color: #1e7062; font-size: 0.73rem; "
            "font-weight: 600; border-radius: 10px; padding: 3px 9px;"
        )
        bot_row.addWidget(pill)
        bot_row.addStretch()

        open_btn = QPushButton("Open Module →")
        open_btn.setObjectName("SecondaryButton")
        open_btn.setCursor(QCursor(Qt.PointingHandCursor))
        open_btn.setStyleSheet(
            "QPushButton { background: rgba(111, 177, 234, 0.16); color: #2f6690; font-size: 0.78rem; "
            "font-weight: 700; border-radius: 8px; padding: 5px 12px; border: 1px solid rgba(111, 177, 234, 0.35); }"
            "QPushButton:hover { background: #6fb1ea; color: white; }"
        )
        open_btn.clicked.connect(self._on_click)
        bot_row.addWidget(open_btn)

        inner_layout.addLayout(bot_row)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_click()
        super().mousePressEvent(event)

    def _on_click(self):
        if self.nav_callback and self.nav_name:
            self.nav_callback(self.nav_name)

    def update_float(self, global_phase):
        # Continuous sinusoidal wavy floating displacement
        amp = 5.0
        y_offset = amp * math.sin(global_phase + self.phase_offset)
        w = self.width()
        # Keep inner card height well-proportioned
        target_h = max(180, self.height() - 14)
        self.inner.setGeometry(0, int(round(7.0 + y_offset)), w, target_h)
        self.wave_accent.set_phase(global_phase)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        target_h = max(180, self.height() - 14)
        self.inner.setGeometry(0, 7, w, target_h)


class HomePage(QWidget):
    def __init__(self, nav_callback=None):
        super().__init__()
        self.nav_callback = nav_callback
        self._phase = 0.0

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 8, 20)

        # Header Hero Card
        header = QFrame()
        header.setObjectName("HomeHeroCard")
        header.setStyleSheet(
            "QFrame#HomeHeroCard { background: rgba(255, 255, 255, 0.88); border-radius: 18px; "
            "border: 1.5px solid rgba(111, 177, 234, 0.32); }"
        )
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(24, 20, 24, 20)
        h_layout.setSpacing(12)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("🌊 ReSonus")
        title.setObjectName("TitleLabel")
        title.setStyleSheet("font-size: 1.85rem; font-weight: 800; color: #1c2a38;")
        title_row.addWidget(title)

        version_badge = QLabel("CSE220 Signals & Linear Systems Demo")
        version_badge.setStyleSheet(
            "background: rgba(111, 177, 234, 0.16); color: #2f6690; font-size: 0.75rem; "
            "font-weight: 700; border-radius: 12px; padding: 4px 12px; letter-spacing: 0.03em;"
        )
        title_row.addWidget(version_badge)
        title_row.addStretch()
        h_layout.addLayout(title_row)

        desc = QLabel(
            "One unified digital signal processing engine powering five live audio applications. "
            "Explore non-destructive temporal editing, interactive 9-band equalization, surgical "
            "spectral denoising, Morse code carrier modulation, and pattern cross-correlation — "
            "executed in real time with live mathematical visualizations."
        )
        desc.setWordWrap(True)
        desc.setObjectName("Caption")
        desc.setStyleSheet("font-size: 0.92rem; line-height: 1.6; color: #43586d;")
        h_layout.addWidget(desc)

        # Pillar Pills
        pills_row = QHBoxLayout()
        pills_row.setSpacing(8)
        pillars = [
            ("🔄 Convolution", "#e7f3ff", "#2f6690"),
            ("📊 Fourier Transform (FFT/STFT)", "#eafaf1", "#1e7062"),
            ("📈 Cross-Correlation", "#f5eef8", "#6c3483"),
            ("⚡ LTI Systems & Biquads", "#fef9e7", "#9a7d0a")
        ]
        for text, bg, fg in pillars:
            p = QLabel(text)
            p.setStyleSheet(
                f"background: {bg}; color: {fg}; font-size: 0.76rem; font-weight: 700; "
                f"border-radius: 10px; padding: 4px 10px;"
            )
            pills_row.addWidget(p)
        pills_row.addStretch()
        h_layout.addLayout(pills_row)

        layout.addWidget(header)

        # Section Header
        sec_header = QHBoxLayout()
        sec_title = QLabel("Toolbox Modules — Select a Module to Launch")
        sec_title.setStyleSheet("font-size: 1.05rem; font-weight: 800; color: #2c3e50; margin-top: 4px;")
        sec_header.addWidget(sec_title)
        sec_header.addStretch()
        layout.addLayout(sec_header)

        # Large Wavy Floating Cards Grid
        self.cards_grid = QGridLayout()
        self.cards_grid.setSpacing(16)
        self.cards = []

        modules_data = [
            (
                "Noise Remover",
                "4 Methods · STFT & Wiener",
                "Suppress AC hum, broadband hiss, and non-stationary ambient noise using moving average, "
                "frequency-domain notch filters, spectral subtraction, and optimal Wiener estimation.",
                "Convolution · FFT · STFT",
                "🔇",
                "Noise Remover"
            ),
            (
                "Equalizer",
                "3 Architectures · 9-Band",
                "Shape tonal spectral balance with 3-band windowed-sinc FIR filters, professional 9-band "
                "IIR peaking biquad cascades (RBJ cookbook), and hybrid low/high shelving filters.",
                "FIR Sinc · IIR Peaking · Cascades",
                "🎛️",
                "Equalizer"
            ),
            (
                "Audio Editor",
                "7 Operations · OLA Time Stretch",
                "Perform precision temporal audio operations: array trimming, concatenation with peak "
                "normalization, time reversal, Overlap-Add (OLA) speed change, fades, and LTI room echo.",
                "Array Slicing · OLA · LTI Echo",
                "✂️",
                "Editor"
            ),
            (
                "Morse Code Converter",
                "OOK · 3 Envelope Extractors",
                "Encode text into international Morse timing with 700 Hz carrier synthesis, and decode "
                "audio via auto-pitch detection and 3 envelope extractors: Hilbert, Rectifier, and RMS energy.",
                "Hilbert · Butterworth · Keying",
                "📡",
                "Morse Code Converter"
            ),
            (
                "Audio Matcher",
                "Pattern · Shazam Fingerprint",
                "Locate acoustic events with matched-filter cross-correlation, compute volume-invariant "
                "L2-normalized clip similarity, and search songs with 2D STFT peak constellation fingerprinting.",
                "Cross-Correlation · STFT Fingerprint",
                "🎯",
                "Audio Matcher"
            ),
        ]

        for i, (title, badge, desc, prim, icon, nav_name) in enumerate(modules_data):
            card = WavyFloatingCard(
                title=title,
                badge=badge,
                desc=desc,
                primitive=prim,
                icon=icon,
                nav_name=nav_name,
                nav_callback=self.nav_callback,
                card_index=i,
                parent=self
            )
            self.cards.append(card)
            # 2 columns for top 4, last card spans or 3 columns
            if i < 3:
                self.cards_grid.addWidget(card, 0, i)
            else:
                self.cards_grid.addWidget(card, 1, i - 3, 1, 1 if i == 3 else 2)

        layout.addLayout(self.cards_grid)
        layout.addStretch()

        # Timer for synchronized wavy floating animation
        self._float_timer = QTimer(self)
        self._float_timer.timeout.connect(self._tick_floating)
        self._float_timer.start(33)  # ~30 fps

    def _tick_floating(self):
        self._phase += 0.045
        for card in self.cards:
            card.update_float(self._phase)
