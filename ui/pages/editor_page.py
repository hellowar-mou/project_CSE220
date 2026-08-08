import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame
from PySide6.QtCore import Qt

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class EditorPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("✂️ Mini Audio Editor")
        title.setObjectName("SectionTitle")
        caption = QLabel("Trim, reverse, fade, and a convolution-based echo effect.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        controls = QHBoxLayout()
        self.delay_label = QLabel("Echo delay: 0.12s")
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setRange(2, 40)
        self.delay_slider.setValue(12)
        self.delay_slider.valueChanged.connect(self._on_change)

        self.decay_label = QLabel("Echo decay: 0.55")
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setRange(10, 90)
        self.decay_slider.setValue(55)
        self.decay_slider.valueChanged.connect(self._on_change)

        col1 = QVBoxLayout(); col1.addWidget(self.delay_label); col1.addWidget(self.delay_slider)
        col2 = QVBoxLayout(); col2.addWidget(self.decay_label); col2.addWidget(self.decay_slider)
        controls.addLayout(col1)
        controls.addLayout(col2)
        layout.addLayout(controls)

        self.canvas = MplCanvas(n_rows=3)
        layout.addWidget(self.canvas)

        players = QHBoxLayout()
        self.p_clip = AudioPlayButton("Trimmed clip")
        self.p_edit = AudioPlayButton("Joined w/ reverse")
        self.p_echo = AudioPlayButton("Echo (convolution)")
        players.addWidget(self.p_clip)
        players.addWidget(self.p_edit)
        players.addWidget(self.p_echo)
        players.addStretch()
        layout.addLayout(players)
        layout.addStretch()

        clean = dsp.make_voice_like()
        self.clip = dsp.fade(dsp.trim(clean, 0.0, 1.0))
        self.p_clip.set_audio(self.clip)
        self.edited = np.concatenate([self.clip, dsp.reverse(self.clip)])
        self.p_edit.set_audio(self.edited)
        self._on_change()

    def _on_change(self):
        delay = self.delay_slider.value() / 100.0
        decay = self.decay_slider.value() / 100.0
        self.delay_label.setText(f"Echo delay: {delay:.2f}s")
        self.decay_label.setText(f"Echo decay: {decay:.2f}")

        echoed = dsp.echo(self.clip, delay_s=delay, decay=decay)
        self.canvas.plot_waveforms(
            [(self.clip, "Clip"), (self.edited, "Clip+reverse"), (echoed, "Echo")],
            "Editor operations")
        self.p_echo.set_audio(echoed)
