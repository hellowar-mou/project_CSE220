import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame,
    QPushButton, QFileDialog
)
from PySide6.QtCore import Qt

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class EditorPage(QWidget):
    """UI/visualization layer only — trim/reverse/fade/echo below call the
    exact same dsp_core functions, unchanged, regardless of whether the
    source audio is the built-in demo or a user-uploaded file."""

    def __init__(self):
        super().__init__()
        self.source_audio = None
        self._build_ui()
        self._load_demo_signal()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("✂️ Editor")
        title.setObjectName("SectionTitle")
        caption = QLabel("Upload your own audio or use the demo clip • "
                          "Trim, reverse, fade, and a convolution-based echo effect.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        # ---- Input row: upload + demo
        input_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self.upload_btn)

        self.demo_btn = QPushButton("🔊 Load Demo Signal")
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.clicked.connect(self._load_demo_signal)
        input_row.addWidget(self.demo_btn)
        input_row.addStretch()
        layout.addLayout(input_row)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("Caption")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

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
        self.p_clip = AudioPlayButton("Input (trimmed clip)")
        self.p_edit = AudioPlayButton("Clip + reverse")
        self.p_echo = AudioPlayButton("Output (echo)")
        players.addWidget(self.p_clip)
        players.addWidget(self.p_edit)
        players.addWidget(self.p_echo)
        players.addStretch()
        layout.addLayout(players)
        layout.addStretch()

        # Live playback position marker, synced to whichever player is active
        for btn in (self.p_clip, self.p_edit, self.p_echo):
            btn.position_changed.connect(self.canvas.update_playhead)
            btn.playback_active_changed.connect(
                lambda active: None if active else self.canvas.clear_playhead())

    # ------------------------------------------------------------- input
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3)")
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            self.source_audio = audio
            self.file_label.setText(f"✅ {os.path.basename(path)}  "
                                     f"({orig_fs} Hz → {dsp.FS} Hz, {len(audio)/dsp.FS:.2f}s)")
            self._rebuild_from_source()
        except Exception as e:
            self.file_label.setText(f"❌ Error loading file: {e}")

    def _load_demo_signal(self):
        self.source_audio = dsp.make_voice_like()
        self.file_label.setText("🔊 Demo: synthetic voice-like tone (2.0s)")
        self._rebuild_from_source()

    def _rebuild_from_source(self):
        # Same original logic: trim to (up to) the first second, then fade —
        # only the trim window adapts to the source's actual duration.
        duration_s = len(self.source_audio) / dsp.FS
        trim_end = min(1.0, duration_s)
        self.clip = dsp.fade(dsp.trim(self.source_audio, 0.0, trim_end))
        self.p_clip.set_audio(self.clip)
        self.edited = np.concatenate([self.clip, dsp.reverse(self.clip)])
        self.p_edit.set_audio(self.edited)
        self._on_change()

    # ------------------------------------------------------------- update
    def _on_change(self):
        if not hasattr(self, "clip"):
            return
        delay = self.delay_slider.value() / 100.0
        decay = self.decay_slider.value() / 100.0
        self.delay_label.setText(f"Echo delay: {delay:.2f}s")
        self.decay_label.setText(f"Echo decay: {decay:.2f}")

        echoed = dsp.echo(self.clip, delay_s=delay, decay=decay)
        self.canvas.plot_waveforms(
            [(self.clip, "Input (clip)"), (self.edited, "Clip+reverse"), (echoed, "Output (echo)")],
            "Editor operations — live")
        self.p_echo.set_audio(echoed)
