import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame,
    QPushButton, QFileDialog, QTabWidget
)
from PySide6.QtCore import Qt

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton, MicRecordWidget


class EqualizerPage(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self._build_ui()
        self._load_demo_signal()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🎚️ 3-Band Equalizer")
        title.setObjectName("SectionTitle")
        caption = QLabel("Upload audio or record from mic • FIR band-split filters, "
                          "recombined via superposition • Live filter response as you move sliders")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        # ---- Input row: upload, mic, demo
        input_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self.upload_btn)

        self.mic_widget = MicRecordWidget()
        self.mic_widget.recording_ready.connect(self._on_recording_ready)
        input_row.addWidget(self.mic_widget)

        self.demo_btn = QPushButton("🔊 Load Demo Signal")
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.clicked.connect(self._load_demo_signal)
        input_row.addWidget(self.demo_btn)
        layout.addLayout(input_row)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("Caption")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        # ---- Gain sliders
        controls = QHBoxLayout()
        self.sliders = {}
        self.labels = {}
        for band in ["Low", "Mid", "High"]:
            col = QVBoxLayout()
            lab = QLabel(f"{band} gain: 1.0")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 20)
            sld.setValue(10)
            sld.valueChanged.connect(self._on_change)
            col.addWidget(lab)
            col.addWidget(sld)
            controls.addLayout(col)
            self.sliders[band] = sld
            self.labels[band] = lab
        layout.addLayout(controls)

        # ---- Visualization tabs: waveform in/out, band spectra, live filter response
        self.tabs = QTabWidget()
        self.wave_canvas = MplCanvas(n_rows=2)
        self.spec_canvas = MplCanvas(n_rows=3)
        self.response_canvas = MplCanvas(n_rows=1)
        self.tabs.addTab(self.wave_canvas, "Input / Output Waveform")
        self.tabs.addTab(self.spec_canvas, "Band-Split Spectra")
        self.tabs.addTab(self.response_canvas, "Live Filter Response")
        layout.addWidget(self.tabs)

        players = QHBoxLayout()
        self.p_orig = AudioPlayButton("Original")
        self.p_eq = AudioPlayButton("EQ'd")
        players.addWidget(self.p_orig)
        players.addWidget(self.p_eq)
        players.addStretch()
        layout.addLayout(players)
        layout.addStretch()

        # Live playback position marker, synced to whichever player is active
        self.p_orig.position_changed.connect(self.wave_canvas.update_playhead)
        self.p_eq.position_changed.connect(self.wave_canvas.update_playhead)
        self.p_orig.playback_active_changed.connect(
            lambda active: None if active else self.wave_canvas.clear_playhead())
        self.p_eq.playback_active_changed.connect(
            lambda active: None if active else self.wave_canvas.clear_playhead())

    # ---------------------------------------------------------- input handlers
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3)")
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            self.audio_data = audio
            self.file_label.setText(f"✅ {os.path.basename(path)}  "
                                     f"({orig_fs} Hz → {dsp.FS} Hz, {len(audio)/dsp.FS:.2f}s)")
            self._on_new_audio()
        except Exception as e:
            self.file_label.setText(f"❌ Error loading file: {e}")

    def _on_recording_ready(self, audio, fs):
        self.audio_data = dsp.normalize(audio)
        self.file_label.setText(f"🎙️ Recorded {len(audio)/fs:.2f}s at {fs} Hz")
        self._on_new_audio()

    def _load_demo_signal(self):
        self.audio_data = dsp.make_voice_like()
        self.file_label.setText("🔊 Demo: synthetic voice-like tone (2.0s)")
        self._on_new_audio()

    def _on_new_audio(self):
        self.p_orig.set_audio(self.audio_data)
        self._on_change()

    # ---------------------------------------------------------- live update
    def _on_change(self):
        if self.audio_data is None:
            return
        gains = {}
        for band, sld in self.sliders.items():
            g = sld.value() / 10.0
            gains[band] = g
            self.labels[band].setText(f"{band} gain: {g:.1f}")
        gain_tuple = (gains["Low"], gains["Mid"], gains["High"])

        eq = dsp.equalize(self.audio_data, gains=gain_tuple)
        self.p_eq.set_audio(eq)

        # Input/output waveform (live)
        self.wave_canvas.plot_waveforms(
            [(self.audio_data, "Input"), (eq, "Output (EQ'd)")], "Input vs. Output")

        # Band-split spectra of the *input* signal
        lo, mid, hi = dsp.band_split(self.audio_data)
        self.spec_canvas.plot_spectra(
            [(lo, "Low"), (mid, "Mid"), (hi, "High")], "Band-split spectra of input")

        # Live combined filter frequency response at current slider gains —
        # this is the "what is the EQ doing right now" curve.
        self.response_canvas.fig.clear()
        ax = self.response_canvas.fig.subplots(1, 1)
        w, h_db = dsp.equalizer_frequency_response(gains=gain_tuple)
        ax.plot(w, h_db, color="#2f6690", linewidth=1.2)
        ax.set_xlim(0, 4000)
        ax.set_ylim(-40, 20)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("Gain (dB)", fontsize=8)
        ax.set_title(f"Combined EQ Response  (L{gain_tuple[0]:.1f} / "
                     f"M{gain_tuple[1]:.1f} / H{gain_tuple[2]:.1f})", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        self.response_canvas.fig.tight_layout()
        self.response_canvas.draw()
