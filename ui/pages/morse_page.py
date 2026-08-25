import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFrame,
    QPushButton, QFileDialog, QTabWidget, QToolButton, QMenu, QSlider
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton, MicRecordWidget


class MorsePage(QWidget):
    """Morse Code Converter — encode (text -> tones) and decode (tones ->
    text) in one page, with a built-in toolbox of sample clips and full
    live visualization of every decoding stage. Auto tone-frequency and
    speed detection (dsp.detect_tone_freq / detect_unit_duration) are the
    logic additions that let decoding work on arbitrary uploaded, recorded,
    or toolbox audio — this feature is explicitly allowed to change logic."""

    def __init__(self):
        super().__init__()
        self.toolbox = dsp.build_morse_toolbox()
        self.decode_source_audio = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QHBoxLayout(header)
        title_col = QVBoxLayout()
        title = QLabel("📡 Morse Code Converter")
        title.setObjectName("SectionTitle")
        caption = QLabel("Encode text to Morse tones, or decode a tone recording back to text. "
                          "Tone pitch and speed are auto-detected, so decoding works on any "
                          "uploaded, recorded, or toolbox clip.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(caption)
        h_layout.addLayout(title_col, 1)

        # ---- Toolbox icon, directly on the page, available from both tabs
        self.toolbox_btn = QToolButton()
        self.toolbox_btn.setText("🧰 Toolbox")
        self.toolbox_btn.setPopupMode(QToolButton.InstantPopup)
        self.toolbox_btn.setObjectName("SecondaryButton")
        menu = QMenu(self.toolbox_btn)
        for label in self.toolbox:
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, l=label: self._load_from_toolbox(l))
            menu.addAction(action)
        self.toolbox_btn.setMenu(menu)
        h_layout.addWidget(self.toolbox_btn)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_encode_tab(), "Encode (Text → Audio)")
        self.tabs.addTab(self._build_decode_tab(), "Decode (Audio → Text)")
        layout.addWidget(self.tabs)

        self._on_encode_change()

    # ------------------------------------------------------------- Encode tab
    def _build_encode_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.encode_input = QLineEdit("SOS HELP")
        self.encode_input.setPlaceholderText("Message to encode")
        self.encode_input.textChanged.connect(self._on_encode_change)
        layout.addWidget(self.encode_input)

        tone_row = QHBoxLayout()
        self.tone_label = QLabel("Tone pitch: 700 Hz")
        self.tone_slider = QSlider(Qt.Horizontal)
        self.tone_slider.setRange(300, 1200)
        self.tone_slider.setValue(700)
        self.tone_slider.valueChanged.connect(self._on_encode_change)
        tone_row.addWidget(self.tone_label)
        tone_row.addWidget(self.tone_slider)
        layout.addLayout(tone_row)

        self.encode_canvas = MplCanvas(n_rows=1)
        layout.addWidget(self.encode_canvas)

        row = QHBoxLayout()
        self.encode_player = AudioPlayButton("Generated Morse tone recording")
        row.addWidget(self.encode_player)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()

        self.encode_player.position_changed.connect(self.encode_canvas.update_playhead)
        self.encode_player.playback_active_changed.connect(
            lambda active: None if active else self.encode_canvas.clear_playhead())
        return tab

    def _on_encode_change(self):
        message = self.encode_input.text().strip()
        tone_freq = self.tone_slider.value()
        self.tone_label.setText(f"Tone pitch: {tone_freq} Hz")
        if not message:
            return
        audio, unit = dsp.synth_morse(message, tone_freq=tone_freq)
        self.encode_canvas.plot_waveforms(
            [(audio, "Generated Morse tone")], f"Encoding: '{message.upper()}'")
        self.encode_player.set_audio(audio)

    # ------------------------------------------------------------- Decode tab
    def _build_decode_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        input_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self.upload_btn)

        self.mic_widget = MicRecordWidget()
        self.mic_widget.recording_ready.connect(self._on_recording_ready)
        input_row.addWidget(self.mic_widget, 1)
        layout.addLayout(input_row)

        self.decode_source_label = QLabel(
            "Use the 🧰 Toolbox above, upload a file, or record from mic to decode.")
        self.decode_source_label.setObjectName("Caption")
        self.decode_source_label.setWordWrap(True)
        layout.addWidget(self.decode_source_label)

        self.decode_canvas = MplCanvas(n_rows=4, figsize=(7, 1.5))
        layout.addWidget(self.decode_canvas)

        row = QHBoxLayout()
        self.decode_player = AudioPlayButton("Decoding input")
        row.addWidget(self.decode_player)
        row.addStretch()
        layout.addLayout(row)

        metrics = QHBoxLayout()
        self.detected_label = QLabel("Detected tone: — Hz, unit: — s")
        self.detected_label.setObjectName("Caption")
        self.decoded_label = QLabel("Decoded text: —")
        self.decoded_label.setObjectName("SectionTitle")
        metrics.addWidget(self.detected_label)
        metrics.addWidget(self.decoded_label)
        metrics.addStretch()
        layout.addLayout(metrics)
        layout.addStretch()

        self.decode_player.position_changed.connect(self.decode_canvas.update_playhead)
        self.decode_player.playback_active_changed.connect(
            lambda active: None if active else self.decode_canvas.clear_playhead())
        return tab

    # ------------------------------------------------------------- decode inputs
    def _load_from_toolbox(self, label):
        item = self.toolbox[label]
        self.decode_source_audio = item["audio"]
        self.decode_source_label.setText(
            f"🧰 Toolbox clip: '{label}'  (expected message: {item['message']})")
        self.tabs.setCurrentIndex(1)  # jump to Decode tab
        self._run_decode()

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3)")
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            self.decode_source_audio = audio
            self.decode_source_label.setText(
                f"✅ {os.path.basename(path)}  ({orig_fs} Hz → {dsp.FS} Hz, {len(audio)/dsp.FS:.2f}s)")
            self._run_decode()
        except Exception as e:
            self.decode_source_label.setText(f"❌ Error loading file: {e}")

    def _on_recording_ready(self, audio, fs):
        self.decode_source_audio = dsp.normalize(audio)
        self.decode_source_label.setText(f"🎙️ Recorded {len(audio)/fs:.2f}s at {fs} Hz")
        self._run_decode()

    # ------------------------------------------------------------- decode + viz
    def _run_decode(self):
        if self.decode_source_audio is None:
            return
        audio = self.decode_source_audio
        decoded, filtered, envelope, keyed, det_freq, det_unit = dsp.decode_morse(audio)

        self.decode_canvas.plot_waveforms(
            [(audio, "Input"), (filtered, "Bandpassed"),
             (envelope, "Envelope"), (keyed.astype(float), "Dots/Dashes")],
            f"Decoding pipeline — result: '{decoded}'")
        self.decode_player.set_audio(audio)
        self.detected_label.setText(
            f"Detected tone: {det_freq:.0f} Hz, unit: {det_unit:.3f} s")
        self.decoded_label.setText(f"Decoded text: {decoded if decoded.strip() else '(none detected)'}")
