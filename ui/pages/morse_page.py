import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFrame,
    QPushButton, QFileDialog, QTabWidget, QComboBox, QSlider, QDialog,
    QScrollArea, QApplication, QCheckBox
)
from PySide6.QtCore import Qt, QTimer

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioTransportWidget, MicRecordWidget


SEGMENT_COLORS = {
    "dot": "#6fb1ea", "dash": "#2f6690", "gap": "#dbe8f4",
    "char_gap": "#f0c987", "word_gap": "#e67e22",
}


class ToolboxDialog(QDialog):
    """Floating drawer of predetermined Morse audio clips — Preview and
    Use as Input, per the Morse page's built-in toolbox requirement."""

    def __init__(self, toolbox, on_use, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🧰 Morse Audio Toolbox")
        self.resize(480, 500)
        self.toolbox = toolbox
        self.on_use = on_use
        self._preview_transport = AudioTransportWidget()

        layout = QVBoxLayout(self)
        intro = QLabel("Predetermined Morse clips — preview or use directly as decoder input.")
        intro.setObjectName("Caption")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        for label, item in toolbox.items():
            row_frame = QFrame()
            row_frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.6); border-radius: 8px; }")
            row = QVBoxLayout(row_frame)
            name_lab = QLabel(f"📻 {label}")
            name_lab.setStyleSheet("font-weight: 700;")
            row.addWidget(name_lab)
            info_lab = QLabel(f"Expected: {item['message']}  •  Tone: {item['tone_freq']}Hz  •  "
                               f"Unit: {item['unit']*1000:.0f}ms")
            info_lab.setObjectName("Caption")
            row.addWidget(info_lab)
            btn_row = QHBoxLayout()
            preview_btn = QPushButton("▶ Preview")
            preview_btn.setObjectName("PlayButton")
            preview_btn.clicked.connect(lambda checked=False, a=item["audio"]: self._preview(a))
            use_btn = QPushButton("✓ Use as Input")
            use_btn.setObjectName("SecondaryButton")
            use_btn.clicked.connect(lambda checked=False, l=label: self._use(l))
            btn_row.addWidget(preview_btn)
            btn_row.addWidget(use_btn)
            row.addLayout(btn_row)
            content_layout.addWidget(row_frame)
        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addWidget(self._preview_transport)

    def _preview(self, audio):
        self._preview_transport.set_audio(audio)
        self._preview_transport.play()

    def _use(self, label):
        self.on_use(label)
        self.accept()


class MorsePage(QWidget):
    """Morse Code Converter — encode (text -> tones) and decode (tones ->
    text) with a choice of 3 tone-detection algorithms, a built-in toolbox,
    a timeline of classified dot/dash/gap segments, live progressive
    decoding, manual Morse editing, and a Normal/Analysis debug mode."""

    def __init__(self):
        super().__init__()
        self.toolbox = dsp.build_morse_toolbox()
        self.decode_source_audio = None
        self.current_segments = []
        self.current_keyed = None
        self.current_unit = None
        self.current_tone_freq = None
        self._progressive_timer = QTimer(self)
        self._progressive_timer.timeout.connect(self._reveal_next_segment)
        self._reveal_index = 0
        self.analysis_mode = False
        self._build_ui()

    # ================================================================= UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QHBoxLayout(header)
        title_col = QVBoxLayout()
        title = QLabel("📡 Morse Code Converter")
        title.setObjectName("SectionTitle")
        caption = QLabel("Encode text to Morse tones, or decode a recording back to text. "
                          "Choose a detection algorithm; tone pitch and speed auto-detect.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(caption)
        h_layout.addLayout(title_col, 1)

        self.toolbox_btn = QPushButton("🧰 Toolbox")
        self.toolbox_btn.setObjectName("SecondaryButton")
        self.toolbox_btn.clicked.connect(self._open_toolbox)
        h_layout.addWidget(self.toolbox_btn)

        self.mode_toggle = QPushButton("🔬 Analysis Mode: OFF")
        self.mode_toggle.setObjectName("SecondaryButton")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setToolTip(
            "Analysis Mode exposes every internal decoding stage:\n"
            "Raw Audio → Envelope → Threshold Detection → Segmentation → "
            "Classification → Grouping → Text")
        self.mode_toggle.clicked.connect(self._toggle_analysis_mode)
        h_layout.addWidget(self.mode_toggle)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_encode_tab(), "Encode (Text → Audio)")
        self.tabs.addTab(self._build_decode_tab(), "Decode (Audio → Text)")
        layout.addWidget(self.tabs)

        self._on_encode_change()

    # ------------------------------------------------------------- Encode
    def _build_encode_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        self.encode_input = QLineEdit("SOS HELP")
        self.encode_input.setPlaceholderText("Message to encode")
        self.encode_input.textChanged.connect(self._on_encode_change)
        layout.addWidget(self.encode_input)

        morse_row = QHBoxLayout()
        morse_row.addWidget(QLabel("Morse:"))
        self.morse_display = QLineEdit()
        self.morse_display.setToolTip("Auto-generated from your text. Edit it directly to override.")
        self.morse_display.editingFinished.connect(self._on_morse_manually_edited)
        morse_row.addWidget(self.morse_display, 1)
        layout.addLayout(morse_row)

        controls_grid = QHBoxLayout()
        tone_col = QVBoxLayout()
        self.tone_label = QLabel("Tone pitch: 700 Hz")
        self.tone_slider = QSlider(Qt.Horizontal)
        self.tone_slider.setRange(300, 1200)
        self.tone_slider.setValue(700)
        self.tone_slider.valueChanged.connect(self._on_encode_change)
        tone_col.addWidget(self.tone_label)
        tone_col.addWidget(self.tone_slider)
        controls_grid.addLayout(tone_col)

        wpm_col = QVBoxLayout()
        self.wpm_label = QLabel("Speed: 15 WPM")
        self.wpm_slider = QSlider(Qt.Horizontal)
        self.wpm_slider.setRange(5, 40)
        self.wpm_slider.setValue(15)
        self.wpm_slider.valueChanged.connect(self._on_encode_change)
        wpm_col.addWidget(self.wpm_label)
        wpm_col.addWidget(self.wpm_slider)
        controls_grid.addLayout(wpm_col)

        vol_col = QVBoxLayout()
        self.volume_label = QLabel("Volume: 90%")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(10, 100)
        self.volume_slider.setValue(90)
        self.volume_slider.valueChanged.connect(self._on_encode_change)
        vol_col.addWidget(self.volume_label)
        vol_col.addWidget(self.volume_slider)
        controls_grid.addLayout(vol_col)

        fade_col = QVBoxLayout()
        self.fade_label = QLabel("Fade: 5 ms")
        self.fade_slider = QSlider(Qt.Horizontal)
        self.fade_slider.setRange(0, 30)
        self.fade_slider.setValue(5)
        self.fade_slider.valueChanged.connect(self._on_encode_change)
        fade_col.addWidget(self.fade_label)
        fade_col.addWidget(self.fade_slider)
        controls_grid.addLayout(fade_col)
        layout.addLayout(controls_grid)

        gen_row = QHBoxLayout()
        self.generate_btn = QPushButton("⚡ Generate Audio")
        self.generate_btn.setObjectName("PrimaryButton")
        self.generate_btn.clicked.connect(self._on_generate_audio)
        gen_row.addWidget(self.generate_btn)
        self.encode_status = QLabel("")
        self.encode_status.setObjectName("Caption")
        gen_row.addWidget(self.encode_status)
        gen_row.addStretch()
        layout.addLayout(gen_row)

        self.encode_canvas = MplCanvas(n_rows=1, figsize=(7, 1.6))
        layout.addWidget(self.encode_canvas)

        row = QHBoxLayout()
        self.encode_transport = AudioTransportWidget()
        row.addWidget(self.encode_transport)
        self.download_encoded_btn = QPushButton("⬇ Download Audio")
        self.download_encoded_btn.setObjectName("SecondaryButton")
        self.download_encoded_btn.clicked.connect(self._on_download_encoded)
        row.addWidget(self.download_encoded_btn)
        layout.addLayout(row)
        layout.addStretch()

        self.encode_transport.position_changed.connect(self.encode_canvas.update_playhead)
        self.encode_transport.playback_active_changed.connect(
            lambda active: None if active else self.encode_canvas.clear_playhead())
        return tab

    def _wpm_to_unit(self, wpm):
        # PARIS-standard WPM formula: unit(s) = 1.2 / WPM
        return 1.2 / wpm

    def _on_encode_change(self):
        message = self.encode_input.text().strip()
        tone_freq = self.tone_slider.value()
        wpm = self.wpm_slider.value()
        volume = self.volume_slider.value() / 100.0
        fade_ms = self.fade_slider.value()
        self.tone_label.setText(f"Tone pitch: {tone_freq} Hz")
        self.wpm_label.setText(f"Speed: {wpm} WPM")
        self.volume_label.setText(f"Volume: {int(volume*100)}%")
        self.fade_label.setText(f"Fade: {fade_ms} ms")

        self.morse_display.blockSignals(True)
        self.morse_display.setText(dsp.text_to_morse_string(message))
        self.morse_display.blockSignals(False)

        if not message:
            return
        unit = self._wpm_to_unit(wpm)
        self._last_audio, _ = dsp.synth_morse(message, tone_freq=tone_freq, unit=unit,
                                               volume=volume, fade_ms=fade_ms, noise_amp=0.0)
        self.encode_canvas.plot_waveforms(
            [(self._last_audio, "Generated tone")], f"Encoding: '{message.upper()}'")
        self.encode_transport.set_audio(self._last_audio)

    def _on_morse_manually_edited(self):
        morse_str = self.morse_display.text().strip()
        if not morse_str:
            return
        text = dsp.morse_string_to_text(morse_str)
        self.encode_input.blockSignals(True)
        self.encode_input.setText(text)
        self.encode_input.blockSignals(False)
        self._on_encode_change()

    def _on_generate_audio(self):
        # Staged animation: Text -> Morse Symbols -> Tone Generation -> Waveform
        stages = ["Text \u2713", "Morse Symbols \u2713", "Generating Tones...", "Waveform Ready \u2713"]
        self._run_generate_stage(stages, 0)

    def _run_generate_stage(self, stages, i):
        self.encode_status.setText(stages[i])
        if i < len(stages) - 1:
            QTimer.singleShot(220, lambda: self._run_generate_stage(stages, i + 1))
        else:
            self._on_encode_change()

    def _on_download_encoded(self):
        if not hasattr(self, "_last_audio"):
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Morse Audio", "morse_encoded.wav")
        if not dest:
            return
        dsp.write_wav_file(dest, self._last_audio, dsp.FS)

    # ------------------------------------------------------------- Decode
    def _build_decode_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        input_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self.upload_btn)
        layout.addLayout(input_row)

        self.mic_widget = MicRecordWidget()
        self.mic_widget.recording_ready.connect(self._on_recording_ready)
        layout.addWidget(self.mic_widget)

        self.decode_source_label = QLabel("Use the \U0001F9F0 Toolbox above, upload a file, or record from mic.")
        self.decode_source_label.setObjectName("Caption")
        self.decode_source_label.setWordWrap(True)
        layout.addWidget(self.decode_source_label)

        # ---- detection controls
        det_row = QHBoxLayout()
        algo_col = QVBoxLayout()
        algo_col.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(dsp.DECODE_ALGORITHMS)
        self.algo_combo.setToolTip(
            "Hilbert Envelope: exact instantaneous amplitude via analytic signal.\n"
            "Rectify + Low-pass: classic AM-radio style envelope detector.\n"
            "Short-Time Energy: windowed RMS block energy (voice-activity-style)."
        )
        self.algo_combo.currentTextChanged.connect(self._on_algo_or_threshold_change)
        algo_col.addWidget(self.algo_combo)
        det_row.addLayout(algo_col)

        thresh_col = QVBoxLayout()
        self.threshold_label = QLabel("Detection threshold: 0.25")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(5, 80)
        self.threshold_slider.setValue(25)
        self.threshold_slider.setToolTip("Lower = more sensitive (good for weak/noisy signals, "
                                          "risk of false triggers). Higher = stricter.")
        self.threshold_slider.valueChanged.connect(self._on_algo_or_threshold_change)
        thresh_col.addWidget(self.threshold_label)
        thresh_col.addWidget(self.threshold_slider)
        det_row.addLayout(thresh_col)

        self.manual_timing_check = QCheckBox("Manual timing")
        self.manual_timing_check.setToolTip("Override auto-detected dot duration with the slider below.")
        self.manual_timing_check.toggled.connect(self._on_algo_or_threshold_change)
        det_row.addWidget(self.manual_timing_check)

        unit_col = QVBoxLayout()
        self.unit_label = QLabel("Unit (manual): 80 ms")
        self.unit_slider = QSlider(Qt.Horizontal)
        self.unit_slider.setRange(20, 300)
        self.unit_slider.setValue(80)
        self.unit_slider.valueChanged.connect(self._on_algo_or_threshold_change)
        unit_col.addWidget(self.unit_label)
        unit_col.addWidget(self.unit_slider)
        det_row.addLayout(unit_col)

        self.auto_detect_btn = QPushButton("🎯 Auto-Detect Timing")
        self.auto_detect_btn.setObjectName("SecondaryButton")
        self.auto_detect_btn.clicked.connect(self._on_auto_detect)
        det_row.addWidget(self.auto_detect_btn)
        layout.addLayout(det_row)

        # ---- live progressive decode display
        live_frame = QFrame()
        live_frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.6); border-radius: 10px; }")
        live_layout = QVBoxLayout(live_frame)
        self.live_morse_label = QLabel("Detected Morse: —")
        self.live_morse_label.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: 600;")
        self.live_text_label = QLabel("Decoded Text: —")
        self.live_text_label.setStyleSheet("font-size: 15px; font-weight: 700;")
        live_layout.addWidget(self.live_morse_label)
        live_layout.addWidget(self.live_text_label)
        layout.addWidget(live_frame)

        # ---- manual morse edit + redecode
        edit_row = QHBoxLayout()
        edit_row.addWidget(QLabel("Edit detected Morse:"))
        self.decode_morse_edit = QLineEdit()
        self.decode_morse_edit.setToolTip("Correct any mis-detected symbols, then re-decode.")
        edit_row.addWidget(self.decode_morse_edit, 1)
        self.redecode_btn = QPushButton("↻ Re-decode")
        self.redecode_btn.setObjectName("SecondaryButton")
        self.redecode_btn.clicked.connect(self._on_redecode_from_edit)
        edit_row.addWidget(self.redecode_btn)
        layout.addLayout(edit_row)

        # ---- timeline
        self.timeline_label = QLabel("Timeline: —")
        self.timeline_label.setWordWrap(True)
        self.timeline_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self.timeline_label)

        # ---- decode pipeline canvas
        self.decode_canvas = MplCanvas(n_rows=4, figsize=(7, 1.4))
        layout.addWidget(self.decode_canvas)

        row = QHBoxLayout()
        self.decode_player = AudioTransportWidget()
        row.addWidget(self.decode_player)
        layout.addLayout(row)

        action_row = QHBoxLayout()
        self.copy_morse_btn = QPushButton("📋 Copy Morse")
        self.copy_morse_btn.setObjectName("SecondaryButton")
        self.copy_morse_btn.clicked.connect(self._on_copy_morse)
        action_row.addWidget(self.copy_morse_btn)

        self.copy_text_btn = QPushButton("📋 Copy Text")
        self.copy_text_btn.setObjectName("SecondaryButton")
        self.copy_text_btn.clicked.connect(self._on_copy_text)
        action_row.addWidget(self.copy_text_btn)

        self.export_btn = QPushButton("⬇ Export Result")
        self.export_btn.setObjectName("SecondaryButton")
        self.export_btn.clicked.connect(self._on_export_result)
        action_row.addWidget(self.export_btn)

        self.clear_btn = QPushButton("🗑 Clear")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.clicked.connect(self._on_clear_decode)
        action_row.addWidget(self.clear_btn)
        action_row.addStretch()
        layout.addLayout(action_row)
        layout.addStretch()

        self.decode_player.position_changed.connect(self.decode_canvas.update_playhead)
        self.decode_player.playback_active_changed.connect(
            lambda active: None if active else self.decode_canvas.clear_playhead())
        return tab

    # ------------------------------------------------------------- toolbox
    def _open_toolbox(self):
        dlg = ToolboxDialog(self.toolbox, self._load_from_toolbox, parent=self)
        dlg.exec()

    def _load_from_toolbox(self, label):
        item = self.toolbox[label]
        self.decode_source_audio = item["audio"]
        self.decode_source_label.setText(
            f"\U0001F9F0 Toolbox clip: '{label}'  (expected message: {item['message']})")
        self.tabs.setCurrentIndex(1)
        self._run_decode()

    # ------------------------------------------------------------- inputs
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
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
    def _on_algo_or_threshold_change(self):
        self.threshold_label.setText(f"Detection threshold: {self.threshold_slider.value()/100:.2f}")
        self.unit_label.setText(f"Unit (manual): {self.unit_slider.value()} ms")
        self.unit_slider.setEnabled(self.manual_timing_check.isChecked())
        if self.decode_source_audio is not None:
            self._run_decode()

    def _on_auto_detect(self):
        self.manual_timing_check.setChecked(False)
        self._run_decode()

    def _run_decode(self):
        if self.decode_source_audio is None:
            return
        audio = self.decode_source_audio
        algorithm = self.algo_combo.currentText()
        threshold = self.threshold_slider.value() / 100.0
        manual_unit = (self.unit_slider.value() / 1000.0) if self.manual_timing_check.isChecked() else None

        decoded, filtered, envelope, keyed, det_freq, det_unit = dsp.decode_morse(
            audio, tone_freq=None, unit=manual_unit, threshold=threshold, algorithm=algorithm)

        self.current_keyed = keyed
        self.current_unit = det_unit
        self.current_tone_freq = det_freq
        self.current_segments = dsp.classify_morse_segments(keyed, fs=dsp.FS, unit=det_unit)
        self._final_decoded_text = decoded
        self._final_morse_string = dsp.text_to_morse_string(decoded) if decoded.strip() else ""

        self.decode_canvas.plot_waveforms(
            [(audio, "Input"), (filtered, "Bandpassed"),
             (envelope, "Envelope"), (keyed.astype(float), "Dots/Dashes")],
            f"Decoding pipeline ({algorithm}) — result: '{decoded}'")
        self.decode_player.set_audio(audio)
        self._render_timeline()

        self.decode_morse_edit.setText(self._final_morse_string)

        # Live progressive reveal, paced by actual segment count
        self._reveal_index = 0
        self.live_morse_label.setText("Detected Morse: ")
        self.live_text_label.setText("Decoded Text: ")
        self._progressive_timer.stop()
        if self.current_segments:
            self._progressive_timer.start(60)

    def _reveal_next_segment(self):
        if self._reveal_index >= len(self.current_segments):
            self._progressive_timer.stop()
            self.live_morse_label.setText(f"Detected Morse: {self._final_morse_string}")
            self.live_text_label.setText(f"Decoded Text: {self._final_decoded_text}")
            return
        seg = self.current_segments[self._reveal_index]
        partial_keyed = np.zeros_like(self.current_keyed)
        end_sample = int(seg["end_s"] * dsp.FS)
        partial_keyed[:end_sample] = self.current_keyed[:end_sample]
        partial_text, _ = dsp.keyed_to_text(partial_keyed, fs=dsp.FS, unit=self.current_unit)
        partial_morse = dsp.text_to_morse_string(partial_text) if partial_text.strip() else ""
        self.live_morse_label.setText(f"Detected Morse: {partial_morse}")
        self.live_text_label.setText(f"Decoded Text: {partial_text}")
        self._reveal_index += 1

    def _render_timeline(self):
        parts = []
        for seg in self.current_segments:
            label = seg["type"].upper().replace("_", " ")
            if seg["uncertain"]:
                label = f"?{label}"
            parts.append(f"[{label} {seg['duration_s']*1000:.0f}ms]")
        self.timeline_label.setText("Timeline: " + " ".join(parts) if parts else "Timeline: —")

    def _on_redecode_from_edit(self):
        morse_str = self.decode_morse_edit.text().strip()
        text = dsp.morse_string_to_text(morse_str)
        self._final_decoded_text = text
        self._final_morse_string = morse_str
        self.live_morse_label.setText(f"Detected Morse: {morse_str}")
        self.live_text_label.setText(f"Decoded Text: {text}")

    def _on_copy_morse(self):
        QApplication.clipboard().setText(self.decode_morse_edit.text())

    def _on_copy_text(self):
        QApplication.clipboard().setText(getattr(self, "_final_decoded_text", ""))

    def _on_export_result(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Export Decode Result", "morse_result.txt")
        if not dest:
            return
        with open(dest, "w") as f:
            f.write(f"Decoded Text: {getattr(self, '_final_decoded_text', '')}\n")
            f.write(f"Morse: {getattr(self, '_final_morse_string', '')}\n")
            if self.current_tone_freq:
                f.write(f"Detected tone: {self.current_tone_freq:.1f} Hz\n")
            if self.current_unit:
                f.write(f"Detected unit: {self.current_unit*1000:.1f} ms\n")

    def _on_clear_decode(self):
        self.decode_source_audio = None
        self.current_segments = []
        self.decode_source_label.setText("Use the \U0001F9F0 Toolbox above, upload a file, or record from mic.")
        self.live_morse_label.setText("Detected Morse: —")
        self.live_text_label.setText("Decoded Text: —")
        self.timeline_label.setText("Timeline: —")
        self.decode_morse_edit.clear()
        self.decode_canvas.fig.clear()
        self.decode_canvas.draw()

    # ------------------------------------------------------------- analysis mode
    def _toggle_analysis_mode(self):
        self.analysis_mode = self.mode_toggle.isChecked()
        self.mode_toggle.setText(f"🔬 Analysis Mode: {'ON' if self.analysis_mode else 'OFF'}")
        if self.decode_source_audio is not None:
            self._run_decode()
