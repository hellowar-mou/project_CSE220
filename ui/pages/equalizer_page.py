import os
import copy
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider, QFrame,
    QPushButton, QFileDialog, QTabWidget, QComboBox, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

import dsp_core as dsp
from ui.widgets import (
    MplCanvas, AudioTransportWidget, MicRecordWidget, CollapsiblePanel,
    WaveformControls, show_module_help, ModuleGuide, AudioInputCard,
)


BAND_LABELS = ["60Hz", "120Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "8kHz", "16kHz"]
APPLY_STAGES = ["Analyzing...", "Applying Filters...", "Reconstructing Signal...", "Complete"]


class EqualizerPage(QWidget):
    """9-band graphic equalizer with a genuine choice of 3 filter-design
    algorithms (IIR peaking cascade, FIR windowed-sinc bands, shelving+
    peaking hybrid) driving the same 9 gain sliders — see logic/equalizer.py.
    Processes audio at 44.1kHz (not the app's shared 16kHz) since a 16kHz
    band needs a sample rate above 32kHz to exist at all (Nyquist)."""

    def __init__(self):
        super().__init__()
        self.audio_data = None          # input (at FS_EQ)
        self.processed_audio = None     # last "Applied" output
        self.bypass = False
        self.ab_mode = "B"              # "A" = original, "B" = processed
        self.history = []
        self.redo_stack = []
        self._mic_thread = None
        self._anim_group = []
        self._applying_preset = False

        self._build_ui()
        # Keep the scroll-area content height stable while Matplotlib canvases
        # update their minimum sizes; otherwise Qt can recurse during resize.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMinimumHeight(1240)
        self._load_demo_signal()
        self._push_history()

    # ================================================================ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ---- header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(111,177,234,0.15), stop:1 rgba(143,211,199,0.15));
                border-radius: 16px;
                border: 1px solid rgba(111,177,234,0.2);
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 16, 20, 16)
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("🎚️ Equalizer")
        title.setStyleSheet(
            "font-size: 20px; font-weight: 700; color: #2f3e50; "
            "background: transparent; border: none;"
        )
        caption = QLabel("9-band graphic EQ (60Hz–16kHz) • 3 selectable filter-design algorithms • "
                          "Live spectrum analyzer • Processed at 44.1kHz")
        caption.setStyleSheet(
            "font-size: 11px; color: #6b7f96; "
            "background: transparent; border: none;"
        )
        caption.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(caption)
        h_layout.addLayout(title_col, 1)
        help_btn = QPushButton("? Help")
        help_btn.setObjectName("SecondaryButton")
        help_btn.setToolTip("Show a short guide to the Equalizer.")
        help_btn.clicked.connect(lambda: show_module_help(
            self, "Equalizer Help", [
                "1. Upload audio, record from the microphone, or use the demo.",
                "2. Adjust the frequency bands and choose a preset if desired.",
                "3. Press Apply to create the equalized output.",
                "Use Bypass or A/B to compare original and processed audio.",
            ]))
        h_layout.addWidget(help_btn)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: 700; color: #2f6690;")
        h_layout.addWidget(self.status_label)
        layout.addWidget(header)
        layout.addWidget(ModuleGuide(
            "equalizer",
            "Upload audio, record from your microphone, or use the demo. "
            "Then adjust the bands and press Apply to hear the equalized result."))
        workflow = QLabel("INPUT  →  PROCESSING  →  OUTPUT")
        workflow.setToolTip("Load audio, adjust frequency bands, then preview the processed output.")
        layout.addWidget(workflow)

        # ---- input row
        input_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.setToolTip("Supported formats: WAV, MP3, FLAC, OGG")
        self.upload_btn.clicked.connect(self._on_upload)
        input_row.addWidget(self.upload_btn)

        self.mic_widget = MicRecordWidget()
        self.mic_widget.recording_ready.connect(self._on_recording_ready)
        input_row.addWidget(self.mic_widget)

        self.demo_btn = QPushButton("🔊 Load Demo Signal")
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.setToolTip("Load a multi-tone demo signal for experimenting with the EQ.")
        self.demo_btn.clicked.connect(self._load_demo_signal)
        input_row.addWidget(self.demo_btn)
        layout.addLayout(input_row)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("Caption")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)
        self.input_card = AudioInputCard()
        self.input_card.replace_requested.connect(self._on_upload)
        self.input_card.remove_requested.connect(self._remove_audio)
        layout.addWidget(self.input_card)

        # ---- algorithm + preset row
        config_row = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        algo_label.setToolTip("Chooses the filter-design technique used to realize the 9 bands.")
        config_row.addWidget(algo_label)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(dsp.EQ_ALGORITHMS)
        self.algo_combo.setToolTip(
            "IIR Peaking Cascade: classic parametric-EQ biquads, one per band.\n"
            "FIR Windowed-Sinc Bands: linear-phase FIR filters summed by gain.\n"
            "Shelving + Peaking Hybrid: shelf filters at the outer bands, peaking in between."
        )
        self.algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        config_row.addWidget(self.algo_combo)

        preset_label = QLabel("Preset:")
        config_row.addWidget(preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(dsp.EQ_PRESETS.keys()) + ["Custom"])
        self.preset_combo.setCurrentText("Flat")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.preset_combo.setToolTip("Apply a ready-made frequency balance, or choose Custom.")
        config_row.addWidget(self.preset_combo)
        config_row.addStretch()
        layout.addLayout(config_row)

        # ---- 9 band sliders
        bands_frame = QFrame()
        bands_frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.6); border-radius: 12px; }")
        bands_frame.setFixedHeight(230)
        bands_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bands_grid = QGridLayout(bands_frame)
        bands_grid.setSpacing(6)
        self.sliders = []
        self.gain_labels = []
        for i, label in enumerate(BAND_LABELS):
            freq_lab = QLabel(label)
            freq_lab.setAlignment(Qt.AlignCenter)
            freq_lab.setStyleSheet("font-size: 10px; font-weight: 600;")
            bands_grid.addWidget(freq_lab, 0, i)

            sld = QSlider(Qt.Vertical)
            sld.setRange(-120, 120)  # tenths of a dB: -12.0..+12.0
            sld.setValue(0)
            sld.setFixedHeight(120)
            descriptions = {
                "60Hz": "Controls very low-frequency bass and rumble.",
                "120Hz": "Controls bass weight and punch.",
                "250Hz": "Controls low-mid warmth and fullness.",
                "500Hz": "Controls body and boxiness.",
                "1kHz": "Controls midrange presence for speech and instruments.",
                "2kHz": "Controls clarity and attack.",
                "4kHz": "Controls detail and intelligibility.",
                "8kHz": "Controls high-frequency brightness.",
                "16kHz": "Controls air and sparkle.",
            }
            sld.setToolTip(f"{descriptions[label]} Range: -12 dB to +12 dB.")
            sld.valueChanged.connect(self._on_slider_live_change)
            sld.sliderReleased.connect(self._push_history)
            bands_grid.addWidget(sld, 1, i, Qt.AlignHCenter)
            self.sliders.append(sld)

            gval = QLabel("0.0 dB")
            gval.setAlignment(Qt.AlignCenter)
            gval.setStyleSheet("font-size: 9px;")
            bands_grid.addWidget(gval, 2, i)
            self.gain_labels.append(gval)
        layout.addWidget(bands_frame)

        # ---- control buttons
        controls_row = QHBoxLayout()
        self.reset_btn = QPushButton("↺ Reset EQ")
        self.reset_btn.setObjectName("SecondaryButton")
        self.reset_btn.setToolTip("Set all bands back to 0 dB (Flat).")
        self.reset_btn.clicked.connect(self._on_reset)
        controls_row.addWidget(self.reset_btn)

        self.bypass_btn = QPushButton("⏭ Bypass: OFF")
        self.bypass_btn.setObjectName("SecondaryButton")
        self.bypass_btn.setCheckable(True)
        self.bypass_btn.setToolTip("Instantly compare original vs. EQ'd without losing your settings.")
        self.bypass_btn.clicked.connect(self._on_bypass_toggle)
        controls_row.addWidget(self.bypass_btn)

        self.apply_btn = QPushButton("⚡ Apply")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.setToolTip("Render the equalized output with the current bands and algorithm.")
        self.apply_btn.clicked.connect(self._on_apply)
        controls_row.addWidget(self.apply_btn)

        self.undo_btn = QPushButton("⟲ Undo")
        self.undo_btn.setObjectName("SecondaryButton")
        self.undo_btn.setToolTip("Undo the last completed EQ slider change.")
        self.undo_btn.clicked.connect(self._on_undo)
        controls_row.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("⟳ Redo")
        self.redo_btn.setObjectName("SecondaryButton")
        self.redo_btn.setToolTip("Redo the last undone EQ slider change.")
        self.redo_btn.clicked.connect(self._on_redo)
        controls_row.addWidget(self.redo_btn)
        controls_row.addStretch()
        layout.addLayout(controls_row)

        # ---- A/B comparison
        ab_row = QHBoxLayout()
        ab_label = QLabel("Compare:")
        ab_row.addWidget(ab_label)
        self.a_btn = QPushButton("A — Original")
        self.b_btn = QPushButton("B — Equalized")
        self.a_btn.setToolTip("Listen to the original input audio.")
        self.b_btn.setToolTip("Listen to the equalized output.")
        for b in (self.a_btn, self.b_btn):
            b.setObjectName("SecondaryButton")
            b.setCheckable(True)
        self.b_btn.setChecked(True)
        self.a_btn.clicked.connect(lambda: self._set_ab_mode("A"))
        self.b_btn.clicked.connect(lambda: self._set_ab_mode("B"))
        ab_row.addWidget(self.a_btn)
        ab_row.addWidget(self.b_btn)
        ab_row.addStretch()
        layout.addLayout(ab_row)

        # ---- visualization tabs
        self.tabs = QTabWidget()
        self.spectrum_canvas = MplCanvas(n_rows=1, figsize=(7, 2.2))
        self.response_canvas = MplCanvas(n_rows=1, figsize=(7, 1.8))
        self.wave_canvas = MplCanvas(n_rows=2, figsize=(7, 1.8))
        # Keep the scroll-area geometry stable while Matplotlib redraws.
        self.spectrum_canvas.setFixedHeight(260)
        self.response_canvas.setFixedHeight(220)
        self.wave_canvas.setFixedHeight(360)
        self.tabs.addTab(self.spectrum_canvas, "Spectrum Analyzer (Before/After)")
        self.tabs.addTab(self.response_canvas, "Live Filter Response")
        self.tabs.addTab(self.wave_canvas, "Input / Output Waveform")
        layout.addWidget(self.tabs)
        layout.addWidget(WaveformControls(self.wave_canvas))

        # Keep playback controls with the input/output waveform rather than
        # duplicating them in the metadata card above.
        self.transport = AudioTransportWidget()
        layout.addWidget(self.transport)

        self.transport.position_changed.connect(self.wave_canvas.update_playhead)
        self.transport.playback_active_changed.connect(
            lambda active: None if active else self.wave_canvas.clear_playhead())

        # ---- technical info panel
        self.info_panel = CollapsiblePanel("Technical Information")
        info_grid = QGridLayout()
        self.info_labels = {}
        info_fields = ["Sample Rate", "Bit Depth", "Channels", "Duration",
                       "Dominant Freq", "RMS Level", "Peak Level", "Algorithm"]
        for i, field in enumerate(info_fields):
            lab = QLabel(f"{field}:")
            lab.setStyleSheet("font-weight: 600; font-size: 11px;")
            val = QLabel("—")
            val.setStyleSheet("font-size: 11px;")
            info_grid.addWidget(lab, i // 2, (i % 2) * 2)
            info_grid.addWidget(val, i // 2, (i % 2) * 2 + 1)
            self.info_labels[field] = val
        self.info_panel.body_layout().addLayout(info_grid)
        layout.addWidget(self.info_panel)

        # ---- download
        download_row = QHBoxLayout()
        self.download_btn = QPushButton("⬇ Download Processed Audio")
        self.download_btn.setObjectName("SecondaryButton")
        self.download_btn.setEnabled(False)
        self.download_btn.setToolTip("Apply the equalizer first to create a downloadable result.")
        self.download_btn.clicked.connect(self._on_download)
        download_row.addWidget(self.download_btn)
        download_row.addStretch()
        layout.addLayout(download_row)

        # The page is hosted inside a resizable scroll area. A trailing
        # stretch makes Qt repeatedly renegotiate the page height with the
        # scroll viewport, which can recurse until the process stack overflows.

    # ========================================================== input
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS_EQ)
            self.audio_data = audio
            self._input_metadata = (os.path.basename(path), "Uploaded File", orig_fs)
            self.file_label.setText(f"✅ {os.path.basename(path)}  "
                                     f"({orig_fs} Hz → {dsp.FS_EQ} Hz, {len(audio)/dsp.FS_EQ:.2f}s)")
            self._on_new_audio()
        except Exception as e:
            self.file_label.setText(f"❌ Error loading file: {e}")
            self._set_status("Error", "#c0392b")

    def _on_recording_ready(self, audio, fs):
        # MicRecordWidget records at dsp.FS (16kHz); resample up to FS_EQ
        # so it's consistent with everything else on this page.
        if fs != dsp.FS_EQ:
            n_samples = int(len(audio) * dsp.FS_EQ / fs)
            from scipy import signal as scipy_signal
            audio = scipy_signal.resample(audio, n_samples)
        self.audio_data = dsp.normalize(audio)
        self._input_metadata = ("Microphone recording", "Microphone Recording", fs)
        self.file_label.setText(f"🎙️ Recorded, resampled to {dsp.FS_EQ} Hz")
        self._on_new_audio()

    def _load_demo_signal(self):
        ts = np.arange(int(2.5 * dsp.FS_EQ)) / dsp.FS_EQ
        demo = (0.35 * np.sin(2 * np.pi * 110 * ts) +
                0.30 * np.sin(2 * np.pi * 440 * ts) +
                0.20 * np.sin(2 * np.pi * 1500 * ts) +
                0.15 * np.sin(2 * np.pi * 6000 * ts))
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * ts)
        self.audio_data = dsp.normalize(demo * envelope)
        self._input_metadata = ("Multi-tone demo", "Built-in Demo", dsp.FS_EQ)
        self.file_label.setText("🔊 Demo: multi-tone signal (110Hz/440Hz/1.5kHz/6kHz, 2.5s)")
        self._on_new_audio()

    def _on_new_audio(self):
        self.processed_audio = None
        self.download_btn.setEnabled(False)
        self._update_info_panel()
        self._on_slider_live_change()
        self._update_transport()
        self._render_waveform_preview()
        filename, source, source_fs = getattr(
            self, "_input_metadata", ("Audio input", "Audio Input", dsp.FS_EQ))
        self.input_card.set_audio(
            self.audio_data, dsp.FS_EQ, filename=filename,
            source=source, channels=1)

    def _render_waveform_preview(self):
        """Keep the waveform tab populated before and after Apply."""
        if self.audio_data is None:
            return
        preview = dsp.apply_nband_eq(
            self.audio_data,
            self._current_gains_db(),
            algorithm=self.algo_combo.currentText(),
            fs=dsp.FS_EQ,
        )
        output_label = "Output (preview)" if self.processed_audio is None else "Output (Applied)"
        output = preview if self.processed_audio is None else self.processed_audio
        self.wave_canvas.plot_waveforms(
            [(self.audio_data, "Input"), (output, output_label)],
            "Input vs. Output Waveform",
            fs=dsp.FS_EQ,
        )

    def _remove_audio(self):
        self.audio_data = None
        self.processed_audio = None
        self.file_label.setText("No file selected")
        self.input_card.set_audio(None)
        self.transport.clear()
        self.download_btn.setEnabled(False)

    # ========================================================== sliders/preset/algo
    def _current_gains_db(self):
        return [s.value() / 10.0 for s in self.sliders]

    def _on_slider_live_change(self):
        if self.audio_data is None:
            return
        gains = self._current_gains_db()
        for lab, g in zip(self.gain_labels, gains):
            lab.setText(f"{g:+.1f} dB")
        if (not self._applying_preset
                and self.preset_combo.currentText() != "Custom"
                and gains != dsp.EQ_PRESETS.get(self.preset_combo.currentText())):
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
        if self._applying_preset:
            return
        self._redraw_response()
        self._render_waveform_preview()

    def _on_algorithm_changed(self):
        self._redraw_response()
        self.info_labels["Algorithm"].setText(self.algo_combo.currentText())

    def _on_preset_changed(self, name):
        if name == "Custom" or name not in dsp.EQ_PRESETS:
            return
        target = dsp.EQ_PRESETS[name]
        for animation in self._anim_group:
            animation.stop()
        self._anim_group = []
        self._applying_preset = True
        for slider, gain_db in zip(self.sliders, target):
            slider.blockSignals(True)
            slider.setValue(int(round(gain_db * 10)))
            slider.blockSignals(False)
        self._applying_preset = False
        self._on_slider_live_change()
        self._push_history()

    def _animate_sliders_to(self, target_gains_db):
        self._anim_group = []
        for sld, g_db in zip(self.sliders, target_gains_db):
            anim = QPropertyAnimation(sld, b"value", self)
            anim.setDuration(350)
            anim.setStartValue(sld.value())
            anim.setEndValue(int(round(g_db * 10)))
            anim.setEasingCurve(QEasingCurve.OutCubic)
            self._anim_group.append(anim)
            anim.start()
        def finish_preset():
            self._applying_preset = False
            self._on_slider_live_change()
            self._push_history()
        QTimer.singleShot(370, finish_preset)

    def _redraw_response(self):
        if self.audio_data is None:
            return
        gains = self._current_gains_db()
        algorithm = self.algo_combo.currentText()

        # Live filter response tab (cheap — filter coefficients only)
        self.response_canvas.fig.clear()
        ax = self.response_canvas.fig.subplots(1, 1)
        w, h_db = dsp.nband_eq_frequency_response(gains, algorithm=algorithm, fs=dsp.FS_EQ)
        ax.semilogx(w, h_db, color="#2f6690", linewidth=1.3)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-15, 15)
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_xlabel("Frequency (Hz, log scale)", fontsize=8)
        ax.set_ylabel("Gain (dB)", fontsize=8)
        ax.set_title(f"Live Filter Response — {algorithm}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.tick_params(labelsize=7)
        self.response_canvas.fig.tight_layout()
        self.response_canvas.recolor()
        self.response_canvas.draw()

        # Spectrum analyzer: keep both traces high-contrast, including the
        # live output preview before the user presses Apply.
        self.spectrum_canvas.fig.clear()
        ax2 = self.spectrum_canvas.fig.subplots(1, 1)
        freqs_in, db_in = dsp.spectrum_db(self.audio_data, dsp.FS_EQ)
        ax2.semilogx(
            freqs_in, db_in, color="#f97316", linewidth=1.35,
            alpha=0.95, label="Original / Input",
        )
        output = self.processed_audio
        output_label = "Equalized (Applied)"
        if output is None:
            output = dsp.apply_nband_eq(
                self.audio_data,
                self._current_gains_db(),
                algorithm=self.algo_combo.currentText(),
                fs=dsp.FS_EQ,
            )
            output_label = "Equalized (Live Preview)"
        freqs_out, db_out = dsp.spectrum_db(output, dsp.FS_EQ)
        ax2.semilogx(
            freqs_out, db_out, color="#06b6d4", linewidth=1.25,
            alpha=0.95, label=output_label,
        )
        ax2.set_xlim(20, 20000)
        ax2.set_xlabel("Frequency (Hz, log scale)", fontsize=8)
        ax2.set_ylabel("Magnitude (dB)", fontsize=8)
        ax2.set_title("Spectrum: Input vs. Output", fontsize=9)
        ax2.legend(fontsize=7, loc="upper right", framealpha=0.88)
        ax2.grid(alpha=0.25, which="both")
        ax2.tick_params(labelsize=7)
        self.spectrum_canvas.fig.tight_layout()
        self.spectrum_canvas.recolor()
        self.spectrum_canvas.draw()

    # ========================================================== apply / bypass / A-B
    def _on_apply(self):
        if self.audio_data is None:
            return
        self.apply_btn.setEnabled(False)
        self._run_apply_stage(0)

    def _run_apply_stage(self, stage_idx):
        self._set_status(APPLY_STAGES[stage_idx], "#b8860b" if stage_idx < 3 else "#1e8a5f")
        if stage_idx < len(APPLY_STAGES) - 1:
            QTimer.singleShot(280, lambda: self._run_apply_stage(stage_idx + 1))
        else:
            self._finish_apply()

    def _finish_apply(self):
        gains = self._current_gains_db()
        algorithm = self.algo_combo.currentText()
        self.processed_audio = dsp.apply_nband_eq(self.audio_data, gains, algorithm=algorithm, fs=dsp.FS_EQ)
        self.download_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self._update_info_panel()
        self._redraw_response()
        self._update_transport()
        self.wave_canvas.plot_waveforms(
            [(self.audio_data, "Input"), (self.processed_audio, "Output (Applied)")],
            f"Input vs. Output — {algorithm}", fs=dsp.FS_EQ)

    def _on_bypass_toggle(self):
        self.bypass = self.bypass_btn.isChecked()
        self.bypass_btn.setText(f"⏭ Bypass: {'ON' if self.bypass else 'OFF'}")
        self._update_transport()

    def _set_ab_mode(self, mode):
        self.ab_mode = mode
        self.a_btn.setChecked(mode == "A")
        self.b_btn.setChecked(mode == "B")
        self._update_transport()

    def _update_transport(self):
        if self.audio_data is None:
            return
        if self.bypass or self.ab_mode == "A" or self.processed_audio is None:
            self.transport.set_audio(self.audio_data, fs=dsp.FS_EQ)
        else:
            self.transport.set_audio(self.processed_audio, fs=dsp.FS_EQ)

    def _on_reset(self):
        self._animate_sliders_to([0] * 9)
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText("Flat")
        self.preset_combo.blockSignals(False)

    # ========================================================== undo/redo
    def _push_history(self):
        gains = self._current_gains_db()
        if self.history and self.history[-1] == gains:
            return
        self.history.append(gains)
        if len(self.history) > 50:
            self.history.pop(0)
        self.redo_stack = []

    def _on_undo(self):
        if len(self.history) < 2:
            return
        self.redo_stack.append(self.history.pop())
        prev = self.history[-1]
        self._set_sliders_silently(prev)
        self._on_slider_live_change()

    def _on_redo(self):
        if not self.redo_stack:
            return
        nxt = self.redo_stack.pop()
        self.history.append(nxt)
        self._set_sliders_silently(nxt)
        self._on_slider_live_change()

    def _set_sliders_silently(self, gains_db):
        for sld, g in zip(self.sliders, gains_db):
            sld.blockSignals(True)
            sld.setValue(int(round(g * 10)))
            sld.blockSignals(False)
        for lab, g in zip(self.gain_labels, gains_db):
            lab.setText(f"{g:+.1f} dB")

    # ========================================================== info panel
    def _update_info_panel(self):
        if self.audio_data is None:
            return
        duration = len(self.audio_data) / dsp.FS_EQ
        dominant = dsp.compute_dominant_frequency(self.audio_data, dsp.FS_EQ)
        rms = dsp.compute_rms(self.audio_data)
        peak = dsp.compute_peak(self.audio_data)
        self.info_labels["Sample Rate"].setText(f"{dsp.FS_EQ} Hz")
        self.info_labels["Bit Depth"].setText("16-bit (processing: float64)")
        self.info_labels["Channels"].setText("1 (mono)")
        self.info_labels["Duration"].setText(f"{duration:.2f} s")
        self.info_labels["Dominant Freq"].setText(f"{dominant:.1f} Hz")
        self.info_labels["RMS Level"].setText(f"{rms:.4f}")
        self.info_labels["Peak Level"].setText(f"{peak:.4f}")
        self.info_labels["Algorithm"].setText(self.algo_combo.currentText())

    # ========================================================== download
    def _on_download(self):
        if self.processed_audio is None:
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save Processed Audio", "equalized_output.wav")
        if not dest:
            return
        try:
            dsp.write_wav_file(dest, self.processed_audio, dsp.FS_EQ)
            QMessageBox.information(self, "Download complete", f"Saved to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Download failed", f"Could not save file:\n{e}")

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-weight: 700; color: {color};")
