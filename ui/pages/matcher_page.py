import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QFileDialog, QTabWidget
)
import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class MatcherPage(QWidget):
    """UI/visualization layer only. Both tabs call dsp_core's unchanged
    correlation primitives (match_template / clip_similarity, both built on
    scipy.signal.correlate) — no matching logic was modified here."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🎯 Audio Matcher")
        title.setObjectName("SectionTitle")
        caption = QLabel("Cross-correlation (matched filtering) locates a reference sound "
                          "inside a longer recording, or scores how similar two clips are.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_demo_tab(), "Built-in Demo")
        self.tabs.addTab(self._build_upload_tab(), "Upload Your Own Clips")
        layout.addWidget(self.tabs)

    # ------------------------------------------------------- Demo tab (original)
    def _build_demo_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.player = AudioPlayButton("Full recording")
        row = QHBoxLayout(); row.addWidget(self.player); row.addStretch()
        layout.addLayout(row)

        self.canvas = MplCanvas(n_rows=4)
        layout.addWidget(self.canvas)

        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setObjectName("Caption")
        layout.addWidget(self.results_label)
        layout.addStretch()

        self._build_demo()

        self.player.position_changed.connect(self.canvas.update_playhead)
        self.player.playback_active_changed.connect(
            lambda active: None if active else self.canvas.clear_playhead())
        return tab

    def _build_demo(self):
        recording, templates, placements = dsp.build_template_recording()
        self.player.set_audio(recording)

        figs_data = [(recording, "Recording")]
        lines = []
        for name, tmpl in templates.items():
            corr = dsp.match_template(recording, tmpl)  # unchanged correlation logic
            peak_t = np.argmax(np.abs(corr)) / dsp.FS
            figs_data.append((corr, f"corr: {name}"))
            lines.append(f"• {name}: detected at {peak_t:.2f}s (true: {placements[name]:.2f}s)")

        self.canvas.plot_waveforms(figs_data, "Recording & correlation scores")
        self.results_label.setText("Detected vs. true placement:<br>" + "<br>".join(lines))

    # ------------------------------------------------------- Upload tab (new UI only)
    def _build_upload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)

        self.ref_audio = None
        self.target_audio = None

        upload_row = QHBoxLayout()
        self.ref_btn = QPushButton("📁 Upload Reference Clip")
        self.ref_btn.setObjectName("PrimaryButton")
        self.ref_btn.clicked.connect(lambda: self._on_upload("ref"))
        upload_row.addWidget(self.ref_btn)

        self.target_btn = QPushButton("📁 Upload Target Recording")
        self.target_btn.setObjectName("PrimaryButton")
        self.target_btn.clicked.connect(lambda: self._on_upload("target"))
        upload_row.addWidget(self.target_btn)
        upload_row.addStretch()
        layout.addLayout(upload_row)

        self.upload_status = QLabel("Upload a short reference clip and a longer target "
                                     "recording to compare.")
        self.upload_status.setObjectName("Caption")
        self.upload_status.setWordWrap(True)
        layout.addWidget(self.upload_status)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run Match")
        self.run_btn.setObjectName("SecondaryButton")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_upload_match)
        run_row.addWidget(self.run_btn)
        run_row.addStretch()
        layout.addLayout(run_row)

        players = QHBoxLayout()
        self.p_ref = AudioPlayButton("Reference clip")
        self.p_target = AudioPlayButton("Target recording")
        players.addWidget(self.p_ref)
        players.addWidget(self.p_target)
        players.addStretch()
        layout.addLayout(players)

        self.upload_canvas = MplCanvas(n_rows=2)
        layout.addWidget(self.upload_canvas)

        self.upload_result_label = QLabel("")
        self.upload_result_label.setObjectName("SectionTitle")
        self.upload_result_label.setWordWrap(True)
        layout.addWidget(self.upload_result_label)
        layout.addStretch()

        self.p_target.position_changed.connect(self.upload_canvas.update_playhead)
        self.p_target.playback_active_changed.connect(
            lambda active: None if active else self.upload_canvas.clear_playhead())
        return tab

    def _on_upload(self, which):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3)")
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            fname = os.path.basename(path)
            if which == "ref":
                self.ref_audio = audio
                self.p_ref.set_audio(audio)
            else:
                self.target_audio = audio
                self.p_target.set_audio(audio)
            self._update_upload_status(fname)
            self.run_btn.setEnabled(self.ref_audio is not None and self.target_audio is not None)
        except Exception as e:
            self.upload_status.setText(f"❌ Error loading {which} file: {e}")

    def _update_upload_status(self, last_loaded_name):
        parts = []
        parts.append("Reference: ✅ loaded" if self.ref_audio is not None else "Reference: not loaded")
        parts.append("Target: ✅ loaded" if self.target_audio is not None else "Target: not loaded")
        self.upload_status.setText(f"Last loaded: {last_loaded_name}  —  " + "  |  ".join(parts))

    def _run_upload_match(self):
        if self.ref_audio is None or self.target_audio is None:
            return
        # Both calls below use the same unchanged correlation primitives as
        # the built-in demo tab (dsp.match_template, dsp.clip_similarity).
        corr = dsp.match_template(self.target_audio, self.ref_audio)
        peak_idx = int(np.argmax(np.abs(corr)))
        peak_t = peak_idx / dsp.FS
        score, offset_s = dsp.clip_similarity(self.ref_audio, self.target_audio)

        self.upload_canvas.plot_waveforms(
            [(self.target_audio, "Target recording"), (corr, "Cross-correlation with reference")],
            "Matching reference against target — live")

        self.upload_result_label.setText(
            f"Best alignment at t = {peak_t:.2f}s in the target recording. "
            f"Normalized similarity score: {score:.3f} "
            f"({'likely match' if score > 0.3 else 'weak/no match'})."
        )
