import os
import shutil
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame, QPushButton,
    QFileDialog, QTabWidget, QComboBox, QProgressBar, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread

import dsp_core as dsp
from ui.widgets import (
    MplCanvas, AudioTransportWidget, MicRecorderThread, WaveformControls,
    show_module_help, ModuleGuide, AudioInputCard,
)


# ============================================================== worker thread
class MatchWorkerThread(QThread):
    """Runs dsp.run_matching() off the UI thread so searching the library
    never freezes the app, and reports genuine per-track progress back to
    the UI as each library item actually finishes being scored."""
    progress = Signal(str, float)      # library item name, fraction complete
    finished_matching = Signal(dict)   # result dict from dsp.run_matching
    failed = Signal(str)

    def __init__(self, query, library, algorithm, parent=None):
        super().__init__(parent)
        self.query = query
        self.library = library
        self.algorithm = algorithm

    def run(self):
        try:
            result = dsp.run_matching(
                self.query, self.library, self.algorithm,
                progress_callback=lambda name, frac: self.progress.emit(name, frac)
            )
            self.finished_matching.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


def _confidence_label(score):
    if score >= 0.6:
        return "High confidence", "#1e8a5f"
    elif score >= 0.35:
        return "Medium confidence", "#b8860b"
    else:
        return "Low confidence", "#c0392b"


PIPELINE_STAGES = [
    "Input", "Preprocessing", "Feature Extraction",
    "Algorithm Matching", "Scoring", "Ranking", "Final Match",
]
MATCH_THRESHOLD = 0.45


# ============================================================== main page
class MatcherPage(QWidget):
    """Audio Matcher — upload or record a query, search the real built-in
    audio library (audios/*.mp3) using one or all of the existing
    correlation-based algorithms (logic/matcher.py), and inspect ranked
    results with a live, genuine processing-pipeline visualization.

    Matching primitives (match_template, clip_similarity, spectrogram_peaks/
    match_song) are unchanged from before; only the orchestration layer
    (logic.matcher.run_matching) and this UI are new."""

    def __init__(self):
        super().__init__()
        self.library = dsp.load_audio_library()
        self.query_audio = None
        self.query_source_label = "No input yet"
        self._mic_thread = None
        self._mic_chunks = []
        self._worker = None
        self._last_result = None

        self._build_ui()
        self._set_status("Ready", "#2f6690")
        self._refresh_library_panel()

    # ------------------------------------------------------------ UI build
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(12)

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
        title = QLabel("🎯 Audio Matcher")
        title.setStyleSheet(
            "font-size: 20px; font-weight: 700; color: #2f3e50; "
            "background: transparent; border: none;"
        )
        caption = QLabel("Upload audio or record from mic, then search the built-in "
                          "song library using cross-correlation, similarity, and "
                          "spectral-fingerprint matching.")
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
        help_btn.setToolTip("Show a short guide to the Audio Matcher.")
        help_btn.clicked.connect(lambda: show_module_help(
            self, "Audio Matcher Help", [
                "1. Upload, record, or select a demo audio sample.",
                "2. Select a matching algorithm.",
                "3. Click Search Library.",
                "4. Review the ranked candidates and similarity scores.",
                "A match is reported only when confidence reaches the threshold.",
            ]))
        h_layout.addWidget(help_btn)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("Caption")
        self.status_label.setStyleSheet("font-weight: 700;")
        h_layout.addWidget(self.status_label)
        outer.addWidget(header)
        outer.addWidget(ModuleGuide(
            "matcher",
            "Provide an audio sample by upload, microphone, or demo. "
            "Choose an algorithm and press Match Audio to search the library."))
        workflow = QLabel("INPUT  →  PROCESSING  →  OUTPUT")
        workflow.setToolTip("Provide audio, select a matching algorithm, then review ranked results.")
        outer.addWidget(workflow)

        # ---- Section 1: Audio Input
        outer.addWidget(self._section_label("① Audio Input"))
        self.input_tabs = QTabWidget()
        self.input_tabs.addTab(self._build_upload_tab(), "📁 Upload Audio")
        self.input_tabs.addTab(self._build_mic_tab(), "🎙️ Record Microphone")
        outer.addWidget(self.input_tabs)

        self.query_info_label = QLabel(
            "🎵 No audio loaded\nUpload an audio file, record from your microphone, "
            "or select a demo to begin.")
        self.query_info_label.setObjectName("Caption")
        self.query_info_label.setWordWrap(True)
        outer.addWidget(self.query_info_label)
        self.input_card = AudioInputCard()
        self.input_card.replace_requested.connect(self._on_upload)
        self.input_card.remove_requested.connect(self._on_remove_input)
        outer.addWidget(self.input_card)

        self.query_canvas = MplCanvas(n_rows=1, figsize=(7, 2.0))
        outer.addWidget(self.query_canvas)
        outer.addWidget(WaveformControls(self.query_canvas))

        self.query_transport = AudioTransportWidget()
        outer.addWidget(self.query_transport)
        self.query_transport.position_changed.connect(self.query_canvas.update_playhead)
        self.query_transport.playback_active_changed.connect(
            lambda active: None if active else self.query_canvas.clear_playhead())

        # ---- Section 2: Built-in Audio Library
        outer.addWidget(self._section_label("② Built-in Audio Library (Demo Mode)"))
        lib_desc = QLabel("This is what your query is searched against. Click 'Use as Demo Query' "
                           "on any track to try the whole pipeline with one click — no upload needed.")
        lib_desc.setObjectName("Caption")
        lib_desc.setWordWrap(True)
        outer.addWidget(lib_desc)
        self.library_grid = QGridLayout()
        outer.addLayout(self.library_grid)

        # ---- Section 3: Matching Configuration
        outer.addWidget(self._section_label("③ Matching Configuration"))
        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("Algorithm:"))
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(dsp.ALGORITHMS)
        self.algo_combo.setCurrentText("Combined (All Algorithms)")
        config_row.addWidget(self.algo_combo, 1)
        self.search_btn = QPushButton("🔍 Match Audio")
        self.search_btn.setObjectName("PrimaryButton")
        self.search_btn.setEnabled(False)
        self.search_btn.setToolTip("Load or record an audio sample before matching.")
        self.algo_combo.setToolTip(
            "Choose the matching technique. Combined averages all available algorithms.")
        self.search_btn.clicked.connect(self._start_matching)
        config_row.addWidget(self.search_btn)
        outer.addLayout(config_row)

        # ---- Section 4: Live Processing Pipeline
        outer.addWidget(self._section_label("④ Live Processing Pipeline"))
        self.stage_row = QHBoxLayout()
        self.stage_labels = []
        for stage in PIPELINE_STAGES:
            lab = QLabel(stage)
            lab.setObjectName("Caption")
            lab.setStyleSheet("padding: 4px 8px; border-radius: 8px; background: #eef5fb;")
            self.stage_row.addWidget(lab)
            self.stage_labels.append(lab)
        outer.addLayout(self.stage_row)

        # MplCanvas multiplies the height by n_rows; use the per-row height
        # here so the backing figure remains close to the widget's 300px size.
        self.pipeline_canvas = MplCanvas(n_rows=3, figsize=(7, 0.85))
        # Reserve enough vertical space for the raw signal, normalized signal,
        # and spectrogram so their labels and title do not overlap.
        self.pipeline_canvas.setMinimumHeight(285)
        self.pipeline_canvas.setFixedHeight(300)
        self.pipeline_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(self.pipeline_canvas)

        self.progress_grid = QGridLayout()
        outer.addLayout(self.progress_grid)
        self.progress_bars = {}

        # ---- Section 5: Results
        outer.addWidget(self._section_label("⑤ Results"))
        self.best_match_frame = QFrame()
        self.best_match_frame.setStyleSheet(
            "QFrame { background: rgba(143, 211, 199, 0.35); border-radius: 14px; "
            "border: 2px solid #8fd3c7; }")
        self.best_match_layout = QVBoxLayout(self.best_match_frame)
        self.best_match_label = QLabel("No results yet — run a search above.")
        self.best_match_label.setObjectName("SectionTitle")
        self.best_match_layout.addWidget(self.best_match_label)
        outer.addWidget(self.best_match_frame)

        self.ranked_list_layout = QVBoxLayout()
        outer.addLayout(self.ranked_list_layout)

        # ---- Section 6: Comparison
        outer.addWidget(self._section_label("⑥ Input vs. Matched Comparison"))
        self.compare_canvas = MplCanvas(n_rows=2, figsize=(7, 2.0))
        outer.addWidget(self.compare_canvas)

        compare_players = QHBoxLayout()
        self.compare_input_transport = AudioTransportWidget()
        self.compare_match_transport = AudioTransportWidget()
        col1 = QVBoxLayout(); col1.addWidget(QLabel("Your Input")); col1.addWidget(self.compare_input_transport)
        col2 = QVBoxLayout(); col2.addWidget(QLabel("Matched Audio")); col2.addWidget(self.compare_match_transport)
        compare_players.addLayout(col1)
        compare_players.addLayout(col2)
        outer.addLayout(compare_players)

        # ---- Section 7: Download
        download_row = QHBoxLayout()
        self.download_btn = QPushButton("⬇ Download Matched Audio (Original)")
        self.download_btn.setObjectName("SecondaryButton")
        self.download_btn.setEnabled(False)
        self.download_btn.setToolTip("Run a match first to enable downloading the library audio.")
        self.download_btn.clicked.connect(self._on_download_original)
        download_row.addWidget(self.download_btn)

        self.download_processed_btn = QPushButton("⬇ Download Processed Version (WAV)")
        self.download_processed_btn.setObjectName("SecondaryButton")
        self.download_processed_btn.setEnabled(False)
        self.download_processed_btn.setToolTip("Run a match first to enable downloading the WAV copy.")
        self.download_processed_btn.clicked.connect(self._on_download_processed)
        download_row.addWidget(self.download_processed_btn)
        download_row.addStretch()
        outer.addLayout(download_row)

        outer.addStretch()

    def _section_label(self, text):
        lab = QLabel(text)
        lab.setObjectName("SectionTitle")
        return lab

    # ------------------------------------------------------------ Upload tab
    def _build_upload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.upload_btn = QPushButton("📁 Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.setToolTip("Supported formats: WAV, MP3, FLAC, OGG")
        self.upload_btn.clicked.connect(self._on_upload)
        row.addWidget(self.upload_btn)

        self.remove_btn = QPushButton("✖ Remove")
        self.remove_btn.setObjectName("DangerButton")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setToolTip("Clear the uploaded input audio.")
        self.remove_btn.clicked.connect(self._on_remove_input)
        row.addWidget(self.remove_btn)
        row.addStretch()
        layout.addLayout(row)

        self.upload_file_label = QLabel("No file selected.")
        self.upload_file_label.setObjectName("Caption")
        self.upload_file_label.setWordWrap(True)
        layout.addWidget(self.upload_file_label)
        layout.addStretch()
        return tab

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        if not path:
            return
        try:
            self._set_status("Analyzing", "#b8860b")
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            ext = os.path.splitext(path)[1].lstrip(".").upper()
            duration_s = len(audio) / dsp.FS
            self.upload_file_label.setText(
                f"✅ {os.path.basename(path)}  |  Format: {ext}  |  "
                f"Original SR: {orig_fs} Hz  |  Duration: {duration_s:.2f}s")
            self.remove_btn.setEnabled(True)
            self._set_query(audio, os.path.basename(path), "Uploaded File", os.path.basename(path), orig_fs)
            self._set_status("Ready", "#2f6690")
        except Exception as e:
            self.upload_file_label.setText(f"❌ Could not load file: {e}")
            self._set_status("Error", "#c0392b")

    def _on_remove_input(self):
        self.query_audio = None
        self.query_source_label = "No input yet"
        self.upload_file_label.setText("No file selected.")
        self.remove_btn.setEnabled(False)
        self.query_transport.clear()
        self.query_canvas.fig.clear()
        self.query_canvas.draw()
        self.query_info_label.setText(
            "🎵 No audio loaded\nUpload an audio file, record from your microphone, "
            "or select a demo to begin.")
        self.search_btn.setEnabled(False)
        self.best_match_label.setText("No results yet — provide input and press Match Audio.")
        self.compare_canvas.fig.clear()
        self.compare_canvas.draw()
        self.compare_match_transport.clear()
        self.download_btn.setEnabled(False)
        self.download_processed_btn.setEnabled(False)
        if hasattr(self, "_matched_library_name"):
            del self._matched_library_name

    # ------------------------------------------------------------ Mic tab
    def _build_mic_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.mic_start_btn = QPushButton("🎙️ Start Recording")
        self.mic_start_btn.setObjectName("PrimaryButton")
        self.mic_start_btn.setToolTip("Start capturing a microphone sample for matching.")
        self.mic_start_btn.clicked.connect(self._on_mic_start)
        row.addWidget(self.mic_start_btn)

        self.mic_pause_btn = QPushButton("⏸ Pause")
        self.mic_pause_btn.setObjectName("SecondaryButton")
        self.mic_pause_btn.setEnabled(False)
        self.mic_pause_btn.setToolTip("Pause or resume microphone capture.")
        self.mic_pause_btn.clicked.connect(self._on_mic_pause)
        row.addWidget(self.mic_pause_btn)

        self.mic_stop_btn = QPushButton("⏹ Stop")
        self.mic_stop_btn.setObjectName("SecondaryButton")
        self.mic_stop_btn.setEnabled(False)
        self.mic_stop_btn.setToolTip("Stop microphone capture and use the recording as input.")
        self.mic_stop_btn.clicked.connect(self._on_mic_stop)
        row.addWidget(self.mic_stop_btn)

        self.mic_clear_btn = QPushButton("🗑 Clear")
        self.mic_clear_btn.setObjectName("SecondaryButton")
        self.mic_clear_btn.setEnabled(False)
        self.mic_clear_btn.setToolTip("Discard the current microphone recording.")
        self.mic_clear_btn.clicked.connect(self._on_mic_clear)
        row.addWidget(self.mic_clear_btn)
        row.addStretch()
        layout.addLayout(row)

        self.mic_level_bar = QProgressBar()
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setTextVisible(False)
        self.mic_level_bar.setFixedHeight(10)
        layout.addWidget(self.mic_level_bar)

        self.mic_status_label = QLabel("Not recording.")
        self.mic_status_label.setObjectName("Caption")
        layout.addWidget(self.mic_status_label)

        self.mic_live_canvas = MplCanvas(n_rows=1, figsize=(7, 1.3))
        layout.addWidget(self.mic_live_canvas)
        layout.addStretch()
        return tab

    def _on_mic_start(self):
        self._mic_chunks = []
        self.mic_start_btn.setEnabled(False)
        self.mic_pause_btn.setEnabled(True)
        self.mic_stop_btn.setEnabled(True)
        self.mic_clear_btn.setEnabled(False)
        self.mic_status_label.setText("🔴 Recording...")
        self._set_status("Recording", "#c0392b")

        self._mic_thread = MicRecorderThread(duration=30, fs=dsp.FS, parent=self)
        self._mic_thread.level_update.connect(self._on_mic_level)
        self._mic_thread.chunk_ready.connect(self._on_mic_chunk)
        self._mic_thread.finished.connect(self._on_mic_finished)
        self._mic_thread.start()

    def _on_mic_pause(self):
        if not self._mic_thread:
            return
        paused = self.mic_pause_btn.text().startswith("⏸")
        self._mic_thread.set_paused(paused)
        self.mic_pause_btn.setText("▶ Resume" if paused else "⏸ Pause")
        self.mic_status_label.setText("⏸ Paused." if paused else "🔴 Recording...")

    def _on_mic_stop(self):
        if self._mic_thread:
            self._mic_thread.stop_recording()
        self.mic_start_btn.setEnabled(True)
        self.mic_pause_btn.setEnabled(False)
        self.mic_pause_btn.setText("⏸ Pause")
        self.mic_stop_btn.setEnabled(False)

    def _on_mic_clear(self):
        self._mic_chunks = []
        self.mic_live_canvas.fig.clear()
        self.mic_live_canvas.draw()
        self.mic_clear_btn.setEnabled(False)
        self.mic_status_label.setText("Not recording.")
        self._on_remove_input()

    def _on_mic_level(self, rms):
        self.mic_level_bar.setValue(min(int(rms * 500), 100))

    def _on_mic_chunk(self, chunk):
        self._mic_chunks.append(chunk)
        # Redraw the live waveform from real captured audio (not simulated)
        if len(self._mic_chunks) % 3 == 0:  # throttle redraws for smoothness
            buf = np.concatenate(self._mic_chunks)
            self.mic_live_canvas.plot_waveforms([(buf, "Live mic input")], "Recording live...")

    def _on_mic_finished(self, audio, fs):
        self.mic_level_bar.setValue(0)
        self.mic_status_label.setText(f"✅ Recorded {len(audio)/fs:.2f}s.")
        self.mic_clear_btn.setEnabled(True)
        self._set_status("Ready", "#2f6690")
        if len(audio) > 1:
            self._set_query(dsp.normalize(audio), "Microphone recording",
                            "Microphone Recording", "Microphone recording", fs)
            self.mic_live_canvas.plot_waveforms([(audio, "Recorded input")], "Final recording")

    # ------------------------------------------------------------ Library panel
    def _refresh_library_panel(self):
        while self.library_grid.count():
            item = self.library_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for i, (name, item) in enumerate(self.library.items()):
            frame = QFrame()
            frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.7); border-radius: 10px; }")
            v = QVBoxLayout(frame)
            name_lab = QLabel(f"🎵 {name}")
            name_lab.setStyleSheet("font-weight: 700;")
            name_lab.setWordWrap(True)
            v.addWidget(name_lab)
            info_lab = QLabel(f"Category: Music  •  Duration: {item['duration_s']:.1f}s")
            info_lab.setObjectName("Caption")
            v.addWidget(info_lab)

            demo_btn = QPushButton("▶ Use as Demo Query")
            demo_btn.setObjectName("SecondaryButton")
            demo_btn.clicked.connect(lambda checked=False, n=name: self._load_demo_query(n))
            v.addWidget(demo_btn)

            self.library_grid.addWidget(frame, i // 3, i % 3)

    def _load_demo_query(self, library_name):
        """One-click demo: carve a short, slightly-noisy excerpt out of a
        real library track and use it as the query — the exact scenario the
        matcher is designed to solve, with zero uploading required."""
        full = self.library[library_name]["audio"]
        fs = dsp.FS
        dur_s = len(full) / fs
        excerpt_len = min(5.0, max(1.0, dur_s * 0.3))
        rng = np.random.default_rng(abs(hash(library_name)) % (2**32))
        max_start = max(0.0, dur_s - excerpt_len)
        start = rng.uniform(0, max_start) if max_start > 0 else 0.0
        excerpt = full[int(start * fs):int((start + excerpt_len) * fs)]
        excerpt = dsp.normalize(excerpt + 0.02 * rng.standard_normal(len(excerpt)))
        self._set_query(
            excerpt, f"Demo excerpt of '{library_name}' ({excerpt_len:.1f}s @ {start:.1f}s)",
            "Built-in Demo", library_name, fs)
        self._true_demo_source = library_name

    # ------------------------------------------------------------ shared query setter
    def _set_query(self, audio, label, source="—", filename="—", sample_rate=None):
        # Bound query length for responsiveness on very long uploads/recordings.
        max_len = int(15.0 * dsp.FS)
        if len(audio) > max_len:
            audio = audio[:max_len]
        self.query_audio = audio
        self.query_source_label = label
        self.query_info_label.setText(
            f"Query source: {label}  |  Duration: {len(audio)/dsp.FS:.2f}s  |  "
            f"RMS: {dsp.compute_rms(audio):.3f}  |  Peak: {dsp.compute_peak(audio):.3f}")
        self.query_canvas.plot_waveforms([(audio, "Input")], "Input Waveform")
        self.query_transport.set_audio(audio)
        self.compare_input_transport.set_audio(audio)
        self.input_card.set_audio(
            audio, sample_rate or dsp.FS, filename=filename or label, source=source)
        self.search_btn.setEnabled(True)
        self._reset_pipeline_visuals()

    # ------------------------------------------------------------ pipeline viz
    def _reset_pipeline_visuals(self):
        self._set_stage(None)
        self.pipeline_canvas.fig.clear()
        self.pipeline_canvas.draw()
        while self.progress_grid.count():
            item = self.progress_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.progress_bars = {}

    def _set_stage(self, active_index):
        for i, lab in enumerate(self.stage_labels):
            if active_index is None:
                lab.setStyleSheet("padding: 4px 8px; border-radius: 8px; background: #eef5fb;")
            elif i < active_index:
                lab.setStyleSheet("padding: 4px 8px; border-radius: 8px; background: #d3f0e8; color: #1e8a5f; font-weight: 600;")
            elif i == active_index:
                lab.setStyleSheet("padding: 4px 8px; border-radius: 8px; background: #6fb1ea; color: white; font-weight: 700;")
            else:
                lab.setStyleSheet("padding: 4px 8px; border-radius: 8px; background: #eef5fb;")

    def _set_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"font-weight: 700; color: {color};")

    # ------------------------------------------------------------ run matching
    def _start_matching(self):
        if self.query_audio is None or len(self.library) == 0:
            return
        self.search_btn.setEnabled(False)
        self._set_status("Processing", "#b8860b")
        self._reset_pipeline_visuals()
        self._set_stage(0)  # Input

        # Real "preprocessing" stage: show the normalized query alongside raw
        preprocessed = dsp.normalize(self.query_audio)
        self._set_stage(1)

        # Real "feature extraction" stage: compute and show the query's spectrogram
        f, t, Sxx = dsp.compute_spectrogram(self.query_audio)
        self.pipeline_canvas.fig.clear()
        axes = self.pipeline_canvas.fig.subplots(3, 1)
        self.pipeline_canvas.fig.subplots_adjust(
            left=0.10, right=0.985, top=0.84, bottom=0.14, hspace=0.72)
        ts = np.arange(len(self.query_audio)) / dsp.FS
        axes[0].plot(ts, self.query_audio, linewidth=0.7, color="#2f6690")
        axes[0].set_ylabel("Raw", fontsize=8, labelpad=5)
        axes[0].tick_params(labelsize=7)
        axes[1].plot(ts, preprocessed, linewidth=0.7, color="#4a8fa3")
        axes[1].set_ylabel("Preprocessed", fontsize=8, labelpad=5)
        axes[1].tick_params(labelsize=7)
        axes[2].pcolormesh(t, f, Sxx, shading='auto', cmap='viridis')
        axes[2].set_ylabel("Features (Hz)", fontsize=8, labelpad=5)
        axes[2].set_ylim(0, 4000)
        axes[2].tick_params(labelsize=7)
        self.pipeline_canvas.fig.suptitle("Preprocessing → Feature Extraction (live query data)", fontsize=9)
        self.pipeline_canvas.draw()
        self._set_stage(2)

        # Progress rows, one per library track — filled in live by the worker
        for i, name in enumerate(self.library.keys()):
            lab = QLabel(name)
            lab.setObjectName("Caption")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            self.progress_grid.addWidget(lab, i, 0)
            self.progress_grid.addWidget(bar, i, 1)
            self.progress_bars[name] = bar

        algorithm = self.algo_combo.currentText()
        self._set_stage(3)  # Algorithm Matching
        self._worker = MatchWorkerThread(self.query_audio, self.library, algorithm, parent=self)
        self._worker.progress.connect(self._on_match_progress)
        self._worker.finished_matching.connect(self._on_match_finished)
        self._worker.failed.connect(self._on_match_failed)
        self._worker.start()

    def _on_match_progress(self, name, fraction):
        if name in self.progress_bars:
            self.progress_bars[name].setValue(int(fraction * 100))

    def _on_match_failed(self, message):
        self._set_status("Error", "#c0392b")
        self.search_btn.setEnabled(True)
        QMessageBox.warning(self, "Matching failed", f"Could not complete matching:\n{message}")

    def _on_match_finished(self, result):
        self._set_stage(4)  # Scoring
        self._last_result = result
        self._set_stage(5)  # Ranking
        self._render_results(result)
        self._set_stage(6)  # Final Match
        self._set_status("Complete", "#1e8a5f")
        self.search_btn.setEnabled(True)

    # ------------------------------------------------------------ results
    def _render_results(self, result):
        ranked = result["ranked"]
        if not ranked:
            self.best_match_label.setText("No library tracks available to match against.")
            return

        best = ranked[0]
        conf_text, conf_color = _confidence_label(best["score"])
        if best["score"] < MATCH_THRESHOLD:
            self.best_match_label.setText(
                "⚠ No Reliable Match Found\n"
                f"Best candidate: {best['name']} — {best['score']*100:.1f}% "
                f"(below the {MATCH_THRESHOLD*100:.0f}% matching threshold)")
            self.best_match_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #c0392b;")
        else:
            self.best_match_label.setText(
                f"🏆 Best Match: {best['name']}\n"
                f"Similarity: {best['score']*100:.1f}%   |   Algorithm: {result['algorithm']}   |   "
                f"Time: {result['elapsed_s']:.2f}s   |   {conf_text}"
            )
            self.best_match_label.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {conf_color};")

        # Ranked alternatives list
        while self.ranked_list_layout.count():
            item = self.ranked_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for i, r in enumerate(ranked[:3]):
            is_best = i == 0
            row = QLabel(f"#{i+1}  {r['name']}   {r['score']*100:.1f}%" + ("   ← Best Match" if is_best else ""))
            row.setStyleSheet(
                "font-weight: 700; color: #1e8a5f; padding: 4px 8px; background: rgba(143,211,199,0.3); border-radius: 8px;"
                if is_best else
                "color: #445872; padding: 4px 8px;"
            )
            self.ranked_list_layout.addWidget(row)

        # Comparison view
        if best["score"] < MATCH_THRESHOLD:
            self.compare_canvas.fig.clear()
            self.compare_canvas.draw()
            self.compare_match_transport.clear()
            self.download_btn.setEnabled(False)
            self.download_processed_btn.setEnabled(False)
            if hasattr(self, "_matched_library_name"):
                del self._matched_library_name
            return

        matched_audio = self.library[best["name"]]["audio"]
        self.compare_canvas.plot_waveforms(
            [(self.query_audio, "Input"), (matched_audio, "Matched")],
            f"Input vs. Matched — {best['name']}")
        self.compare_match_transport.set_audio(matched_audio)
        self.compare_input_transport.position_changed.connect(self.compare_canvas.update_playhead)
        self.compare_match_transport.position_changed.connect(self.compare_canvas.update_playhead)

        self._matched_library_name = best["name"]
        self.download_btn.setEnabled(True)
        self.download_processed_btn.setEnabled(True)

    # ------------------------------------------------------------ download
    def _on_download_original(self):
        if not hasattr(self, "_matched_library_name"):
            return
        name = self._matched_library_name
        src_path = self.library[name]["path"]
        ext = os.path.splitext(src_path)[1]
        default_name = f"{name}{ext}"
        dest, _ = QFileDialog.getSaveFileName(self, "Save Matched Audio", default_name)
        if not dest:
            return
        try:
            shutil.copyfile(src_path, dest)
            QMessageBox.information(self, "Download complete", f"Saved to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Download failed", f"Could not save file:\n{e}")

    def _on_download_processed(self):
        if not hasattr(self, "_matched_library_name"):
            return
        name = self._matched_library_name
        audio = self.library[name]["audio"]
        default_name = f"{name}_processed.wav"
        dest, _ = QFileDialog.getSaveFileName(self, "Save Processed Audio (WAV)", default_name)
        if not dest:
            return
        try:
            dsp.write_wav_file(dest, audio, dsp.FS)
            QMessageBox.information(self, "Download complete", f"Saved to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Download failed", f"Could not save file:\n{e}")
