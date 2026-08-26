"""
Audio Editor Page – full-featured audio editing with:
  ✂️  Trim & Join    – cut segments or concatenate multiple files
  🔄 Reverse        – play audio backwards
  ⏩ Time-scale     – change playback speed
  🔊 Fade In/Out    – gradual volume ramp at start/end
  🌊 Convolution FX – smoothing (moving-average) and echo (impulse-response)

Songs from the audios/ folder are shown in a library panel for quick loading.
"""

import os
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame,
    QPushButton, QFileDialog, QTabWidget, QListWidget, QListWidgetItem,
    QScrollArea, QSizePolicy, QSpacerItem,
)
from PySide6.QtCore import Qt, Signal

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton

# Path to the project's built-in audio library
AUDIOS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "audios")
)

# ---------------------------------------------------------------------------
#  Helper: styled card frame
# ---------------------------------------------------------------------------

def _card_frame(bg_alpha=0.82, radius=14):
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background: rgba(255,255,255,{bg_alpha});"
        f" border-radius: {radius}px; }}"
    )
    return frame


# ---------------------------------------------------------------------------
#  Clickable song card for the library row
# ---------------------------------------------------------------------------

class _SongCard(QFrame):
    clicked = Signal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.path = file_path
        self._selected = False
        self._base_style = (
            "QFrame { background: rgba(255,255,255,0.88);"
            " border-radius: 12px; border: 1.5px solid #d6e2ef; }"
            "QFrame:hover { background: #eaf4ff; border: 1.5px solid #6fb1ea; }"
        )
        self._sel_style = (
            "QFrame { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            " stop:0 #6fb1ea, stop:1 #8fd3c7);"
            " border-radius: 12px; border: none; }"
        )
        self.setStyleSheet(self._base_style)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(64)
        self.setMinimumWidth(170)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(2)

        name = os.path.splitext(os.path.basename(file_path))[0]
        self._title = QLabel(f"🎵  {name}")
        self._title.setStyleSheet("font-weight:600; font-size:11px; color:#2f3e50;")
        self._title.setWordWrap(True)

        ext = os.path.splitext(file_path)[1].upper().lstrip(".")
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            info_text = f"{ext} • {size_mb:.1f} MB"
        except OSError:
            info_text = ext
        self._info = QLabel(info_text)
        self._info.setStyleSheet("font-size:10px; color:#6b7f96;")

        lay.addWidget(self._title)
        lay.addWidget(self._info)

    def set_selected(self, sel):
        self._selected = sel
        if sel:
            self.setStyleSheet(self._sel_style)
            self._title.setStyleSheet("font-weight:700; font-size:11px; color:white;")
            self._info.setStyleSheet("font-size:10px; color:rgba(255,255,255,0.85);")
        else:
            self.setStyleSheet(self._base_style)
            self._title.setStyleSheet("font-weight:600; font-size:11px; color:#2f3e50;")
            self._info.setStyleSheet("font-size:10px; color:#6b7f96;")

    def mousePressEvent(self, ev):
        self.clicked.emit(self.path)
        super().mousePressEvent(ev)


# ===================================================================
#  Main Editor Page
# ===================================================================

class EditorPage(QWidget):
    """Full-featured audio editor with library, trim/join, reverse,
    time-scale, fade, and convolution effects."""

    # Tab indices
    _TAB_TRIM = 0
    _TAB_REVERSE = 1
    _TAB_SPEED = 2
    _TAB_FADE = 3
    _TAB_FX = 4

    def __init__(self):
        super().__init__()
        self.source_audio = None
        self.processed_audio = None
        self.join_items = []           # list of (name, np.ndarray)
        self._library_cards = []       # _SongCard references
        self._current_lib_path = None  # which library song is selected
        self._build_ui()
        self._scan_library()
        self._load_demo_signal()

    # ---------------------------------------------------------------
    #  UI construction
    # ---------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ---- scrollable content ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{background:transparent; border:none;}")
        container = QWidget()
        container.setStyleSheet("background:transparent;")
        self.layout_inner = QVBoxLayout(container)
        self.layout_inner.setSpacing(10)
        scroll.setWidget(container)
        root.addWidget(scroll)

        lay = self.layout_inner

        # ===== HEADER CARD =====
        hdr = _card_frame()
        hdr_l = QVBoxLayout(hdr)
        title = QLabel("✂️  Audio Editor")
        title.setObjectName("TitleLabel")
        sub = QLabel(
            "Load a song from your library or upload your own  •  "
            "Trim & Join · Reverse · Time-scale · Fade · Convolution Effects"
        )
        sub.setObjectName("Caption")
        sub.setWordWrap(True)
        hdr_l.addWidget(title)
        hdr_l.addWidget(sub)
        lay.addWidget(hdr)

        # ===== AUDIO LIBRARY CARD =====
        lib_card = _card_frame(bg_alpha=0.72)
        lib_lay = QVBoxLayout(lib_card)
        lib_lay.setSpacing(8)
        lib_title = QLabel("🎵  Audio Library")
        lib_title.setObjectName("SectionTitle")
        lib_lay.addWidget(lib_title)

        # horizontal scrollable row for song cards
        self.lib_scroll = QScrollArea()
        self.lib_scroll.setFixedHeight(80)
        self.lib_scroll.setWidgetResizable(True)
        self.lib_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.lib_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lib_scroll.setStyleSheet(
            "QScrollArea{background:transparent; border:none;}"
        )
        self.lib_row_widget = QWidget()
        self.lib_row_widget.setStyleSheet("background:transparent;")
        self.lib_row_layout = QHBoxLayout(self.lib_row_widget)
        self.lib_row_layout.setContentsMargins(0, 0, 0, 0)
        self.lib_row_layout.setSpacing(10)
        self.lib_row_layout.addStretch()
        self.lib_scroll.setWidget(self.lib_row_widget)
        lib_lay.addWidget(self.lib_scroll)

        # upload + demo buttons
        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁  Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.clicked.connect(self._on_upload)
        btn_row.addWidget(self.upload_btn)

        self.demo_btn = QPushButton("🔊  Demo Signal")
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.clicked.connect(self._load_demo_signal)
        btn_row.addWidget(self.demo_btn)
        btn_row.addStretch()
        lib_lay.addLayout(btn_row)

        self.file_label = QLabel("No audio loaded")
        self.file_label.setObjectName("Caption")
        self.file_label.setWordWrap(True)
        lib_lay.addWidget(self.file_label)
        lay.addWidget(lib_card)

        # ===== EFFECT TABS =====
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_trim_tab(),    "✂️ Trim && Join")
        self.tabs.addTab(self._build_reverse_tab(), "🔄 Reverse")
        self.tabs.addTab(self._build_speed_tab(),   "⏩ Time-scale")
        self.tabs.addTab(self._build_fade_tab(),    "🔊 Fade In/Out")
        self.tabs.addTab(self._build_fx_tab(),      "🌊 Convolution FX")
        self.tabs.currentChanged.connect(lambda _: self._refresh())
        lay.addWidget(self.tabs)

        # ===== WAVEFORM CANVAS =====
        self.canvas = MplCanvas(n_rows=2, figsize=(7, 1.6))
        lay.addWidget(self.canvas)

        # ===== PLAY BUTTONS + SAVE =====
        play_row = QHBoxLayout()
        self.p_original = AudioPlayButton("▶ Original")
        self.p_processed = AudioPlayButton("▶ Processed")
        play_row.addWidget(self.p_original)
        play_row.addWidget(self.p_processed)

        self.save_btn = QPushButton("💾  Save Processed Audio")
        self.save_btn.setObjectName("SecondaryButton")
        self.save_btn.clicked.connect(self._on_save)
        play_row.addWidget(self.save_btn)
        play_row.addStretch()
        lay.addLayout(play_row)

        # connect playhead sync
        for btn in (self.p_original, self.p_processed):
            btn.position_changed.connect(self.canvas.update_playhead)
            btn.playback_active_changed.connect(
                lambda active: None if active else self.canvas.clear_playhead()
            )

        lay.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

    # ---- Tab builders ----

    def _build_trim_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        # -- Trim controls --
        trim_title = QLabel("✂️  Trim Audio")
        trim_title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(trim_title)
        trim_hint = QLabel("Select start and end points to keep a portion of the audio.")
        trim_hint.setObjectName("Caption")
        lay.addWidget(trim_hint)

        row = QHBoxLayout()

        col1 = QVBoxLayout()
        self.trim_start_label = QLabel("Start: 0.00 s")
        self.trim_start_label.setObjectName("Caption")
        self.trim_start_slider = QSlider(Qt.Horizontal)
        self.trim_start_slider.setRange(0, 1000)
        self.trim_start_slider.setValue(0)
        self.trim_start_slider.valueChanged.connect(self._on_trim_changed)
        col1.addWidget(self.trim_start_label)
        col1.addWidget(self.trim_start_slider)
        row.addLayout(col1)

        col2 = QVBoxLayout()
        self.trim_end_label = QLabel("End: 1.00 s")
        self.trim_end_label.setObjectName("Caption")
        self.trim_end_slider = QSlider(Qt.Horizontal)
        self.trim_end_slider.setRange(0, 1000)
        self.trim_end_slider.setValue(1000)
        self.trim_end_slider.valueChanged.connect(self._on_trim_changed)
        col2.addWidget(self.trim_end_label)
        col2.addWidget(self.trim_end_slider)
        row.addLayout(col2)

        lay.addLayout(row)

        # -- separator --
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #d6e2ef;")
        lay.addWidget(sep)

        # -- Join controls --
        join_title = QLabel("📎  Join Multiple Audio Files")
        join_title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(join_title)
        join_hint = QLabel(
            "Add audio files to the list below, then click 'Join All' to "
            "concatenate them into one continuous file."
        )
        join_hint.setObjectName("Caption")
        join_hint.setWordWrap(True)
        lay.addWidget(join_hint)

        self.join_list_widget = QListWidget()
        self.join_list_widget.setMaximumHeight(90)
        self.join_list_widget.setStyleSheet(
            "QListWidget { background: rgba(255,255,255,0.7);"
            " border-radius: 8px; border: 1px solid #d6e2ef; padding: 4px; }"
        )
        lay.addWidget(self.join_list_widget)

        jbtn_row = QHBoxLayout()
        add_current = QPushButton("➕ Add Current Audio")
        add_current.setObjectName("SecondaryButton")
        add_current.clicked.connect(self._join_add_current)
        jbtn_row.addWidget(add_current)

        add_file = QPushButton("📁 Add From File")
        add_file.setObjectName("SecondaryButton")
        add_file.clicked.connect(self._join_add_file)
        jbtn_row.addWidget(add_file)

        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setObjectName("SecondaryButton")
        clear_btn.clicked.connect(self._join_clear)
        jbtn_row.addWidget(clear_btn)

        join_btn = QPushButton("🔗  Join All")
        join_btn.setObjectName("PrimaryButton")
        join_btn.clicked.connect(self._apply_join)
        jbtn_row.addWidget(join_btn)
        jbtn_row.addStretch()
        lay.addLayout(jbtn_row)

        lay.addStretch()
        return w

    def _build_reverse_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        title = QLabel("🔄  Reverse Audio")
        title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(title)

        desc = QLabel(
            "Reverses the audio so it plays from end to start. "
            "The effect is applied instantly to the loaded audio."
        )
        desc.setObjectName("Caption")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        hint = QLabel(
            "💡 Tip: Reversing audio is used in sound design, "
            "creating build-ups, and analysing hidden messages!"
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        lay.addStretch()
        return w

    def _build_speed_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        title = QLabel("⏩  Time-scale (Playback Speed)")
        title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(title)

        desc = QLabel(
            "Change the playback speed. Faster speeds shorten the audio; "
            "slower speeds stretch it. Uses resampling (pitch also shifts)."
        )
        desc.setObjectName("Caption")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        row = QHBoxLayout()
        slow_label = QLabel("🐢 0.25×")
        slow_label.setObjectName("Caption")
        row.addWidget(slow_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(25, 400)   # 0.25x to 4.0x
        self.speed_slider.setValue(100)        # 1.0x default
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        row.addWidget(self.speed_slider, 1)

        fast_label = QLabel("🐇 4.0×")
        fast_label.setObjectName("Caption")
        row.addWidget(fast_label)
        lay.addLayout(row)

        self.speed_value_label = QLabel("Speed: 1.00×")
        self.speed_value_label.setStyleSheet(
            "font-weight:700; font-size:15px; color:#2f6690; padding: 4px;"
        )
        self.speed_value_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.speed_value_label)

        lay.addStretch()
        return w

    def _build_fade_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        title = QLabel("🔊  Fade In / Fade Out")
        title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(title)

        desc = QLabel(
            "Gradually increase volume at the start (Fade In) or "
            "decrease it at the end (Fade Out). Avoids harsh audio clicks."
        )
        desc.setObjectName("Caption")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        row = QHBoxLayout()

        col1 = QVBoxLayout()
        self.fadein_label = QLabel("Fade In: 0.00 s")
        self.fadein_label.setObjectName("Caption")
        self.fadein_slider = QSlider(Qt.Horizontal)
        self.fadein_slider.setRange(0, 500)  # 0 – 5.0s in 10ms steps
        self.fadein_slider.setValue(0)
        self.fadein_slider.valueChanged.connect(self._on_fade_changed)
        col1.addWidget(self.fadein_label)
        col1.addWidget(self.fadein_slider)
        row.addLayout(col1)

        col2 = QVBoxLayout()
        self.fadeout_label = QLabel("Fade Out: 0.00 s")
        self.fadeout_label.setObjectName("Caption")
        self.fadeout_slider = QSlider(Qt.Horizontal)
        self.fadeout_slider.setRange(0, 500)
        self.fadeout_slider.setValue(0)
        self.fadeout_slider.valueChanged.connect(self._on_fade_changed)
        col2.addWidget(self.fadeout_label)
        col2.addWidget(self.fadeout_slider)
        row.addLayout(col2)

        lay.addLayout(row)
        lay.addStretch()
        return w

    def _build_fx_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)

        title = QLabel("🌊  Convolution-based Effects")
        title.setStyleSheet("font-weight:700; font-size:13px; color:#2f3e50;")
        lay.addWidget(title)

        desc = QLabel(
            "Apply effects that use convolution with a kernel (impulse response). "
            "Smoothing uses a uniform box kernel; Echo uses a decaying impulse train."
        )
        desc.setObjectName("Caption")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        # -- effect selector --
        sel_row = QHBoxLayout()
        self.fx_smooth_btn = QPushButton("🫧  Smoothing")
        self.fx_smooth_btn.setObjectName("PrimaryButton")
        self.fx_smooth_btn.setCheckable(True)
        self.fx_smooth_btn.setChecked(True)
        self.fx_smooth_btn.clicked.connect(lambda: self._select_fx("smooth"))
        sel_row.addWidget(self.fx_smooth_btn)

        self.fx_echo_btn = QPushButton("🔁  Echo")
        self.fx_echo_btn.setObjectName("SecondaryButton")
        self.fx_echo_btn.setCheckable(True)
        self.fx_echo_btn.clicked.connect(lambda: self._select_fx("echo"))
        sel_row.addWidget(self.fx_echo_btn)
        sel_row.addStretch()
        lay.addLayout(sel_row)

        # -- Smoothing controls --
        self.smooth_frame = QFrame()
        sl = QVBoxLayout(self.smooth_frame)
        sl.setContentsMargins(0, 4, 0, 0)
        self.smooth_label = QLabel("Kernel size: 50 samples")
        self.smooth_label.setObjectName("Caption")
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(2, 500)
        self.smooth_slider.setValue(50)
        self.smooth_slider.valueChanged.connect(self._on_fx_changed)
        sl.addWidget(self.smooth_label)
        sl.addWidget(self.smooth_slider)
        smooth_hint = QLabel(
            "💡 A larger kernel produces stronger smoothing (low-pass filtering). "
            "This convolves the signal with a uniform averaging window."
        )
        smooth_hint.setObjectName("HintLabel")
        smooth_hint.setWordWrap(True)
        sl.addWidget(smooth_hint)
        lay.addWidget(self.smooth_frame)

        # -- Echo controls --
        self.echo_frame = QFrame()
        el = QVBoxLayout(self.echo_frame)
        el.setContentsMargins(0, 4, 0, 0)

        erow = QHBoxLayout()
        ecol1 = QVBoxLayout()
        self.echo_delay_label = QLabel("Delay: 0.15 s")
        self.echo_delay_label.setObjectName("Caption")
        self.echo_delay_slider = QSlider(Qt.Horizontal)
        self.echo_delay_slider.setRange(2, 60)   # 0.02 – 0.60s
        self.echo_delay_slider.setValue(15)
        self.echo_delay_slider.valueChanged.connect(self._on_fx_changed)
        ecol1.addWidget(self.echo_delay_label)
        ecol1.addWidget(self.echo_delay_slider)
        erow.addLayout(ecol1)

        ecol2 = QVBoxLayout()
        self.echo_decay_label = QLabel("Decay: 0.50")
        self.echo_decay_label.setObjectName("Caption")
        self.echo_decay_slider = QSlider(Qt.Horizontal)
        self.echo_decay_slider.setRange(5, 95)
        self.echo_decay_slider.setValue(50)
        self.echo_decay_slider.valueChanged.connect(self._on_fx_changed)
        ecol2.addWidget(self.echo_decay_label)
        ecol2.addWidget(self.echo_decay_slider)
        erow.addLayout(ecol2)

        el.addLayout(erow)
        echo_hint = QLabel(
            "💡 Echo convolves the signal with a decaying impulse train — "
            "each repeat is quieter by the decay factor."
        )
        echo_hint.setObjectName("HintLabel")
        echo_hint.setWordWrap(True)
        el.addWidget(echo_hint)
        self.echo_frame.hide()
        lay.addWidget(self.echo_frame)

        self._active_fx = "smooth"

        lay.addStretch()
        return w

    # ---------------------------------------------------------------
    #  Library scanning
    # ---------------------------------------------------------------

    def _scan_library(self):
        if not os.path.isdir(AUDIOS_DIR):
            return
        exts = (".wav", ".mp3", ".flac", ".ogg")
        files = sorted(
            f for f in os.listdir(AUDIOS_DIR)
            if os.path.splitext(f)[1].lower() in exts
        )
        # Remove the trailing stretch first
        stretch = self.lib_row_layout.takeAt(self.lib_row_layout.count() - 1)

        for fname in files:
            path = os.path.join(AUDIOS_DIR, fname)
            card = _SongCard(path)
            card.clicked.connect(self._on_library_song)
            self._library_cards.append(card)
            self.lib_row_layout.addWidget(card)

        # Re-add stretch
        self.lib_row_layout.addStretch()

    def _on_library_song(self, path):
        try:
            audio, orig_fs = dsp.load_sample_song(path, target_fs=dsp.FS, max_duration=30)
            self.source_audio = audio
            self._current_lib_path = path
            name = os.path.splitext(os.path.basename(path))[0]
            dur = len(audio) / dsp.FS
            self.file_label.setText(
                f"✅  {name}   ({orig_fs} Hz → {dsp.FS} Hz,  {dur:.1f} s)"
            )
            # highlight card
            for c in self._library_cards:
                c.set_selected(c.path == path)
            self._update_trim_range()
            self._refresh()
        except Exception as e:
            self.file_label.setText(f"❌  Error loading: {e}")

    # ---------------------------------------------------------------
    #  Upload / Demo
    # ---------------------------------------------------------------

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;"
            "WAV (*.wav);;MP3 (*.mp3);;All Files (*)"
        )
        if not path:
            return
        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)
            self.source_audio = audio
            self._current_lib_path = None
            for c in self._library_cards:
                c.set_selected(False)
            name = os.path.basename(path)
            dur = len(audio) / dsp.FS
            self.file_label.setText(
                f"✅  {name}   ({orig_fs} Hz → {dsp.FS} Hz,  {dur:.2f} s)"
            )
            self._update_trim_range()
            self._refresh()
        except Exception as e:
            self.file_label.setText(f"❌  Error: {e}")

    def _load_demo_signal(self):
        self.source_audio = dsp.make_voice_like()
        self._current_lib_path = None
        for c in self._library_cards:
            c.set_selected(False)
        dur = len(self.source_audio) / dsp.FS
        self.file_label.setText(f"🔊  Demo signal — synthetic voice-like tone  ({dur:.1f} s)")
        self._update_trim_range()
        self._refresh()

    # ---------------------------------------------------------------
    #  Trim helpers
    # ---------------------------------------------------------------

    def _update_trim_range(self):
        if self.source_audio is None:
            return
        dur = len(self.source_audio) / dsp.FS
        # slider values map to 0..1000 → 0..duration
        self.trim_start_slider.blockSignals(True)
        self.trim_end_slider.blockSignals(True)
        self.trim_start_slider.setValue(0)
        self.trim_end_slider.setValue(1000)
        self.trim_start_slider.blockSignals(False)
        self.trim_end_slider.blockSignals(False)
        self._update_trim_labels()

    def _update_trim_labels(self):
        if self.source_audio is None:
            return
        dur = len(self.source_audio) / dsp.FS
        s = self.trim_start_slider.value() / 1000.0 * dur
        e = self.trim_end_slider.value() / 1000.0 * dur
        self.trim_start_label.setText(f"Start: {s:.2f} s")
        self.trim_end_label.setText(f"End: {e:.2f} s")

    # ---------------------------------------------------------------
    #  Control callbacks → _refresh()
    # ---------------------------------------------------------------

    def _on_trim_changed(self, _=None):
        self._update_trim_labels()
        if self.tabs.currentIndex() == self._TAB_TRIM:
            self._refresh()

    def _on_speed_changed(self, val):
        speed = val / 100.0
        self.speed_value_label.setText(f"Speed: {speed:.2f}×")
        if self.tabs.currentIndex() == self._TAB_SPEED:
            self._refresh()

    def _on_fade_changed(self, _=None):
        fi = self.fadein_slider.value() / 100.0
        fo = self.fadeout_slider.value() / 100.0
        self.fadein_label.setText(f"Fade In: {fi:.2f} s")
        self.fadeout_label.setText(f"Fade Out: {fo:.2f} s")
        if self.tabs.currentIndex() == self._TAB_FADE:
            self._refresh()

    def _on_fx_changed(self, _=None):
        ks = self.smooth_slider.value()
        self.smooth_label.setText(f"Kernel size: {ks} samples")
        d = self.echo_delay_slider.value() / 100.0
        dc = self.echo_decay_slider.value() / 100.0
        self.echo_delay_label.setText(f"Delay: {d:.2f} s")
        self.echo_decay_label.setText(f"Decay: {dc:.2f}")
        if self.tabs.currentIndex() == self._TAB_FX:
            self._refresh()

    def _select_fx(self, which):
        self._active_fx = which
        if which == "smooth":
            self.fx_smooth_btn.setChecked(True)
            self.fx_smooth_btn.setObjectName("PrimaryButton")
            self.fx_echo_btn.setChecked(False)
            self.fx_echo_btn.setObjectName("SecondaryButton")
            self.smooth_frame.show()
            self.echo_frame.hide()
        else:
            self.fx_smooth_btn.setChecked(False)
            self.fx_smooth_btn.setObjectName("SecondaryButton")
            self.fx_echo_btn.setChecked(True)
            self.fx_echo_btn.setObjectName("PrimaryButton")
            self.smooth_frame.hide()
            self.echo_frame.show()
        # Re-apply style (objectName changed)
        self.fx_smooth_btn.style().unpolish(self.fx_smooth_btn)
        self.fx_smooth_btn.style().polish(self.fx_smooth_btn)
        self.fx_echo_btn.style().unpolish(self.fx_echo_btn)
        self.fx_echo_btn.style().polish(self.fx_echo_btn)
        if self.tabs.currentIndex() == self._TAB_FX:
            self._refresh()

    # ---------------------------------------------------------------
    #  Join operations
    # ---------------------------------------------------------------

    def _join_add_current(self):
        if self.source_audio is None or len(self.source_audio) < 2:
            return
        name = "Current audio"
        if self._current_lib_path:
            name = os.path.splitext(os.path.basename(self._current_lib_path))[0]
        dur = len(self.source_audio) / dsp.FS
        self.join_items.append((name, self.source_audio.copy()))
        self.join_list_widget.addItem(
            f"🎵 {name}  ({dur:.1f}s)"
        )

    def _join_add_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio to Join", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        if not path:
            return
        try:
            audio, _ = dsp.load_audio_file(path, target_fs=dsp.FS)
            name = os.path.splitext(os.path.basename(path))[0]
            dur = len(audio) / dsp.FS
            self.join_items.append((name, audio))
            self.join_list_widget.addItem(
                f"🎵 {name}  ({dur:.1f}s)"
            )
        except Exception as e:
            self.file_label.setText(f"❌  Join add error: {e}")

    def _join_clear(self):
        self.join_items.clear()
        self.join_list_widget.clear()

    def _apply_join(self):
        if len(self.join_items) < 2:
            self.file_label.setText("⚠️  Add at least 2 audio files to join.")
            return
        arrays = [arr for _, arr in self.join_items]
        joined = dsp.join(*arrays)
        self.processed_audio = joined
        dur = len(joined) / dsp.FS
        names = " + ".join(n for n, _ in self.join_items)
        self.file_label.setText(f"🔗  Joined {len(self.join_items)} files: {names}  ({dur:.1f}s)")
        self.p_processed.set_audio(joined)

        # Show a combined waveform
        self.canvas.plot_waveforms(
            [(self.source_audio if self.source_audio is not None else joined, "Current Source"),
             (joined, "Joined Result")],
            "Join — concatenated audio"
        )

    # ---------------------------------------------------------------
    #  Core refresh: applies the active effect
    # ---------------------------------------------------------------

    def _refresh(self):
        if self.source_audio is None:
            return

        src = self.source_audio
        tab = self.tabs.currentIndex()
        title = "Editor"

        if tab == self._TAB_TRIM:
            dur = len(src) / dsp.FS
            s = self.trim_start_slider.value() / 1000.0 * dur
            e = self.trim_end_slider.value() / 1000.0 * dur
            if e <= s:
                e = s + 0.01
            processed = dsp.trim(src, s, e)
            title = f"Trim — keeping [{s:.2f}s – {e:.2f}s]"
            proc_label = f"Trimmed ({len(processed)/dsp.FS:.2f}s)"

        elif tab == self._TAB_REVERSE:
            processed = dsp.reverse(src)
            title = "Reverse — audio played backwards"
            proc_label = "Reversed"

        elif tab == self._TAB_SPEED:
            speed = self.speed_slider.value() / 100.0
            processed = dsp.time_scale(src, speed)
            title = f"Time-scale — {speed:.2f}× speed"
            proc_label = f"Scaled ({len(processed)/dsp.FS:.2f}s)"

        elif tab == self._TAB_FADE:
            fi = self.fadein_slider.value() / 100.0
            fo = self.fadeout_slider.value() / 100.0
            processed = dsp.fade(src, fade_in_s=fi, fade_out_s=fo)
            title = f"Fade — In: {fi:.2f}s, Out: {fo:.2f}s"
            proc_label = "Faded"

        elif tab == self._TAB_FX:
            if self._active_fx == "smooth":
                ks = self.smooth_slider.value()
                processed = dsp.smooth(src, kernel_size=ks)
                title = f"Smoothing — kernel size {ks}"
                proc_label = "Smoothed"
            else:
                d = self.echo_delay_slider.value() / 100.0
                dc = self.echo_decay_slider.value() / 100.0
                processed = dsp.echo(src, delay_s=d, decay=dc)
                title = f"Echo — delay {d:.2f}s, decay {dc:.2f}"
                proc_label = "Echo"
        else:
            processed = src
            proc_label = "Processed"

        self.processed_audio = processed
        self.p_original.set_audio(src)
        self.p_processed.set_audio(processed)

        self.canvas.plot_waveforms(
            [(src, "Original"), (processed, proc_label)],
            title
        )

    # ---------------------------------------------------------------
    #  Save processed audio
    # ---------------------------------------------------------------

    def _on_save(self):
        if self.processed_audio is None:
            self.file_label.setText("⚠️  Nothing to save — apply an effect first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Audio", "processed_audio.wav",
            "WAV Files (*.wav)"
        )
        if not path:
            return
        try:
            from scipy.io import wavfile
            x16 = np.clip(self.processed_audio, -1, 1)
            x16 = (x16 * 32767).astype(np.int16)
            wavfile.write(path, dsp.FS, x16)
            self.file_label.setText(f"💾  Saved → {os.path.basename(path)}")
        except Exception as e:
            self.file_label.setText(f"❌  Save error: {e}")
