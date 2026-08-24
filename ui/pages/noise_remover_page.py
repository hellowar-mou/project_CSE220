import os
import tempfile
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame,
    QPushButton, QFileDialog, QComboBox, QGroupBox, QGridLayout,
    QProgressBar, QSplitter, QTabWidget, QSizePolicy, QSpacerItem,
    QScrollArea
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor, QPalette

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


# ─────────────────────────────────────────── Smooth Scroll Area
class SmoothScrollArea(QScrollArea):
    """QScrollArea with animated smooth scrolling on mouse wheel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target_value = 0
        self._anim = QPropertyAnimation(self.verticalScrollBar(), b"value", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        # Modern styled scrollbar
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 4px 2px 4px 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(111, 177, 234, 0.35);
                min-height: 40px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(111, 177, 234, 0.6);
            }
            QScrollBar::handle:vertical:pressed {
                background: rgba(111, 177, 234, 0.8);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
                border: none;
                height: 0px;
            }
        """)

    def wheelEvent(self, event):
        bar = self.verticalScrollBar()

        # If animation is running, use its target as the base
        if self._anim.state() == QPropertyAnimation.Running:
            current = self._target_value
        else:
            current = bar.value()

        # Calculate step
        delta = event.angleDelta().y()
        step = int(bar.singleStep() * 2.5)
        self._target_value = max(bar.minimum(),
                                 min(bar.maximum(), current - (delta // 120) * step))

        # Animate to target
        self._anim.stop()
        self._anim.setStartValue(bar.value())
        self._anim.setEndValue(self._target_value)
        self._anim.start()

        event.accept()


# ─────────────────────────────────────────── Mic Recorder Thread
class MicRecorderThread(QThread):
    """Records audio from the microphone in a background thread."""
    finished = Signal(np.ndarray, int)  # audio_data, sample_rate
    level_update = Signal(float)         # RMS level for VU meter

    def __init__(self, duration=5, fs=16000, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.fs = fs
        self._stop_flag = False

    def stop_recording(self):
        self._stop_flag = True

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.finished.emit(np.zeros(1), self.fs)
            return

        frames = []
        block_size = int(self.fs * 0.05)  # 50ms blocks
        total_blocks = int(self.duration * self.fs / block_size)

        def callback(indata, frame_count, time_info, status):
            frames.append(indata[:, 0].copy())
            rms = np.sqrt(np.mean(indata ** 2))
            self.level_update.emit(float(rms))

        try:
            with sd.InputStream(samplerate=self.fs, channels=1,
                                blocksize=block_size, callback=callback):
                for _ in range(total_blocks):
                    if self._stop_flag:
                        break
                    self.msleep(50)
        except Exception:
            pass

        if frames:
            audio = np.concatenate(frames)
            audio = dsp.normalize(audio)
            self.finished.emit(audio, self.fs)
        else:
            self.finished.emit(np.zeros(1), self.fs)


# ─────────────────────────────────────────── Status Pill Widget
class StatusPill(QLabel):
    """A small colored status indicator pill."""
    def __init__(self, text="Ready", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(26)
        self.setMinimumWidth(80)
        self._set_style("#2ecc71", "#e8faf1")

    def set_status(self, text, color="green"):
        colors = {
            "green": ("#2ecc71", "#e8faf1"),
            "blue": ("#3498db", "#ebf5fb"),
            "orange": ("#e67e22", "#fef5e7"),
            "red": ("#e74c3c", "#fdedec"),
            "purple": ("#9b59b6", "#f5eef8"),
        }
        fg, bg = colors.get(color, colors["green"])
        self._set_style(fg, bg)
        self.setText(text)

    def _set_style(self, fg, bg):
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {fg};
                font-size: 11px;
                font-weight: 700;
                border-radius: 13px;
                padding: 4px 14px;
                border: 1px solid {fg}40;
            }}
        """)


# ─────────────────────────────────────────── Metric Card
class MetricCard(QFrame):
    """A small card showing a metric value with label."""
    def __init__(self, title, value="—", icon="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.85);
                border-radius: 12px;
                border: 1px solid rgba(111,177,234,0.2);
            }
        """)
        self.setFixedHeight(80)
        self.setMinimumWidth(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        header = QLabel(f"{icon} {title}")
        header.setStyleSheet("font-size: 10px; color: #6b7f96; font-weight: 600; border: none; background: transparent;")
        layout.addWidget(header)

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #2f3e50; border: none; background: transparent;")
        layout.addWidget(self.value_label)

    def set_value(self, val):
        self.value_label.setText(val)


# ─────────────────────────────────────────── Song Card Widget
class SongCard(QFrame):
    """A clickable card representing a sample song."""
    clicked = Signal(str, str)  # song_key, file_path

    def __init__(self, song_key, title, artist, icon, file_path, parent=None):
        super().__init__(parent)
        self.song_key = song_key
        self.file_path = file_path
        self._selected = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(110)
        self.setMinimumWidth(200)
        self._apply_style(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        # Icon + Title row
        top_row = QHBoxLayout()
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 28px; background: transparent; border: none;")
        top_row.addWidget(icon_label)
        top_row.addStretch()

        # File exists indicator
        exists = os.path.exists(file_path)
        status_dot = QLabel("●" if exists else "○")
        status_dot.setStyleSheet(f"font-size: 10px; color: {'#2ecc71' if exists else '#e74c3c'}; background: transparent; border: none;")
        top_row.addWidget(status_dot)
        layout.addLayout(top_row)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 13px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Artist
        artist_label = QLabel(artist)
        artist_label.setStyleSheet("font-size: 10px; color: #6b7f96; font-weight: 500; background: transparent; border: none;")
        layout.addWidget(artist_label)

        # Duration hint
        dur_label = QLabel("30s clip" if exists else "File not found")
        dur_label.setStyleSheet(f"font-size: 9px; color: {'#8a9bb0' if exists else '#e74c3c'}; background: transparent; border: none;")
        layout.addWidget(dur_label)

    def _apply_style(self, selected):
        if selected:
            self.setStyleSheet("""
                QFrame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgba(111,177,234,0.25), stop:1 rgba(143,211,199,0.25));
                    border-radius: 14px;
                    border: 2px solid #6fb1ea;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background: rgba(255,255,255,0.85);
                    border-radius: 14px;
                    border: 1px solid rgba(111,177,234,0.15);
                }
                QFrame:hover {
                    background: rgba(255,255,255,0.95);
                    border: 1px solid rgba(111,177,234,0.4);
                }
            """)

    def set_selected(self, selected):
        self._selected = selected
        self._apply_style(selected)

    def mousePressEvent(self, event):
        if os.path.exists(self.file_path):
            self.clicked.emit(self.song_key, self.file_path)
        super().mousePressEvent(event)


# ─────────────────────────────────────────── Main Page
class NoiseRemoverPage(QWidget):
    def __init__(self):
        super().__init__()
        self.clean_data = None        # original clean audio (from song/file)
        self.audio_data = None        # currently active audio (could be noisy)
        self.audio_fs = dsp.FS
        self.denoised_data = None     # result after denoising
        self._clean_ref = None        # clean reference for SNR computation
        self._recording = False
        self._recorder_thread = None
        self._current_song_key = None
        self._song_cards = {}

        # Resolve audios directory path
        self._audios_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "audios"
        )

        self._build_ui()

    # ────────────────────────────────────────────────────────────────
    #  UI Construction
    # ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Main scrollable layout
        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(4, 4, 4, 4)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
        scroll.setWidget(scroll_content)

        # ═══════════════════ Header Card ═══════════════════
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
        title = QLabel("🧹 Professional Noise Remover")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        caption = QLabel("Load sample songs → Add noise → Apply denoising → Compare results in real-time")
        caption.setStyleSheet("font-size: 11px; color: #6b7f96; background: transparent; border: none;")
        caption.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(caption)
        h_layout.addLayout(title_col, 1)

        self.status_pill = StatusPill("Ready")
        h_layout.addWidget(self.status_pill)

        main_layout.addWidget(header)

        # ═══════════════════ Sample Songs Library ═══════════════════
        songs_card = QFrame()
        songs_card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.82);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.6);
            }
        """)
        songs_layout = QVBoxLayout(songs_card)
        songs_layout.setContentsMargins(16, 14, 16, 14)
        songs_layout.setSpacing(10)

        songs_header = QLabel("🎵 Sample Songs Library")
        songs_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        songs_layout.addWidget(songs_header)

        songs_desc = QLabel("Select a song to load as clean audio • Then add noise and test denoising algorithms")
        songs_desc.setStyleSheet("font-size: 11px; color: #6b7f96; background: transparent; border: none;")
        songs_desc.setWordWrap(True)
        songs_layout.addWidget(songs_desc)

        # Song cards row
        songs_row = QHBoxLayout()
        songs_row.setSpacing(12)

        sample_songs = [
            ("photograph", "Photograph", "Ed Sheeran", "📸",
             os.path.join(self._audios_dir, "Ed Sheeran - Photograph.mp3")),
            ("thousand_years", "A Thousand Years", "Christina Perri", "💫",
             os.path.join(self._audios_dir, "Cristina Perry - A Thousand Years.mp3")),
            ("memories", "Memories", "Maroon 5", "🎶",
             os.path.join(self._audios_dir, "Maroon 5 - Memories.mp3")),
        ]

        for key, title_text, artist, icon, fpath in sample_songs:
            card = SongCard(key, title_text, artist, icon, fpath)
            card.clicked.connect(self._on_song_selected)
            self._song_cards[key] = card
            songs_row.addWidget(card)

        songs_row.addStretch()
        songs_layout.addLayout(songs_row)

        main_layout.addWidget(songs_card)

        # ═══════════════════ Audio Input (Upload / Record) ═══════════════════
        input_card = QFrame()
        input_card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.82);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.6);
            }
        """)
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(16, 14, 16, 14)
        input_layout.setSpacing(10)

        input_header = QLabel("📥 Custom Audio Input")
        input_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        input_layout.addWidget(input_header)

        # File upload row
        file_row = QHBoxLayout()
        self.upload_btn = QPushButton("📁  Upload Audio File")
        self.upload_btn.setObjectName("PrimaryButton")
        self.upload_btn.setMinimumHeight(40)
        self.upload_btn.setCursor(Qt.PointingHandCursor)
        self.upload_btn.clicked.connect(self._on_upload)
        file_row.addWidget(self.upload_btn)

        self.file_label = QLabel("No file selected — use a sample song above or upload your own")
        self.file_label.setStyleSheet("""
            font-size: 11px; color: #6b7f96; background: #f7fafd;
            border-radius: 8px; padding: 8px 12px;
            border: 1px dashed #d6e2ef;
        """)
        self.file_label.setMinimumWidth(200)
        file_row.addWidget(self.file_label, 1)
        input_layout.addLayout(file_row)

        # Mic recording row
        mic_row = QHBoxLayout()
        self.record_btn = QPushButton("🎙️  Record from Microphone")
        self.record_btn.setObjectName("SecondaryButton")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.setCursor(Qt.PointingHandCursor)
        self.record_btn.clicked.connect(self._on_record_toggle)
        mic_row.addWidget(self.record_btn)

        self.rec_duration_label = QLabel("Duration:")
        self.rec_duration_label.setStyleSheet("font-size: 11px; color: #6b7f96; background: transparent; border: none;")
        mic_row.addWidget(self.rec_duration_label)

        self.rec_duration_slider = QSlider(Qt.Horizontal)
        self.rec_duration_slider.setRange(1, 15)
        self.rec_duration_slider.setValue(5)
        self.rec_duration_slider.setFixedWidth(120)
        self.rec_duration_slider.valueChanged.connect(
            lambda v: self.rec_duration_val.setText(f"{v}s"))
        mic_row.addWidget(self.rec_duration_slider)

        self.rec_duration_val = QLabel("5s")
        self.rec_duration_val.setStyleSheet("font-size: 12px; font-weight: 600; color: #2f3e50; background: transparent; border: none;")
        mic_row.addWidget(self.rec_duration_val)

        # VU meter
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setFixedWidth(100)
        self.vu_meter.setFixedHeight(16)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setStyleSheet("""
            QProgressBar {
                background: #e8eef5;
                border-radius: 8px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:0.7 #f1c40f, stop:1.0 #e74c3c);
                border-radius: 8px;
            }
        """)
        mic_row.addWidget(self.vu_meter)

        mic_row.addStretch()

        # Demo button
        self.demo_btn = QPushButton("🔊 Load Demo Signal")
        self.demo_btn.setObjectName("SecondaryButton")
        self.demo_btn.setMinimumHeight(40)
        self.demo_btn.setCursor(Qt.PointingHandCursor)
        self.demo_btn.clicked.connect(self._load_demo_signal)
        mic_row.addWidget(self.demo_btn)

        input_layout.addLayout(mic_row)
        main_layout.addWidget(input_card)

        # ═══════════════════ Add Noise Section ═══════════════════
        noise_card = QFrame()
        noise_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(231,76,60,0.06), stop:1 rgba(243,156,18,0.06));
                border-radius: 14px;
                border: 1px solid rgba(231,76,60,0.15);
            }
        """)
        noise_layout = QVBoxLayout(noise_card)
        noise_layout.setContentsMargins(16, 14, 16, 14)
        noise_layout.setSpacing(10)

        noise_header = QLabel("🔊 Add Noise to Audio")
        noise_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        noise_layout.addWidget(noise_header)

        noise_desc = QLabel("Add controlled noise to the clean audio — then use denoising to remove it and compare the results")
        noise_desc.setStyleSheet("font-size: 11px; color: #6b7f96; background: transparent; border: none;")
        noise_desc.setWordWrap(True)
        noise_layout.addWidget(noise_desc)

        # Noise controls row
        noise_ctrl_row = QHBoxLayout()
        noise_ctrl_row.setSpacing(16)

        # Noise type selector
        type_col = QVBoxLayout()
        type_label = QLabel("Noise Type:")
        type_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #445872; background: transparent; border: none;")
        type_col.addWidget(type_label)

        self.noise_type_combo = QComboBox()
        self.noise_type_combo.addItems([
            "White Noise (Gaussian)",
            "Pink Noise (1/f)",
            "Power Line Hum (50/60 Hz)",
            "Traffic Rumble (Low-freq)",
            "Crowd Babble (Broadband)",
        ])
        self.noise_type_combo.setStyleSheet("""
            QComboBox {
                background: white;
                border: 1px solid #d6e2ef;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
                color: #2f3e50;
                min-width: 220px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: white;
                border: 1px solid #d6e2ef;
                border-radius: 8px;
                selection-background-color: #eaf4ff;
                selection-color: #2f3e50;
                padding: 4px;
            }
        """)
        type_col.addWidget(self.noise_type_combo)
        noise_ctrl_row.addLayout(type_col)

        # Noise level slider
        level_col = QVBoxLayout()
        self.noise_level_label = QLabel("Noise Level: 25%")
        self.noise_level_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #445872; background: transparent; border: none;")
        level_col.addWidget(self.noise_level_label)

        self.noise_level_slider = QSlider(Qt.Horizontal)
        self.noise_level_slider.setRange(1, 100)
        self.noise_level_slider.setValue(25)
        self.noise_level_slider.setMinimumWidth(200)
        self.noise_level_slider.valueChanged.connect(
            lambda v: self.noise_level_label.setText(f"Noise Level: {v}%"))
        level_col.addWidget(self.noise_level_slider)
        noise_ctrl_row.addLayout(level_col)

        noise_ctrl_row.addStretch()
        noise_layout.addLayout(noise_ctrl_row)

        # Action buttons row
        noise_btn_row = QHBoxLayout()
        noise_btn_row.setSpacing(10)

        self.add_noise_btn = QPushButton("🔊  Add Noise")
        self.add_noise_btn.setMinimumHeight(40)
        self.add_noise_btn.setCursor(Qt.PointingHandCursor)
        self.add_noise_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #e74c3c, stop:1 #e67e22);
                color: white;
                font-weight: 700;
                font-size: 12px;
                border: none;
                border-radius: 12px;
                padding: 10px 24px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #c0392b, stop:1 #d35400);
            }
            QPushButton:disabled {
                background: #bdc3c7;
                color: #95a5a6;
            }
        """)
        self.add_noise_btn.clicked.connect(self._on_add_noise)
        self.add_noise_btn.setEnabled(False)
        noise_btn_row.addWidget(self.add_noise_btn)

        self.reset_clean_btn = QPushButton("✨  Reset to Clean")
        self.reset_clean_btn.setMinimumHeight(40)
        self.reset_clean_btn.setCursor(Qt.PointingHandCursor)
        self.reset_clean_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.9);
                color: #27ae60;
                border: 2px solid #27ae60;
                border-radius: 12px;
                padding: 10px 24px;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #e8faf1;
            }
            QPushButton:disabled {
                background: #ecf0f1;
                color: #bdc3c7;
                border-color: #bdc3c7;
            }
        """)
        self.reset_clean_btn.clicked.connect(self._on_reset_clean)
        self.reset_clean_btn.setEnabled(False)
        noise_btn_row.addWidget(self.reset_clean_btn)

        noise_btn_row.addStretch()

        # Noise status label
        self.noise_status_label = QLabel("")
        self.noise_status_label.setStyleSheet("font-size: 11px; color: #6b7f96; background: transparent; border: none;")
        noise_btn_row.addWidget(self.noise_status_label)

        noise_layout.addLayout(noise_btn_row)
        main_layout.addWidget(noise_card)

        # ═══════════════════ Denoising Controls Card ═══════════════════
        controls_card = QFrame()
        controls_card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.82);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.6);
            }
        """)
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setSpacing(10)

        ctrl_header = QLabel("⚙️ Denoising Parameters")
        ctrl_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        controls_layout.addWidget(ctrl_header)

        # Algorithm selector
        algo_row = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        algo_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #2f3e50; background: transparent; border: none;")
        algo_row.addWidget(algo_label)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "Moving Average (Time-Domain LTI)",
            "FFT Notch + Hiss Cut (Freq-Domain)",
            "Spectral Subtraction (STFT)",
            "Wiener Filter (STFT)",
            "Combined: All Methods"
        ])
        self.algo_combo.setStyleSheet("""
            QComboBox {
                background: white;
                border: 1px solid #d6e2ef;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
                color: #2f3e50;
                min-width: 280px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: white;
                border: 1px solid #d6e2ef;
                border-radius: 8px;
                selection-background-color: #eaf4ff;
                selection-color: #2f3e50;
                padding: 4px;
            }
        """)
        self.algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_row.addWidget(self.algo_combo)
        algo_row.addStretch()
        controls_layout.addLayout(algo_row)

        # Parameter sliders grid
        param_grid = QGridLayout()
        param_grid.setSpacing(12)

        # Moving Average N
        self.n_label = QLabel("Window N: 9")
        self.n_label.setStyleSheet("font-size: 11px; color: #445872; font-weight: 600; background: transparent; border: none;")
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(1, 30)
        self.n_slider.setValue(4)
        self.n_slider.valueChanged.connect(self._on_param_change)
        param_grid.addWidget(self.n_label, 0, 0)
        param_grid.addWidget(self.n_slider, 0, 1)

        # Hiss attenuation
        self.hiss_label = QLabel("Hiss Attenuation: 0.15")
        self.hiss_label.setStyleSheet("font-size: 11px; color: #445872; font-weight: 600; background: transparent; border: none;")
        self.hiss_slider = QSlider(Qt.Horizontal)
        self.hiss_slider.setRange(0, 20)
        self.hiss_slider.setValue(3)
        self.hiss_slider.valueChanged.connect(self._on_param_change)
        param_grid.addWidget(self.hiss_label, 0, 2)
        param_grid.addWidget(self.hiss_slider, 0, 3)

        # Spectral subtraction alpha
        self.alpha_label = QLabel("Over-subtraction α: 2.0")
        self.alpha_label.setStyleSheet("font-size: 11px; color: #445872; font-weight: 600; background: transparent; border: none;")
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(5, 50)
        self.alpha_slider.setValue(20)
        self.alpha_slider.valueChanged.connect(self._on_param_change)
        param_grid.addWidget(self.alpha_label, 1, 0)
        param_grid.addWidget(self.alpha_slider, 1, 1)

        # Spectral floor beta
        self.beta_label = QLabel("Spectral Floor β: 0.02")
        self.beta_label.setStyleSheet("font-size: 11px; color: #445872; font-weight: 600; background: transparent; border: none;")
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(1, 20)
        self.beta_slider.setValue(2)
        self.beta_slider.valueChanged.connect(self._on_param_change)
        param_grid.addWidget(self.beta_label, 1, 2)
        param_grid.addWidget(self.beta_slider, 1, 3)

        controls_layout.addLayout(param_grid)

        # Apply button
        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("🚀  Apply Denoising")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.setMinimumHeight(42)
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6fb1ea, stop:1 #8fd3c7);
                color: white;
                font-weight: 700;
                font-size: 13px;
                border: none;
                border-radius: 12px;
                padding: 10px 28px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5fa3e0, stop:1 #7cc9bc);
            }
        """)
        self.apply_btn.clicked.connect(self._on_apply)
        apply_row.addWidget(self.apply_btn)
        apply_row.addStretch()
        controls_layout.addLayout(apply_row)

        main_layout.addWidget(controls_card)

        # ═══════════════════ Metrics Row ═══════════════════
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self.metric_duration = MetricCard("Duration", "—", "⏱️")
        self.metric_sample_rate = MetricCard("Sample Rate", "—", "🎵")
        self.metric_samples = MetricCard("Samples", "—", "📊")
        self.metric_snr = MetricCard("Est. SNR Gain", "—", "📈")
        metrics_row.addWidget(self.metric_duration)
        metrics_row.addWidget(self.metric_sample_rate)
        metrics_row.addWidget(self.metric_samples)
        metrics_row.addWidget(self.metric_snr)
        metrics_row.addStretch()
        main_layout.addLayout(metrics_row)

        # ═══════════════════ Visualization Tabs ═══════════════════
        viz_card = QFrame()
        viz_card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.82);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.6);
            }
        """)
        viz_layout = QVBoxLayout(viz_card)
        viz_layout.setContentsMargins(16, 14, 16, 14)

        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                background: transparent;
                padding: 8px 20px;
                color: #6b7f96;
                font-weight: 600;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                color: #2f6690;
                border-bottom: 3px solid #6fb1ea;
            }
            QTabBar::tab:hover:!selected {
                color: #445872;
            }
        """)

        # Tab 1: Waveform comparison
        wave_tab = QWidget()
        wave_layout = QVBoxLayout(wave_tab)
        wave_layout.setContentsMargins(0, 8, 0, 0)
        self.wave_canvas = MplCanvas(n_rows=2, figsize=(8, 2.2))
        wave_layout.addWidget(self.wave_canvas)
        self.viz_tabs.addTab(wave_tab, "📊 Waveforms")

        # Tab 2: Spectrum comparison
        spec_tab = QWidget()
        spec_layout = QVBoxLayout(spec_tab)
        spec_layout.setContentsMargins(0, 8, 0, 0)
        self.spec_canvas = MplCanvas(n_rows=2, figsize=(8, 2.2))
        spec_layout.addWidget(self.spec_canvas)
        self.viz_tabs.addTab(spec_tab, "🌈 Spectrum")

        # Tab 3: Spectrogram (Before / After)
        spectro_tab = QWidget()
        spectro_layout = QVBoxLayout(spectro_tab)
        spectro_layout.setContentsMargins(0, 8, 0, 0)
        self.spectro_canvas = MplCanvas(n_rows=2, figsize=(8, 2.5))
        spectro_layout.addWidget(self.spectro_canvas)
        self.viz_tabs.addTab(spectro_tab, "🔥 Spectrogram")

        viz_layout.addWidget(self.viz_tabs)
        main_layout.addWidget(viz_card)

        # ═══════════════════ Playback Section ═══════════════════
        playback_card = QFrame()
        playback_card.setStyleSheet("""
            QFrame {
                background: rgba(255,255,255,0.82);
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.6);
            }
        """)
        play_layout = QVBoxLayout(playback_card)
        play_layout.setContentsMargins(16, 14, 16, 14)
        play_layout.setSpacing(10)

        play_header = QLabel("🔊 Playback Comparison")
        play_header.setStyleSheet("font-size: 14px; font-weight: 700; color: #2f3e50; background: transparent; border: none;")
        play_layout.addWidget(play_header)

        players = QHBoxLayout()
        players.setSpacing(16)
        self.p_clean = AudioPlayButton("Clean Original")
        self.p_noisy = AudioPlayButton("Noisy Signal")
        self.p_denoised = AudioPlayButton("Denoised Output")
        players.addWidget(self.p_clean)
        players.addWidget(self.p_noisy)
        players.addWidget(self.p_denoised)
        players.addStretch()

        # Export button
        self.export_btn = QPushButton("💾  Export Denoised WAV")
        self.export_btn.setObjectName("SecondaryButton")
        self.export_btn.setMinimumHeight(36)
        self.export_btn.setCursor(Qt.PointingHandCursor)
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        players.addWidget(self.export_btn)

        play_layout.addLayout(players)
        main_layout.addWidget(playback_card)

        main_layout.addStretch()

        # Initial param visibility
        self._on_algo_changed(0)

    # ────────────────────────────────────────────────────────────────
    #  Sample Song Selection
    # ────────────────────────────────────────────────────────────────
    def _on_song_selected(self, song_key, file_path):
        """Load a sample song from the audios directory."""
        self.status_pill.set_status("Loading Song...", "blue")

        # Deselect all cards, select current
        for key, card in self._song_cards.items():
            card.set_selected(key == song_key)
        self._current_song_key = song_key

        try:
            audio, orig_fs = dsp.load_sample_song(file_path, target_fs=dsp.FS, max_duration=30.0)
            self.clean_data = audio.copy()
            self.audio_data = audio.copy()
            self._clean_ref = audio.copy()
            self.audio_fs = dsp.FS

            song_name = os.path.basename(file_path)
            duration = len(audio) / dsp.FS
            self.file_label.setText(
                f"🎵 {song_name}  ({orig_fs} Hz → {dsp.FS} Hz, {duration:.1f}s)")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #1e8a5f; background: #e8faf1;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #b7ebd8;
            """)

            self._update_metrics()
            self._update_visualizations()
            self.p_clean.set_audio(self.clean_data, dsp.FS)
            self.p_noisy.set_audio(self.audio_data, dsp.FS)
            self.denoised_data = None
            self.export_btn.setEnabled(False)
            self.add_noise_btn.setEnabled(True)
            self.reset_clean_btn.setEnabled(False)
            self.noise_status_label.setText("✅ Clean audio loaded — ready to add noise")
            self.noise_status_label.setStyleSheet("font-size: 11px; color: #27ae60; background: transparent; border: none;")
            self.status_pill.set_status("Song Loaded ✓", "green")
        except Exception as e:
            self.file_label.setText(f"❌ Error loading song: {str(e)[:80]}")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #c0392b; background: #fdedec;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #f5c6cb;
            """)
            self.status_pill.set_status("Error", "red")
            print(f"Song load error: {e}")

    # ────────────────────────────────────────────────────────────────
    #  Add Noise
    # ────────────────────────────────────────────────────────────────
    def _on_add_noise(self):
        """Add selected noise type to the clean audio."""
        if self.clean_data is None:
            return

        noise_types = ['white', 'pink', 'hum', 'traffic', 'crowd']
        noise_idx = self.noise_type_combo.currentIndex()
        noise_type = noise_types[noise_idx]
        amplitude = self.noise_level_slider.value() / 200.0  # maps 1-100 to 0.005-0.5

        self.status_pill.set_status("Adding Noise...", "orange")

        try:
            self.audio_data = dsp.add_combined_noise(
                self.clean_data.copy(), noise_type=noise_type,
                amplitude=amplitude, fs=dsp.FS)

            noise_name = self.noise_type_combo.currentText().split("(")[0].strip()
            level_pct = self.noise_level_slider.value()
            self.noise_status_label.setText(
                f"🔊 {noise_name} added at {level_pct}% — now apply denoising ↓")
            self.noise_status_label.setStyleSheet(
                "font-size: 11px; color: #e67e22; font-weight: 600; background: transparent; border: none;")

            self._update_metrics()
            self._update_visualizations()
            self.p_noisy.set_audio(self.audio_data, dsp.FS)
            self.denoised_data = None
            self.export_btn.setEnabled(False)
            self.reset_clean_btn.setEnabled(True)
            self.status_pill.set_status("Noise Added", "orange")
        except Exception as e:
            self.status_pill.set_status("Error", "red")
            self.noise_status_label.setText(f"❌ Error: {str(e)[:60]}")
            self.noise_status_label.setStyleSheet(
                "font-size: 11px; color: #e74c3c; background: transparent; border: none;")
            print(f"Add noise error: {e}")

    def _on_reset_clean(self):
        """Reset audio back to clean (no noise)."""
        if self.clean_data is None:
            return

        self.audio_data = self.clean_data.copy()
        self._update_metrics()
        self._update_visualizations()
        self.p_noisy.set_audio(self.audio_data, dsp.FS)
        self.denoised_data = None
        self.export_btn.setEnabled(False)
        self.reset_clean_btn.setEnabled(False)
        self.noise_status_label.setText("✅ Reset to clean audio")
        self.noise_status_label.setStyleSheet(
            "font-size: 11px; color: #27ae60; background: transparent; border: none;")
        self.status_pill.set_status("Clean ✓", "green")

    # ────────────────────────────────────────────────────────────────
    #  Audio Input Handlers (Upload / Record / Demo)
    # ────────────────────────────────────────────────────────────────
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3)")
        if not path:
            return
        self.status_pill.set_status("Loading...", "blue")

        # Deselect song cards
        for card in self._song_cards.values():
            card.set_selected(False)
        self._current_song_key = None

        try:
            audio, orig_fs = dsp.load_audio_file(path, target_fs=dsp.FS)

            # Trim to 30s max
            max_samples = int(30.0 * dsp.FS)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            self.clean_data = audio.copy()
            self.audio_data = audio.copy()
            self._clean_ref = audio.copy()
            self.audio_fs = dsp.FS
            fname = os.path.basename(path)
            self.file_label.setText(f"✅ {fname}  ({orig_fs} Hz → {dsp.FS} Hz, {len(audio)/dsp.FS:.2f}s)")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #1e8a5f; background: #e8faf1;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #b7ebd8;
            """)
            self._update_metrics()
            self._update_visualizations()
            self.p_clean.set_audio(self.clean_data, dsp.FS)
            self.p_noisy.set_audio(self.audio_data, dsp.FS)
            self.denoised_data = None
            self.export_btn.setEnabled(False)
            self.add_noise_btn.setEnabled(True)
            self.reset_clean_btn.setEnabled(False)
            self.noise_status_label.setText("✅ Audio loaded — ready to add noise")
            self.noise_status_label.setStyleSheet(
                "font-size: 11px; color: #27ae60; background: transparent; border: none;")
            self.status_pill.set_status("Loaded", "green")
        except Exception as e:
            self.file_label.setText(f"❌ Error: {str(e)[:60]}")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #c0392b; background: #fdedec;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #f5c6cb;
            """)
            self.status_pill.set_status("Error", "red")

    def _on_record_toggle(self):
        if self._recording:
            # Stop recording
            self._recording = False
            self.record_btn.setText("🎙️  Record from Microphone")
            self.record_btn.setObjectName("SecondaryButton")
            self.record_btn.style().unpolish(self.record_btn)
            self.record_btn.style().polish(self.record_btn)
            if self._recorder_thread:
                self._recorder_thread.stop_recording()
        else:
            # Start recording
            self._recording = True
            self.record_btn.setText("⏹️  Stop Recording")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background: #fdedec;
                    color: #e74c3c;
                    border: 2px solid #e74c3c;
                    border-radius: 10px;
                    padding: 8px 16px;
                    font-weight: 700;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: #f5c6cb;
                }
            """)
            self.status_pill.set_status("Recording...", "red")
            self.vu_meter.setValue(0)

            duration = self.rec_duration_slider.value()
            self._recorder_thread = MicRecorderThread(
                duration=duration, fs=dsp.FS, parent=self)
            self._recorder_thread.level_update.connect(self._on_vu_update)
            self._recorder_thread.finished.connect(self._on_recording_done)
            self._recorder_thread.start()

    def _on_vu_update(self, rms):
        level = min(int(rms * 500), 100)
        self.vu_meter.setValue(level)

    def _on_recording_done(self, audio, fs):
        self._recording = False
        self.record_btn.setText("🎙️  Record from Microphone")
        self.record_btn.setObjectName("SecondaryButton")
        self.record_btn.setStyleSheet("")  # Reset to default
        self.record_btn.style().unpolish(self.record_btn)
        self.record_btn.style().polish(self.record_btn)
        self.vu_meter.setValue(0)

        # Deselect song cards
        for card in self._song_cards.values():
            card.set_selected(False)
        self._current_song_key = None

        if len(audio) > 1:
            self.clean_data = audio.copy()
            self.audio_data = audio.copy()
            self._clean_ref = audio.copy()
            self.audio_fs = fs
            self.file_label.setText(f"🎙️ Recorded {len(audio)/fs:.2f}s at {fs} Hz")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #8e44ad; background: #f5eef8;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #d2b4de;
            """)
            self._update_metrics()
            self._update_visualizations()
            self.p_clean.set_audio(self.clean_data, dsp.FS)
            self.p_noisy.set_audio(self.audio_data, dsp.FS)
            self.denoised_data = None
            self.export_btn.setEnabled(False)
            self.add_noise_btn.setEnabled(True)
            self.reset_clean_btn.setEnabled(False)
            self.noise_status_label.setText("✅ Recording saved — ready to add noise")
            self.noise_status_label.setStyleSheet(
                "font-size: 11px; color: #27ae60; background: transparent; border: none;")
            self.status_pill.set_status("Recorded", "purple")
        else:
            self.status_pill.set_status("Mic Error", "red")
            self.file_label.setText("❌ Could not record. Install: pip install sounddevice")
            self.file_label.setStyleSheet("""
                font-size: 11px; color: #c0392b; background: #fdedec;
                border-radius: 8px; padding: 8px 12px;
                border: 1px solid #f5c6cb;
            """)

    def _load_demo_signal(self):
        """Load the built-in synthetic noisy demo signal."""
        # Deselect song cards
        for card in self._song_cards.values():
            card.set_selected(False)
        self._current_song_key = None

        clean = dsp.make_voice_like()
        self.clean_data = clean.copy()
        self.audio_data = dsp.make_noisy(clean)
        self._clean_ref = clean
        self.audio_fs = dsp.FS
        self.file_label.setText("🔊 Demo: Synthetic voice + 60 Hz hum + hiss  (2.0s)")
        self.file_label.setStyleSheet("""
            font-size: 11px; color: #2980b9; background: #ebf5fb;
            border-radius: 8px; padding: 8px 12px;
            border: 1px solid #aed6f1;
        """)
        self._update_metrics()
        self._update_visualizations()
        self.p_clean.set_audio(self.clean_data, dsp.FS)
        self.p_noisy.set_audio(self.audio_data, dsp.FS)
        self.denoised_data = None
        self.export_btn.setEnabled(False)
        self.add_noise_btn.setEnabled(True)
        self.reset_clean_btn.setEnabled(True)
        self.noise_status_label.setText("🔊 Demo loaded with built-in noise")
        self.noise_status_label.setStyleSheet(
            "font-size: 11px; color: #2980b9; background: transparent; border: none;")
        self.status_pill.set_status("Demo Loaded", "blue")

    # ────────────────────────────────────────────────────────────────
    #  Algorithm Controls Visibility
    # ────────────────────────────────────────────────────────────────
    def _on_algo_changed(self, index):
        # Show/hide relevant sliders based on algorithm
        is_ma = index in (0, 4)
        is_fft = index in (1, 4)
        is_spectral = index in (2, 4)
        is_wiener = index in (3, 4)

        self.n_label.setVisible(is_ma)
        self.n_slider.setVisible(is_ma)
        self.hiss_label.setVisible(is_fft)
        self.hiss_slider.setVisible(is_fft)
        self.alpha_label.setVisible(is_spectral)
        self.alpha_slider.setVisible(is_spectral)
        self.beta_label.setVisible(is_spectral)
        self.beta_slider.setVisible(is_spectral)

    def _on_param_change(self):
        N = self.n_slider.value() * 2 + 1
        hiss = self.hiss_slider.value() / 20.0
        alpha = self.alpha_slider.value() / 10.0
        beta = self.beta_slider.value() / 100.0
        self.n_label.setText(f"Window N: {N}")
        self.hiss_label.setText(f"Hiss Attenuation: {hiss:.2f}")
        self.alpha_label.setText(f"Over-subtraction α: {alpha:.1f}")
        self.beta_label.setText(f"Spectral Floor β: {beta:.2f}")

    # ────────────────────────────────────────────────────────────────
    #  Apply Denoising
    # ────────────────────────────────────────────────────────────────
    def _on_apply(self):
        if self.audio_data is None:
            return

        self.status_pill.set_status("Processing...", "orange")
        self.apply_btn.setEnabled(False)

        # Use a timer to allow UI to update before heavy computation
        QTimer.singleShot(50, self._run_denoising)

    def _run_denoising(self):
        algo_idx = self.algo_combo.currentIndex()
        x = self.audio_data
        N = self.n_slider.value() * 2 + 1
        hiss_atten = self.hiss_slider.value() / 20.0
        alpha = self.alpha_slider.value() / 10.0
        beta = self.beta_slider.value() / 100.0

        try:
            if algo_idx == 0:
                # Moving Average
                self.denoised_data = dsp.moving_average_filter(x, N=N)
            elif algo_idx == 1:
                # FFT Notch + Hiss Cut
                self.denoised_data = dsp.freq_domain_denoise(x, hiss_atten=hiss_atten)
            elif algo_idx == 2:
                # Spectral Subtraction
                self.denoised_data = dsp.spectral_subtraction_denoise(
                    x, fs=dsp.FS, alpha=alpha, beta=beta)
            elif algo_idx == 3:
                # Wiener Filter
                self.denoised_data = dsp.wiener_denoise(x, fs=dsp.FS)
            elif algo_idx == 4:
                # Combined: apply all in sequence
                step1 = dsp.freq_domain_denoise(x, hiss_atten=hiss_atten)
                step2 = dsp.spectral_subtraction_denoise(
                    step1, fs=dsp.FS, alpha=alpha, beta=beta)
                step3 = dsp.wiener_denoise(step2, fs=dsp.FS)
                self.denoised_data = dsp.moving_average_filter(step3, N=N)

            self.denoised_data = dsp.normalize(self.denoised_data)
            self._update_visualizations_with_denoised()
            self.p_denoised.set_audio(self.denoised_data, dsp.FS)
            self.export_btn.setEnabled(True)

            # SNR estimation
            if self._clean_ref is not None and len(self._clean_ref) == len(x):
                snr_before = dsp.compute_snr(self._clean_ref, x)
                snr_after = dsp.compute_snr(self._clean_ref, self.denoised_data)
                self.metric_snr.set_value(f"+{snr_after - snr_before:.1f} dB")
            else:
                # Estimate improvement by power ratio
                power_orig = np.mean(x ** 2)
                power_den = np.mean(self.denoised_data ** 2)
                ratio_db = 10 * np.log10(power_den / (power_orig + 1e-12))
                self.metric_snr.set_value(f"{ratio_db:+.1f} dB")

            self.status_pill.set_status("Done ✓", "green")
        except Exception as e:
            self.status_pill.set_status("Error", "red")
            print(f"Denoising error: {e}")

        self.apply_btn.setEnabled(True)

    # ────────────────────────────────────────────────────────────────
    #  Metrics
    # ────────────────────────────────────────────────────────────────
    def _update_metrics(self):
        if self.audio_data is None:
            return
        dur = len(self.audio_data) / dsp.FS
        self.metric_duration.set_value(f"{dur:.2f}s")
        self.metric_sample_rate.set_value(f"{dsp.FS} Hz")
        self.metric_samples.set_value(f"{len(self.audio_data):,}")
        self.metric_snr.set_value("—")

    # ────────────────────────────────────────────────────────────────
    #  Visualizations
    # ────────────────────────────────────────────────────────────────
    def _update_visualizations(self):
        """Update visualizations with the loaded (original) audio only."""
        if self.audio_data is None:
            return

        x = self.audio_data

        # Waveform: show full signal
        xlim = (0, len(x) / dsp.FS)
        self.wave_canvas.fig.clear()
        ax = self.wave_canvas.fig.subplots(1, 1)
        ts = np.arange(len(x)) / dsp.FS
        ax.plot(ts, x, linewidth=0.6, color="#e74c3c", alpha=0.8)
        ax.set_ylabel("Original", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_title("Waveform — Input Signal", fontsize=11, fontweight='bold')
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
        self.wave_canvas.fig.tight_layout()
        self.wave_canvas.draw()

        # Spectrum
        self.spec_canvas.fig.clear()
        ax_sp = self.spec_canvas.fig.subplots(1, 1)
        freqs, db = dsp.spectrum_db(x, dsp.FS)
        ax_sp.plot(freqs, db, linewidth=0.7, color="#e74c3c", alpha=0.8, label="Original")
        ax_sp.set_xlim(0, min(dsp.FS // 2, 4000))
        ax_sp.set_ylabel("Magnitude (dB)", fontsize=9)
        ax_sp.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_sp.set_title("Frequency Spectrum — Input Signal", fontsize=11, fontweight='bold')
        ax_sp.legend(fontsize=8, loc='upper right')
        ax_sp.grid(alpha=0.2)
        ax_sp.tick_params(labelsize=8)
        self.spec_canvas.fig.tight_layout()
        self.spec_canvas.draw()

        # Spectrogram
        self._plot_spectrogram_single(x, "Spectrogram — Input Signal")

    def _update_visualizations_with_denoised(self):
        """Update all visualizations with before/after comparison."""
        if self.audio_data is None or self.denoised_data is None:
            return

        x = self.audio_data
        d = self.denoised_data
        xlim = (0, len(x) / dsp.FS)

        # ── Waveform comparison
        self.wave_canvas.fig.clear()
        axes = self.wave_canvas.fig.subplots(2, 1, sharex=True)
        ts = np.arange(len(x)) / dsp.FS

        axes[0].plot(ts, x, linewidth=0.6, color="#e74c3c", alpha=0.8)
        axes[0].set_ylabel("Noisy", fontsize=9)
        axes[0].set_title("Waveform Comparison", fontsize=11, fontweight='bold')
        axes[0].grid(alpha=0.2)
        axes[0].set_xlim(xlim)
        axes[0].tick_params(labelsize=8)

        ts_d = np.arange(len(d)) / dsp.FS
        axes[1].plot(ts_d, d, linewidth=0.6, color="#2ecc71", alpha=0.8)
        axes[1].set_ylabel("Denoised", fontsize=9)
        axes[1].set_xlabel("Time (s)", fontsize=9)
        axes[1].grid(alpha=0.2)
        axes[1].set_xlim(xlim)
        axes[1].tick_params(labelsize=8)

        self.wave_canvas.fig.tight_layout()
        self.wave_canvas.draw()

        # ── Spectrum comparison (overlay)
        self.spec_canvas.fig.clear()
        ax_sp = self.spec_canvas.fig.subplots(1, 1)
        freqs_o, db_o = dsp.spectrum_db(x, dsp.FS)
        freqs_d, db_d = dsp.spectrum_db(d, dsp.FS)
        ax_sp.plot(freqs_o, db_o, linewidth=1.2, color="#e74c3c", alpha=0.9, label="Noisy", zorder=2)
        ax_sp.plot(freqs_d, db_d, linewidth=0.8, color="#2ecc71", alpha=0.75, label="Denoised", zorder=3)
        ax_sp.set_xlim(0, min(dsp.FS // 2, 4000))
        ax_sp.set_ylabel("Magnitude (dB)", fontsize=9)
        ax_sp.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_sp.set_title("Spectrum Comparison", fontsize=11, fontweight='bold')
        ax_sp.legend(fontsize=9, loc='upper right',
                     framealpha=0.8, edgecolor='#ddd')
        ax_sp.grid(alpha=0.2)
        ax_sp.tick_params(labelsize=8)
        self.spec_canvas.fig.tight_layout()
        self.spec_canvas.draw()

        # ── Spectrogram comparison (before / after)
        self._plot_spectrogram_comparison(x, d)

    def _plot_spectrogram_single(self, x, title):
        self.spectro_canvas.fig.clear()
        ax = self.spectro_canvas.fig.subplots(1, 1)
        f, t, Sxx = dsp.compute_spectrogram(x, dsp.FS)
        ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma',
                      vmin=-80, vmax=0)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylim(0, min(dsp.FS // 2, 4000))
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        self.spectro_canvas.fig.tight_layout()
        self.spectro_canvas.draw()

    def _plot_spectrogram_comparison(self, original, denoised):
        self.spectro_canvas.fig.clear()
        axes = self.spectro_canvas.fig.subplots(2, 1, sharex=True)

        f_o, t_o, Sxx_o = dsp.compute_spectrogram(original, dsp.FS)
        axes[0].pcolormesh(t_o, f_o, Sxx_o, shading='gouraud', cmap='magma',
                           vmin=-80, vmax=0)
        axes[0].set_ylabel("Freq (Hz)", fontsize=9)
        axes[0].set_ylim(0, min(dsp.FS // 2, 4000))
        axes[0].set_title("Spectrogram: Noisy vs Denoised", fontsize=11, fontweight='bold')
        axes[0].tick_params(labelsize=8)

        f_d, t_d, Sxx_d = dsp.compute_spectrogram(denoised, dsp.FS)
        axes[1].pcolormesh(t_d, f_d, Sxx_d, shading='gouraud', cmap='magma',
                           vmin=-80, vmax=0)
        axes[1].set_ylabel("Freq (Hz)", fontsize=9)
        axes[1].set_xlabel("Time (s)", fontsize=9)
        axes[1].set_ylim(0, min(dsp.FS // 2, 4000))
        axes[1].tick_params(labelsize=8)

        self.spectro_canvas.fig.tight_layout()
        self.spectro_canvas.draw()

    # ────────────────────────────────────────────────────────────────
    #  Export
    # ────────────────────────────────────────────────────────────────
    def _on_export(self):
        if self.denoised_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Audio", "denoised_output.wav",
            "WAV Files (*.wav)")
        if path:
            wav_bytes = dsp.to_wav_bytes(self.denoised_data, dsp.FS)
            with open(path, 'wb') as f:
                f.write(wav_bytes)
            self.status_pill.set_status("Exported ✓", "green")
