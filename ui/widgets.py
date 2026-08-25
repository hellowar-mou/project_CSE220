import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QPushButton, QWidget, QHBoxLayout, QLabel, QProgressBar, QSlider
from PySide6.QtCore import QUrl, Signal, Qt, QThread
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import dsp_core as dsp


class MicRecorderThread(QThread):
    """Records audio from the microphone in a background thread.
    Shared helper used by pages that need microphone input (Equalizer,
    Morse Code Converter) — mirrors the recorder already used privately by
    the Noise Remover page, without touching that page's own copy."""
    finished = Signal(np.ndarray, int)  # audio_data, sample_rate
    level_update = Signal(float)         # RMS level for a VU meter

    def __init__(self, duration=5, fs=None, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.fs = fs or dsp.FS
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
        total_blocks = max(1, int(self.duration * self.fs / block_size))

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


class MicRecordWidget(QWidget):
    """A compact 'record from mic' control: button + duration slider + VU
    meter, wrapping MicRecorderThread. Emits recording_ready(audio, fs)
    when a recording finishes."""

    recording_ready = Signal(np.ndarray, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recording = False
        self._thread = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.record_btn = QPushButton("🎙️ Record from Microphone")
        self.record_btn.setObjectName("SecondaryButton")
        self.record_btn.clicked.connect(self._toggle)
        layout.addWidget(self.record_btn)

        dur_label = QLabel("Duration:")
        dur_label.setObjectName("Caption")
        layout.addWidget(dur_label)

        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(1, 15)
        self.duration_slider.setValue(5)
        self.duration_slider.setFixedWidth(100)
        self.duration_value = QLabel("5s")
        self.duration_value.setObjectName("Caption")
        self.duration_slider.valueChanged.connect(
            lambda v: self.duration_value.setText(f"{v}s"))
        layout.addWidget(self.duration_slider)
        layout.addWidget(self.duration_value)

        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setFixedWidth(90)
        self.vu_meter.setFixedHeight(14)
        self.vu_meter.setTextVisible(False)
        layout.addWidget(self.vu_meter)
        layout.addStretch()

    def _toggle(self):
        if self._recording:
            self._recording = False
            self.record_btn.setText("🎙️ Record from Microphone")
            if self._thread:
                self._thread.stop_recording()
        else:
            self._recording = True
            self.record_btn.setText("⏹️ Stop Recording")
            self.vu_meter.setValue(0)
            self._thread = MicRecorderThread(
                duration=self.duration_slider.value(), fs=dsp.FS, parent=self)
            self._thread.level_update.connect(self._on_level)
            self._thread.finished.connect(self._on_finished)
            self._thread.start()

    def _on_level(self, rms):
        self.vu_meter.setValue(min(int(rms * 500), 100))

    def _on_finished(self, audio, fs):
        self._recording = False
        self.record_btn.setText("🎙️ Record from Microphone")
        self.vu_meter.setValue(0)
        if len(audio) > 1:
            self.recording_ready.emit(audio, fs)


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib Figure embedded as a Qt widget, styled to match the light theme."""

    def __init__(self, n_rows=1, figsize=(7, 1.9), parent=None):
        self.fig = Figure(figsize=(figsize[0], figsize[1] * n_rows), dpi=100)
        self.fig.patch.set_alpha(0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")
        self._playhead_axes = []   # time-domain axes eligible for a playhead
        self._playhead_duration_s = 0
        self._playhead_lines = []

    def plot_waveforms(self, signals_labels, title, fs=dsp.FS, xlim=None):
        self.fig.clear()
        n = len(signals_labels)
        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (x, lab) in zip(axes, signals_labels):
            import numpy as np
            ts = np.arange(len(x)) / fs
            ax.plot(ts, x, linewidth=0.8, color="#2f6690")
            ax.set_ylabel(lab, fontsize=8)
            ax.grid(alpha=0.25)
            if xlim:
                ax.set_xlim(xlim)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("Time (s)", fontsize=8)
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout()

        # Remember these axes are time-domain, so a playback position marker
        # can be drawn on them later via update_playhead().
        max_len = max((len(x) for x, _ in signals_labels), default=0)
        self._playhead_axes = list(axes)
        self._playhead_duration_s = max_len / fs
        self._playhead_lines = []

        self.draw()

    def update_playhead(self, fraction):
        """Draw/move a vertical line at `fraction` (0-1) of playback across
        the axes from the most recent plot_waveforms() call. Called from a
        page's AudioPlayButton.position_changed signal, so the marker tracks
        real playback position rather than a simulated animation."""
        if not self._playhead_axes or self._playhead_duration_s <= 0:
            return
        x_pos = fraction * self._playhead_duration_s
        if not self._playhead_lines:
            for ax in self._playhead_axes:
                line = ax.axvline(x_pos, color="#e67e22", linewidth=1.4, alpha=0.85, zorder=5)
                self._playhead_lines.append(line)
        else:
            for line in self._playhead_lines:
                line.set_xdata([x_pos, x_pos])
        self.draw_idle()

    def clear_playhead(self):
        for line in self._playhead_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._playhead_lines = []
        self.draw_idle()

    def plot_spectra(self, signals_labels, title, fs=dsp.FS, xlim=(0, 4000)):
        self.fig.clear()
        n = len(signals_labels)
        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (x, lab) in zip(axes, signals_labels):
            freqs, db = dsp.spectrum_db(x, fs)
            ax.plot(freqs, db, linewidth=0.8, color="#4a8fa3")
            ax.set_ylabel(lab, fontsize=8)
            ax.grid(alpha=0.25)
            ax.set_xlim(xlim)
            ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("Frequency (Hz)", fontsize=8)
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout()
        self.draw()


class AudioPlayButton(QWidget):
    """A play button + caption that plays a generated waveform via QtMultimedia.

    Emits `position_changed(fraction)` (0.0-1.0) while playing and
    `playback_active_changed(bool)` on start/stop, so a page can draw a live
    moving playhead on its waveform/spectrum plot synced to actual playback,
    not a simulated animation.
    """

    _tempfiles = []  # keep references alive for the app's lifetime

    position_changed = Signal(float)
    playback_active_changed = Signal(bool)

    def __init__(self, caption="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton("▶ Play")
        self.button.setObjectName("PlayButton")
        self.button.clicked.connect(self._play)
        layout.addWidget(self.button)
        if caption:
            lab = QLabel(caption)
            lab.setObjectName("Caption")
            layout.addWidget(lab)
        layout.addStretch()

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._path = None
        self._duration_ms = 0

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    def set_audio(self, x, fs=dsp.FS):
        wav_bytes = dsp.to_wav_bytes(x, fs)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav_bytes)
        f.close()
        self._path = f.name
        AudioPlayButton._tempfiles.append(f.name)

    def _play(self):
        if self._path and os.path.exists(self._path):
            self._player.setSource(QUrl.fromLocalFile(self._path))
            self._player.play()

    def _on_duration_changed(self, duration_ms):
        self._duration_ms = duration_ms

    def _on_position_changed(self, position_ms):
        if self._duration_ms > 0:
            self.position_changed.emit(min(1.0, position_ms / self._duration_ms))

    def _on_state_changed(self, state):
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.playback_active_changed.emit(is_playing)
        if not is_playing:
            self.position_changed.emit(0.0)
