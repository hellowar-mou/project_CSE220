import tempfile
import os
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import QPushButton, QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import dsp_core as dsp


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib Figure embedded as a Qt widget, styled to match the light theme."""

    def __init__(self, n_rows=1, figsize=(7, 1.9), parent=None):
        self.fig = Figure(figsize=(figsize[0], figsize[1] * n_rows), dpi=100)
        self.fig.patch.set_alpha(0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")

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
        self.draw()

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
    """A small play button + caption that plays a generated waveform via QtMultimedia."""

    _tempfiles = []  # keep references alive for the app's lifetime

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
