import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar,
    QSlider, QCheckBox, QMessageBox, QDialog, QStackedWidget, QDialogButtonBox,
    QFrame,
)
from PySide6.QtCore import QUrl, Signal, Qt, QThread
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import dsp_core as dsp


# Chrome colors (background/ticks/spines/text) applied on top of whatever
# is plotted, by MplCanvas.recolor() / ui.theme.recolor_figure() — the
# plotted data-line colors themselves are left as each call site chose,
# since the existing blue/teal tones already read fine on both themes.
_PLOT_COLORS = {
    "light": {"text": "#33465c", "grid": "#8395ac", "face": "none"},
    "dark": {"text": "#dce6f2", "grid": "#3a4a5e", "face": "#131a24"},
}


class MicRecorderThread(QThread):
    """Records audio from the microphone in a background thread.
    Shared helper used by pages that need microphone input (Equalizer,
    Morse Code Converter, Audio Matcher) — mirrors the recorder already
    used privately by the Noise Remover page, without touching that page's
    own copy."""
    finished = Signal(np.ndarray, int)  # audio_data, sample_rate
    level_update = Signal(float)         # RMS level for a VU meter
    chunk_ready = Signal(np.ndarray)     # raw ~50ms chunk, for live waveform draw

    def __init__(self, duration=5, fs=None, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.fs = fs or dsp.FS
        self._stop_flag = False
        self._paused = False

    def stop_recording(self):
        self._stop_flag = True

    def set_paused(self, paused):
        self._paused = paused

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
            if self._paused:
                return
            chunk = indata[:, 0].copy()
            frames.append(chunk)
            rms = np.sqrt(np.mean(indata ** 2))
            self.level_update.emit(float(rms))
            self.chunk_ready.emit(chunk)

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
        self.record_btn.setToolTip("Record audio from the selected microphone.")
        self.record_btn.clicked.connect(self._toggle)
        layout.addWidget(self.record_btn)

        dur_label = QLabel("Duration:")
        dur_label.setObjectName("Caption")
        layout.addWidget(dur_label)

        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(1, 15)
        self.duration_slider.setValue(5)
        self.duration_slider.setFixedWidth(100)
        self.duration_slider.setToolTip("Choose the microphone recording duration in seconds.")
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
        self.vu_meter.setVisible(False)
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
            self.vu_meter.setVisible(True)
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
        self.vu_meter.setVisible(False)
        if len(audio) > 1:
            self.recording_ready.emit(audio, fs)


class CollapsiblePanel(QWidget):
    """A titled panel that expands/collapses on click — used for secondary
    details (like a technical info panel) that shouldn't always take up
    screen space."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QVBoxLayout, QFrame
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self.toggle_btn = QPushButton(f"▸ {title}")
        self.toggle_btn.setObjectName("SecondaryButton")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._on_toggle)
        outer.addWidget(self.toggle_btn)

        self.content = QFrame()
        self.content.setStyleSheet("QFrame { background: rgba(255,255,255,0.5); border-radius: 10px; }")
        self.content_layout = QVBoxLayout(self.content)
        self.content.setVisible(False)
        outer.addWidget(self.content)

        self._title = title

    def _on_toggle(self, checked):
        self.content.setVisible(checked)
        arrow = "▾" if checked else "▸"
        self.toggle_btn.setText(f"{arrow} {self._title}")

    def body_layout(self):
        return self.content_layout


class MplCanvas(FigureCanvasQTAgg):
    """A matplotlib Figure embedded as a Qt widget, styled to match the light theme."""

    def __init__(self, n_rows=1, figsize=(7, 1.9), parent=None):
        self.fig = Figure(figsize=(figsize[0], figsize[1] * n_rows), dpi=100)
        self.fig.patch.set_alpha(0)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")
        self.setToolTip("Waveform view: amplitude over time. Use the controls below to zoom.")
        self._playhead_axes = []   # time-domain axes eligible for a playhead
        self._playhead_duration_s = 0
        self._playhead_lines = []
        self._waveform_view = None

        from ui import theme as theme_module
        theme_module.register_canvas(self)
        if theme_module.get_current_mode() == "dark":
            theme_module.recolor_figure(self.fig, "dark")

    def plot_waveforms(self, signals_labels, title, fs=dsp.FS, xlim=None):
        self.fig.clear()
        n = len(signals_labels)
        compare_axes = n > 1
        axes = self.fig.subplots(n + int(compare_axes), 1, sharex=True)
        axes = list(np.atleast_1d(axes))
        comparison_ax = axes[-1] if compare_axes else None
        input_color = "#f97316"
        output_color = "#06b6d4"
        line_colors = []
        for ax, (x, lab) in zip(axes, signals_labels):
            if ax is comparison_ax:
                break
            values = np.asarray(x)
            # Plot a bounded number of points so long recordings remain
            # readable while the axis still represents the complete signal.
            max_points = 8000
            if len(values) > max_points:
                indices = np.linspace(0, len(values) - 1, max_points).astype(int)
                values = values[indices]
                ts = indices / fs
            else:
                ts = np.arange(len(values)) / fs
            color = input_color if len(line_colors) == 0 else output_color
            line_colors.append((ts, values, lab, color))
            ax.plot(ts, values, linewidth=0.9, color=color)
            ax.axhline(0, color="#8395ac", linewidth=0.6, alpha=0.7)
            ax.set_ylabel(lab, fontsize=8)
            ax.grid(alpha=0.25)
            if xlim:
                ax.set_xlim(xlim)
            ax.tick_params(labelsize=7)
        if comparison_ax is not None:
            for ts, values, lab, color in line_colors:
                comparison_ax.plot(
                    ts, values, linewidth=1.0, color=color, alpha=0.95,
                    label=lab, zorder=3,
                )
            comparison_ax.axhline(0, color="#8395ac", linewidth=0.6, alpha=0.7)
            comparison_ax.set_ylabel("Overlay", fontsize=8)
            comparison_ax.set_title(
                "Input / Output Overlay (live preview)",
                fontsize=8,
                pad=2,
            )
            comparison_ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
            comparison_ax.grid(alpha=0.25)
            comparison_ax.tick_params(labelsize=7)
        axes[-1].set_xlabel("Time (s)", fontsize=8)
        self.fig.suptitle(title, fontsize=10)
        self.fig.tight_layout()
        self.setMinimumHeight(max(240, 150 * len(axes)))
        self._waveform_view = {
            "duration_s": max((len(x) for x, _ in signals_labels), default=0) / fs,
            "xlim": xlim,
        }

        # Remember these axes are time-domain, so a playback position marker
        # can be drawn on them later via update_playhead().
        max_len = max((len(x) for x, _ in signals_labels), default=0)
        self._playhead_axes = list(axes)
        self._playhead_duration_s = max_len / fs
        self._playhead_lines = []

        from ui import theme as theme_module
        theme_module.recolor_figure(self.fig, theme_module.get_current_mode())
        self.draw()

    def zoom_waveform(self, factor):
        """Zoom the time axis while retaining a meaningful centered view."""
        if not self._waveform_view or not self._playhead_axes:
            return
        duration = self._waveform_view["duration_s"]
        current = self._playhead_axes[0].get_xlim()
        center = (current[0] + current[1]) / 2
        width = max(duration / 1000, (current[1] - current[0]) * factor)
        left = max(0.0, min(duration - width, center - width / 2))
        right = min(duration, left + width)
        for ax in self._playhead_axes:
            ax.set_xlim(left, right)
        self.draw_idle()

    def fit_waveform(self):
        if not self._waveform_view:
            return
        duration = self._waveform_view["duration_s"]
        for ax in self._playhead_axes:
            ax.set_xlim(0, duration)
        self.draw_idle()
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
        from ui import theme as theme_module
        theme_module.recolor_figure(self.fig, theme_module.get_current_mode())
        self.draw()

    def recolor(self):
        """Public hook: re-tint this canvas's chrome to the current theme
        without touching whatever data is plotted. Safe to call any time,
        including from pages that don't otherwise know theming exists."""
        from ui import theme as theme_module
        theme_module.recolor_figure(self.fig, theme_module.get_current_mode())


class WaveformControls(QWidget):
    """Small reusable view controls for time-domain canvases."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        for label, factor, tooltip in (
            ("Zoom −", 1.8, "Show more of the waveform."),
            ("Zoom +", 0.55, "Zoom in to inspect waveform detail."),
        ):
            button = QPushButton(label)
            button.setObjectName("SecondaryButton")
            button.setToolTip(tooltip)
            button.clicked.connect(lambda checked=False, f=factor: canvas.zoom_waveform(f))
            layout.addWidget(button)
        fit = QPushButton("Fit")
        fit.setObjectName("SecondaryButton")
        fit.setToolTip("Fit the complete audio signal in the waveform.")
        fit.clicked.connect(canvas.fit_waveform)
        layout.addWidget(fit)
        reset = QPushButton("Reset View")
        reset.setObjectName("SecondaryButton")
        reset.setToolTip("Return to the complete-signal overview.")
        reset.clicked.connect(canvas.fit_waveform)
        layout.addWidget(reset)
        layout.addStretch()


def show_module_help(parent, title, steps):
    """Display concise in-app help without sending users to documentation."""
    QMessageBox.information(parent, title, "\n".join(steps))


class GuidanceBubble(QWidget):
    """Reusable first-visit guide with session close and persistent dismissal."""

    def __init__(self, key, message, parent=None):
        super().__init__(parent)
        from ui import theme as theme_module
        self._key = key
        self._theme_module = theme_module
        self.setObjectName("GuidanceBubble")
        self.setStyleSheet(
            "QWidget#GuidanceBubble { background: #eef7ff; border: 1px solid #b8d8f2; "
            "border-radius: 12px; }")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(8)
        icon = QLabel("💡")
        layout.addWidget(icon)
        text = QLabel(f"<b>Start here</b><br>{message}")
        text.setWordWrap(True)
        text.setObjectName("Caption")
        layout.addWidget(text, 1)

        self.dont_show = QCheckBox("Don't show again")
        self.dont_show.setToolTip("Hide this guide on future visits.")
        layout.addWidget(self.dont_show)
        close_btn = QPushButton("×")
        close_btn.setObjectName("GuidanceCloseButton")
        close_btn.setFixedSize(28, 28)
        close_btn.setToolTip("Close this guide for now.")
        close_btn.clicked.connect(self._close)
        layout.addWidget(close_btn)

        if theme_module.is_guidance_dismissed(key):
            self.hide()

    def _close(self):
        if self.dont_show.isChecked():
            self._theme_module.set_guidance_dismissed(self._key)
        self.hide()


ModuleGuide = GuidanceBubble


class FirstRunWalkthrough(QDialog):
    """One-time in-app orientation for users opening the toolbox for the first time."""

    def __init__(self, user_key="default", parent=None):
        super().__init__(parent)
        self._user_key = user_key
        self.setWindowTitle("Welcome to Audio Signals Toolbox")
        self.setMinimumWidth(560)
        self._pages = [
            ("Welcome",
             "Explore five practical audio tools built on convolution, FFT, "
             "correlation, and LTI systems."),
            ("Choose an input",
             "Upload a file, record from your microphone, or use a built-in demo "
             "where available. The current audio is shown in an input card."),
            ("Process and inspect",
             "Adjust controls, run the module action, and follow the waveform, "
             "spectrum, or intermediate processing visualizations."),
            ("Review and save",
             "Compare original and processed audio, inspect scores or decoded text, "
             "then download the result when it is ready."),
        ]
        root = QVBoxLayout(self)
        self.stack = QStackedWidget()
        for title, text in self._pages:
            page = QFrame()
            layout = QVBoxLayout(page)
            heading = QLabel(title)
            heading.setObjectName("TitleLabel")
            body = QLabel(text)
            body.setObjectName("Caption")
            body.setWordWrap(True)
            layout.addWidget(heading)
            layout.addWidget(body)
            layout.addStretch()
            self.stack.addWidget(page)
        root.addWidget(self.stack)

        self.page_label = QLabel()
        self.page_label.setObjectName("Caption")
        root.addWidget(self.page_label)
        buttons = QDialogButtonBox()
        self.back_btn = buttons.addButton("Back", QDialogButtonBox.ButtonRole.ActionRole)
        self.next_btn = buttons.addButton("Next", QDialogButtonBox.ButtonRole.ActionRole)
        self.skip_btn = buttons.addButton("Skip", QDialogButtonBox.ButtonRole.DestructiveRole)
        self.finish_btn = buttons.addButton("Finish", QDialogButtonBox.ButtonRole.AcceptRole)
        self.back_btn.clicked.connect(self._back)
        self.next_btn.clicked.connect(self._next)
        self.skip_btn.clicked.connect(self._finish)
        self.finish_btn.clicked.connect(self._finish)
        root.addWidget(buttons)
        self._update_controls()

    def _update_controls(self):
        index = self.stack.currentIndex()
        self.page_label.setText(f"Step {index + 1} of {len(self._pages)}")
        self.back_btn.setEnabled(index > 0)
        self.next_btn.setVisible(index < len(self._pages) - 1)
        self.finish_btn.setVisible(index == len(self._pages) - 1)

    def _back(self):
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._update_controls()

    def _next(self):
        self.stack.setCurrentIndex(min(len(self._pages) - 1, self.stack.currentIndex() + 1))
        self._update_controls()

    def _finish(self):
        from ui import theme as theme_module
        theme_module.set_walkthrough_completed(self._user_key)
        self.accept()


class AudioInputCard(QFrame):
    """Reusable metadata card for the currently selected audio input."""

    replace_requested = Signal()
    remove_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("AudioInputCard")
        self.setMinimumHeight(118)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel("Input Audio")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.details = QLabel("No audio loaded. Upload, record, or choose a demo to begin.")
        self.details.setObjectName("Caption")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.transport = AudioTransportWidget()
        self.transport.setMinimumHeight(34)
        # Playback is presented below the module waveform. Keeping the
        # transport here would duplicate the same controls in the metadata
        # card and make the input area visually noisy.
        self.transport.setVisible(False)
        buttons = QHBoxLayout()
        self.replace_btn = QPushButton("Replace")
        self.replace_btn.setObjectName("SecondaryButton")
        self.replace_btn.setToolTip("Choose a different audio input.")
        self.replace_btn.clicked.connect(self.replace_requested)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setObjectName("SecondaryButton")
        self.remove_btn.setToolTip("Clear the current audio input.")
        self.remove_btn.clicked.connect(self.remove_requested)
        buttons.addWidget(self.replace_btn)
        buttons.addWidget(self.remove_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.replace_btn.setMinimumHeight(32)
        self.remove_btn.setMinimumHeight(32)
        self.set_audio(None)

    def set_audio(self, audio, sample_rate=None, filename="—", source="—", channels=1):
        if audio is None or len(audio) == 0:
            self.details.setText("No audio loaded. Upload, record, or choose a demo to begin.")
            self.transport.clear()
            self.replace_btn.setEnabled(True)
            self.remove_btn.setEnabled(False)
            return
        duration = len(audio) / float(sample_rate)
        channel_text = "Mono" if channels == 1 else f"{channels} channels"
        self.details.setText(
            f"<b>File:</b> {filename}  |  <b>Duration:</b> {duration:.2f}s  |  "
            f"<b>Sample rate:</b> {sample_rate / 1000:.1f} kHz  |  "
            f"<b>Channels:</b> {channel_text}  |  <b>Source:</b> {source}")
        self.transport.set_audio(audio, sample_rate)
        self.remove_btn.setEnabled(True)


class AudioPlayButton(QWidget):
    """A compact Play/Pause/Stop control for a generated waveform.

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
        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.setObjectName("PlayButton")
        self.pause_button.setToolTip("Pause this waveform.")
        self.pause_button.clicked.connect(self._pause)
        layout.addWidget(self.pause_button)
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setObjectName("PlayButton")
        self.stop_button.setToolTip("Stop playback and return to the beginning.")
        self.stop_button.clicked.connect(self._stop)
        layout.addWidget(self.stop_button)
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
        self._set_controls_enabled(False)

    def set_audio(self, x, fs=dsp.FS):
        wav_bytes = dsp.to_wav_bytes(x, fs)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav_bytes)
        f.close()
        self._path = f.name
        AudioPlayButton._tempfiles.append(f.name)
        self._player.setSource(QUrl.fromLocalFile(self._path))
        self._set_controls_enabled(True)

    def _play(self):
        if self._path and os.path.exists(self._path):
            self._player.play()

    def _pause(self):
        self._player.pause()

    def _stop(self):
        self._player.stop()
        self._player.setPosition(0)

    def _set_controls_enabled(self, enabled):
        for button in (self.button, self.pause_button, self.stop_button):
            button.setEnabled(enabled)

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


class AudioTransportWidget(QWidget):
    """A fuller audio transport control: Play / Pause / Stop / Replay,
    an elapsed/duration label, and the same position_changed /
    playback_active_changed signals as AudioPlayButton for playhead sync.
    Used where a page needs real Play/Pause/Stop/Replay controls (Audio
    Matcher's uploaded/recorded query and matched-result playback), not
    just a single play button."""

    position_changed = Signal(float)
    playback_active_changed = Signal(bool)

    _tempfiles = []

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.play_btn = QPushButton("▶ Play")
        self.pause_btn = QPushButton("⏸ Pause")
        self.stop_btn = QPushButton("⏹ Stop")
        self.replay_btn = QPushButton("⟲ Replay")
        self.play_btn.setToolTip("Play this audio.")
        self.pause_btn.setToolTip("Pause playback.")
        self.stop_btn.setToolTip("Stop playback and return to the beginning.")
        self.replay_btn.setToolTip("Replay from the beginning.")
        for b in (self.play_btn, self.pause_btn, self.stop_btn, self.replay_btn):
            b.setObjectName("PlayButton")
            layout.addWidget(b)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("Caption")
        layout.addWidget(self.time_label)
        layout.addStretch()

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.replay_btn.clicked.connect(self.replay)

        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._path = None
        self._duration_ms = 0

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._set_controls_enabled(False)

    def set_audio(self, x, fs=dsp.FS):
        wav_bytes = dsp.to_wav_bytes(x, fs)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav_bytes)
        f.close()
        self._path = f.name
        AudioTransportWidget._tempfiles.append(f.name)
        self._player.setSource(QUrl.fromLocalFile(self._path))
        self._set_controls_enabled(True)
        self.time_label.setText("0:00 / 0:00")

    def clear(self):
        self._player.stop()
        self._player.setSource(QUrl())
        self._path = None
        self._duration_ms = 0
        self._set_controls_enabled(False)
        self.time_label.setText("0:00 / 0:00")

    def _set_controls_enabled(self, enabled):
        for b in (self.play_btn, self.pause_btn, self.stop_btn, self.replay_btn):
            b.setEnabled(enabled)

    def play(self):
        if self._path:
            self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()
        self._player.setPosition(0)

    def replay(self):
        if self._path:
            self._player.setPosition(0)
            self._player.play()

    @staticmethod
    def _fmt(ms):
        s = max(0, int(ms / 1000))
        return f"{s // 60}:{s % 60:02d}"

    def _on_duration_changed(self, duration_ms):
        self._duration_ms = duration_ms
        self.time_label.setText(f"{self._fmt(0)} / {self._fmt(duration_ms)}")

    def _on_position_changed(self, position_ms):
        self.time_label.setText(f"{self._fmt(position_ms)} / {self._fmt(self._duration_ms)}")
        if self._duration_ms > 0:
            self.position_changed.emit(min(1.0, position_ms / self._duration_ms))

    def _on_state_changed(self, state):
        is_playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.playback_active_changed.emit(is_playing)
