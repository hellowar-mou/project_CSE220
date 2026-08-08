from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame)
from PySide6.QtCore import Qt

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class NoiseRemoverPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🧹 Noise Remover")
        title.setObjectName("SectionTitle")
        caption = QLabel("Moving-average (convolution) vs. FFT bin-zeroing — "
                          "two LTI approaches to the same problem.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        controls = QHBoxLayout()
        self.n_label = QLabel("Moving-average window N: 9")
        self.n_slider = QSlider(Qt.Horizontal)
        self.n_slider.setRange(1, 20)   # maps to N = 3..41 step 2
        self.n_slider.setValue(3)
        self.n_slider.valueChanged.connect(self._on_change)

        self.hiss_label = QLabel("Freq-domain hiss attenuation: 0.15")
        self.hiss_slider = QSlider(Qt.Horizontal)
        self.hiss_slider.setRange(0, 20)
        self.hiss_slider.setValue(3)
        self.hiss_slider.valueChanged.connect(self._on_change)

        col1 = QVBoxLayout(); col1.addWidget(self.n_label); col1.addWidget(self.n_slider)
        col2 = QVBoxLayout(); col2.addWidget(self.hiss_label); col2.addWidget(self.hiss_slider)
        controls.addLayout(col1)
        controls.addLayout(col2)
        layout.addLayout(controls)

        self.wave_canvas = MplCanvas(n_rows=3)
        self.spec_canvas = MplCanvas(n_rows=2)
        layout.addWidget(self.wave_canvas)
        layout.addWidget(self.spec_canvas)

        players = QHBoxLayout()
        self.p_noisy = AudioPlayButton("Noisy")
        self.p_time = AudioPlayButton("Denoised (time)")
        self.p_freq = AudioPlayButton("Denoised (freq)")
        players.addWidget(self.p_noisy)
        players.addWidget(self.p_time)
        players.addWidget(self.p_freq)
        players.addStretch()
        layout.addLayout(players)
        layout.addStretch()

        self.clean = dsp.make_voice_like()
        self.noisy = dsp.make_noisy(self.clean)
        self._on_change()

    def _on_change(self):
        N = self.n_slider.value() * 2 + 1
        hiss_atten = self.hiss_slider.value() / 20.0
        self.n_label.setText(f"Moving-average window N: {N}")
        self.hiss_label.setText(f"Freq-domain hiss attenuation: {hiss_atten:.2f}")

        den_time = dsp.moving_average_filter(self.noisy, N=N)
        den_freq = dsp.freq_domain_denoise(self.noisy, hiss_atten=hiss_atten)

        self.wave_canvas.plot_waveforms(
            [(self.noisy, "Noisy"), (den_time, "Time-domain"), (den_freq, "Freq-domain")],
            "Waveform comparison", xlim=(0, 0.05))
        self.spec_canvas.plot_spectra(
            [(self.noisy, "Noisy"), (den_freq, "Freq-domain denoised")], "Spectrum (dB)")

        self.p_noisy.set_audio(self.noisy)
        self.p_time.set_audio(dsp.normalize(den_time))
        self.p_freq.set_audio(dsp.normalize(den_freq))
