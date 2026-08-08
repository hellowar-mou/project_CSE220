from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QFrame
from PySide6.QtCore import Qt

import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class EqualizerPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🎚️ 3-Band Equalizer")
        title.setObjectName("SectionTitle")
        caption = QLabel("FIR band-split filters, recombined via superposition.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        controls = QHBoxLayout()
        self.sliders = {}
        self.labels = {}
        for band in ["Low", "Mid", "High"]:
            col = QVBoxLayout()
            lab = QLabel(f"{band} gain: 1.0")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 20)
            sld.setValue(10)
            sld.valueChanged.connect(self._on_change)
            col.addWidget(lab)
            col.addWidget(sld)
            controls.addLayout(col)
            self.sliders[band] = sld
            self.labels[band] = lab
        layout.addLayout(controls)

        self.spec_canvas = MplCanvas(n_rows=3)
        layout.addWidget(self.spec_canvas)

        players = QHBoxLayout()
        self.p_orig = AudioPlayButton("Original")
        self.p_eq = AudioPlayButton("EQ'd")
        players.addWidget(self.p_orig)
        players.addWidget(self.p_eq)
        players.addStretch()
        layout.addLayout(players)
        layout.addStretch()

        self.clean = dsp.make_voice_like()
        lo, mid, hi = dsp.band_split(self.clean)
        self.spec_canvas.plot_spectra([(lo, "Low"), (mid, "Mid"), (hi, "High")], "Band-split spectra")
        self.p_orig.set_audio(self.clean)
        self._on_change()

    def _on_change(self):
        gains = {}
        for band, sld in self.sliders.items():
            g = sld.value() / 10.0
            gains[band] = g
            self.labels[band].setText(f"{band} gain: {g:.1f}")
        eq = dsp.equalize(self.clean, gains=(gains["Low"], gains["Mid"], gains["High"]))
        self.p_eq.set_audio(eq)
