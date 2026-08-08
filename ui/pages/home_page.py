from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QFrame, QHBoxLayout
from PySide6.QtCore import Qt


def _card(title, desc):
    frame = QFrame()
    frame.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
    layout = QVBoxLayout(frame)
    t = QLabel(title)
    t.setStyleSheet("font-weight: 700; font-size: 13px; color: #2f3e50;")
    d = QLabel(desc)
    d.setObjectName("Caption")
    d.setWordWrap(True)
    layout.addWidget(t)
    layout.addWidget(d)
    return frame


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 16px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🌊 Audio Signals Toolbox")
        title.setObjectName("TitleLabel")
        h_layout.addWidget(title)

        pills_row = QHBoxLayout()
        for text in ["Convolution", "Fourier Transform", "Correlation", "LTI Systems"]:
            p = QLabel(text)
            p.setProperty("class", "Pill")
            p.setStyleSheet("background:#e7f3ff; color:#2f6690; font-size:11px; font-weight:600;"
                            "border-radius:10px; padding:4px 10px;")
            pills_row.addWidget(p)
        pills_row.addStretch()
        h_layout.addLayout(pills_row)

        desc = QLabel("One shared DSP engine, six live applications. Pick a module from the "
                       "sidebar — every demo runs the real NumPy/SciPy signal processing "
                       "pipeline in real time.")
        desc.setWordWrap(True)
        desc.setObjectName("Caption")
        h_layout.addWidget(desc)

        layout.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(12)
        items = [
            ("🧹 Noise Remover", "Time-domain & frequency-domain LTI filtering"),
            ("🎚️ Equalizer", "3-band FIR filters + superposition"),
            ("✂️ Mini Editor", "Trim/reverse/fade + convolution echo"),
            ("📡 Morse Decoder", "Bandpass filter + Hilbert envelope detection"),
            ("🎯 Template Matcher", "Cross-correlation / matched filtering"),
            ("🎵 Mini Shazam", "Spectrogram peaks + correlation song ID"),
        ]
        for i, (t, d) in enumerate(items):
            grid.addWidget(_card(t, d), i // 3, i % 3)
        layout.addLayout(grid)
        layout.addStretch()
