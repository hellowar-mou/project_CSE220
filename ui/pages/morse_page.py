from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFrame
import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class MorsePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("📡 Morse Code Decoder")
        title.setObjectName("SectionTitle")
        caption = QLabel("Bandpass filter → Hilbert envelope → threshold → text.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        self.input = QLineEdit("SOS HELP")
        self.input.setPlaceholderText("Message to encode & decode")
        self.input.textChanged.connect(self._on_change)
        layout.addWidget(self.input)

        self.canvas = MplCanvas(n_rows=4)
        layout.addWidget(self.canvas)

        row = QHBoxLayout()
        self.player = AudioPlayButton("Encoded tone recording")
        row.addWidget(self.player)
        row.addStretch()
        layout.addLayout(row)

        metrics = QHBoxLayout()
        self.sent_label = QLabel("Sent: —")
        self.decoded_label = QLabel("Decoded: —")
        self.sent_label.setObjectName("SectionTitle")
        self.decoded_label.setObjectName("SectionTitle")
        metrics.addWidget(self.sent_label)
        metrics.addWidget(self.decoded_label)
        metrics.addStretch()
        layout.addLayout(metrics)
        layout.addStretch()

        self._on_change()

    def _on_change(self):
        message = self.input.text().strip()
        if not message:
            return
        audio, unit = dsp.synth_morse(message)
        decoded, filtered, envelope, keyed = dsp.decode_morse(audio, unit=unit)

        self.canvas.plot_waveforms(
            [(audio, "Raw tone"), (filtered, "Bandpassed"),
             (envelope, "Envelope"), (keyed.astype(float), "Keyed")],
            f"Decoded: '{decoded}'")
        self.player.set_audio(audio)
        self.sent_label.setText(f"Sent: {message.upper()}")
        self.decoded_label.setText(f"Decoded: {decoded}")
