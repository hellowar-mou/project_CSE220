import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
import dsp_core as dsp
from ui.widgets import MplCanvas, AudioPlayButton


class MatcherPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🎯 Sound Template Matcher")
        title.setObjectName("SectionTitle")
        caption = QLabel("Cross-correlation locates known sounds (clap/bell/tap) "
                          "hidden in a noisy recording.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        self.player = AudioPlayButton("Full recording")
        row = QHBoxLayout(); row.addWidget(self.player); row.addStretch()
        layout.addLayout(row)

        self.canvas = MplCanvas(n_rows=4)
        layout.addWidget(self.canvas)

        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setObjectName("Caption")
        layout.addWidget(self.results_label)
        layout.addStretch()

        self._build()

    def _build(self):
        recording, templates, placements = dsp.build_template_recording()
        self.player.set_audio(recording)

        figs_data = [(recording, "Recording")]
        lines = []
        for name, tmpl in templates.items():
            corr = dsp.match_template(recording, tmpl)
            peak_t = np.argmax(np.abs(corr)) / dsp.FS
            figs_data.append((corr, f"corr: {name}"))
            lines.append(f"• {name}: detected at {peak_t:.2f}s (true: {placements[name]:.2f}s)")

        self.canvas.plot_waveforms(figs_data, "Recording & correlation scores")
        self.results_label.setText("Detected vs. true placement:<br>" + "<br>".join(lines))
