import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QFrame
import dsp_core as dsp
from ui.widgets import AudioPlayButton


class ShazamPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("🎵 Mini Shazam — Song Identification")
        title.setObjectName("SectionTitle")
        caption = QLabel("STFT peak fingerprints + offset-correlation matching "
                          "against a 3-song library.")
        caption.setObjectName("Caption")
        caption.setWordWrap(True)
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        self.library = dsp.build_song_library()
        self.combo = QComboBox()
        self.combo.addItems(list(self.library.keys()))
        self.combo.currentTextChanged.connect(self._on_change)
        layout.addWidget(QLabel("Pick the 'true' song to sample a noisy snippet from:"))
        layout.addWidget(self.combo)

        self.snippet_player = AudioPlayButton("Unknown noisy snippet")
        row = QHBoxLayout(); row.addWidget(self.snippet_player); row.addStretch()
        layout.addLayout(row)

        self.scores_label = QLabel("")
        self.scores_label.setObjectName("SectionTitle")
        layout.addWidget(self.scores_label)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        lib_row = QHBoxLayout()
        self.lib_players = {}
        for name, audio in self.library.items():
            col = QVBoxLayout()
            p = AudioPlayButton(name)
            p.set_audio(audio)
            self.lib_players[name] = p
            col.addWidget(p)
            lib_row.addLayout(col)
        layout.addLayout(lib_row)
        layout.addStretch()

        self._on_change(self.combo.currentText())

    def _on_change(self, true_song):
        full = self.library[true_song]
        start = min(0.4, max(0, len(full) / dsp.FS - 0.9))
        snippet = full[int(start * dsp.FS):int(start * dsp.FS) + int(0.9 * dsp.FS)]
        rng = np.random.default_rng(1)
        snippet = dsp.normalize(snippet + 0.08 * rng.standard_normal(len(snippet)))
        self.snippet_player.set_audio(snippet)

        library_fps = {name: dsp.spectrogram_peaks(audio)[0] for name, audio in self.library.items()}
        scores = dsp.match_song(snippet, library_fps)
        best = max(scores, key=scores.get)

        score_text = "  |  ".join(
            f"{name}: {sc}{' ⭐' if name == best else ''}"
            for name, sc in sorted(scores.items(), key=lambda kv: -kv[1])
        )
        self.scores_label.setText(score_text)

        if best == true_song:
            self.result_label.setObjectName("SuccessLabel")
            self.result_label.setText(f"Correctly identified: {best} ✅")
        else:
            self.result_label.setObjectName("ErrorLabel")
            self.result_label.setText(f"Identified as {best}, true song was {true_song}")
        self.result_label.style().unpolish(self.result_label)
        self.result_label.style().polish(self.result_label)
