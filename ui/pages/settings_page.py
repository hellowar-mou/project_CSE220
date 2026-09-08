from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QTextBrowser, QScrollArea,
)

from ui import theme as theme_module
from ui.widgets import FirstRunWalkthrough


class SettingsPage(QWidget):
    """Global app settings. Currently: dark/light theme toggle.

    Toggling here switches the ENTIRE running app immediately — every page
    already built, not just this one — via ui.theme.set_current_mode(),
    which re-applies every registered widget's stylesheet and recolors
    every registered plot canvas app-wide."""

    def __init__(self, username="default", walkthrough_callback=None):
        super().__init__()
        self.username = username
        self.walkthrough_callback = walkthrough_callback
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("⚙️ Settings")
        title.setObjectName("SectionTitle")
        caption = QLabel("App-wide preferences.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        theme_card = QFrame()
        theme_card.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        tc_layout = QVBoxLayout(theme_card)
        theme_title = QLabel("Appearance")
        theme_title.setObjectName("SectionTitle")
        tc_layout.addWidget(theme_title)

        theme_desc = QLabel("Choose Light or Dark. Applies immediately across the whole "
                             "application — every page, every plot, every panel.")
        theme_desc.setObjectName("Caption")
        theme_desc.setWordWrap(True)
        tc_layout.addWidget(theme_desc)

        btn_row = QHBoxLayout()
        self.light_btn = QPushButton("☀️  Light")
        self.dark_btn = QPushButton("🌙  Dark")
        for b in (self.light_btn, self.dark_btn):
            b.setObjectName("SecondaryButton")
            b.setCheckable(True)
            b.setMinimumWidth(120)
        self.light_btn.clicked.connect(lambda: self._set_theme("light"))
        self.dark_btn.clicked.connect(lambda: self._set_theme("dark"))
        btn_row.addWidget(self.light_btn)
        btn_row.addWidget(self.dark_btn)
        btn_row.addStretch()
        tc_layout.addLayout(btn_row)

        layout.addWidget(theme_card)

        help_card = QFrame()
        help_card.setStyleSheet(
            "QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        help_layout = QVBoxLayout(help_card)
        help_title = QLabel("Help & Documentation")
        help_title.setObjectName("SectionTitle")
        help_layout.addWidget(help_title)
        help_desc = QLabel(
            "Review the workflow, algorithms, and equations used by each toolbox module.")
        help_desc.setObjectName("Caption")
        help_desc.setWordWrap(True)
        help_layout.addWidget(help_desc)
        walkthrough_btn = QPushButton("🎓 Get Walkthrough")
        walkthrough_btn.setObjectName("PrimaryButton")
        walkthrough_btn.setToolTip("Open the guided walkthrough again for this account.")
        walkthrough_btn.clicked.connect(self._open_walkthrough)
        help_layout.addWidget(walkthrough_btn)

        documentation = QTextBrowser()
        documentation.setOpenExternalLinks(False)
        documentation.setMinimumHeight(360)
        documentation.setHtml(self._documentation_html())
        help_layout.addWidget(documentation)
        layout.addWidget(help_card)
        layout.addStretch()

        theme_module.add_listener(self._sync_buttons)
        self._sync_buttons()

    def _sync_buttons(self):
        mode = theme_module.get_current_mode()
        self.light_btn.setChecked(mode == "light")
        self.dark_btn.setChecked(mode == "dark")
        # Re-polish so the :checked pseudo-state style actually repaints —
        # Qt doesn't always repaint checkable-button state changes made
        # from code without this nudge.
        for b in (self.light_btn, self.dark_btn):
            b.style().unpolish(b)
            b.style().polish(b)

    def _set_theme(self, mode):
        theme_module.set_current_mode(mode)
        theme_module.save_theme(mode)

    def _open_walkthrough(self):
        if self.walkthrough_callback:
            self.walkthrough_callback()
        else:
            FirstRunWalkthrough(self.username, self).exec()

    @staticmethod
    def _documentation_html():
        return """
        <h3>How to use the toolbox</h3>
        <p>Choose a module, provide an uploaded file, microphone recording, or demo,
        adjust its controls, inspect the visualizations, then play or download the result.</p>
        <h3>Noise Remover</h3>
        <p>Reduces unwanted components while preserving the signal. Moving-average
        smoothing uses convolution:</p>
        <p><b>y[n] = (1/M) Σ x[n-k]</b></p>
        <p>Frequency filtering uses the Fourier transform, where multiplication in
        frequency corresponds to convolution in time: <b>Y(f) = X(f)H(f)</b>.</p>
        <h3>Equalizer</h3>
        <p>Splits audio into frequency bands and applies gain settings. The output is
        a superposition of band responses:</p>
        <p><b>y[n] = Σ gᵢ (x * hᵢ)[n]</b></p>
        <p>The available implementations are IIR peaking filters, FIR windowed-sinc
        filters, and a shelving/peaking hybrid.</p>
        <h3>Audio Editor</h3>
        <p>Trim selects a time interval; reverse maps <b>y[n] = x[N−1−n]</b>;
        fade applies a time-varying gain; echo uses an impulse response such as
        <b>h[n] = δ[n] + αδ[n−D]</b>.</p>
        <h3>Morse Code Converter</h3>
        <p>Encoding maps symbols to keyed tones. Decoding applies band-pass filtering,
        envelope detection, thresholding, timing classification, and Morse lookup.
        A dot is approximately one timing unit and a dash three units.</p>
        <h3>Audio Matcher</h3>
        <p>Compares the current input against every library track using normalized
        cross-correlation, clip similarity, and STFT spectral fingerprints.
        Cross-correlation is:</p>
        <p><b>r[k] = Σ x[n]y[n+k]</b></p>
        <p>Combined mode averages the enabled normalized scores. Results below the
        confidence threshold are reported as <b>No Reliable Match Found</b>.</p>
        <h3>Visualizations</h3>
        <p>Waveforms show amplitude over time. Spectra show magnitude by frequency.
        Spectrograms show frequency energy over time. Use the waveform zoom controls
        for long recordings.</p>
        """
