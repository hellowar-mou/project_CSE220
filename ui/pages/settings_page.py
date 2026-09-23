from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QTextBrowser, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt

from ui import theme as theme_module
from ui.widgets import FirstRunWalkthrough
from ui.algorithm_visualizer import AlgorithmVisualizer


class SettingsPage(QWidget):
    """Global app settings and interactive Help & Documentation section.

    Toggling the theme here switches the entire running app immediately.
    The Help & Documentation section provides written explanations of all 5
    toolbox modules, each accompanied by an 'Explain the Algorithm' button
    that opens an interactive, animated visualizer video/slideshow.
    """

    def __init__(self, username="default", walkthrough_callback=None):
        super().__init__()
        self.username = username
        self.walkthrough_callback = walkthrough_callback

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(8, 8, 8, 16)

        # Header card
        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; padding: 4px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("⚙️ Settings & Documentation")
        title.setObjectName("SectionTitle")
        caption = QLabel("App preferences, guided tour, and interactive algorithm documentation.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        # Appearance / Theme card
        theme_card = QFrame()
        theme_card.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        tc_layout = QVBoxLayout(theme_card)
        theme_title = QLabel("Appearance")
        theme_title.setObjectName("SectionTitle")
        tc_layout.addWidget(theme_title)

        theme_desc = QLabel("Choose Light or Dark mode. Applies immediately across the whole "
                            "application — every page, every plot, and every panel.")
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

        # Help & Documentation section header
        help_header = QFrame()
        help_header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        hh_layout = QVBoxLayout(help_header)

        help_title = QLabel("📚 Help & Documentation")
        help_title.setObjectName("SectionTitle")
        hh_layout.addWidget(help_title)

        help_desc = QLabel(
            "Review the core DSP workflow, mathematical formulas, and algorithms used by each SonusCure module. "
            "Click <b>Explain the Algorithm</b> under any module to launch an interactive, step-by-step animated video visualizer."
        )
        help_desc.setObjectName("Caption")
        help_desc.setWordWrap(True)
        hh_layout.addWidget(help_desc)

        walkthrough_row = QHBoxLayout()
        presentation_btn = QPushButton("🎬 Open Presentation Slides")
        presentation_btn.setObjectName("PrimaryButton")
        presentation_btn.setStyleSheet("QPushButton { background: #2f6690; color: white; font-weight: 700; "
                                       "padding: 8px 16px; border-radius: 8px; font-size: 0.88rem; }"
                                       "QPushButton:hover { background: #3b7bb0; }")
        presentation_btn.setToolTip("Open the interactive animated presentation slide deck in your browser.")
        presentation_btn.clicked.connect(self._open_presentation)
        walkthrough_row.addWidget(presentation_btn)

        walkthrough_btn = QPushButton("🎓 Launch App Walkthrough")
        walkthrough_btn.setObjectName("SecondaryButton")
        walkthrough_btn.setToolTip("Open the interactive first-run guided tour again.")
        walkthrough_btn.clicked.connect(self._open_walkthrough)
        walkthrough_row.addWidget(walkthrough_btn)
        walkthrough_row.addStretch()
        hh_layout.addLayout(walkthrough_row)

        layout.addWidget(help_header)

        # Individual Module Documentation Cards
        for info in self._module_documentation():
            card = self._create_module_card(info)
            layout.addWidget(card)

        layout.addStretch()

        theme_module.add_listener(self._sync_buttons)
        self._sync_buttons()

    def _create_module_card(self, info):
        module_name = info["name"]
        icon = info["icon"]
        badge_text = info["badge"]
        html_content = info["html"]

        card = QFrame()
        card.setObjectName("ModuleDocCard")
        card.setStyleSheet(
            "QFrame#ModuleDocCard { background: rgba(255,255,255,0.82); border-radius: 14px; "
            "border: 1px solid rgba(111,177,234,0.22); }"
        )
        c_layout = QVBoxLayout(card)
        c_layout.setSpacing(10)
        c_layout.setContentsMargins(16, 14, 16, 14)

        # Card Title Row
        head_row = QHBoxLayout()
        title_label = QLabel(f"<span style='font-size:1.15rem; font-weight:800;'>{icon} {module_name}</span>")
        title_label.setObjectName("SectionTitle")
        head_row.addWidget(title_label)

        badge_label = QLabel(badge_text)
        badge_label.setStyleSheet(
            "background: rgba(111,177,234,0.18); color: #2f6690; padding: 3px 10px; "
            "border-radius: 12px; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.03em;"
        )
        head_row.addWidget(badge_label)
        head_row.addStretch()
        c_layout.addLayout(head_row)

        # Written Documentation Label (natural word-wrapped layout, no clipping or fixed height)
        doc_label = QLabel()
        doc_label.setWordWrap(True)
        doc_label.setTextFormat(Qt.RichText)
        doc_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        doc_label.setStyleSheet(
            "QLabel { background: transparent; font-size: 0.88rem; line-height: 1.6; color: inherit; }"
        )
        doc_label.setText(html_content)
        c_layout.addWidget(doc_label)

        # Prominent "Explain the Algorithm" Button directly beneath the explanation
        btn_row = QHBoxLayout()
        explain_btn = QPushButton("▶  Explain the Algorithm")
        explain_btn.setObjectName("SecondaryButton")
        explain_btn.setStyleSheet(
            "QPushButton { background: rgba(111,177,234,0.16); color: #2f6690; font-weight: 700; "
            "padding: 8px 18px; border-radius: 8px; border: 1px solid rgba(111,177,234,0.35); font-size: 0.88rem; }"
            "QPushButton:hover { background: rgba(111,177,234,0.30); border-color: #6fb1ea; }"
        )
        explain_btn.setToolTip(f"Open the interactive {module_name} algorithm visualizer video/slideshow.")
        explain_btn.clicked.connect(lambda checked=False, name=module_name: self._open_visualizer(name))
        btn_row.addWidget(explain_btn)
        btn_row.addStretch()
        c_layout.addLayout(btn_row)

        return card

    def _sync_buttons(self):
        mode = theme_module.get_current_mode()
        self.light_btn.setChecked(mode == "light")
        self.dark_btn.setChecked(mode == "dark")
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

    def _open_presentation(self):
        import os
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "presentation.html")
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_visualizer(self, module):
        AlgorithmVisualizer(module, self).exec()

    @staticmethod
    def _module_documentation():
        return [
            {
                "name": "Noise Remover",
                "icon": "🔇",
                "badge": "4 Methods · Convolution, FFT & STFT",
                "html": """
                <p><b>Goal:</b> Suppress unwanted interference (power-line hum, broadband hiss, ambient noise) while preserving original speech and instrument audio fidelity.</p>
                <div style="margin: 6px 0;">
                <p><b>• Moving Average Filter:</b> Time-domain convolution smoothing: <code>y[n] = (1/N) Σ x[n−k] = (x * h)[n]</code>. Slides a uniform kernel of size <i>N=9</i> across samples, acting as an elementary low-pass filter to attenuate rapid noise spikes without requiring an FFT.</p>
                <p><b>• Frequency Domain Denoise:</b> The Fourier Transform decomposes the signal: <code>X(f) = ℱ{x[n]}</code>. Surgical notch filters zero out bins at <b>60 Hz, 120 Hz, and 180 Hz</b> (eliminating AC hum), while bins above 4000 Hz are attenuated by <code>0.15×</code> to silence hiss before Inverse FFT reconstruction.</p>
                <p><b>• Spectral Subtraction:</b> Uses the Short-Time Fourier Transform (STFT) with 512-sample frames. The first 10 frames are assumed noise-only to estimate the noise power spectrum <code>|N̂(f)|²</code>. Noise power is subtracted from every frame: <code>|Ŝ|² = max(|X|² − α|N̂|², β|N̂|²)</code> with over-subtraction <i>α=2.0</i> and spectral floor <i>β=0.02</i> to eliminate musical noise.</p>
                <p><b>• Wiener Filter:</b> The optimal linear estimator in the minimum mean squared error (MMSE) sense. Calculates an adaptive gain: <code>G(f,t) = max(1 − |N̂(f)|² / |X(f,t)|², 0)</code>. Bins with strong speech remain untouched (<i>G ≈ 1</i>) while noise bins are smoothly attenuated (<i>G ≈ 0</i>).</p>
                </div>
                """
            },
            {
                "name": "Equalizer",
                "icon": "🎛️",
                "badge": "3 Filter Architectures · FIR, IIR & Biquads",
                "html": """
                <p><b>Goal:</b> Shape the spectral balance of audio across frequency bands via gain adjustments (boosting or cutting specific tonal regions).</p>
                <div style="margin: 6px 0;">
                <p><b>• 3-Band FIR Equalizer:</b> Designs 3 windowed-sinc FIR filters (Lowpass &lt;300 Hz, Bandpass 300–3000 Hz, Highpass &gt;3000 Hz) using <code>signal.firwin</code>. Convolves each band independently, scales by user gains, and sums: <code>y[n] = Σ gᵢ (x * hᵢ)[n]</code>.</p>
                <p><b>• 9-Band IIR Peaking Cascade:</b> Professional graphic EQ using industry-standard ISO frequencies (60Hz to 16kHz). Each band implements a 2nd-order peaking biquad filter from the Robert Bristow-Johnson Audio EQ Cookbook with <i>Q=1.0</i>, cascaded in series via <code>signal.lfilter</code>.</p>
                <p><b>• 9-Band FIR Windowed-Sinc Bands:</b> Computes geometric-mean crossovers <code>f_edge = √(f_i · f_{i+1})</code> to create 9 contiguous FIR bandpass filters. Offers strictly <b>linear phase</b> (zero phase distortion), ideal for audio mastering.</p>
                <p><b>• Shelving + Peaking Hybrid:</b> Utilizes low-shelf biquads at 60 Hz for deep bass shaping, high-shelf biquads at 16 kHz for air/treble, and peaking biquads for the 7 middle bands.</p>
                </div>
                """
            },
            {
                "name": "Audio Editor",
                "icon": "✂️",
                "badge": "7 Features · Array Slicing, OLA & LTI",
                "html": """
                <p><b>Goal:</b> Non-destructive, high-precision temporal and acoustic manipulation of audio signals using standard DSP primitives.</p>
                <div style="margin: 6px 0;">
                <p><b>• Trim & Join:</b> <code>trim</code> maps seconds to array indices <code>start_idx = int(start_s * fs)</code> and extracts <code>x[start:end]</code>. <code>join</code> concatenates multiple clips end-to-end and normalizes peak amplitude to prevent clipping distortion.</p>
                <p><b>• Reverse:</b> Performs time-domain reflection: <code>y[n] = x[N − 1 − n]</code>. Inverts the phase spectrum while maintaining identical frequency magnitudes.</p>
                <p><b>• Time Scale (Speed Change):</b> Changes playback speed without altering vocal pitch using <b>Overlap-Add (OLA)</b> with 30ms Hann-windowed frames. Shifting synthesis hop distance <code>hop_out = hop_in / speed</code> shortens or lengthens duration without chipmunk pitch shifts.</p>
                <p><b>• Fade In / Fade Out:</b> Multiplies signal boundaries by linear amplitude envelopes (<code>0 → 1</code> and <code>1 → 0</code>), eliminating transient boundary clicks.</p>
                <p><b>• Smooth & Echo (LTI Systems):</b> <code>smooth</code> convolves audio with a uniform moving-average kernel to tame harsh transients. <code>echo</code> convolves with a discrete impulse response <code>h[n] = Σ αᵏ δ[n − kD]</code> simulating decaying room acoustic reflections.</p>
                </div>
                """
            },
            {
                "name": "Morse Code Converter",
                "icon": "📡",
                "badge": "On-Off Keying (OOK) · Synthesis & 3-Way Decoding",
                "html": """
                <p><b>Goal:</b> Bidirectional conversion between text and Morse code audio tones, incorporating carrier synthesis and robust multi-algorithm decoding.</p>
                <div style="margin: 6px 0;">
                <p><b>• Tone Synthesis:</b> Encodes characters according to international Morse timing: Dot = 1 unit (approx 70ms), Dash = 3 units, element space = 1 unit, letter space = 3 units, word space = 7 units. Modulates a 700 Hz sine wave carrier with 5ms fade envelopes to avoid audible switching transients.</p>
                <p><b>• Auto Frequency Detection:</b> Performs an FFT over the 200–2000 Hz band to identify the dominant spectral peak <code>f_tone = argmax |X(f)|</code>, then designs an FIR bandpass filter (±80 Hz) to isolate the carrier.</p>
                <p><b>• Envelope Detection (3 Choices):</b> Extracts the instantaneous loudness curve using: 1. <b>Hilbert Transform</b> (analytic signal magnitude), 2. <b>Rectification + Butterworth Low-pass</b> (classic AM radio demodulation), or 3. <b>Short-Time RMS Energy</b> (moving convolution).</p>
                <p><b>• Keyed Classification & Decoding:</b> Thresholds the envelope into binary states. Uses discrete differentiation <code>np.diff()</code> to find edges, measures pulse and gap durations to classify dots and dashes, and reconstructs the decoded text via dictionary lookup.</p>
                </div>
                """
            },
            {
                "name": "Audio Matcher",
                "icon": "🔍",
                "badge": "Pattern Matching · Correlation & STFT Fingerprinting",
                "html": """
                <p><b>Goal:</b> Identify audio clips and locate sound patterns within recordings using matched filtering and time-frequency fingerprinting.</p>
                <div style="margin: 6px 0;">
                <p><b>• Template Sounds:</b> Synthesizes physical acoustic models: Bell (880 Hz sine wave, decay τ=0.12s), Clap (broadband noise burst, decay τ=0.01s), and Tap (transient percussive click, decay τ=0.003s) using exponential decay envelopes <code>e^(−t/τ)</code>.</p>
                <p><b>• Cross-Correlation Matched Filter:</b> Slides a template along a recording and computes inner products: <code>r_xy[k] = Σ x[n] y[n+k]</code>. A prominent correlation spike reveals the exact sample offset where the sound occurs.</p>
                <p><b>• Normalized Clip Similarity:</b> Normalizes cross-correlation by Euclidean L2 norms <code>||x||₂ · ||y||₂</code>, generating a volume-invariant score between 0.0 (unrelated) and 1.0 (identical).</p>
                <p><b>• Spectral Fingerprint (Shazam Algorithm):</b> Computes an STFT spectrogram, extracts the top 3 loudest peaks per frame, and hashes pairs. When matching, consistent time-difference alignments <code>Δt = t_lib − t_query</code> produce a distinctive histogram spike indicating a confirmed track match.</p>
                </div>
                """
            }
        ]
