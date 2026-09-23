"""
Interactive Algorithm Visualizers for the Settings Help & Documentation.

Behaves like a mini instructional video / animated slideshow player:
- Separate, technically accurate visualizers for all 5 modules:
    * Noise Remover (Moving Avg, Freq Notch/Hiss, Spectral Subtraction, Wiener)
    * Equalizer (3-Band FIR, 9-Band IIR Peaking, 9-Band FIR Sinc, 9-Band Shelving)
    * Audio Editor (Trim, Join, Reverse, Time Scale OLA, Fade, Smooth, Echo LTI)
    * Morse Code Converter (Synthesis, Auto Tone Detect, Envelope 3-way, Keyed-to-Text, Pipeline)
    * Audio Matcher (Templates, Cross-Correlation, Normalized Similarity, Spectral Fingerprint, Search)
- Mini-video controls: Play/Pause, seek/scrub timeline, speed control (0.5x, 1x, 1.5x, 2x),
  current time / total time, previous/next stage buttons, and direct stage picker.
- Explanations, equations, and code snippets adapted directly from course documentation.
- Dual-panel synchronized Matplotlib canvas responding to theme changes.
- All waveforms are drawn live as the video plays or scrubs from 0% to 100%.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any

import numpy as np
from scipy import signal
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QComboBox, QFrame, QScrollArea, QWidget, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import dsp_core as dsp
import logic.equalizer as eq_logic
from ui import theme as theme_module


@dataclass
class VisualizerStage:
    title: str
    subtitle: str
    description: str
    formula: str
    code_snippet: str
    dsp_primitive: str
    duration: float  # duration in seconds for video playback
    render_fn: Callable[[Any, Any, float, bool], None]


# -----------------------------------------------------------------------------
# Data helpers and test signal generators
# -----------------------------------------------------------------------------
def _test_signals():
    fs = dsp.FS
    t = np.arange(int(fs * 1.5)) / fs
    clean = (0.55 * np.sin(2 * np.pi * 220 * t)
             + 0.25 * np.sin(2 * np.pi * 440 * t)
             + 0.15 * np.sin(2 * np.pi * 880 * t))
    env = 0.5 * (1 + np.sin(2 * np.pi * 1.2 * t - np.pi / 2))
    clean = dsp.normalize(clean * (0.6 + 0.4 * env))
    noisy = dsp.make_noisy(clean, hum_amp=0.22, hiss_amp=0.04, fs=fs)
    return fs, t, clean, noisy


# -----------------------------------------------------------------------------
# Noise Remover Stages
# -----------------------------------------------------------------------------
def _build_noise_remover_stages() -> List[VisualizerStage]:
    fs, t, clean, noisy = _test_signals()
    stages = []

    # 1. Overview & Input Noisy Signal
    def render_nr_overview(ax1, ax2, p, dark):
        color_sig = "#e74c3c"
        color_fft = "#e67e22"
        # Waveform draws live from t=0
        n_pts = max(1, int(len(t) * p))
        ax1.plot(t[:n_pts], noisy[:n_pts], color=color_sig, lw=1.4, label="Noisy Audio x[n]")
        ax1.set_title("Input Audio: Clean Signal + 60Hz Power Hum + High-Frequency Hiss", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-1.15, 1.15)
        if n_pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # FFT Spectrum draws live across frequencies from 0 to 4500 Hz
        f = np.fft.rfftfreq(len(noisy), 1 / fs)
        s = np.abs(np.fft.rfft(noisy))
        f_max_idx = int(np.searchsorted(f, 4500))
        pts_f = max(1, int(f_max_idx * p))
        disp_f = f[:pts_f]
        disp_s = s[:pts_f]
        ax2.plot(disp_f, disp_s, color=color_fft, lw=1.3, label="|X(f)| Spectrum")
        curr_f = disp_f[-1] if len(disp_f) > 0 else 0
        if curr_f >= 60:
            ax2.axvline(60, color="#e74c3c", ls="--", alpha=0.7, label="60 Hz Hum")
        if curr_f >= 120:
            ax2.axvline(120, color="#e74c3c", ls=":", alpha=0.5, label="120 Hz Harmonic")
        if curr_f >= 3500:
            ax2.axvspan(3500, min(curr_f, 4500), color="#f1c40f", alpha=0.15, label="Hiss Zone (>3.5kHz)")
        ax2.set_title("Frequency Domain: Unwanted Hum Spikes & High Hiss Exposed by FFT", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_xlim(0, 4500)
        ax2.set_ylim(0, np.max(s[:f_max_idx]) * 1.15)
        if pts_f > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Input & Spectrum Analysis",
        subtitle="Signal Inspection · Time & Frequency Domain",
        description="The incoming audio signal contains the desired voice harmonics alongside 60Hz/120Hz power-line hum and broadband high-frequency hiss. Taking the Fourier Transform exposes precisely which frequencies carry unwanted noise.",
        formula=r"X(f) = \mathcal{F}\{x[n]\} = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi f n / f_s}",
        code_snippet="X = np.fft.rfft(x)\nfreqs = np.fft.rfftfreq(len(x), d=1/fs)",
        dsp_primitive="Fourier Transform",
        duration=5.0,
        render_fn=render_nr_overview,
    ))

    # 2. Method 1: Moving Average Filter (Convolution)
    def render_nr_mov_avg(ax1, ax2, p, dark):
        N = 11
        h = np.ones(N) / N
        filtered = np.convolve(noisy, h, mode="same")
        curr_idx = max(1, int(len(t) * p))

        # Both noisy input and filtered output draw live together
        ax1.plot(t[:curr_idx], noisy[:curr_idx], color="#94a3b8", lw=0.9, alpha=0.45, label="Noisy Input")
        ax1.plot(t[:curr_idx], filtered[:curr_idx], color="#2ecc71", lw=2.0, label="Filtered Output y[n]")
        # Sliding window indicator at current position
        win_start = max(0, t[curr_idx - 1] - (N / 2) / fs)
        win_end = min(1.5, t[curr_idx - 1] + (N / 2) / fs)
        ax1.axvspan(win_start, win_end, color="#6fb1ea", alpha=0.35, label=f"Sliding Window (N={N})")
        ax1.scatter([t[curr_idx - 1]], [filtered[curr_idx - 1]], color="#2ecc71", s=36, zorder=5)
        ax1.set_title(f"Sliding Window Convolution: Averaging {N} Neighboring Samples Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-1.15, 1.15)
        if curr_idx > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Sinc-shaped frequency response draws live across frequencies
        w, H = signal.freqz(h, 1, worN=512, fs=fs)
        f_lim_idx = 180
        pts_w = max(1, int(f_lim_idx * p))
        ax2.plot(w[:pts_w], np.abs(H[:pts_w]), color="#6fb1ea", lw=2.0, label="|H(f)| Low-pass Response")
        notch_f = fs / (2 * N)
        if pts_w > 1 and w[pts_w - 1] >= notch_f:
            ax2.axvline(notch_f, color="#e67e22", ls="--", label="First Notch f ≈ fs/(2N)")
        ax2.set_title("Moving Average Frequency Response: Sinc-shaped Low-Pass Filter", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Gain")
        ax2.set_xlim(0, 2000)
        ax2.set_ylim(0, 1.1)
        if pts_w > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Method 1: Moving Average Filter",
        subtitle="Time-Domain Discrete Convolution",
        description="A sliding boxcar window of size N=9 averages adjacent samples. Rapid high-frequency fluctuations (noise) cancel out while slowly varying voice fundamentals pass through. This is pure time-domain convolution (no FFT required).",
        formula=r"y[n] = (x * h)[n] = \frac{1}{N} \sum_{k=0}^{N-1} x[n - k], \quad h[k] = \frac{1}{N}",
        code_snippet="def moving_average_filter(x, N=9):\n    h = np.ones(N) / N\n    return np.convolve(x, h, mode='same')",
        dsp_primitive="Convolution",
        duration=5.5,
        render_fn=render_nr_mov_avg,
    ))

    # 3. Method 2: Frequency Domain Denoise (FFT & Notch / Hiss Filtering)
    def render_nr_freq_domain(ax1, ax2, p, dark):
        X = np.fft.rfft(noisy)
        freqs = np.fft.rfftfreq(len(noisy), d=1 / fs)
        X_clean = X.copy()
        for f0 in (60, 120, 180):
            X_clean[np.abs(freqs - f0) < 4] = 0
        X_clean[freqs > 4000] *= 0.15
        clean_time = np.fft.irfft(X_clean, n=len(noisy))

        disp_len = int(np.searchsorted(freqs, 5000))
        pts_f = max(1, int(disp_len * p))
        ax1.plot(freqs[:pts_f], np.abs(X)[:pts_f], color="#94a3b8", lw=0.9, alpha=0.5, label="Raw Spectrum |X(f)|")
        if p >= 0.35:
            prog_clean = min(1.0, (p - 0.35) / 0.65)
            pts_clean = max(1, int(disp_len * prog_clean))
            ax1.plot(freqs[:pts_clean], np.abs(X_clean)[:pts_clean], color="#8fd3c7", lw=1.8, label="Notched & Attenuated |X'(f)|")
            curr_freq = freqs[pts_clean - 1]
            for f0 in (60, 120, 180):
                if curr_freq >= f0:
                    ax1.scatter([f0], [0], color="#e74c3c", marker="x", s=64, lw=2.5, zorder=5)
            if curr_freq >= 4000:
                ax1.axvline(4000, color="#f1c40f", ls="--", lw=1.5, label="4 kHz Cutoff")
        ax1.set_title("Frequency Modification: Zeroing Hum Bins (60, 120, 180 Hz) & Attenuating Hiss (>4kHz)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude")
        ax1.set_xlim(0, 5000)
        ax1.set_ylim(0, np.max(np.abs(X)[:disp_len]) * 1.15)
        if pts_f > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Bottom plot: Waveform draws live
        pts_t = max(1, int(len(t) * p))
        if p < 0.4:
            ax2.plot(t[:pts_t], noisy[:pts_t], color="#e74c3c", lw=1.3, label="Original Noisy Signal x[n]")
            ax2.set_title("Time Domain: Noisy Signal", fontsize=10, fontweight="bold")
        else:
            prog = min(1.0, (p - 0.4) / 0.6)
            pts_c = max(1, int(len(t) * prog))
            ax2.plot(t[:pts_t], noisy[:pts_t], color="#94a3b8", alpha=0.35, lw=0.9, label="Original Noisy")
            ax2.plot(t[:pts_c], clean_time[:pts_c], color="#2ecc71", lw=1.8, label="Reconstructed Clean Signal y[n]")
            ax2.set_title("Time Domain: Live Inverse FFT (IFFT) Reconstruction", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(-1.15, 1.15)
        if pts_t > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Method 2: Frequency Domain Denoise",
        subtitle="FFT · Selective Notch Filters & High-Frequency Attenuation",
        description="The FFT transforms the entire audio track into frequency bins. Narrow notches at 60 Hz, 120 Hz, and 180 Hz zero out power-line hum with surgical precision, while bins above 4000 Hz are attenuated by 0.15× to suppress hiss. The Inverse FFT reconstructs the pristine audio.",
        formula=r"X[f_0] = 0 \quad (f_0 \in \{60, 120, 180\}), \quad X[f > 4000] \leftarrow 0.15 X[f], \quad y[n] = \mathcal{F}^{-1}\{X'[f]\}",
        code_snippet="X = np.fft.rfft(x)\nfor f0 in (60, 120, 180):\n    X[np.abs(freqs - f0) < 3] = 0\nX[freqs > 4000] *= 0.15\ny = np.fft.irfft(X, n=len(x))",
        dsp_primitive="Fourier Transform",
        duration=5.5,
        render_fn=render_nr_freq_domain,
    ))

    # 4. Method 3: Spectral Subtraction (STFT & Power Subtraction)
    def render_nr_spectral_sub(ax1, ax2, p, dark):
        nperseg = 512
        noverlap = 256
        f, tt, Zxx = signal.stft(noisy, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        noise_power = np.mean(mag[:, :10] ** 2, axis=1, keepdims=True)
        clean_power = np.maximum(mag ** 2 - 2.0 * noise_power, 0.02 * noise_power)

        # STFT Spectrogram streams live frame-by-frame
        disp_frames = max(1, int(mag.shape[1] * p))
        t_right = max(0.05, tt[min(disp_frames - 1, len(tt) - 1)])
        ax1.imshow(
            20 * np.log10(mag[:80, :disp_frames] + 1e-4),
            aspect="auto", origin="lower", extent=[0, t_right, f[0], f[80]],
            cmap="viridis" if not dark else "plasma"
        )
        if disp_frames >= 10:
            ax1.axvline(tt[min(9, len(tt) - 1)], color="#e74c3c", lw=2.0, ls="--", label="Noise-Only Reference (Frames 0–9)")
            ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("STFT Spectrogram: Streaming Time-Frequency Frames Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlim(0, tt[-1])

        # Bottom: Power subtraction slice drawing live across frequencies
        frame_idx = min(25, mag.shape[1] - 1)
        sub_p = clean_power[:80, frame_idx]
        raw_p = (mag[:80, frame_idx] ** 2)
        n_est = noise_power[:80, 0]
        pts_f = max(1, int(80 * p))
        ax2.plot(f[:pts_f], raw_p[:pts_f], color="#e74c3c", lw=1.5, label="Observed Power |X|²")
        ax2.plot(f[:pts_f], 2.0 * n_est[:pts_f], color="#f1c40f", ls="--", lw=1.5, label="Over-subtracted Noise α|N|² (α=2.0)")
        ax2.plot(f[:pts_f], sub_p[:pts_f], color="#2ecc71", lw=2.0, label="Clean Power |Ŝ|² (with β Floor)")
        ax2.set_title("Power Subtraction per Frequency Bin: |Ŝ|² = max(|X|² − α|N|², β|N|²)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Spectral Power")
        ax2.set_xlim(0, f[80])
        ax2.set_ylim(0, np.max(raw_p) * 1.15)
        if pts_f > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Method 3: Spectral Subtraction",
        subtitle="STFT-Domain Power Spectrum Subtraction",
        description="The signal is divided into overlapping 512-sample frames via STFT. The first 10 frames are assumed noise-only to calculate the average noise power spectrum. For subsequent frames, noise power is subtracted with an over-subtraction factor α=2.0, and a spectral floor β=0.02 prevents musical noise artifacts.",
        formula=r"|\hat{S}(f,t)|^2 = \max\left( |X(f,t)|^2 - \alpha |\hat{N}(f)|^2, \; \beta |\hat{N}(f)|^2 \right), \quad \alpha=2.0, \; \beta=0.02",
        code_snippet="noise_power = np.mean(mag[:, :10] ** 2, axis=1, keepdims=True)\nclean_power = np.maximum(mag ** 2 - alpha * noise_power, beta * noise_power)\nclean_mag = np.sqrt(clean_power)\nclean_Zxx = clean_mag * np.exp(1j * phase)",
        dsp_primitive="Fourier Transform (STFT)",
        duration=6.0,
        render_fn=render_nr_spectral_sub,
    ))

    # 5. Method 4: Wiener Filter (Optimal Estimation)
    def render_nr_wiener(ax1, ax2, p, dark):
        nperseg = 512
        noverlap = 256
        f, tt, Zxx = signal.stft(noisy, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        noise_power = np.mean(mag[:, :10] ** 2, axis=1, keepdims=True)
        sig_power = mag ** 2
        gain = np.maximum(1.0 - noise_power / (sig_power + 1e-10), 0.0)

        mid_frame = min(22, mag.shape[1] - 1)
        g_slice = gain[:100, mid_frame]
        freq_slice = f[:100]

        # Top: Wiener Gain Curve drawn live across frequencies
        draw_pts = max(1, int(len(freq_slice) * p))
        ax1.plot(freq_slice[:draw_pts], g_slice[:draw_pts], color="#f1c40f", lw=2.2, label="Wiener Gain G(f) ∈ [0, 1]")
        ax1.axhline(0.5, color="#94a3b8", ls=":", alpha=0.5)
        if freq_slice[draw_pts - 1] >= 950:
            ax1.axvspan(150, 950, color="#2ecc71", alpha=0.15, label="High SNR Voice Bins (Gain ≈ 1.0, Keep)")
        if freq_slice[draw_pts - 1] >= 2500:
            ax1.axvspan(2500, freq_slice[draw_pts - 1], color="#e74c3c", alpha=0.15, label="Low SNR Noise Bins (Gain ≈ 0.0, Suppress)")
        ax1.set_title("Adaptive Wiener Gain: G = max(1 − Noise_Power / Signal_Power, 0)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Gain G(f)")
        ax1.set_ylim(-0.05, 1.1)
        ax1.set_xlim(0, freq_slice[-1])
        if draw_pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Bottom: Comparing Clean Estimation with Original live
        clean_time = dsp.wiener_denoise(noisy, fs=fs)
        n_show = max(1, int(len(t) * p))
        ax2.plot(t[:n_show], noisy[:n_show], color="#94a3b8", alpha=0.45, lw=1.0, label="Input Noisy")
        ax2.plot(t[:n_show], clean_time[:n_show], color="#2ecc71", lw=1.8, label="Wiener Denoised y[n]")
        ax2.set_title("Wiener Filter Output: Minimum Mean Squared Error (MMSE) Optimal Estimate", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(-1.15, 1.15)
        if n_show > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Method 4: Wiener Filter",
        subtitle="Optimal Least-Squares Estimation (MMSE)",
        description="Rather than brutally subtracting noise, the Wiener filter computes a smooth, frequency-adaptive multiplier G(f) between 0 and 1. Where the signal is strong relative to noise (high SNR), G ≈ 1 (keeps the voice). Where noise dominates (low SNR), G ≈ 0 (suppresses noise). This mathematically minimizes Mean Squared Error.",
        formula=r"G(f,t) = \max\left(1 - \frac{P_{\text{noise}}(f)}{P_{\text{signal}}(f,t) + \epsilon}, \; 0\right), \quad \hat{S}(f,t) = G(f,t) \cdot X(f,t)",
        code_snippet="gain = np.maximum(1.0 - noise_power / (signal_power + 1e-10), 0.0)\nclean_mag = mag * gain\nclean_Zxx = clean_mag * np.exp(1j * phase)",
        dsp_primitive="Optimal Filtering / MMSE",
        duration=6.0,
        render_fn=render_nr_wiener,
    ))

    return stages


# -----------------------------------------------------------------------------
# Equalizer Stages
# -----------------------------------------------------------------------------
def _build_equalizer_stages() -> List[VisualizerStage]:
    fs_eq = dsp.FS_EQ
    t_eq = np.arange(int(fs_eq * 1.2)) / fs_eq
    eq_in = 0.5 * np.sin(2 * np.pi * 120 * t_eq) + 0.35 * np.sin(2 * np.pi * 1000 * t_eq) + 0.25 * np.sin(2 * np.pi * 6000 * t_eq)
    stages = []

    # 1. 3-Band FIR Equalizer
    def render_eq_3band(ax1, ax2, p, dark):
        lo_filt = signal.firwin(101, 300, fs=fs_eq, pass_zero="lowpass")
        mid_filt = signal.firwin(101, [300, 3000], fs=fs_eq, pass_zero="bandpass")
        hi_filt = signal.firwin(101, 3000, fs=fs_eq, pass_zero="highpass")

        w, h_lo = signal.freqz(lo_filt, worN=512, fs=fs_eq)
        _, h_mid = signal.freqz(mid_filt, worN=512, fs=fs_eq)
        _, h_hi = signal.freqz(hi_filt, worN=512, fs=fs_eq)

        g_lo, g_mid, g_hi = 2.0, 0.8, 1.5
        pts_w = max(2, int(len(w) * p))
        ax1.plot(w[1:pts_w], np.abs(h_lo[1:pts_w]) * g_lo, color="#e74c3c", lw=1.8, label=f"Bass Band (<300Hz, ×{g_lo})")
        ax1.plot(w[1:pts_w], np.abs(h_mid[1:pts_w]) * g_mid, color="#2ecc71", lw=1.8, label=f"Mid Band (300–3kHz, ×{g_mid})")
        ax1.plot(w[1:pts_w], np.abs(h_hi[1:pts_w]) * g_hi, color="#6fb1ea", lw=1.8, label=f"Treble Band (>3kHz, ×{g_hi})")
        ax1.set_xlim(40, 20000)
        ax1.set_xscale("log")
        ax1.set_title("3-Band FIR Filter Bank (Windowed-Sinc firwin, 101 taps) - Live Frequency Scan", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Filter Magnitude")
        ax1.set_xlim(40, 20000)
        ax1.set_ylim(0, 2.3)
        if pts_w > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Output waveform convolution draws live from t=0
        lo_sig, mid_sig, hi_sig = dsp.band_split(eq_in, fs=fs_eq)
        out_sig = dsp.normalize(g_lo * lo_sig + g_mid * mid_sig + g_hi * hi_sig)
        max_idx = int(fs_eq * 0.05)  # 50ms window
        pts = max(1, int(max_idx * p))
        ax2.plot(t_eq[:pts], eq_in[:pts], color="#94a3b8", alpha=0.45, label="Raw Input")
        ax2.plot(t_eq[:pts], out_sig[:pts], color="#c9b6f2", lw=1.8, label="Equalized Sum y[n]")
        ax2.set_title("Time-Domain Output: Weighted Superposition of Processed Bands", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 0.05)
        ax2.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="3-Band FIR Equalizer",
        subtitle="Band-Split Convolution & Superposition",
        description="The audio is decomposed into 3 independent frequency streams (Bass <300Hz, Mid 300–3000Hz, Treble >3000Hz) using windowed-sinc FIR filters convolved with the audio. Each band is scaled by its gain slider and summed into the output.",
        formula=r"y[n] = g_L (x * h_L)[n] + g_M (x * h_M)[n] + g_H (x * h_H)[n]",
        code_snippet="lo = signal.firwin(101, 300, fs=fs, pass_zero='lowpass')\nmid = signal.firwin(101, [300, 3000], fs=fs, pass_zero='bandpass')\nhi = signal.firwin(101, 3000, fs=fs, pass_zero='highpass')\nreturn normalize(gL*lo_sig + gM*mid_sig + gH*hi_sig)",
        dsp_primitive="Convolution / FIR Filtering",
        duration=5.5,
        render_fn=render_eq_3band,
    ))

    # 2. 9-Band IIR Peaking Cascade
    def render_eq_iir(ax1, ax2, p, dark):
        bands = dsp.EQ_BANDS
        gains_db = [6, 4, 1, -2, -3, 0, 3, 5, 4]
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c", "#3498db", "#9b59b6", "#e84393", "#fd79a8"]

        total_h = np.ones(512, dtype=complex)
        worN = np.logspace(np.log10(30), np.log10(20000), 512)

        pts_w = max(1, int(len(worN) * p))
        for i, (f0, g, c) in enumerate(zip(bands, gains_db, colors)):
            b, a = eq_logic._peaking_biquad(fs_eq, f0, g)
            w, h = signal.freqz(b, a, worN=worN, fs=fs_eq)
            total_h *= h
            ax1.plot(w[:pts_w], 20 * np.log10(np.abs(h[:pts_w]) + 1e-9), color=c, lw=1.2, alpha=0.7, label=f"{f0}Hz ({g:+d}dB)" if p > 0.8 else None)

        ax1.set_xscale("log")
        ax1.set_title("9-Band IIR Biquad Peaking Filters (RBJ Audio EQ Cookbook)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Gain (dB)")
        ax1.set_xlim(30, 20000)
        ax1.set_ylim(-8, 10)
        ax1.axhline(0, color="#94a3b8", ls=":", alpha=0.5)

        # Bottom: Combined Cascade Response drawn live
        ax2.plot(worN[:pts_w], 20 * np.log10(np.abs(total_h[:pts_w]) + 1e-9), color="#6fb1ea", lw=2.2, label="Cascaded Overall Response")
        ax2.set_xscale("log")
        ax2.set_title("Cascaded Series Response: H(z) = ∏ H_i(z) Applied via signal.lfilter", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Total Gain (dB)")
        ax2.set_xlim(30, 20000)
        ax2.set_ylim(-8, 12)
        if pts_w > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="9-Band IIR Peaking Cascade",
        subtitle="Series Biquad Architecture (RBJ Audio EQ Cookbook)",
        description="The industry standard for graphic equalizers. Each band uses a 2nd-order recursive IIR biquad peaking filter with Q=1.0. All 9 biquads are connected in series (cascaded), so the output of filter 1 feeds filter 2, and so forth.",
        formula=r"H(z) = \prod_{k=0}^{8} \frac{b_{0,k} + b_{1,k} z^{-1} + b_{2,k} z^{-2}}{1 + a_{1,k} z^{-1} + a_{2,k} z^{-2}}, \quad A = 10^{\text{gain}_{\text{dB}} / 40}",
        code_snippet="for b, a in coeffs:\n    out = signal.lfilter(b, a, out)  # cascade each biquad in series",
        dsp_primitive="IIR Filtering / Biquad",
        duration=6.0,
        render_fn=render_eq_iir,
    ))

    # 3. 9-Band FIR Windowed-Sinc Bands
    def render_eq_fir_bands(ax1, ax2, p, dark):
        bands = dsp.EQ_BANDS
        edges = eq_logic._band_edges(bands)
        gains_db = [5, 4, 2, -1, -2, 1, 3, 4, 3]

        curr_freq_limit = 30 * (20000 / 30) ** p
        visible_bands = [b for b in bands if b <= curr_freq_limit]
        if visible_bands:
            ax1.scatter(visible_bands, [0] * len(visible_bands), color="#6fb1ea", s=40, zorder=5, label="ISO Center Frequencies")
        for e in edges:
            if e <= curr_freq_limit:
                ax1.axvline(e, color="#f1c40f", ls="--", alpha=0.5)
        ax1.set_xscale("log")
        ax1.set_title("Geometric Mean Crossovers: f_edge = √(f_i · f_{i+1}) Log-Spaced Splits", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_xlim(30, 20000)
        ax1.set_yticks([])
        if visible_bands:
            ax1.legend(loc="upper right", fontsize=8)

        # Plot the combined FIR response live
        f_resp, db_resp = dsp.nband_eq_frequency_response(gains_db, algorithm="FIR Windowed-Sinc Bands", fs=fs_eq)
        disp_pts = max(2, int(len(f_resp) * p))
        ax2.plot(f_resp[1:disp_pts], db_resp[1:disp_pts], color="#8fd3c7", lw=2.0, label="FIR Combined Frequency Response")
        ax2.set_xlim(30, 20000)
        ax2.set_xscale("log")
        ax2.set_title("Parallel FIR Filter Bank: Perfect Linear Phase (Zero Phase Distortion)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Gain (dB)")
        ax2.set_xlim(30, 20000)
        ax2.set_ylim(-8, 12)
        if disp_pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="9-Band FIR Windowed-Sinc Bands",
        subtitle="Linear-Phase Parallel Convolution Architecture",
        description="Unlike recursive IIR filters, FIR filters have strictly linear phase — meaning every frequency component experiences the exact same time delay (zero phase distortion). The spectrum is split at geometric-mean crossovers into 9 non-overlapping FIR bandpass filters.",
        formula=r"f_{\text{crossover}} = \sqrt{f_i \cdot f_{i+1}}, \quad y[n] = \sum_{k=0}^{8} 10^{G_k / 20} (x * h_k)[n]",
        code_snippet="edges = _band_edges(bands)\nfor h, g_db in zip(filters, gains_db):\n    gain_linear = 10 ** (g_db / 20.0)\n    out += gain_linear * np.convolve(x, h, mode='same')",
        dsp_primitive="Linear Phase FIR Convolution",
        duration=5.5,
        render_fn=render_eq_fir_bands,
    ))

    # 4. Shelving + Peaking Hybrid
    def render_eq_hybrid(ax1, ax2, p, dark):
        f0_lo, f0_hi = 60, 16000
        b_lo, a_lo = eq_logic._low_shelf_biquad(fs_eq, f0_lo, 6.0)
        b_hi, a_hi = eq_logic._high_shelf_biquad(fs_eq, f0_hi, 5.0)

        worN = np.logspace(np.log10(30), np.log10(20000), 512)
        w, h_lo = signal.freqz(b_lo, a_lo, worN=worN, fs=fs_eq)
        _, h_hi = signal.freqz(b_hi, a_hi, worN=worN, fs=fs_eq)

        pts_w = max(1, int(len(w) * p))
        ax1.plot(w[:pts_w], 20 * np.log10(np.abs(h_lo[:pts_w]) + 1e-9), color="#e74c3c", lw=2.0, label="Low-Shelf (60 Hz, +6dB)")
        ax1.plot(w[:pts_w], 20 * np.log10(np.abs(h_hi[:pts_w]) + 1e-9), color="#3498db", lw=2.0, label="High-Shelf (16 kHz, +5dB)")
        ax1.set_xscale("log")
        ax1.set_title("Edge Shelving Filters: Boosting/Cutting Full Low & High Shelves", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Gain (dB)")
        ax1.set_xlim(30, 20000)
        ax1.set_ylim(-8, 12)
        if pts_w > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Full Hybrid curve live
        gains = [6, 4, 1, 0, -1, 2, 3, 4, 5]
        f_resp, db_resp = dsp.nband_eq_frequency_response(gains, algorithm="Shelving + Peaking Hybrid", fs=fs_eq)
        disp_pts = max(2, int(len(f_resp) * p))
        ax2.plot(f_resp[1:disp_pts], db_resp[1:disp_pts], color="#f1c40f", lw=2.2, label="Hybrid Total EQ Response")
        ax2.set_xlim(30, 20000)
        ax2.set_xscale("log")
        ax2.set_title("Hybrid Architecture: Natural Bass/Treble Shelves + Resonant Peaking Mids", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Gain (dB)")
        ax2.set_xlim(30, 20000)
        ax2.set_ylim(-8, 12)
        if disp_pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Shelving + Peaking Hybrid",
        subtitle="Low-Shelf + Peaking Mids + High-Shelf Biquads",
        description="Combines the best of both worlds: the 60 Hz band uses a low-shelf filter (shaping all deep sub-bass frequencies below 60 Hz), the 16 kHz band uses a high-shelf filter (shaping air and sparkle above 16 kHz), and the 7 middle bands use peaking filters.",
        formula=r"H_{\text{hybrid}}(z) = H_{\text{low-shelf}}(z) \cdot \left( \prod_{k=1}^7 H_{\text{peaking}, k}(z) \right) \cdot H_{\text{high-shelf}}(z)",
        code_snippet="if i == 0:\n    coeffs.append(_low_shelf_biquad(fs, f0, g))\nelif i == n - 1:\n    coeffs.append(_high_shelf_biquad(fs, f0, g))\nelse:\n    coeffs.append(_peaking_biquad(fs, f0, g, Q))",
        dsp_primitive="Biquad Shelving & Peaking",
        duration=5.5,
        render_fn=render_eq_hybrid,
    ))

    return stages


# -----------------------------------------------------------------------------
# Audio Editor Stages
# -----------------------------------------------------------------------------
def _build_editor_stages() -> List[VisualizerStage]:
    fs = dsp.FS
    t = np.arange(int(fs * 1.5)) / fs
    sig = dsp.normalize(0.6 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 700 * t))
    stages = []

    # 1. Trim (Array Slicing)
    def render_ed_trim(ax1, ax2, p, dark):
        start_s, end_s = 0.35, 1.15
        trimmed = dsp.trim(sig, start_s, end_s, fs=fs)
        trimmed_t = np.arange(len(trimmed)) / fs + start_s

        pts_in = max(1, int(len(t) * p))
        ax1.plot(t[:pts_in], sig[:pts_in], color="#94a3b8", lw=1.2, label="Audio Stream")
        curr_t = t[pts_in - 1]
        if curr_t >= start_s:
            ax1.axvspan(0, start_s, color="#e74c3c", alpha=0.2, label="Discarded")
            ax1.axvspan(start_s, min(curr_t, end_s), color="#2ecc71", alpha=0.25, label="Retained Segment")
        if curr_t >= end_s:
            ax1.axvspan(end_s, curr_t, color="#e74c3c", alpha=0.2)
        ax1.set_title("Input Waveform: Trim Interval Slicing Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-1.15, 1.15)
        if pts_in > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Sliced output draws live
        if curr_t >= start_s:
            prog_slice = min(1.0, (curr_t - start_s) / (end_s - start_s))
            pts_out = max(1, int(len(trimmed) * prog_slice))
            ax2.plot(trimmed_t[:pts_out], trimmed[:pts_out], color="#2ecc71", lw=1.8, label="Trimmed Output x[start:end]")
            ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title("Extracted Audio Slice: Precise Sample Index Boundary Mapping", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(start_s, end_s)
        ax2.set_ylim(-1.15, 1.15)

    stages.append(VisualizerStage(
        title="Feature 1: Trim",
        subtitle="Time-to-Sample Index Slicing",
        description="Trim extracts a specific time segment [start_s, end_s]. Seconds are converted directly to discrete sample indices start_idx = int(start_s * fs) and end_idx = int(end_s * fs), returning an independent NumPy array slice.",
        formula=r"\text{start\_idx} = \max(0, \lfloor t_{\text{start}} \cdot f_s \rfloor), \quad y = x[\text{start\_idx} : \text{end\_idx}]",
        code_snippet="def trim(x, start_s, end_s, fs=FS):\n    start_idx = max(0, int(start_s * fs))\n    end_idx = min(len(x), int(end_s * fs))\n    return x[start_idx:end_idx].copy()",
        dsp_primitive="Array Slicing",
        duration=4.5,
        render_fn=render_ed_trim,
    ))

    # 2. Reverse (Time Reversal)
    def render_ed_reverse(ax1, ax2, p, dark):
        reversed_sig = dsp.reverse(sig)
        pts = max(1, int(len(t) * p))

        ax1.plot(t[:pts], sig[:pts], color="#6fb1ea", lw=1.5, label="Forward Signal x[n]")
        ax1.set_title("Forward Signal: Played Normal Direction Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(t[:pts], reversed_sig[:pts], color="#c9b6f2", lw=1.8, label="Reversed Signal y[n] = x[N-1-n]")
        ax2.set_title("Time Reversal: Horizontally Mirrored Phase Spectrum", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Feature 2: Reverse",
        subtitle="Time-Domain Reflection y[n] = x[N − 1 − n]",
        description="Flips the audio sequence end-to-start. The frequency magnitudes remain identical (the same pitches exist), but the phase spectrum is inverted. Executed efficiently in Python via slicing with a step of -1.",
        formula=r"y[n] = x[N - 1 - n] \iff Y(e^{j\omega}) = X^*(e^{j\omega})",
        code_snippet="def reverse(x):\n    return x[::-1].copy()",
        dsp_primitive="Time Reversal",
        duration=4.5,
        render_fn=render_ed_reverse,
    ))

    # 3. Time Scale (Overlap-Add OLA)
    def render_ed_timescale(ax1, ax2, p, dark):
        speed = 1.6
        scaled = dsp.time_scale(sig, speed=speed, fs=fs)
        scaled_t = np.arange(len(scaled)) / fs

        pts_orig = max(1, int(len(t) * p))
        pts_scale = max(1, int(len(scaled_t) * p))

        ax1.plot(t[:pts_orig], sig[:pts_orig], color="#94a3b8", alpha=0.5, label=f"Original (1.50s)")
        ax1.plot(scaled_t[:pts_scale], scaled[:pts_scale], color="#e67e22", lw=1.8, label=f"Scaled ({speed}×, {scaled_t[-1]:.2f}s)")
        ax1.set_title(f"OLA Time-Stretching: Pitch-Preserved Speed Adjustment ({speed}×) Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-1.15, 1.15)
        if pts_orig > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Bottom: Hann window frames sliding in live
        frame_len = int(0.030 * fs)
        win = np.hanning(frame_len)
        hop_in = frame_len // 2
        hop_out = int(hop_in / speed)

        pts_f = max(1, int(frame_len * p))
        f_t = np.arange(frame_len) / fs
        ax2.plot(f_t[:pts_f], win[:pts_f], color="#6fb1ea", lw=1.8, label="30ms Hann Window")
        if pts_f > hop_in:
            ax2.axvline(hop_in / fs, color="#2ecc71", ls="--", label=f"Hop In: {hop_in / fs * 1000:.1f}ms")
        if pts_f > hop_out:
            ax2.axvline(hop_out / fs, color="#e74c3c", ls=":", label=f"Hop Out: {hop_out / fs * 1000:.1f}ms")
        ax2.set_title("Overlap-Add (OLA) Synthesis: Modifying Synthesis Hop Distance", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frame Time (s)")
        ax2.set_ylabel("Window Weight")
        ax2.set_xlim(0, frame_len / fs)
        ax2.set_ylim(0, 1.1)
        if pts_f > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Feature 3: Time Scale (Speed Change)",
        subtitle="Overlap-Add (OLA) Pitch-Preserving Time Stretching",
        description="Changes playback speed without altering vocal pitch. The signal is segmented into 30ms Hann-windowed frames read at fixed hop_in. Frames are then accumulated in the output buffer with modified hop_out = hop_in / speed, followed by normalization by the window sum.",
        formula=r"\text{hop}_{\text{out}} = \lfloor \text{hop}_{\text{in}} / \text{speed} \rfloor, \quad y[n] = \frac{\sum_m x_m[n]}{\sum_m w_m[n]}",
        code_snippet="frame_len = int(0.030 * fs)\nhop_in = frame_len // 2\nhop_out = max(1, int(hop_in / speed))\n# accumulate frames in output and divide by win_sum",
        dsp_primitive="Overlap-Add (OLA)",
        duration=5.5,
        render_fn=render_ed_timescale,
    ))

    # 4. Fade In / Fade Out
    def render_ed_fade(ax1, ax2, p, dark):
        faded = dsp.fade(sig, fade_in_s=0.25, fade_out_s=0.35, fs=fs)
        n_in = int(0.25 * fs)
        n_out = int(0.35 * fs)
        env = np.ones(len(sig))
        env[:n_in] = np.linspace(0, 1, n_in)
        env[-n_out:] = np.linspace(1, 0, n_out)

        pts = max(1, int(len(t) * p))
        ax1.plot(t[:pts], env[:pts], color="#f1c40f", lw=2.2, label="Amplitude Modulation Envelope")
        curr_t = t[pts - 1]
        if curr_t >= 0.25:
            ax1.axvline(0.25, color="#2ecc71", ls="--", label="Fade-In End")
        if curr_t >= (t[-1] - 0.35):
            ax1.axvline(t[-1] - 0.35, color="#e74c3c", ls="--", label="Fade-Out Start")
        ax1.set_title("Envelope Shaping: Linear Ramps (0 → 1 and 1 → 0) Drawn Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Gain")
        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(-0.05, 1.1)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(t[:pts], faded[:pts], color="#2ecc71", lw=1.8, label="Faded Audio Waveform")
        ax2.set_title("Click-Free Waveform: Smooth Boundary Attenuation", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Feature 4: Fade In / Fade Out",
        subtitle="Linear Amplitude Envelope Modulation",
        description="Multiplies the boundaries of the audio array by linear ramps. Fade-in gradually increases gain from 0 to 1 at the start; fade-out decreases gain from 1 to 0 at the end. This prevents abrupt discontinuities that cause audible clicks/pops.",
        formula=r"y[n] = x[n] \cdot \text{envelope}[n], \quad \text{envelope} = [0 \to 1, \; 1, \dots, 1, \; 1 \to 0]",
        code_snippet="if n_in > 0:\n    x[:n_in] *= np.linspace(0, 1, n_in)\nif n_out > 0:\n    x[-n_out:] *= np.linspace(1, 0, n_out)",
        dsp_primitive="Amplitude Modulation",
        duration=5.0,
        render_fn=render_ed_fade,
    ))

    # 5. Echo & Reverb (Convolution with Impulse Response)
    def render_ed_echo(ax1, ax2, p, dark):
        delay_s = 0.15
        decay = 0.5
        n_repeats = 3
        h_len = int(delay_s * fs * n_repeats) + 1
        h = np.zeros(h_len)
        for k in range(n_repeats + 1):
            h[int(k * delay_s * fs)] = decay ** k
        h_t = np.arange(h_len) / fs

        pts_h = max(1, int(h_len * p))
        markerline, stemlines, baseline = ax1.stem(h_t[:pts_h], h[:pts_h])
        stemlines.set_color("#e74c3c")
        markerline.set_color("#e74c3c")
        baseline.set_color("#94a3b8")
        ax1.set_title("Impulse Response h[n]: Multi-Tap Decaying Impulses (LTI System)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Delay (s)")
        ax1.set_ylabel("Gain")
        ax1.set_xlim(0, h_t[-1])
        ax1.set_ylim(-0.05, 1.15)

        echoed = dsp.echo(sig, delay_s=delay_s, decay=decay, n_repeats=n_repeats, fs=fs)
        pts = max(1, int(len(t) * p))
        ax2.plot(t[:pts], sig[:pts], color="#94a3b8", alpha=0.4, label="Dry Signal")
        ax2.plot(t[:pts], echoed[:pts], color="#e74c3c", lw=1.8, label="Echo Output y[n] = (x * h)[n]")
        ax2.set_title("Convolved Output: Room Acoustic Reflection Simulation", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 1.5)
        ax2.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Feature 5: Echo & Reverb",
        subtitle="LTI System Simulation via Discrete Convolution",
        description="Models room acoustics as a Linear Time-Invariant (LTI) system. An impulse response h[n] is created with discrete spikes of decaying amplitude (1.0, 0.5, 0.25...) placed at intervals of delay_s. Convolving x[n] with h[n] creates repeating echoes.",
        formula=r"h[n] = \sum_{k=0}^{M} \alpha^k \delta[n - k D], \quad y[n] = (x * h)[n] = \sum_{k=0}^M \alpha^k x[n - kD]",
        code_snippet="for k in range(n_repeats + 1):\n    h[int(k * delay_s * fs)] = decay ** k\nreturn normalize(np.convolve(x, h, mode='full')[:len(x)])",
        dsp_primitive="Convolution / LTI System",
        duration=5.5,
        render_fn=render_ed_echo,
    ))

    return stages


# -----------------------------------------------------------------------------
# Morse Code Converter Stages
# -----------------------------------------------------------------------------
def _build_morse_stages() -> List[VisualizerStage]:
    fs = dsp.FS
    text = "SOS"
    encoded, unit = dsp.synth_morse(text, fs=fs, tone_freq=700, unit=0.07, noise_amp=0.03)
    stages = []

    # 1. Synthesis: Text -> Audio
    def render_mo_synth(ax1, ax2, p, dark):
        n = min(len(encoded), int(fs * 1.6))
        tt = np.arange(n) / fs
        pts = max(1, int(n * p))

        ax1.plot(tt[:pts], encoded[:pts], color="#6fb1ea", lw=1.2, label="Morse Audio (700 Hz Carrier)")
        ax1.set_title("Synthesized Morse Stream for 'SOS': · · ·   — — —   · · · Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, tt[-1])
        ax1.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Carrier zoom (10ms) drawing live
        carrier_n = int(fs * 0.01)
        carrier_t = np.arange(carrier_n) / fs
        tone = np.sin(2 * np.pi * 700 * carrier_t)
        pts_c = max(1, int(carrier_n * p))
        ax2.plot(carrier_t[:pts_c] * 1000, tone[:pts_c], color="#f1c40f", lw=2.0, label="700 Hz Sine Wave Carrier")
        ax2.set_title("Carrier Detail: 700 Hz Pure Tone with 5ms Anti-Click Fade Ramps", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 10)
        ax2.set_ylim(-1.15, 1.15)
        if pts_c > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Morse Synthesis (Text → Audio)",
        subtitle="On-Off Keying (OOK) Tone Generation",
        description="International Morse timing rules dictate: Dot = 1 unit (0.07s), Dash = 3 units, element space = 1 unit, letter space = 3 units, word space = 7 units. Each pulse is a 700 Hz sine wave windowed with 5ms fade ramps to eliminate switching clicks.",
        formula=r"s(t) = A \sin(2\pi f_0 t) \cdot \text{env}(t), \quad \text{Dot} = 1u, \; \text{Dash} = 3u, \; \text{Gap} = 1u",
        code_snippet="for sym in code:\n    chunks.append(tone(unit if sym=='.' else unit*3))\n    chunks.append(silence(unit))",
        dsp_primitive="On-Off Keying Synthesis",
        duration=5.0,
        render_fn=render_mo_synth,
    ))

    # 2. Auto Tone-Frequency Detection
    def render_mo_detect_freq(ax1, ax2, p, dark):
        f_detected = dsp.detect_tone_freq(encoded, fs=fs)
        X = np.fft.rfft(encoded)
        freqs = np.fft.rfftfreq(len(encoded), d=1 / fs)
        mag = np.abs(X)

        band = (freqs >= 200) & (freqs <= 2000)
        disp_f = freqs[band]
        disp_m = mag[band]

        pts = max(1, int(len(disp_f) * p))
        ax1.plot(disp_f[:pts], disp_m[:pts], color="#8fd3c7", lw=1.8, label="|X(f)| Magnitude Spectrum")
        if disp_f[pts - 1] >= f_detected:
            ax1.scatter([f_detected], [np.max(disp_m)], color="#e74c3c", s=70, zorder=5, label=f"Detected Tone Peak: {f_detected:.1f} Hz")
        ax1.set_title("Auto Frequency Detection: FFT Peak Picking within 200–2000 Hz", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude")
        ax1.set_xlim(200, 2000)
        ax1.set_ylim(0, np.max(disp_m) * 1.15)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # FIR Bandpass filter centered at detected tone
        bp_filt = signal.firwin(101, [f_detected - 80, f_detected + 80], fs=fs, pass_zero="bandpass")
        w, h = signal.freqz(bp_filt, worN=512, fs=fs)
        pts_w = max(1, int(len(w) * p))
        ax2.plot(w[:pts_w], np.abs(h[:pts_w]), color="#6fb1ea", lw=2.0, label=f"FIR Bandpass ({f_detected-80:.0f}–{f_detected+80:.0f} Hz)")
        ax2.set_title("Pre-filtering: Narrow Bandpass Filter Isolates Morse Tone", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Gain")
        ax2.set_xlim(200, 1600)
        ax2.set_ylim(0, 1.1)
        if pts_w > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Tone Frequency Detection & Pre-Filtering",
        subtitle="Spectral Peak Identification & Narrow FIR Bandpass",
        description="The decoder doesn't assume a fixed frequency: it runs an FFT across the audio and picks the dominant peak between 200 Hz and 2000 Hz. An FIR bandpass filter (±80 Hz bandwidth) is then convolved with the audio to isolate the tone and reject noise.",
        formula=r"f_{\text{tone}} = \arg\max_{f \in [200, 2000]} |\mathcal{F}\{x[n]\}|",
        code_snippet="X = np.fft.rfft(x)\nfreqs = np.fft.rfftfreq(len(x), d=1/fs)\nband = (freqs >= 200) & (freqs <= 2000)\ntone_freq = freqs[band][np.argmax(mag[band])]",
        dsp_primitive="Fourier Transform / Peak Detection",
        duration=5.5,
        render_fn=render_mo_detect_freq,
    ))

    # 3. Envelope Detection (3 Algorithms Comparison)
    def render_mo_envelope(ax1, ax2, p, dark):
        filtered, env_hilbert, _ = dsp.compute_envelope(encoded, fs=fs, tone_freq=700, algorithm="Hilbert")
        _, env_rect, _ = dsp.compute_envelope(encoded, fs=fs, tone_freq=700, algorithm="Rectify + Low-pass")
        _, env_rms, _ = dsp.compute_envelope(encoded, fs=fs, tone_freq=700, algorithm="Short-Time Energy (RMS)")

        n = min(len(filtered), int(fs * 1.5))
        tt = np.arange(n) / fs
        pts = max(1, int(n * p))

        ax1.plot(tt[:pts], filtered[:pts], color="#94a3b8", alpha=0.5, label="Filtered Carrier Signal")
        ax1.plot(tt[:pts], env_hilbert[:pts], color="#6fb1ea", lw=1.8, label="1. Hilbert Envelope |hilbert(x)|")
        ax1.set_title("Envelope Extraction: Converting High-Frequency Carrier into Slow Amplitude Envelope", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, tt[-1])
        ax1.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Comparing 3 methods live
        ax2.plot(tt[:pts], env_hilbert[:pts], color="#6fb1ea", lw=1.5, label="Hilbert Envelope (Analytic Signal)")
        ax2.plot(tt[:pts], env_rect[:pts], color="#2ecc71", lw=1.5, label="Rectify + Butterworth Low-pass (30Hz)")
        ax2.plot(tt[:pts], env_rms[:pts], color="#f1c40f", lw=1.5, label="RMS Moving Window Energy")
        ax2.set_title("Three Implemented Envelope Algorithms Comparison", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Envelope Level")
        ax2.set_xlim(0, tt[-1])
        ax2.set_ylim(0, 1.15)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Envelope Detection (3 Algorithms)",
        subtitle="Analytic Signal, AM Demodulation & Moving RMS",
        description="To detect dots and dashes, we extract the envelope (instantaneous loudness curve). 3 algorithms are implemented: 1. Hilbert Transform (analytic signal magnitude), 2. Full-wave Rectification + Butterworth Low-pass (classic AM radio detector), and 3. Moving Window RMS Energy.",
        formula=r"\text{Env}_{\text{Hilbert}} = |x(t) + j \mathcal{H}\{x(t)\}|, \quad \text{Env}_{\text{Rect}} = \text{LPF}\{|x(t)|\}, \quad \text{Env}_{\text{RMS}} = \sqrt{x^2 * \mathbf{1}_W / W}",
        code_snippet="# 1. Hilbert: np.abs(signal.hilbert(filtered))\n# 2. Rectify: b, a = butter(2, 30, fs=fs); signal.filtfilt(b, a, np.abs(filtered))\n# 3. RMS: np.sqrt(np.convolve(filtered**2, np.ones(W)/W, 'same'))",
        dsp_primitive="Analytic Signal / Demodulation",
        duration=6.0,
        render_fn=render_mo_envelope,
    ))

    # 4. Thresholding & Keyed Classification -> Decoded Text
    def render_mo_decode(ax1, ax2, p, dark):
        filtered, env, _ = dsp.compute_envelope(encoded, fs=fs, tone_freq=700, algorithm="Hilbert")
        thresh = 0.25 * np.max(env)
        keyed = (env > thresh).astype(float)
        n = min(len(keyed), int(fs * 1.5))
        tt = np.arange(n) / fs

        pts = max(1, int(n * p))
        ax1.plot(tt[:pts], env[:pts], color="#94a3b8", label="Continuous Envelope")
        ax1.axhline(thresh, color="#e74c3c", ls="--", lw=1.5, label=f"Detection Threshold ({thresh:.2f})")
        ax1.plot(tt[:pts], keyed[:pts] * 0.8, color="#2ecc71", lw=2.0, label="Binary Keyed State (On/Off)")
        ax1.set_title("Thresholding: Converting Envelope to Binary Keyed Sequence", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("State")
        ax1.set_xlim(0, tt[-1])
        ax1.set_ylim(-0.1, 1.2)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Pulse duration classification edges live
        changes = np.diff(keyed[:pts].astype(int))
        if len(changes) > 0:
            markerline, stemlines, baseline = ax2.stem(tt[:len(changes)], changes)
            stemlines.set_color("#6fb1ea")
            markerline.set_color("#6fb1ea")
            baseline.set_color("#94a3b8")
        ax2.set_title("Edge Transitions: +1 (Tone ON), -1 (Tone OFF) via np.diff() → Dot vs Dash Classifier", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Transition")
        ax2.set_xlim(0, tt[-1])
        ax2.set_ylim(-1.5, 1.5)

    stages.append(VisualizerStage(
        title="Keyed Thresholding & Text Decoding",
        subtitle="Edge Transition Detection & Timing Classification",
        description="The continuous envelope is compared against a threshold to create a binary on/off square wave. Discrete derivative (np.diff) finds transitions: +1 starts a tone, -1 stops it. Pulse lengths <2 units become dots (·), ≥2 units become dashes (—). Gaps >2 units separate letters, looking up '...' → S and '---' → O to output 'SOS'.",
        formula=r"\text{keyed}[n] = (\text{env}[n] > \theta), \quad \Delta t < 2u \implies \text{Dot}, \; \Delta t \ge 2u \implies \text{Dash}",
        code_snippet="changes = np.diff(keyed.astype(int))\nstarts = np.where(changes == 1)[0] + 1\nends = np.where(changes == -1)[0] + 1\ntext, unit = keyed_to_text(keyed, fs)",
        dsp_primitive="Discrete Differentiation / FSM",
        duration=5.5,
        render_fn=render_mo_decode,
    ))

    return stages


# -----------------------------------------------------------------------------
# Audio Matcher Stages
# -----------------------------------------------------------------------------
def _build_matcher_stages() -> List[VisualizerStage]:
    fs = dsp.FS
    clap = dsp.make_clap(fs=fs)
    bell = dsp.make_bell(fs=fs)
    tap = dsp.make_tap(fs=fs)
    recording, templates, placements = dsp.build_template_recording(total_dur=3.5, fs=fs)
    stages = []

    # 1. Template Sounds
    def render_ma_templates(ax1, ax2, p, dark):
        t_clap = np.arange(len(clap)) / fs
        t_bell = np.arange(len(bell)) / fs
        t_tap = np.arange(len(tap)) / fs

        pts_bell = max(1, int(len(t_bell) * p))
        pts_clap = max(1, int(len(t_clap) * p))
        pts_tap = max(1, int(len(t_tap) * p))

        ax1.plot(t_bell[:pts_bell] * 1000, bell[:pts_bell], color="#6fb1ea", lw=1.5, label="Bell: 880 Hz Tone × Slower Decay (τ=0.12s)")
        ax1.plot(t_clap[:pts_clap] * 1000, clap[:pts_clap], color="#e74c3c", lw=1.2, label="Clap: Noise Burst × Fast Decay (τ=0.01s)")
        ax1.plot(t_tap[:pts_tap] * 1000, tap[:pts_tap], color="#2ecc71", lw=1.0, label="Tap: 2kHz Spike × Ultra-Short Decay (τ=0.003s)")
        ax1.set_title("Synthetic Sound Templates with Exponential Decay Envelopes e^(-t/τ) Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 400)
        ax1.set_ylim(-1.15, 1.15)
        if pts_bell > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Full multi-template recording drawn live
        t_rec = np.arange(len(recording)) / fs
        pts = max(1, int(len(t_rec) * p))
        ax2.plot(t_rec[:pts], recording[:pts], color="#94a3b8", lw=1.2, label="Full Recording")
        curr_t = t_rec[pts - 1]
        if curr_t >= 0.5:
            ax2.scatter([0.5], [0.9], color="#6fb1ea", s=50, zorder=5)
            ax2.text(0.5, 0.95, "Bell (0.5s)", color="#6fb1ea", fontsize=8, ha="center")
        if curr_t >= 1.8:
            ax2.scatter([1.8], [0.9], color="#e74c3c", s=50, zorder=5)
            ax2.text(1.8, 0.95, "Clap (1.8s)", color="#e74c3c", fontsize=8, ha="center")
        if curr_t >= 3.0:
            ax2.scatter([3.0], [0.9], color="#2ecc71", s=50, zorder=5)
            ax2.text(3.0, 0.95, "Tap (3.0s)", color="#2ecc71", fontsize=8, ha="center")
        ax2.set_title("Test Timeline with Multiple Embedded Sound Events", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.set_xlim(0, 3.5)
        ax2.set_ylim(-1.15, 1.25)
        if pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Template Sound Generation",
        subtitle="Exponential Decay Envelopes & Test Recordings",
        description="Creates synthetic test sounds (Clap, Bell, Tap) to evaluate matching algorithms. Each sound models physical acoustics using exponential decay envelopes e^(-t/τ): Bell rings out at 880 Hz, Clap is a broadband noise burst, and Tap is an ultra-fast transient click.",
        formula=r"\text{Bell}(t) = \sin(2\pi \cdot 880 t) \cdot e^{-t / 0.12}, \quad \text{Clap}(t) = \text{noise}(t) \cdot e^{-t / 0.01}",
        code_snippet="clap = normalize(random_noise * np.exp(-t / 0.01))\nbell = normalize(np.sin(2*pi*880*t) * np.exp(-t / 0.12))",
        dsp_primitive="Synthesis / Envelope Shaping",
        duration=5.0,
        render_fn=render_ma_templates,
    ))

    # 2. Cross-Correlation Matched Filtering
    def render_ma_cross_corr(ax1, ax2, p, dark):
        corr_bell = dsp.match_template(recording, bell)
        t_rec = np.arange(len(recording)) / fs

        pts_rec = max(1, int(len(t_rec) * p))
        ax1.plot(t_rec[:pts_rec], recording[:pts_rec], color="#94a3b8", alpha=0.45, label="Composite Recording")
        # Sliding template position moves live
        curr_t = min(3.4, max(0.01, p * 3.4))
        ax1.axvspan(curr_t, min(3.5, curr_t + len(bell) / fs), color="#6fb1ea", alpha=0.35, label="Sliding Template Bell")
        ax1.set_title("Sliding Template Across Recording Live (Matched Filter)", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, 3.5)
        ax1.set_ylim(-1.15, 1.15)
        if pts_rec > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Cross-correlation response draws live as template slides
        disp_pts = max(1, int(len(t_rec) * p))
        ax2.plot(t_rec[:disp_pts], corr_bell[:disp_pts], color="#e67e22", lw=1.8, label="Cross-Correlation r[k]")
        peak_idx = int(np.argmax(corr_bell))
        if disp_pts > peak_idx:
            ax2.scatter([t_rec[peak_idx]], [corr_bell[peak_idx]], color="#e74c3c", s=80, zorder=5, label=f"Detection Spike at {t_rec[peak_idx]:.2f}s!")
        ax2.set_title("Cross-Correlation Output: Prominent Spike Reveals Exact Match Location", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Lag / Time Offset (s)")
        ax2.set_ylabel("Normalized Correlation")
        ax2.set_xlim(0, 3.5)
        ax2.set_ylim(np.min(corr_bell) * 1.1, np.max(corr_bell) * 1.25)
        if disp_pts > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Cross-Correlation Matched Filter",
        subtitle="Time-Domain Sliding Pattern Detection",
        description="The third fundamental DSP primitive of CSE220. Slides the template along the recording and computes the inner product at every displacement. Where the template matches an embedded sound, the correlation exhibits a massive spike, locating the event with sample-accurate precision.",
        formula=r"r_{xy}[k] = \sum_{n=-\infty}^{\infty} x[n] \cdot y[n + k] = (x * y_{\text{reversed}})[k]",
        code_snippet="def match_template(recording, template):\n    corr = signal.correlate(recording, template, mode='same')\n    return corr / (np.std(corr) + 1e-9)",
        dsp_primitive="Cross-Correlation",
        duration=5.5,
        render_fn=render_ma_cross_corr,
    ))

    # 3. Normalized Clip Similarity
    def render_ma_similarity(ax1, ax2, p, dark):
        clip_a = bell
        clip_b = bell * 0.4 + 0.05 * np.random.standard_normal(len(bell))
        score, offset = dsp.clip_similarity(clip_a, clip_b)

        t_clip = np.arange(len(clip_a)) / fs
        pts = max(1, int(len(t_clip) * p))
        ax1.plot(t_clip[:pts], clip_a[:pts], color="#6fb1ea", lw=1.8, label="Clip A (Original)")
        ax1.plot(t_clip[:pts], clip_b[:pts], color="#2ecc71", lw=1.5, alpha=0.8, label="Clip B (Different Volume + Noise)")
        ax1.set_title("Comparing Two Arbitrary Audio Clips Live at Different Volumes", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.set_xlim(0, t_clip[-1])
        ax1.set_ylim(-1.15, 1.15)
        if pts > 1:
            ax1.legend(loc="upper right", fontsize=8)

        # Bar chart grows live
        scores = [score * p, 0.12 * p, 0.05 * p]
        labels = ["Match (Bell vs Noisy Bell)", "Clap vs Bell", "Tap vs Bell"]
        ax2.barh(labels, scores, color=["#2ecc71", "#e74c3c", "#94a3b8"], height=0.5)
        ax2.set_xlim(0, 1.0)
        ax2.set_title(f"Normalized Similarity Score = {score:.2f} ∈ [0, 1] (L2-Norm Energy Invariant)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Similarity Score")

    stages.append(VisualizerStage(
        title="Normalized Clip Similarity",
        subtitle="L2-Normalized Cross-Correlation Score ∈ [0, 1]",
        description="Normalizes the correlation by both signals' Euclidean L2 norms (total energy). This produces an invariant similarity score between 0.0 (unrelated) and 1.0 (identical), making match scores independent of recording volume or loudness differences.",
        formula=r"\rho_{xy} = \frac{\sum x[n] y[n + k]}{\|x\|_2 \|y\|_2} \in [0, 1], \quad \|x\|_2 = \sqrt{\sum |x[n]|^2}",
        code_snippet="corr = signal.correlate(longer, shorter, mode='valid')\nnorm = np.linalg.norm(longer) * np.linalg.norm(shorter) + 1e-9\nscore = abs((corr / norm)[argmax])",
        dsp_primitive="Normalized Correlation",
        duration=5.0,
        render_fn=render_ma_similarity,
    ))

    # 4. Spectral Fingerprint (STFT Peaks - Shazam Algorithm)
    def render_ma_fingerprint(ax1, ax2, p, dark):
        peaks, f, tt, mag = dsp.spectrogram_peaks(recording[:int(fs * 2.0)], fs=fs)

        disp_frames = max(1, int(mag.shape[1] * p))
        t_right = max(0.05, tt[min(disp_frames - 1, len(tt) - 1)])
        ax1.imshow(
            20 * np.log10(mag[:90, :disp_frames] + 1e-4),
            aspect="auto", origin="lower", extent=[0, t_right, f[0], f[90]],
            cmap="viridis" if not dark else "plasma"
        )
        if len(peaks) > 0 and disp_frames > 5:
            valid_p = [(pt, pf) for pt, pf in peaks if pt <= tt[min(disp_frames - 1, len(tt) - 1)]]
            if valid_p:
                pts_t, pts_f = zip(*valid_p)
                ax1.scatter(pts_t, pts_f, color="#e74c3c", s=28, marker="o", facecolors="none", lw=1.5, label="STFT Spectral Peaks")
                ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("STFT Constellation Map: Streaming Spectral Peaks Live", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlim(0, tt[-1])

        # Bottom: Delta-Time Alignment Histogram builds up live
        offsets = np.random.normal(loc=1.2, scale=0.03, size=45)
        offsets = np.concatenate([offsets, np.random.uniform(0, 3, size=20)])
        n_offsets = max(1, int(len(offsets) * p))
        counts, bins = np.histogram(offsets[:n_offsets], bins=30, range=(0, 3))
        ax2.bar(bins[:-1], counts, width=(bins[1] - bins[0]) * 0.85, color="#f1c40f", alpha=0.85, label="Time Offset Histogram Δt")
        if p >= 0.4:
            ax2.axvline(1.2, color="#2ecc71", lw=2.0, ls="--", label="True Alignment Spike (Δt = 1.2s)")
        ax2.set_title("Fingerprint Matching: Prominent Histogram Peak at Consistent Offset = Confident Match", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time Difference Δt = t_lib − t_query (s)")
        ax2.set_ylabel("Matching Peak Count")
        ax2.set_xlim(0, 3.0)
        ax2.set_ylim(0, 25)
        if n_offsets > 1:
            ax2.legend(loc="upper right", fontsize=8)

    stages.append(VisualizerStage(
        title="Spectral Fingerprint (Shazam Principle)",
        subtitle="STFT Peak Constellation & Time-Delta Histograms",
        description="Extracts the top 3 loudest frequency peaks in every STFT frame to form an audio 'fingerprint'. When matching a query against a library song, pairs with matching frequencies yield time offsets Δt = t_lib − t_query. When many peak pairs align at the SAME time offset, a dramatic histogram spike confirms the match.",
        formula=r"\text{Peaks} = \{(t_i, f_i)\}, \quad \Delta t = t_{\text{lib}} - t_{\text{query}}, \quad \text{Confidence} = \max\{ \text{Hist}(\Delta t) \}",
        code_snippet="f, tt, Zxx = signal.stft(x, nperseg=512)\nfor frame in mag.T:\n    top_bins = np.argsort(frame)[-3:]\n    peaks.append((time, freq))\n# Match via histogram of time-difference alignments",
        dsp_primitive="STFT Peak Picking / Hashing",
        duration=6.0,
        render_fn=render_ma_fingerprint,
    ))

    return stages


def _get_stages_for_module(module: str) -> List[VisualizerStage]:
    if module == "Noise Remover":
        return _build_noise_remover_stages()
    elif module == "Equalizer":
        return _build_equalizer_stages()
    elif module == "Audio Editor":
        return _build_editor_stages()
    elif module == "Morse Code Converter":
        return _build_morse_stages()
    elif module == "Audio Matcher":
        return _build_matcher_stages()
    return []


# =============================================================================
# Main Interactive Visualizer Dialog
# =============================================================================
class AlgorithmVisualizer(QDialog):
    """Interactive, animated algorithm visualizer dialog behaving like a mini
    instructional video/slideshow player with Play/Pause, Timeline/Seek bar,
    speed control, stage navigation, and formula/code display."""

    def __init__(self, module: str, parent=None):
        super().__init__(parent)
        self.module = module
        self.setWindowTitle(f"{module} — Algorithm Visualizer")
        self.resize(960, 680)
        self.setMinimumSize(840, 580)

        self.stages = _get_stages_for_module(module)
        if not self.stages:
            self.stages = [
                VisualizerStage(
                    title="Overview",
                    subtitle="Module Walkthrough",
                    description=f"Overview of the {module} signal processing stages.",
                    formula=r"",
                    code_snippet="",
                    dsp_primitive="DSP Core",
                    duration=5.0,
                    render_fn=lambda ax1, ax2, p, d: None,
                )
            ]

        self.stage_index = 0
        self.stage_elapsed = 0.0  # seconds elapsed in current stage
        self.playback_speed = 1.0
        self.timer_interval_ms = 40  # 25 fps

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(self.timer_interval_ms)
        self.timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._apply_theme()
        theme_module.add_listener(self._apply_theme)
        self._update_stage_ui()

    def _total_duration(self) -> float:
        return sum(s.duration for s in self.stages)

    def _current_global_time(self) -> float:
        prev = sum(self.stages[i].duration for i in range(self.stage_index))
        return prev + self.stage_elapsed

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Top Bar: Module Title, Stage Info, and Stage Selector Dropdown
        top_bar = QHBoxLayout()
        icon_map = {
            "Noise Remover": "🔇",
            "Equalizer": "🎛️",
            "Audio Editor": "✂️",
            "Morse Code Converter": "📡",
            "Audio Matcher": "🔍",
        }
        icon = icon_map.get(self.module, "🌊")
        self.header_label = QLabel(f"<span style='font-size:1.15rem; font-weight:800;'>{icon} {self.module}</span>"
                                   f" <span style='font-size:0.85rem; color:#6fb1ea;'>· Algorithm Visualizer</span>")
        top_bar.addWidget(self.header_label)
        top_bar.addStretch()

        self.jump_label = QLabel("Jump to:")
        top_bar.addWidget(self.jump_label)

        self.stage_combo = QComboBox()
        self.stage_combo.setMinimumWidth(260)
        for i, st in enumerate(self.stages):
            self.stage_combo.addItem(f"{i + 1}. {st.title}")
        self.stage_combo.currentIndexChanged.connect(self._on_stage_combo_changed)
        top_bar.addWidget(self.stage_combo)

        root.addLayout(top_bar)

        # Canvas for animated visualization
        self.figure = Figure(figsize=(9, 4.4), facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(320)
        root.addWidget(self.canvas, 1)

        # Instructional Info Card
        self.info_card = QFrame()
        self.info_card.setObjectName("VisualizerInfoCard")
        info_layout = QVBoxLayout(self.info_card)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(4)

        info_header = QHBoxLayout()
        self.stage_title_label = QLabel()
        self.stage_title_label.setStyleSheet("font-weight: 700; font-size: 0.95rem; color: #2f6690;")
        info_header.addWidget(self.stage_title_label)
        info_header.addStretch()

        self.primitive_badge = QLabel()
        info_header.addWidget(self.primitive_badge)
        info_layout.addLayout(info_header)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("font-size: 0.84rem; line-height: 1.5;")
        info_layout.addWidget(self.description_label)

        self.formula_label = QLabel()
        info_layout.addWidget(self.formula_label)

        root.addWidget(self.info_card)

        # Video Player Controls Bar
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setObjectName("PrimaryButton")
        self.play_btn.setMinimumWidth(85)
        self.play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_btn)

        self.prev_btn = QPushButton("⏮ Prev")
        self.prev_btn.setObjectName("SecondaryButton")
        self.prev_btn.clicked.connect(self._prev_stage)
        controls.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ⏭")
        self.next_btn.setObjectName("SecondaryButton")
        self.next_btn.clicked.connect(self._next_stage)
        controls.addWidget(self.next_btn)

        # Seek Bar
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        controls.addWidget(self.seek_slider, 1)

        # Time Readout
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 0.84rem; min-width: 95px; font-weight: 600;")
        controls.addWidget(self.time_label)

        # Speed Dropdown with explicit label
        self.speed_label = QLabel("Speed:")
        self.speed_label.setStyleSheet("font-weight: 700; font-size: 0.82rem;")
        controls.addWidget(self.speed_label)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setMinimumWidth(80)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_combo)

        root.addLayout(controls)

    def _apply_theme(self):
        dark = theme_module.get_current_mode() == "dark"
        bg_col = "#0f172a" if dark else "#f8fafc"
        text_col = "#f1f5f9" if dark else "#0f172a"
        muted_text = "#cbd5e1" if dark else "#475569"
        card_bg = "rgba(30, 41, 59, 0.85)" if dark else "rgba(255, 255, 255, 0.9)"
        badge_bg = "rgba(111,177,234,0.22)" if dark else "rgba(111,177,234,0.18)"
        badge_text = "#6fb1ea" if dark else "#2f6690"
        formula_col = "#7dd3fc" if dark else "#0369a1"

        # Explicit High-Contrast Styling for ComboBoxes (speed and stage selector)
        combo_bg = "#1e293b" if dark else "#ffffff"
        combo_text = "#f8fafc" if dark else "#0f172a"
        combo_border = "rgba(111,177,234,0.5)" if dark else "rgba(111,177,234,0.6)"
        combo_popup_bg = "#1e293b" if dark else "#ffffff"
        combo_popup_sel = "#2563eb" if dark else "#3b82f6"

        combo_css = f"""
            QComboBox {{
                background-color: {combo_bg};
                color: {combo_text};
                border: 1px solid {combo_border};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 0.84rem;
                font-weight: 700;
                min-height: 22px;
            }}
            QComboBox:hover {{
                border-color: #38bdf8;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {combo_popup_bg};
                color: {combo_text};
                selection-background-color: {combo_popup_sel};
                selection-color: #ffffff;
                border: 1px solid {combo_border};
                padding: 4px;
            }}
        """
        self.speed_combo.setStyleSheet(combo_css)
        self.stage_combo.setStyleSheet(combo_css)

        self.setStyleSheet(
            f"QDialog {{ background: {bg_col}; color: {text_col}; }}"
            f"QLabel {{ color: {text_col}; }}"
        )
        self.speed_label.setStyleSheet(f"color: {text_col}; font-weight: 700; font-size: 0.82rem;")
        self.jump_label.setStyleSheet(f"color: {text_col}; font-weight: 600; font-size: 0.82rem;")
        self.time_label.setStyleSheet(f"color: {'#38bdf8' if dark else '#2563eb'}; font-family: monospace; font-size: 0.84rem; font-weight: 700;")

        self.info_card.setStyleSheet(
            f"QFrame#VisualizerInfoCard {{ background: {card_bg}; border-radius: 10px; border: 1px solid rgba(111,177,234,0.3); padding: 8px; }}"
        )
        self.stage_title_label.setStyleSheet(f"font-weight: 800; font-size: 0.96rem; color: {'#38bdf8' if dark else '#1e40af'};")
        self.description_label.setStyleSheet(f"font-size: 0.84rem; line-height: 1.5; color: {text_col};")
        self.formula_label.setStyleSheet(f"font-family: 'Cambria Math', 'DejaVu Math TeX Gyre', serif; font-size: 0.88rem; color: {formula_col}; padding: 2px 0;")
        self.primitive_badge.setStyleSheet(
            f"background: {badge_bg}; color: {badge_text}; padding: 2px 10px; border-radius: 10px; font-weight: 700; font-size: 0.72rem;"
        )
        self._render_current_frame()

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.timer.start()
            self.play_btn.setText("⏸ Pause")

    def _on_speed_changed(self, index):
        speeds = [0.5, 1.0, 1.5, 2.0]
        self.playback_speed = speeds[index]

    def _prev_stage(self):
        if self.stage_index > 0:
            self._set_stage(self.stage_index - 1)
        else:
            self.stage_elapsed = 0.0
            self._render_current_frame()

    def _next_stage(self):
        if self.stage_index < len(self.stages) - 1:
            self._set_stage(self.stage_index + 1)
        else:
            self.stage_elapsed = self.stages[-1].duration
            self._render_current_frame()

    def _set_stage(self, new_index: int):
        self.stage_index = max(0, min(len(self.stages) - 1, new_index))
        self.stage_elapsed = 0.0
        self.stage_combo.blockSignals(True)
        self.stage_combo.setCurrentIndex(self.stage_index)
        self.stage_combo.blockSignals(False)
        self._update_stage_ui()

    def _on_stage_combo_changed(self, index):
        self._set_stage(index)

    def _on_seek_moved(self, value):
        ratio = value / 1000.0
        target_time = ratio * self._total_duration()

        acc = 0.0
        for i, st in enumerate(self.stages):
            if acc + st.duration >= target_time or i == len(self.stages) - 1:
                self.stage_index = i
                self.stage_elapsed = target_time - acc
                break
            acc += st.duration

        self.stage_combo.blockSignals(True)
        self.stage_combo.setCurrentIndex(self.stage_index)
        self.stage_combo.blockSignals(False)
        self._update_stage_ui()

    def _on_timer_tick(self):
        step_dt = (self.timer_interval_ms / 1000.0) * self.playback_speed
        self.stage_elapsed += step_dt

        curr_stage = self.stages[self.stage_index]
        if self.stage_elapsed >= curr_stage.duration:
            if self.stage_index < len(self.stages) - 1:
                self._set_stage(self.stage_index + 1)
            else:
                self._set_stage(0)

        self._update_seek_bar()
        self._render_current_frame()

    def _update_seek_bar(self):
        tot = self._total_duration()
        curr = self._current_global_time()
        slider_val = int((curr / max(0.1, tot)) * 1000)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(slider_val)
        self.seek_slider.blockSignals(False)

        curr_min = int(curr) // 60
        curr_sec = int(curr) % 60
        tot_min = int(tot) // 60
        tot_sec = int(tot) % 60
        self.time_label.setText(f"{curr_min:02d}:{curr_sec:02d} / {tot_min:02d}:{tot_sec:02d}")

    def _update_stage_ui(self):
        stage = self.stages[self.stage_index]
        self.stage_title_label.setText(f"Stage {self.stage_index + 1} of {len(self.stages)}: {stage.title}")
        self.primitive_badge.setText(stage.dsp_primitive)
        self.description_label.setText(stage.description)
        if stage.formula:
            self.formula_label.setText(f"<b>Key Formula:</b> {stage.formula}")
            self.formula_label.show()
        else:
            self.formula_label.hide()
        self._update_seek_bar()
        self._render_current_frame()

    def _render_current_frame(self):
        dark = theme_module.get_current_mode() == "dark"
        bg_col = "#0f172a" if dark else "#ffffff"
        text_col = "#cbd5e1" if dark else "#334155"
        spine_col = "#334155" if dark else "#cbd5e1"
        grid_col = "#1e293b" if dark else "#e2e8f0"

        stage = self.stages[self.stage_index]
        progress = max(0.0, min(1.0, self.stage_elapsed / max(0.1, stage.duration)))

        self.figure.clear()
        self.figure.set_facecolor(bg_col)

        # Dual subplot grid: Top (ax1) and Bottom (ax2)
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)

        for ax in (ax1, ax2):
            ax.set_facecolor(bg_col)
            ax.tick_params(colors=text_col, labelsize=8)
            ax.xaxis.label.set_color(text_col)
            ax.yaxis.label.set_color(text_col)
            ax.title.set_color(text_col)
            for spine in ax.spines.values():
                spine.set_color(spine_col)
            ax.grid(True, alpha=0.3, color=grid_col)

        # Execute stage rendering callback
        stage.render_fn(ax1, ax2, progress, dark)

        for ax in (ax1, ax2):
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(bg_col)
                leg.get_frame().set_edgecolor(spine_col)
                for t in leg.get_texts():
                    t.set_color(text_col)

        self.figure.tight_layout(pad=1.4)
        self.canvas.draw_idle()

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)
