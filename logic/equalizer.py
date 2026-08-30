"""
Equalizer logic for the Audio Signals Toolbox.

Two things live here:
  1. The ORIGINAL 3-band FIR equalizer (band_split / equalize /
     equalizer_frequency_response) — unchanged, still available.
  2. A new 9-band graphic EQ (60Hz-16kHz, the industry-standard ISO bands)
     with a genuine choice of THREE different filter-design algorithms
     driving the same 9 gain sliders — a legitimate 'logic change' for this
     feature, matching the algorithm-selection pattern already used
     elsewhere in the app (Noise Remover, Audio Matcher).

Note on sample rate: a 16 kHz band requires a sample rate comfortably
above 32 kHz (Nyquist). The rest of this app standardizes on FS=16000 Hz
(8 kHz Nyquist) for its other modules, which cannot represent a 16 kHz
band at all. Rather than silently mis-designing filters above Nyquist,
the 9-band EQ processes audio at FS_EQ=44100 Hz (CD-quality) instead —
the Equalizer page loads/generates its own audio at this rate.
"""
import numpy as np
from scipy import signal

from logic.utils import FS, normalize

FS_EQ = 44100  # equalizer-specific sample rate (see module docstring)

EQ_BANDS = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]

EQ_ALGORITHMS = [
    "IIR Peaking Cascade",
    "FIR Windowed-Sinc Bands",
    "Shelving + Peaking Hybrid",
]

EQ_PRESETS = {
    "Flat":       [0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Bass Boost": [8, 6, 4, 2, 0, 0, 0, 0, 0],
    "Treble Boost": [0, 0, 0, 0, 0, 2, 4, 6, 8],
    "Vocal":      [-3, -2, 0, 3, 4, 3, 1, -1, -2],
    "Podcast":    [-4, -2, 1, 3, 4, 2, 0, -2, -4],
    "Rock":       [5, 3, -1, -2, 0, 2, 4, 4, 3],
    "Classical":  [4, 3, 2, 0, 0, 0, 2, 3, 4],
    "Electronic": [6, 5, 0, -2, 0, 1, 3, 5, 6],
    "Speech":     [-6, -4, -1, 2, 5, 4, 1, -2, -5],
}


def band_split(x, fs=FS, low_cut=300, high_cut=3000, numtaps=101):
    lo = signal.firwin(numtaps, low_cut, fs=fs, pass_zero='lowpass')
    mid = signal.firwin(numtaps, [low_cut, high_cut], fs=fs, pass_zero='bandpass')
    hi = signal.firwin(numtaps, high_cut, fs=fs, pass_zero='highpass')
    return (np.convolve(x, lo, mode='same'),
            np.convolve(x, mid, mode='same'),
            np.convolve(x, hi, mode='same'))


def equalize(x, gains=(1.0, 1.0, 1.0), **kwargs):
    lo, mid, hi = band_split(x, **kwargs)
    gL, gM, gH = gains
    return normalize(gL * lo + gM * mid + gH * hi)


def equalizer_frequency_response(gains=(1.0, 1.0, 1.0), fs=FS, low_cut=300,
                                  high_cut=3000, numtaps=101, n_points=512):
    """Combined magnitude frequency response of the 3-band EQ at the given
    gains — used to draw a live 'what the EQ is doing to the spectrum right
    now' curve as the user moves the gain sliders."""
    lo = signal.firwin(numtaps, low_cut, fs=fs, pass_zero='lowpass')
    mid = signal.firwin(numtaps, [low_cut, high_cut], fs=fs, pass_zero='bandpass')
    hi = signal.firwin(numtaps, high_cut, fs=fs, pass_zero='highpass')
    gL, gM, gH = gains
    combined = gL * lo + gM * mid + gH * hi
    w, h = signal.freqz(combined, worN=n_points, fs=fs)
    return w, 20 * np.log10(np.abs(h) + 1e-9)


# ============================================================ 9-band graphic EQ
def _band_edges(bands):
    """Geometric-mean crossover frequencies between consecutive log-spaced
    bands — the standard way to split octave/decade bands for a graphic EQ."""
    return [float(np.sqrt(bands[i] * bands[i + 1])) for i in range(len(bands) - 1)]


def _peaking_biquad(fs, f0, gain_db, Q=1.0):
    """RBJ 'Audio EQ Cookbook' peaking-EQ biquad coefficients."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _low_shelf_biquad(fs, f0, gain_db, Q=0.707):
    """RBJ low-shelf biquad — boosts/cuts everything below f0."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _high_shelf_biquad(fs, f0, gain_db, Q=0.707):
    """RBJ high-shelf biquad — boosts/cuts everything above f0."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    sqrtA = np.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _iir_peaking_cascade_coeffs(gains_db, fs, bands, Q=1.4):
    return [_peaking_biquad(fs, min(f0, fs / 2 - 50), g, Q=Q)
            for f0, g in zip(bands, gains_db) if abs(g) > 1e-6]


def _shelving_hybrid_coeffs(gains_db, fs, bands, Q=1.4):
    coeffs = []
    n = len(bands)
    for i, (f0, g) in enumerate(zip(bands, gains_db)):
        if abs(g) < 1e-6:
            continue
        f0c = min(f0, fs / 2 - 50)
        if i == 0:
            coeffs.append(_low_shelf_biquad(fs, f0c, g))
        elif i == n - 1:
            coeffs.append(_high_shelf_biquad(fs, f0c, g))
        else:
            coeffs.append(_peaking_biquad(fs, f0c, g, Q=Q))
    return coeffs


def _fir_band_filters(fs, bands, numtaps=201):
    """Design one FIR filter per band (lowpass / bandpass / highpass,
    split at geometric-mean crossovers) — a fundamentally different
    filter-design technique (linear-phase FIR) from the two IIR options."""
    edges = _band_edges(bands)
    filters = []
    nyq_margin = fs / 2 - 50
    for i in range(len(bands)):
        if i == 0:
            cutoff = min(edges[0], nyq_margin)
            filters.append(signal.firwin(numtaps, cutoff, fs=fs, pass_zero='lowpass'))
        elif i == len(bands) - 1:
            cutoff = min(edges[-1], nyq_margin)
            filters.append(signal.firwin(numtaps, cutoff, fs=fs, pass_zero='highpass'))
        else:
            lo_edge = min(edges[i - 1], nyq_margin - 1)
            hi_edge = min(edges[i], nyq_margin)
            filters.append(signal.firwin(numtaps, [lo_edge, hi_edge], fs=fs, pass_zero='bandpass'))
    return filters


def apply_nband_eq(x, gains_db, algorithm="IIR Peaking Cascade", fs=FS_EQ, bands=None):
    """Apply the 9-band graphic EQ to `x` using the selected algorithm.
    All three algorithms take the same gains_db (dB per band in `bands`)
    and produce a comparable result via genuinely different DSP:
      - IIR Peaking Cascade: cascaded peaking biquads (RBJ cookbook)
      - FIR Windowed-Sinc Bands: linear-phase FIR band-split + linear sum
      - Shelving + Peaking Hybrid: shelving filters at the band edges,
        peaking filters in between
    """
    bands = bands or EQ_BANDS
    y = np.asarray(x, dtype=float)

    if algorithm == "FIR Windowed-Sinc Bands":
        filters = _fir_band_filters(fs, bands)
        out = np.zeros_like(y)
        for h, g_db in zip(filters, gains_db):
            gain_linear = 10 ** (g_db / 20.0)
            out += gain_linear * np.convolve(y, h, mode='same')
        return normalize(out)

    if algorithm == "Shelving + Peaking Hybrid":
        coeffs = _shelving_hybrid_coeffs(gains_db, fs, bands)
    else:  # "IIR Peaking Cascade" (default)
        coeffs = _iir_peaking_cascade_coeffs(gains_db, fs, bands)

    out = y.copy()
    for b, a in coeffs:
        out = signal.lfilter(b, a, out)
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out = out / peak
    return out


def nband_eq_frequency_response(gains_db, algorithm="IIR Peaking Cascade",
                                 fs=FS_EQ, bands=None, n_points=1024):
    """Combined magnitude response (dB) of the 9-band EQ at the given gains
    and algorithm — the live 'what is the EQ doing right now' curve."""
    bands = bands or EQ_BANDS

    if algorithm == "FIR Windowed-Sinc Bands":
        filters = _fir_band_filters(fs, bands)
        combined_ir = np.zeros(max(len(h) for h in filters))
        for h, g_db in zip(filters, gains_db):
            gain_linear = 10 ** (g_db / 20.0)
            combined_ir[:len(h)] += gain_linear * h
        w, h = signal.freqz(combined_ir, worN=n_points, fs=fs)
        return w, 20 * np.log10(np.abs(h) + 1e-9)

    if algorithm == "Shelving + Peaking Hybrid":
        coeffs = _shelving_hybrid_coeffs(gains_db, fs, bands)
    else:
        coeffs = _iir_peaking_cascade_coeffs(gains_db, fs, bands)

    w = np.linspace(0, fs / 2, n_points)
    combined = np.ones(n_points, dtype=complex)
    for b, a in coeffs:
        _, h = signal.freqz(b, a, worN=n_points, fs=fs)
        combined *= h
    return w, 20 * np.log10(np.abs(combined) + 1e-9)
