"""
Equalizer logic for the Audio Signals Toolbox.
3-band FIR equalizer: band splitting, gain-weighted recombination,
and combined frequency response.
"""
import numpy as np
from scipy import signal

from logic.utils import FS, normalize


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
