"""
Audio Editor logic for the Audio Signals Toolbox.
Basic audio operations: trim, reverse, fade in/out, and echo (via convolution).
"""
import numpy as np

from logic.utils import FS, normalize


def trim(x, start_s, end_s, fs=FS):
    return x[int(start_s * fs):int(end_s * fs)]


def reverse(x):
    return x[::-1]


def fade(x, fade_in_s=0.05, fade_out_s=0.05, fs=FS):
    x = x.copy().astype(float)
    n_in = min(int(fade_in_s * fs), len(x) // 2)
    n_out = min(int(fade_out_s * fs), len(x) // 2)
    if n_in > 0:
        x[:n_in] *= np.linspace(0, 1, n_in)
    if n_out > 0:
        x[-n_out:] *= np.linspace(1, 0, n_out)
    return x


def echo(x, delay_s=0.15, decay=0.5, n_repeats=4, fs=FS):
    h = np.zeros(int(delay_s * fs * n_repeats) + 1)
    for k in range(n_repeats + 1):
        idx = int(k * delay_s * fs)
        if idx < len(h):
            h[idx] = decay ** k
    return normalize(np.convolve(x, h, mode='full')[:len(x)])
