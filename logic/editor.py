"""
Audio Editor logic for the Audio Signals Toolbox.
Operations: trim, join, reverse, time_scale, fade in/out,
smooth (convolution-based), and echo (convolution-based).
"""
import numpy as np
from scipy import signal as sp_signal

from logic.utils import FS, normalize


def trim(x, start_s, end_s, fs=FS):
    """Trim audio to the segment [start_s, end_s] in seconds."""
    start_idx = max(0, int(start_s * fs))
    end_idx = min(len(x), int(end_s * fs))
    if end_idx <= start_idx:
        return x[:1].copy()
    return x[start_idx:end_idx].copy()


def join(*arrays):
    """Concatenate multiple audio arrays into one continuous signal."""
    valid = [a for a in arrays if a is not None and len(a) > 0]
    if not valid:
        return np.zeros(1)
    return normalize(np.concatenate(valid))


def reverse(x):
    """Reverse the audio signal (play backwards)."""
    return x[::-1].copy()


def time_scale(x, speed=1.0, fs=FS):
    """Change playback speed WITHOUT changing pitch using OLA time-stretching.
    speed > 1.0 = faster playback, speed < 1.0 = slower playback.
    Uses Overlap-Add with Hann-windowed frames so the voice stays natural."""
    speed = max(0.25, min(speed, 4.0))
    if abs(speed - 1.0) < 0.01:
        return x.copy()

    frame_len = int(0.030 * fs)          # 30 ms frames
    hop_in = frame_len // 2              # input hop  (50 % overlap)
    hop_out = max(1, int(hop_in / speed))  # output hop

    window = np.hanning(frame_len)

    n_frames = (len(x) - frame_len) // hop_in + 1
    out_len = (n_frames - 1) * hop_out + frame_len
    output = np.zeros(out_len)
    win_sum = np.zeros(out_len)

    for i in range(n_frames):
        si = i * hop_in
        if si + frame_len > len(x):
            break
        frame = x[si : si + frame_len] * window

        so = i * hop_out
        if so + frame_len > out_len:
            break
        output[so : so + frame_len] += frame
        win_sum[so : so + frame_len] += window

    # avoid divide-by-zero in gaps
    win_sum[win_sum < 1e-8] = 1e-8
    output /= win_sum

    return normalize(output)


def fade(x, fade_in_s=0.0, fade_out_s=0.0, fs=FS):
    """Apply fade-in and/or fade-out amplitude envelope."""
    x = x.copy().astype(float)
    n_in = min(int(fade_in_s * fs), len(x) // 2)
    n_out = min(int(fade_out_s * fs), len(x) // 2)
    if n_in > 0:
        x[:n_in] *= np.linspace(0, 1, n_in)
    if n_out > 0:
        x[-n_out:] *= np.linspace(1, 0, n_out)
    return x


def smooth(x, kernel_size=50):
    """Convolution-based smoothing using a uniform (moving-average) kernel.
    Larger kernel_size = stronger smoothing effect."""
    kernel_size = max(1, int(kernel_size))
    kernel = np.ones(kernel_size) / kernel_size
    return normalize(np.convolve(x, kernel, mode='same'))


def echo(x, delay_s=0.15, decay=0.5, n_repeats=4, fs=FS):
    """Convolution-based echo/reverb effect.
    Creates an impulse response h[n] with decaying copies at regular intervals,
    then convolves the signal with it."""
    h_len = int(delay_s * fs * n_repeats) + 1
    h = np.zeros(h_len)
    for k in range(n_repeats + 1):
        idx = int(k * delay_s * fs)
        if idx < len(h):
            h[idx] = decay ** k
    return normalize(np.convolve(x, h, mode='full')[:len(x)])
