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
    """Change playback speed WITHOUT changing pitch using WSOLA time-stretching.
    speed > 1.0 = faster playback, speed < 1.0 = slower playback.
    Uses Waveform Similarity Overlap-Add (WSOLA) — cross-correlation finds
    the best overlap position so the voice/pitch stays completely natural."""
    speed = max(0.25, min(speed, 4.0))
    if abs(speed - 1.0) < 0.01:
        return x.copy()

    frame_len = int(0.050 * fs)            # 50 ms analysis frames
    hop_syn = frame_len // 2               # synthesis hop (50 % overlap)
    hop_ana = max(1, int(hop_syn * speed)) # analysis hop  (scaled by speed)
    tolerance = frame_len // 4             # WSOLA search radius (samples)

    window = np.hanning(frame_len)

    # estimate output length
    n_frames = 1 + (len(x) - frame_len) // hop_ana
    out_len = (n_frames + 1) * hop_syn + frame_len
    output = np.zeros(out_len)
    win_sum = np.zeros(out_len)

    # first frame — copy directly
    if frame_len > len(x):
        return x.copy()
    output[:frame_len] += x[:frame_len] * window
    win_sum[:frame_len] += window

    ana_pos = 0  # tracks the "natural" analysis position in input

    for i in range(1, n_frames):
        # nominal next analysis position
        ana_pos_nominal = int(i * hop_ana)

        # search window: look around the nominal position for the best match
        search_lo = max(0, ana_pos_nominal - tolerance)
        search_hi = min(len(x) - frame_len, ana_pos_nominal + tolerance)
        if search_hi < search_lo:
            break

        # the previously-synthesised tail that we must overlap with
        so = i * hop_syn
        if so + frame_len > out_len:
            break

        # overlap region from the already-written output
        prev_tail = output[so : so + frame_len]

        # find the shift that maximises cross-correlation with prev_tail
        best_pos = ana_pos_nominal
        best_corr = -np.inf
        for candidate in range(search_lo, search_hi + 1):
            seg = x[candidate : candidate + frame_len]
            if len(seg) < frame_len:
                break
            # normalised dot-product (fast enough for the short search range)
            corr = np.dot(prev_tail, seg)
            if corr > best_corr:
                best_corr = corr
                best_pos = candidate

        frame = x[best_pos : best_pos + frame_len]
        if len(frame) < frame_len:
            break

        output[so : so + frame_len] += frame * window
        win_sum[so : so + frame_len] += window

    # normalise by the accumulated window sum (avoids amplitude modulation)
    win_sum[win_sum < 1e-8] = 1e-8
    # trim to actual content
    last_nonzero = np.max(np.nonzero(win_sum > 1e-7)[0]) + 1 if np.any(win_sum > 1e-7) else len(output)
    output = output[:last_nonzero]
    win_sum = win_sum[:last_nonzero]
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
