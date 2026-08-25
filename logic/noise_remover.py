"""
Noise Remover logic for the Audio Signals Toolbox.
Test signal generation, denoising algorithms (moving average, frequency-domain,
spectral subtraction, Wiener), and noise addition (white, pink, hum, traffic, crowd).
"""
import numpy as np
from scipy import signal

from logic.utils import FS, _rng, t_axis, normalize


# ---------------------------------------------------------- test signal gen
def make_voice_like(duration=2.0, f0=180, fs=FS):
    ts = t_axis(duration, fs)
    x = (1.0 * np.sin(2 * np.pi * f0 * ts)
         + 0.6 * np.sin(2 * np.pi * 2 * f0 * ts)
         + 0.35 * np.sin(2 * np.pi * 3 * f0 * ts)
         + 0.15 * np.sin(2 * np.pi * 5 * f0 * ts))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.8 * ts - np.pi / 2))
    return normalize(x * (0.5 + 0.5 * envelope))


def make_noisy(clean, hum_amp=0.25, hiss_amp=0.05, fs=FS):
    ts = t_axis(len(clean) / fs, fs)
    hum = hum_amp * np.sin(2 * np.pi * 60 * ts) + hum_amp * 0.4 * np.sin(2 * np.pi * 120 * ts)
    hiss = hiss_amp * _rng.standard_normal(len(clean))
    return normalize(clean + hum + hiss)


# --------------------------------------------------------------- denoising
def moving_average_filter(x, N=9):
    h = np.ones(N) / N
    return np.convolve(x, h, mode='same')


def freq_domain_denoise(x, fs=FS, notch_freqs=(60, 120, 180), notch_width=3, hiss_cutoff=4000, hiss_atten=0.15):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    for f0 in notch_freqs:
        X[np.abs(freqs - f0) < notch_width] = 0
    X[freqs > hiss_cutoff] *= hiss_atten
    return np.fft.irfft(X, n=len(x))


def spectral_subtraction_denoise(x, fs=FS, noise_frames=10, alpha=2.0, beta=0.02):
    """Spectral subtraction noise reduction.
    Estimates noise from first `noise_frames` STFT frames,
    then subtracts scaled noise spectrum from each frame.
    alpha: over-subtraction factor
    beta: spectral floor (prevents musical noise)
    """
    nperseg = 512
    noverlap = nperseg // 2
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise power from initial frames
    noise_power = np.mean(mag[:, :noise_frames] ** 2, axis=1, keepdims=True)

    # Spectral subtraction
    clean_power = mag ** 2 - alpha * noise_power
    clean_power = np.maximum(clean_power, beta * noise_power)  # spectral floor
    clean_mag = np.sqrt(clean_power)

    # Reconstruct
    clean_Zxx = clean_mag * np.exp(1j * phase)
    _, clean_signal = signal.istft(clean_Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Match original length
    if len(clean_signal) > len(x):
        clean_signal = clean_signal[:len(x)]
    elif len(clean_signal) < len(x):
        clean_signal = np.pad(clean_signal, (0, len(x) - len(clean_signal)))

    return clean_signal


def wiener_denoise(x, fs=FS, noise_frames=10):
    """Simple Wiener filter-based denoising.
    Uses initial frames to estimate noise PSD."""
    nperseg = 512
    noverlap = nperseg // 2
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Noise power estimate from initial frames
    noise_power = np.mean(mag[:, :noise_frames] ** 2, axis=1, keepdims=True)
    signal_power = mag ** 2

    # Wiener gain
    gain = np.maximum(1.0 - noise_power / (signal_power + 1e-10), 0.0)

    clean_mag = mag * gain
    clean_Zxx = clean_mag * np.exp(1j * phase)
    _, clean_signal = signal.istft(clean_Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if len(clean_signal) > len(x):
        clean_signal = clean_signal[:len(x)]
    elif len(clean_signal) < len(x):
        clean_signal = np.pad(clean_signal, (0, len(x) - len(clean_signal)))

    return clean_signal


# --------------------------------------------------------------- noise addition
def add_white_noise(x, amplitude=0.1):
    return normalize(x + amplitude * _rng.standard_normal(len(x)))


def add_pink_noise(x, amplitude=0.1):
    white = _rng.standard_normal(len(x))
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(x))
    freqs[0] = 1e-9
    X = X / np.sqrt(freqs)
    pink = np.fft.irfft(X, n=len(x))
    pink = amplitude * pink / np.max(np.abs(pink) + 1e-12)
    return normalize(x + pink)


def add_hum_noise(x, amplitude=0.15, base_freq=50, fs=FS):
    ts = t_axis(len(x) / fs, fs)
    hum = (amplitude * np.sin(2 * np.pi * base_freq * ts) +
           amplitude * 0.4 * np.sin(2 * np.pi * 2 * base_freq * ts) +
           amplitude * 0.2 * np.sin(2 * np.pi * 3 * base_freq * ts))
    return normalize(x + hum)


def add_traffic_noise(x, amplitude=0.15, fs=FS):
    white = _rng.standard_normal(len(x))
    numtaps = 101
    h = signal.firwin(numtaps, [20, 300], fs=fs, pass_zero='bandpass')
    traffic = np.convolve(white, h, mode='same')
    traffic = amplitude * traffic / np.max(np.abs(traffic) + 1e-12)
    return normalize(x + traffic)


def add_crowd_noise(x, amplitude=0.1, fs=FS):
    white = _rng.standard_normal(len(x))
    numtaps = 101
    h = signal.firwin(numtaps, [300, 3000], fs=fs, pass_zero='bandpass')
    crowd = np.convolve(white, h, mode='same')

    ts = t_axis(len(x) / fs, fs)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 2 * ts)) + 0.5 * _rng.standard_normal(len(x))
    crowd = crowd * np.abs(modulator)
    crowd = amplitude * crowd / np.max(np.abs(crowd) + 1e-12)
    return normalize(x + crowd)


def add_combined_noise(x, noise_type='white', amplitude=0.1, fs=FS):
    if noise_type == 'white':
        return add_white_noise(x, amplitude)
    elif noise_type == 'pink':
        return add_pink_noise(x, amplitude)
    elif noise_type == 'hum':
        return add_hum_noise(x, amplitude, base_freq=50, fs=fs)
    elif noise_type == 'traffic':
        return add_traffic_noise(x, amplitude, fs=fs)
    elif noise_type == 'crowd':
        return add_crowd_noise(x, amplitude, fs=fs)
    else:
        return add_white_noise(x, amplitude)
