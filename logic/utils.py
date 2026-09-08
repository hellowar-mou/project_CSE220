"""
Shared utilities for the Audio Signals Toolbox.
Constants, normalization, WAV I/O, file loading, spectrogram, and SNR.
"""
import numpy as np
import io

from scipy import signal
from scipy.io import wavfile

FS = 16000  # sample rate (Hz)

_rng = np.random.default_rng(0)


def t_axis(duration, fs=FS):
    return np.arange(int(duration * fs)) / fs


def normalize(x):
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak


def to_wav_bytes(x, fs=FS):
    """Convert a float array in [-1, 1] to 16-bit PCM WAV bytes for st.audio()."""
    buf = io.BytesIO()
    x16 = np.clip(x, -1, 1)
    x16 = (x16 * 32767).astype(np.int16)
    wavfile.write(buf, fs, x16)
    buf.seek(0)
    return buf.read()


def write_wav_file(path, x, fs=FS):
    """Write a float array in [-1, 1] to a 16-bit PCM .wav file on disk —
    used by 'Download' actions (e.g. Audio Matcher's Download Matched Audio)."""
    with open(path, "wb") as f:
        f.write(to_wav_bytes(x, fs))


def compute_rms(x):
    """RMS level of a signal (linear 0-1 scale)."""
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def compute_peak(x):
    """Peak absolute amplitude of a signal (linear 0-1 scale)."""
    if len(x) == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def load_audio_file(file_path, target_fs=FS):
    """Load WAV/MP3/FLAC/OGG as mono normalized audio and resample it."""
    try:
        import soundfile as sf
        data, orig_fs = sf.read(file_path, dtype="float64", always_2d=True)
        samples = np.mean(data, axis=1)
    except Exception as soundfile_error:
        # Keep MP3 support available when the installed libsndfile lacks an
        # MP3 decoder; pydub delegates that format to the system decoder.
        import os
        if os.path.splitext(file_path)[1].lower() != ".mp3":
            raise soundfile_error
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(file_path)
        except ImportError as exc:
            raise ImportError(
                "Audio decoding requires soundfile or pydub with an MP3 decoder."
            ) from exc
        seg = seg.set_channels(1)
        samples = np.asarray(seg.get_array_of_samples(), dtype=np.float64)
        samples /= float(1 << (8 * seg.sample_width - 1))
        orig_fs = seg.frame_rate

    # Resample to target sample rate if different
    if orig_fs != target_fs:
        num_samples = int(len(samples) * target_fs / orig_fs)
        samples = signal.resample(samples, num_samples)

    return normalize(samples), orig_fs


def load_sample_song(file_path, target_fs=FS, max_duration=30.0):
    """Load an audio file (WAV/MP3/FLAC/OGG), resample to target_fs, trim to
    max_duration seconds.  Returns (normalized_audio, original_sample_rate).
    Uses soundfile as the primary reader (handles MP3/WAV/FLAC natively)."""
    import soundfile as sf

    data, orig_fs = sf.read(file_path, dtype='float64', always_2d=True)

    # Convert to mono by averaging channels
    samples = np.mean(data, axis=1)

    # Resample to target sample rate if different
    if orig_fs != target_fs:
        num_samples = int(len(samples) * target_fs / orig_fs)
        samples = signal.resample(samples, num_samples)

    # Trim to max_duration
    max_samples = int(max_duration * target_fs)
    if len(samples) > max_samples:
        samples = samples[:max_samples]

    return normalize(samples), orig_fs


def spectrum_db(x, fs=FS):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return freqs, 20 * np.log10(np.abs(X) + 1e-9)


def compute_spectrogram(x, fs=FS, nperseg=512, noverlap=384):
    """Compute spectrogram for visualization."""
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 20 * np.log10(np.abs(Zxx) + 1e-9)


def compute_snr(clean, noisy):
    """Compute Signal-to-Noise Ratio in dB."""
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def compute_dominant_frequency(x, fs=FS):
    """Frequency (Hz) of the strongest spectral component — used in
    technical info panels ('current dominant frequency')."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    mag = np.abs(X)
    if len(mag) <= 1:
        return 0.0
    mag[0] = 0  # ignore DC
    return float(freqs[np.argmax(mag)])
