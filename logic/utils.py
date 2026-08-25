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


def load_audio_file(file_path, target_fs=FS):
    """Load a .wav or .mp3 file and return mono float array normalized to [-1,1].
    Resamples to target_fs if needed. Returns (audio_array, original_fs)."""
    import os
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.mp3':
        # Use scipy to read wav; for mp3 we need pydub or ffmpeg
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(file_path)
            seg = seg.set_channels(1)  # mono
            samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
            samples = samples / (2**15)  # 16-bit normalization
            orig_fs = seg.frame_rate
        except ImportError:
            raise ImportError("pydub is required for MP3 support. Install with: pip install pydub")
    else:
        # WAV file
        orig_fs, data = wavfile.read(file_path)
        if data.dtype == np.int16:
            data = data.astype(np.float64) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float64) / 2147483648.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            data = data.astype(np.float64)
        else:
            data = data.astype(np.float64) / np.max(np.abs(data) + 1e-12)

        # Convert to mono if stereo
        if len(data.shape) > 1:
            samples = np.mean(data, axis=1)
        else:
            samples = data

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
