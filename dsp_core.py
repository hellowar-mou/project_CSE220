"""
Shared DSP core for the Audio Signals Toolbox.
Every app in this project (Noise Remover, Equalizer, Editor, Morse Code
Converter, Audio Matcher) is built from three primitives:
  - Convolution (LTI systems)
  - The Fourier Transform (via FFT / STFT)
  - Correlation (matched filtering)
"""
import numpy as np
import io

from scipy import signal
from scipy.io import wavfile

FS = 16000  # sample rate (Hz)


# ---------------------------------------------------------------- utilities
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


# ---------------------------------------------------------- test signal gen
_rng = np.random.default_rng(0)


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


# --------------------------------------------------------------- 1. denoise
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


def spectrum_db(x, fs=FS):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return freqs, 20 * np.log10(np.abs(X) + 1e-9)


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


def compute_snr(clean, noisy):
    """Compute Signal-to-Noise Ratio in dB."""
    noise = noisy - clean
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def compute_spectrogram(x, fs=FS, nperseg=512, noverlap=384):
    """Compute spectrogram for visualization."""
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 20 * np.log10(np.abs(Zxx) + 1e-9)


# --------------------------------------------------------------- 2. eq
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


# --------------------------------------------------------------- 3. editor
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


# --------------------------------------------------------------- 4. morse
MORSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
    '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z'
}
TEXT_TO_MORSE = {v: k for k, v in MORSE.items()}


def synth_morse(text, fs=FS, tone_freq=700, unit=0.08):
    chunks = []

    def tone(dur):
        ts = t_axis(dur, fs)
        return 0.9 * np.sin(2 * np.pi * tone_freq * ts)

    def silence(dur):
        return np.zeros(int(dur * fs))

    for ch in text.upper():
        if ch == ' ':
            chunks.append(silence(unit * 7))
            continue
        if ch not in TEXT_TO_MORSE:
            continue
        code_str = TEXT_TO_MORSE[ch]
        for i, sym in enumerate(code_str):
            chunks.append(tone(unit if sym == '.' else unit * 3))
            if i < len(code_str) - 1:
                chunks.append(silence(unit))
        chunks.append(silence(unit * 3))
    total = np.concatenate(chunks) if chunks else np.zeros(1)
    audio = normalize(total + 0.01 * _rng.standard_normal(len(total)))
    return audio, unit


def detect_tone_freq(x, fs=FS, search_range=(200, 2000)):
    """Auto-detect the dominant Morse tone frequency in a recording by
    finding the strongest spectral peak within a plausible tone range.
    This lets the decoder work on uploaded/toolbox clips whose tone
    frequency and speed aren't known in advance."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    mag = np.abs(X)
    band = (freqs >= search_range[0]) & (freqs <= search_range[1])
    if not np.any(band):
        return 700.0
    band_freqs = freqs[band]
    band_mag = mag[band]
    return float(band_freqs[np.argmax(band_mag)])


def detect_unit_duration(keyed, fs=FS):
    """Estimate the Morse 'unit' (dot) duration from the shortest sustained
    on-pulse in a keyed (on/off) signal, so decoding doesn't depend on a
    hardcoded speed."""
    changes = np.diff(keyed.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if keyed[0]:
        starts = np.r_[0, starts]
    if keyed[-1]:
        ends = np.r_[ends, len(keyed)]
    if len(starts) == 0:
        return 0.08
    durations = (ends - starts) / fs
    durations = durations[durations > 0.01]
    if len(durations) == 0:
        return 0.08
    # The shortest common pulse length is a good estimate of one "dot" unit.
    return float(np.percentile(durations, 20))


def build_morse_toolbox(fs=FS):
    """A small library of predetermined Morse audio clips (different
    messages, tone pitches and speeds) the user can pick from on the Morse
    Code page instead of typing their own message or uploading a file."""
    presets = [
        ("SOS",     "SOS",     600, 0.08),
        ("HELLO",   "HELLO",   700, 0.07),
        ("HELP ME", "HELP ME", 550, 0.09),
        ("PARIS",   "PARIS",   650, 0.06),
        ("TEST",    "TEST",    800, 0.10),
    ]
    toolbox = {}
    for label, message, tone_freq, unit in presets:
        audio, used_unit = synth_morse(message, fs=fs, tone_freq=tone_freq, unit=unit)
        toolbox[label] = {
            "audio": audio,
            "message": message,
            "tone_freq": tone_freq,
            "unit": used_unit,
        }
    return toolbox

def decode_morse(x, fs=FS, tone_freq=None, unit=None):
    """Decode a Morse tone recording into text.
    tone_freq=None auto-detects the tone frequency; unit=None auto-estimates
    the dot duration — both needed to decode arbitrary uploaded/toolbox
    clips whose speed and pitch aren't known ahead of time.
    """
    if tone_freq is None:
        tone_freq = detect_tone_freq(x, fs=fs)

    bp = signal.firwin(201, [max(tone_freq - 80, 1), min(tone_freq + 80, fs / 2 - 1)],
                        fs=fs, pass_zero='bandpass')
    filtered = np.convolve(x, bp, mode='same')
    envelope = np.abs(signal.hilbert(filtered))
    envelope = envelope / (np.max(envelope) + 1e-9)
    keyed = envelope > 0.25

    if unit is None:
        unit = detect_unit_duration(keyed, fs=fs)

    changes = np.diff(keyed.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if keyed[0]:
        starts = np.r_[0, starts]
    if keyed[-1]:
        ends = np.r_[ends, len(keyed)]
    on_runs = list(zip(starts, ends))

    symbols = []
    prev_end = 0
    for s, e in on_runs:
        gap = (s - prev_end) / fs
        dur = (e - s) / fs
        if prev_end > 0:
            if gap > unit * 5:
                symbols.append(' ')
            elif gap > unit * 2:
                symbols.append('/')
        symbols.append('.' if dur < unit * 2 else '-')
        prev_end = e

    text, letter = [], ''
    for s in symbols:
        if s in ('/', ' '):
            if letter:
                text.append(MORSE.get(letter, '?'))
                letter = ''
            if s == ' ':
                text.append(' ')
        else:
            letter += s
    if letter:
        text.append(MORSE.get(letter, '?'))
    return ''.join(text), filtered, envelope, keyed, tone_freq, unit


# --------------------------------------------------------------- 5. matcher
def make_clap(fs=FS, dur=0.08):
    n = int(dur * fs)
    env = np.exp(-np.arange(n) / (0.01 * fs))
    return normalize(_rng.standard_normal(n) * env)


def make_bell(fs=FS, dur=0.4, f0=880):
    ts = t_axis(dur, fs)
    env = np.exp(-ts / 0.12)
    return normalize(np.sin(2 * np.pi * f0 * ts) * env)


def make_tap(fs=FS, dur=0.02):
    n = int(dur * fs)
    env = np.exp(-np.arange(n) / (0.003 * fs))
    return normalize((_rng.standard_normal(n) * 0.3 + np.sign(np.sin(2 * np.pi * 2000 * np.arange(n) / fs))) * env)


def build_template_recording(total_dur=4.0, fs=FS):
    templates = {"Clap": make_clap(fs=fs), "Bell": make_bell(fs=fs), "Tap": make_tap(fs=fs)}
    recording = 0.02 * _rng.standard_normal(int(total_dur * fs))
    placements = {"Bell": 0.5, "Clap": 1.8, "Tap": 3.0}
    for name, start_s in placements.items():
        snd = templates[name]
        start = int(start_s * fs)
        recording[start:start + len(snd)] += snd
    return recording, templates, placements


def match_template(recording, template):
    corr = signal.correlate(recording, template, mode='same')
    corr = corr / (np.std(corr) + 1e-9)
    return corr


def clip_similarity(clip_a, clip_b):
    """Compare two arbitrary-length audio clips via cross-correlation
    (the same matched-filter primitive as match_template) and report a
    normalized similarity score and the best alignment offset in seconds.
    This is a thin convenience wrapper — it does not change the underlying
    correlation logic, only packages it for 'compare two clips' use."""
    if len(clip_a) == 0 or len(clip_b) == 0:
        return 0.0, 0.0
    # Correlate the shorter against the longer so 'mode=valid' has a sensible
    # sweep range regardless of which clip the user uploaded first.
    if len(clip_a) >= len(clip_b):
        longer, shorter = clip_a, clip_b
    else:
        longer, shorter = clip_b, clip_a
    corr = signal.correlate(longer, shorter, mode='valid')
    norm = (np.linalg.norm(longer) * np.linalg.norm(shorter)) + 1e-9
    normalized = corr / norm
    best_idx = int(np.argmax(np.abs(normalized)))
    score = float(np.abs(normalized[best_idx]))  # 0..1-ish similarity
    offset_s = best_idx / FS
    return score, offset_s
def compute_spectrogram(x, fs=FS, nperseg=512, noverlap=384):
    """Compute spectrogram for visualization."""
    f, t, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, 20 * np.log10(np.abs(Zxx) + 1e-9)


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


# --------------------------------------------------------------- 2. eq
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


# --------------------------------------------------------------- 3. editor
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


# --------------------------------------------------------------- 4. morse
MORSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
    '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z'
}
TEXT_TO_MORSE = {v: k for k, v in MORSE.items()}


def synth_morse(text, fs=FS, tone_freq=700, unit=0.08):
    chunks = []

    def tone(dur):
        ts = t_axis(dur, fs)
        return 0.9 * np.sin(2 * np.pi * tone_freq * ts)

    def silence(dur):
        return np.zeros(int(dur * fs))

    for ch in text.upper():
        if ch == ' ':
            chunks.append(silence(unit * 7))
            continue
        if ch not in TEXT_TO_MORSE:
            continue
        code_str = TEXT_TO_MORSE[ch]
        for i, sym in enumerate(code_str):
            chunks.append(tone(unit if sym == '.' else unit * 3))
            if i < len(code_str) - 1:
                chunks.append(silence(unit))
        chunks.append(silence(unit * 3))
    total = np.concatenate(chunks) if chunks else np.zeros(1)
    audio = normalize(total + 0.01 * _rng.standard_normal(len(total)))
    return audio, unit


def decode_morse(x, fs=FS, tone_freq=700, unit=0.08):
    bp = signal.firwin(201, [max(tone_freq - 80, 1), tone_freq + 80], fs=fs, pass_zero='bandpass')
    filtered = np.convolve(x, bp, mode='same')
    envelope = np.abs(signal.hilbert(filtered))
    envelope = envelope / (np.max(envelope) + 1e-9)
    keyed = envelope > 0.25

    changes = np.diff(keyed.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if keyed[0]:
        starts = np.r_[0, starts]
    if keyed[-1]:
        ends = np.r_[ends, len(keyed)]
    on_runs = list(zip(starts, ends))

    symbols = []
    prev_end = 0
    for s, e in on_runs:
        gap = (s - prev_end) / fs
        dur = (e - s) / fs
        if prev_end > 0:
            if gap > unit * 5:
                symbols.append(' ')
            elif gap > unit * 2:
                symbols.append('/')
        symbols.append('.' if dur < unit * 2 else '-')
        prev_end = e

    text, letter = [], ''
    for s in symbols:
        if s in ('/', ' '):
            if letter:
                text.append(MORSE.get(letter, '?'))
                letter = ''
            if s == ' ':
                text.append(' ')
        else:
            letter += s
    if letter:
        text.append(MORSE.get(letter, '?'))
    return ''.join(text), filtered, envelope, keyed


# --------------------------------------------------------------- 5. matcher
def make_clap(fs=FS, dur=0.08):
    n = int(dur * fs)
    env = np.exp(-np.arange(n) / (0.01 * fs))
    return normalize(_rng.standard_normal(n) * env)


def make_bell(fs=FS, dur=0.4, f0=880):
    ts = t_axis(dur, fs)
    env = np.exp(-ts / 0.12)
    return normalize(np.sin(2 * np.pi * f0 * ts) * env)


def make_tap(fs=FS, dur=0.02):
    n = int(dur * fs)
    env = np.exp(-np.arange(n) / (0.003 * fs))
    return normalize((_rng.standard_normal(n) * 0.3 + np.sign(np.sin(2 * np.pi * 2000 * np.arange(n) / fs))) * env)


def build_template_recording(total_dur=4.0, fs=FS):
    templates = {"Clap": make_clap(fs=fs), "Bell": make_bell(fs=fs), "Tap": make_tap(fs=fs)}
    recording = 0.02 * _rng.standard_normal(int(total_dur * fs))
    placements = {"Bell": 0.5, "Clap": 1.8, "Tap": 3.0}
    for name, start_s in placements.items():
        snd = templates[name]
        start = int(start_s * fs)
        recording[start:start + len(snd)] += snd
    return recording, templates, placements


def match_template(recording, template):
    corr = signal.correlate(recording, template, mode='same')
    corr = corr / (np.std(corr) + 1e-9)
    return corr


# --------------------------------------------------------------- 6. shazam
def make_melody(freqs, note_dur=0.25, fs=FS):
    chunks = [np.sin(2 * np.pi * f * t_axis(note_dur, fs)) * np.hanning(int(note_dur * fs)) for f in freqs]
    return normalize(np.concatenate(chunks))


def build_song_library(fs=FS):
    return {
        "Song A": make_melody([392, 440, 494, 440, 392, 330, 294], fs=fs),
        "Song B": make_melody([523, 494, 440, 392, 440, 494, 523], fs=fs),
        "Song C": make_melody([262, 330, 392, 523, 392, 330, 262], fs=fs),
    }


def spectrogram_peaks(x, fs=FS, nperseg=512, noverlap=384, n_peaks_per_frame=3):
    f, tt, Zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    peaks = []
    for frame_idx in range(mag.shape[1]):
        frame = mag[:, frame_idx]
        if frame.max() < 1e-6:
            continue
        top_bins = np.argsort(frame)[-n_peaks_per_frame:]
        for b in top_bins:
            peaks.append((tt[frame_idx], f[b]))
    return peaks, f, tt, mag


def match_song(snippet, library_fps, fs=FS):
    snip_peaks, *_ = spectrogram_peaks(snippet, fs=fs)
    snip_fp = np.array(snip_peaks) if snip_peaks else np.zeros((0, 2))
    scores = {}
    for name, song_peaks in library_fps.items():
        song_fp = np.array(song_peaks)
        offsets = []
        for st, sf in snip_fp:
            close = np.abs(song_fp[:, 1] - sf) < 5
            for lt in song_fp[close, 0]:
                offsets.append(lt - st)
        if offsets:
            hist, edges = np.histogram(offsets, bins=40)
            scores[name] = int(hist.max())
        else:
            scores[name] = 0
    return scores
