"""
Shared DSP core for the Audio Signals Toolbox.
Every app in this project (Noise Remover, Equalizer, Editor, Morse Decoder,
Template Matcher, Mini Shazam) is built from three primitives:
  - Convolution (LTI systems)
  - The Fourier Transform (via FFT / STFT)
  - Correlation (matched filtering)
"""
import numpy as np
from scipy import signal
import io
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
