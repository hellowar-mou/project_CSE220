"""
Morse Code logic for the Audio Signals Toolbox.

Text-to-Morse synthesis (with tone/speed/volume/fade control), Morse-to-
text decoding via a choice of THREE tone-detection algorithms sharing one
symbol-grouping backend, auto tone-frequency/speed detection, a timeline
segment classifier, and a preset toolbox of sample clips.
"""
import numpy as np
from scipy import signal

from logic.utils import FS, _rng, t_axis, normalize

MORSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
    '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
}
TEXT_TO_MORSE = {v: k for k, v in MORSE.items()}

DECODE_ALGORITHMS = [
    "Hilbert Envelope (Auto-Detect)",
    "Rectify + Low-pass Envelope",
    "Short-Time Energy (RMS) Detection",
]


def text_to_morse_string(text):
    """Text -> Morse string with spaces between letters and ' / ' between
    words (display-only helper for the Encoder's live Morse preview)."""
    words = text.upper().split(' ')
    out_words = []
    for word in words:
        letters = [TEXT_TO_MORSE[ch] for ch in word if ch in TEXT_TO_MORSE]
        out_words.append(' '.join(letters))
    return '  /  '.join(w for w in out_words if w)


def morse_string_to_text(morse_str):
    """Reverse of text_to_morse_string — used when the user manually edits
    the Morse stream and wants it re-decoded immediately."""
    words = [w.strip() for w in morse_str.split('/')]
    out = []
    for word in words:
        letters = word.split()
        out.append(''.join(MORSE.get(l, '?') for l in letters))
    return ' '.join(out)


def synth_morse(text, fs=FS, tone_freq=700, unit=0.08, volume=0.9,
                 fade_ms=5, noise_amp=0.01):
    """Encode text as a Morse tone recording.
    volume: peak tone amplitude (0-1). fade_ms: per-tone fade-in/out in ms
    (click-free tones). noise_amp: background hiss mixed in (0 for a clean
    tone, higher to build 'noisy' toolbox demos)."""
    chunks = []
    fade_n = max(1, int(fs * fade_ms / 1000.0))

    def tone(dur):
        ts = t_axis(dur, fs)
        wave = volume * np.sin(2 * np.pi * tone_freq * ts)
        n = len(wave)
        if n > 2 * fade_n:
            env = np.ones(n)
            env[:fade_n] = np.linspace(0, 1, fade_n)
            env[-fade_n:] = np.linspace(1, 0, fade_n)
            wave = wave * env
        return wave

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
    noise = noise_amp * _rng.standard_normal(len(total)) if noise_amp > 0 else 0
    audio = normalize(total + noise)
    return audio, unit


def detect_tone_freq(x, fs=FS, search_range=(200, 2000)):
    """Auto-detect the dominant Morse tone frequency in a recording by
    finding the strongest spectral peak within a plausible tone range."""
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
    return float(np.percentile(durations, 20))


def build_morse_toolbox(fs=FS):
    """A library of predetermined Morse audio clips — different messages,
    tone pitches, speeds, and noise conditions — for the Morse page's
    built-in toolbox."""
    presets = [
        ("SOS",                       "SOS",          600, 0.08, 0.01),
        ("HELLO",                     "HELLO",        700, 0.07, 0.01),
        ("CQ (Calling Any Station)",  "CQ",           650, 0.08, 0.01),
        ("Five Short Tones",          "EEEEE",        700, 0.07, 0.01),
        ("Beginner Exercise",         "SOS",          500, 0.15, 0.005),
        ("Fast Morse (High WPM)",     "HELLO WORLD",  700, 0.035, 0.01),
        ("Slow Morse (Low WPM)",      "HELP",         650, 0.16, 0.01),
        ("Noisy Morse",               "TEST",         700, 0.08, 0.08),
        ("Noisy w/ Background",       "HELP ME",      600, 0.09, 0.15),
        ("Different Tone (1200 Hz)",  "PARIS",        1200, 0.07, 0.01),
    ]
    toolbox = {}
    for label, message, tone_freq, unit, noise_amp in presets:
        audio, used_unit = synth_morse(message, fs=fs, tone_freq=tone_freq,
                                        unit=unit, noise_amp=noise_amp)
        toolbox[label] = {
            "audio": audio, "message": message,
            "tone_freq": tone_freq, "unit": used_unit,
        }
    return toolbox


# ======================================================= envelope algorithms
# Three genuinely different ways to turn a raw tone recording into a keyed
# (on/off) signal. All three feed the SAME downstream symbol-grouping code
# below, so the choice of algorithm only affects *how the tone is detected*,
# not how dots/dashes are turned into letters.

def _envelope_hilbert(filtered):
    """Analytic-signal (Hilbert transform) envelope — the original method.
    Exact instantaneous amplitude; best on clean-ish recordings."""
    env = np.abs(signal.hilbert(filtered))
    return env / (np.max(env) + 1e-9)


def _envelope_rectify_lowpass(filtered, fs):
    """Classic AM-radio envelope detector: full-wave rectify, then
    low-pass filter to smooth it into an envelope. Simpler and cheaper
    than Hilbert, and more forgiving of some kinds of broadband noise."""
    rectified = np.abs(filtered)
    cutoff = min(30.0, fs / 2 - 1)
    b, a = signal.butter(2, cutoff, fs=fs, btype='low')
    env = signal.filtfilt(b, a, rectified)
    env = np.clip(env, 0, None)
    return env / (np.max(env) + 1e-9)


def _envelope_short_time_energy(filtered, fs, window_ms=10):
    """Short-time energy (RMS) detection: a classic voice/tone-activity-
    detection technique — compute a moving-window RMS of the filtered
    signal directly (block energy), rather than tracking a continuous
    envelope via Hilbert transform or a separate rectify+smooth filter.
    Simple, cheap, and very robust for on/off tone detection."""
    win = max(4, int(fs * window_ms / 1000.0))
    energy = np.convolve(filtered ** 2, np.ones(win) / win, mode='same')
    energy = np.sqrt(np.clip(energy, 0, None))
    return energy / (np.max(energy) + 1e-9)


def compute_envelope(x, fs=FS, tone_freq=None,
                      algorithm="Hilbert Envelope (Auto-Detect)"):
    """Bandpass-filter around tone_freq (auto-detected if None), then
    extract an envelope using the selected algorithm. Returns
    (filtered, envelope, tone_freq)."""
    if tone_freq is None:
        tone_freq = detect_tone_freq(x, fs=fs)
    bp = signal.firwin(201, [max(tone_freq - 80, 1), min(tone_freq + 80, fs / 2 - 1)],
                        fs=fs, pass_zero='bandpass')
    filtered = np.convolve(x, bp, mode='same')

    if algorithm == "Rectify + Low-pass Envelope":
        envelope = _envelope_rectify_lowpass(filtered, fs)
    elif algorithm == "Short-Time Energy (RMS) Detection":
        envelope = _envelope_short_time_energy(filtered, fs)
    else:
        envelope = _envelope_hilbert(filtered)

    return filtered, envelope, tone_freq


def keyed_to_text(keyed, fs=FS, unit=None):
    """Shared symbol-grouping backend: on/off keyed signal -> Morse text.
    Used by every decode algorithm so the choice of envelope-detection
    method never changes how dots/dashes/gaps are interpreted."""
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
    return ''.join(text), unit


def decode_morse(x, fs=FS, tone_freq=None, unit=None, threshold=0.25,
                  algorithm="Hilbert Envelope (Auto-Detect)"):
    """Decode a Morse tone recording into text.
    tone_freq=None auto-detects the tone frequency; unit=None auto-estimates
    the dot duration. `threshold` (0-1) is user-adjustable detection
    sensitivity. `algorithm` selects which of the 3 envelope-detection
    techniques above is used; all share the same symbol-grouping logic.
    Returns (text, filtered, envelope, keyed, tone_freq, unit) — unchanged
    signature from the original single-algorithm version.
    """
    filtered, envelope, tone_freq = compute_envelope(x, fs=fs, tone_freq=tone_freq, algorithm=algorithm)
    keyed = envelope > threshold
    text, unit = keyed_to_text(keyed, fs=fs, unit=unit)
    return text, filtered, envelope, keyed, tone_freq, unit


def classify_morse_segments(keyed, fs=FS, unit=0.08, ambiguity_margin=0.35):
    """Break a keyed (on/off) signal into labeled timeline segments: dot,
    dash, gap, char_gap, word_gap — each with start/end/duration and an
    'uncertain' flag when the duration sits too close to a classification
    boundary to be confident, so the UI can show '? UNCERTAIN' and let the
    user reclassify it by hand instead of silently guessing."""
    changes = np.diff(keyed.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if keyed[0]:
        starts = np.r_[0, starts]
    if keyed[-1]:
        ends = np.r_[ends, len(keyed)]
    on_runs = list(zip(starts, ends))

    segments = []
    prev_end = 0
    for s, e in on_runs:
        if prev_end > 0:
            gap_dur = (s - prev_end) / fs
            if gap_dur > unit * 0.5:
                if gap_dur > unit * 5:
                    gap_type, boundary = "word_gap", unit * 5
                elif gap_dur > unit * 2:
                    gap_type, boundary = "char_gap", unit * 2
                else:
                    gap_type, boundary = "gap", unit
                uncertain = abs(gap_dur - boundary) < unit * ambiguity_margin
                segments.append({"type": gap_type, "start_s": prev_end / fs,
                                  "end_s": s / fs, "duration_s": gap_dur,
                                  "uncertain": uncertain})
        dur = (e - s) / fs
        tone_type = "dot" if dur < unit * 2 else "dash"
        uncertain = abs(dur - unit * 2) < unit * ambiguity_margin
        segments.append({"type": tone_type, "start_s": s / fs, "end_s": e / fs,
                          "duration_s": dur, "uncertain": uncertain})
        prev_end = e
    return segments
