"""
Morse Code logic for the Audio Signals Toolbox.
Text-to-Morse synthesis, Morse-to-text decoding with auto tone-frequency
and unit-duration detection, and a preset toolbox of sample clips.
"""
import numpy as np
from scipy import signal

from logic.utils import FS, _rng, t_axis, normalize


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
