"""
Audio Matcher logic for the Audio Signals Toolbox.
Template matching (correlation-based), clip similarity comparison,
sound template generation (clap/bell/tap), and spectrogram-based
song fingerprinting and matching.
"""
import numpy as np
from scipy import signal

from logic.utils import FS, _rng, t_axis, normalize


# --------------------------------------------------------------- templates
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


# --------------------------------------------------------------- correlation
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


# --------------------------------------------------------------- song matching
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
