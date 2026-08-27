"""
Audio Matcher logic for the Audio Signals Toolbox.

Existing, unchanged primitives:
    make_clap / make_bell / make_tap / build_template_recording
    match_template   — cross-correlation (scipy.signal.correlate)
    clip_similarity  — normalized cross-correlation for two arbitrary clips
    spectrogram_peaks / match_song — STFT peak-fingerprint correlation

These are the ONLY matching primitives in this file. What's new below is an
orchestration layer (`load_audio_library`, `run_matching`) that packages
those existing algorithms into a proper "search the built-in library"
pipeline for the Audio Matcher UI — it does not add any new signal-
processing method, only combines/normalizes the outputs of the ones above.

Note: `make_melody` / `build_song_library` (a fake, synthetic 3-note-melody
"song library") have been removed — they were unused leftovers from an
earlier, now-removed feature. The real built-in library is the actual MP3s
in `audios/`, loaded via `load_audio_library()` below.
"""
import os
import time
import numpy as np
from scipy import signal

from logic.utils import FS, _rng, t_axis, normalize, load_sample_song


# --------------------------------------------------------------- templates
# (unchanged — used by the built-in clap/bell/tap correlation demo)
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
# (unchanged — the core matched-filter primitive used throughout)
def match_template(recording, template):
    corr = signal.correlate(recording, template, mode='same')
    corr = corr / (np.std(corr) + 1e-9)
    return corr


def clip_similarity(clip_a, clip_b):
    """Compare two arbitrary-length audio clips via cross-correlation
    (the same matched-filter primitive as match_template) and report a
    normalized similarity score and the best alignment offset in seconds."""
    if len(clip_a) == 0 or len(clip_b) == 0:
        return 0.0, 0.0
    if len(clip_a) >= len(clip_b):
        longer, shorter = clip_a, clip_b
    else:
        longer, shorter = clip_b, clip_a
    corr = signal.correlate(longer, shorter, mode='valid')
    norm = (np.linalg.norm(longer) * np.linalg.norm(shorter)) + 1e-9
    normalized = corr / norm
    best_idx = int(np.argmax(np.abs(normalized)))
    score = float(np.abs(normalized[best_idx]))
    offset_s = best_idx / FS
    return score, offset_s


# --------------------------------------------------------- spectral fingerprint
# (unchanged — STFT peak-fingerprint correlation, formerly used for song ID)
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
    """Score a query snippet's spectral-peak fingerprint against a dict of
    {name: peaks_list} fingerprints. Returns {name: raw_histogram_score}."""
    snip_peaks, *_ = spectrogram_peaks(snippet, fs=fs)
    snip_fp = np.array(snip_peaks) if snip_peaks else np.zeros((0, 2))
    scores = {}
    for name, song_peaks in library_fps.items():
        song_fp = np.array(song_peaks)
        offsets = []
        for st, sf_ in snip_fp:
            close = np.abs(song_fp[:, 1] - sf_) < 5
            for lt in song_fp[close, 0]:
                offsets.append(lt - st)
        if offsets:
            hist, edges = np.histogram(offsets, bins=40)
            scores[name] = int(hist.max())
        else:
            scores[name] = 0
    return scores


# ------------------------------------------------------- built-in audio library
AUDIOS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audios")


def load_audio_library(folder=None, target_fs=FS, max_duration=30.0):
    """Scan `audios/` for real audio files and load each one via
    load_sample_song. Returns an ordered dict:
        {display_name: {"audio": np.ndarray, "fs": int, "duration_s": float,
                         "path": str, "id": str}}
    This is the actual built-in library the Matcher searches — real songs,
    not synthetic placeholders."""
    folder = folder or AUDIOS_DIR
    library = {}
    if not os.path.isdir(folder):
        return library
    supported_ext = {".mp3", ".wav", ".flac", ".ogg"}
    for i, fname in enumerate(sorted(os.listdir(folder))):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported_ext:
            continue
        path = os.path.join(folder, fname)
        try:
            audio, orig_fs = load_sample_song(path, target_fs=target_fs, max_duration=max_duration)
        except Exception:
            continue
        display_name = os.path.splitext(fname)[0]
        library[display_name] = {
            "audio": audio,
            "fs": target_fs,
            "duration_s": len(audio) / target_fs,
            "path": path,
            "id": f"lib_{i:02d}",
        }
    return library


# ------------------------------------------------------- multi-algorithm matching
ALGORITHMS = [
    "Cross-Correlation (Time-Domain)",
    "Normalized Similarity",
    "Spectral Fingerprint (STFT Peaks)",
    "Combined (All Algorithms)",
]


def _score_cross_correlation(query, library_audio):
    """Peak correlation strength of the query against a library track,
    using the existing match_template primitive. The query is correlated
    against the FULL library track (not truncated) so a match anywhere in
    a longer track is still found, then squashed to ~[0,1)."""
    if len(query) < 8 or len(library_audio) < 8:
        return 0.0
    q = query if len(query) <= len(library_audio) else query[:len(library_audio)]
    corr = match_template(library_audio, q)
    peak = float(np.max(np.abs(corr)))
    return float(2.0 / (1.0 + np.exp(-peak / 8.0)) - 1.0)  # squashed to [0,1)


def _score_similarity(query, library_audio):
    """clip_similarity is already normalized to roughly [0,1]."""
    score, _offset = clip_similarity(query, library_audio)
    return min(1.0, score)


def _score_spectral_fingerprint(query, library_audio, n_bins=60):
    """Spectral peak-fingerprint correlation (formerly the song-ID engine),
    now scoring a query against one real library track.

    match_song's raw histogram-bin count has no fixed scale and saturates
    for dense, broadband real music (unlike the sparse pure-tone melodies
    it was originally built for) — so instead of using the raw count, we
    measure how much taller the best-aligned offset bin is than a 'flat'
    (no-alignment) baseline. A true match produces a sharp peak in the
    offset histogram; unrelated audio produces a roughly uniform one. This
    is a normalization change only — match_song's own histogram computation
    is untouched."""
    query_peaks, *_ = spectrogram_peaks(query)
    if not query_peaks:
        return 0.0
    lib_peaks, *_ = spectrogram_peaks(library_audio)
    song_fp = np.array(lib_peaks)
    snip_fp = np.array(query_peaks)
    if len(song_fp) == 0:
        return 0.0
    offsets = []
    for st, sf_ in snip_fp:
        close = np.abs(song_fp[:, 1] - sf_) < 5
        for lt in song_fp[close, 0]:
            offsets.append(lt - st)
    if not offsets:
        return 0.0
    hist, _edges = np.histogram(offsets, bins=n_bins)
    peak = float(hist.max())
    mean = float(hist.mean()) + 1e-9
    ratio = peak / mean  # 1.0 = perfectly flat/no alignment; higher = sharper peak
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def run_matching(query_audio, library, algorithm="Combined (All Algorithms)",
                  progress_callback=None):
    """Search `library` (as returned by load_audio_library) for the best
    match to `query_audio` using the selected algorithm, or all of them
    combined. Returns:
        {
          "ranked": [ {"name", "score", "per_algorithm": {...}}, ... ]  # sorted desc
          "elapsed_s": float,
          "algorithm": str,
        }
    progress_callback(stage_name, fraction_0_to_1) is called as each library
    item / stage completes, so the UI can show genuine (non-fake) progress.
    """
    start = time.time()
    names = list(library.keys())
    n = max(1, len(names))
    results = {name: {} for name in names}

    run_all = algorithm == "Combined (All Algorithms)"
    do_corr = run_all or algorithm == "Cross-Correlation (Time-Domain)"
    do_sim = run_all or algorithm == "Normalized Similarity"
    do_spec = run_all or algorithm == "Spectral Fingerprint (STFT Peaks)"

    for i, name in enumerate(names):
        lib_audio = library[name]["audio"]
        if do_corr:
            results[name]["Cross-Correlation (Time-Domain)"] = _score_cross_correlation(query_audio, lib_audio)
        if do_sim:
            results[name]["Normalized Similarity"] = _score_similarity(query_audio, lib_audio)
        if do_spec:
            results[name]["Spectral Fingerprint (STFT Peaks)"] = _score_spectral_fingerprint(query_audio, lib_audio)
        if progress_callback:
            progress_callback(name, (i + 1) / n)

    ranked = []
    for name in names:
        per_algo = results[name]
        if run_all:
            combined = float(np.mean(list(per_algo.values()))) if per_algo else 0.0
            final_score = combined
        else:
            final_score = list(per_algo.values())[0] if per_algo else 0.0
        ranked.append({"name": name, "score": final_score, "per_algorithm": per_algo})

    ranked.sort(key=lambda r: r["score"], reverse=True)
    elapsed = time.time() - start
    return {"ranked": ranked, "elapsed_s": elapsed, "algorithm": algorithm}
