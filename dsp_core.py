"""
Shared DSP core for ReSonus.
Every app in this project (Noise Remover, Equalizer, Editor, Morse Code
Converter, Audio Matcher) is built from three primitives:
  - Convolution (LTI systems)
  - The Fourier Transform (via FFT / STFT)
  - Correlation (matched filtering)

This file re-exports all public symbols from the logic/ package so that
existing `import dsp_core as dsp` statements continue to work unchanged.
"""

# ---- utilities
from logic.utils import (
    FS, _rng, t_axis, normalize, to_wav_bytes, write_wav_file,
    load_audio_file, load_sample_song,
    spectrum_db, compute_spectrogram, compute_snr,
    compute_rms, compute_peak, compute_dominant_frequency,
)

# ---- noise remover
from logic.noise_remover import (
    make_voice_like, make_noisy,
    moving_average_filter, freq_domain_denoise,
    spectral_subtraction_denoise, wiener_denoise,
    add_white_noise, add_pink_noise, add_hum_noise,
    add_traffic_noise, add_crowd_noise, add_combined_noise,
)

# ---- equalizer
from logic.equalizer import (
    band_split, equalize, equalizer_frequency_response,
    FS_EQ, EQ_BANDS, EQ_ALGORITHMS, EQ_PRESETS,
    apply_nband_eq, nband_eq_frequency_response,
)

# ---- editor
from logic.editor import (
    trim, join, reverse, time_scale, fade, smooth, echo,
)

# ---- morse
from logic.morse import (
    MORSE, TEXT_TO_MORSE,
    synth_morse, decode_morse,
    detect_tone_freq, detect_unit_duration, build_morse_toolbox,
    DECODE_ALGORITHMS, text_to_morse_string, morse_string_to_text,
    compute_envelope, keyed_to_text, classify_morse_segments,
)

# ---- matcher
from logic.matcher import (
    make_clap, make_bell, make_tap,
    build_template_recording, match_template, clip_similarity,
    spectrogram_peaks, match_song,
    load_audio_library, run_matching, ALGORITHMS,
)
