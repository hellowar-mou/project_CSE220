"""
Shared DSP core for the Audio Signals Toolbox.
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
    FS, _rng, t_axis, normalize, to_wav_bytes,
    load_audio_file, load_sample_song,
    spectrum_db, compute_spectrogram, compute_snr,
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
)

# ---- matcher
from logic.matcher import (
    make_clap, make_bell, make_tap,
    build_template_recording, match_template, clip_similarity,
    make_melody, build_song_library, spectrogram_peaks, match_song,
)
