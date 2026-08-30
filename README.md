# Audio Signals Toolbox — Desktop App

A native PySide6 desktop application for the Signals & Linear Systems course demo: real
SQLite + bcrypt login, a light theme with a slow-moving animated wave background (pure Qt,
no browser), and five live audio DSP modules built on one shared engine. Audio upload is
supported wherever it makes sense, and every module shows live, calibrated visualizations
of the input, output, and intermediate processing stages.

## Setup

```bash
git clone <your-repo-url>
cd audio_toolbox_desktop
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

A native window opens directly — no browser, no local server.

## Logging in

- **Seeded demo account** — Username: `signals` / Password: `lti2026`
  (created automatically in `users.db` on first run)
- **Or create your own account** from the "Create account" tab. Passwords are hashed with
  bcrypt before being stored in the local SQLite database (`users.db`) — never in plaintext.

`users.db` is per-machine and gitignored, so each collaborator gets their own local account
store — no shared credentials file to merge or conflict over.

## Project structure

```
audio_toolbox_desktop/
├── main.py                 # entry point — QApplication, login/main-window handoff
├── auth_db.py               # SQLite + bcrypt auth (register_user, verify_login)
├── dsp_core.py               # shared DSP engine — convolution/FFT/correlation primitives
├── requirements.txt
├── ui/
│   ├── theme.qss             # light-theme stylesheet (shared across all windows)
│   ├── wave_widget.py         # animated wave background (QPainter + QTimer)
│   ├── widgets.py              # MplCanvas (matplotlib embed), AudioPlayButton
│   ├── login_window.py          # sign-in / create-account window
│   ├── main_window.py            # sidebar nav + stacked pages
│   └── pages/
│       ├── home_page.py
│       ├── noise_remover_page.py
│       ├── equalizer_page.py
│       ├── editor_page.py
│       ├── morse_page.py
│       └── matcher_page.py
└── users.db                  # created on first run, gitignored
```

## Modules

| Module | Logic | Upload | Live visualization |
|---|---|---|---|
| Noise Remover | **Unchanged** (moving-average, FFT notch) | ✅ file + mic | input/output waveform, spectrum, spectrogram |
| Equalizer | Logic changes allowed | ✅ file | live band spectra + combined frequency response as sliders move |
| Editor | **Unchanged** (trim/reverse/fade/echo) | ✅ file | input/output waveform |
| Morse Code Converter | Logic changes allowed | ✅ file + built-in toolbox | tone → bandpass → envelope → keying → text, all shown live |
| Audio Matcher | **Unchanged** (cross-correlation) | ✅ file (reference + target) | correlation trace with detected alignment |

## Working with a partner

This is a normal git repo — push it to GitHub/GitLab and your partner clones it:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

Partner's side:
```bash
git clone <your-repo-url>
cd audio_toolbox_desktop
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Good places to split work by file, so you don't collide on the same lines:
- One person on `ui/pages/*.py` (the five module UIs)
- One person on `dsp_core.py` (the shared DSP engine — careful, all pages import this)
- `ui/theme.qss` / `ui/wave_widget.py` for anyone doing visual polish

## Presentation tips

- Every page has live sliders/inputs and live visualizations — moving them during the demo
  makes the filter/effect cause-and-effect visible in real time, not just a before/after
  snapshot.
- The sidebar's Home page lists the four theory pillars (Convolution / FFT / Correlation / LTI)
  — good for framing the whole toolbox in one sentence before diving into modules.
- Pages build lazily on first visit, so the very first click into each tab may take a beat
  longer than switching back to an already-visited one — that's expected, not a bug.
