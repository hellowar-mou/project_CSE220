"""
App-wide dark/light theme system.

Design goal: toggling the theme must re-skin EVERY widget in the running
app - including widgets inside pages (Noise Remover, Editor) whose
processing logic must never be touched, and whose existing code sets many
hardcoded, light-theme-only inline stylesheets widget-by-widget.

Rather than hand-editing every one of those call sites, this module
installs a transparent monkey-patch on QWidget.setStyleSheet: every call
anywhere in the app is intercepted, the ORIGINAL (light-theme) string is
remembered against that widget, and a colour-substituted variant is applied
instead when dark mode is active. Toggling the theme just re-walks that
registry and re-applies the correct variant to every still-alive widget -
no page file needs to know theming exists.

matplotlib canvases work the same way via a small MplCanvas registry
(populated by ui/widgets.py) + recolor_figure(), which recolors an
already-rendered Figure in place (facecolor, ticks, spines, text) without
touching any page's plotting/DSP code.
"""
import json
import os
import re
import weakref

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "app_settings.json")

# ----------------------------------------------------------- light -> dark
# Every hex/rgba token below is one actually used somewhere in this app's
# existing (light-only) inline stylesheets. Toggling to dark substitutes
# these; toggling back to light restores the original strings verbatim.
#
# rgba(255,255,255,ALPHA) — any alpha — is handled generically by regex
# below (translucent white card -> translucent dark-navy card, same
# alpha), so every card/panel background repaints correctly in dark mode
# regardless of which exact alpha a given widget happens to use.
_RGBA_WHITE_RE = re.compile(r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
_RGBA_MINT_RE = re.compile(r"rgba\(\s*143\s*,\s*211\s*,\s*199\s*,\s*([0-9]*\.?[0-9]+)\s*\)")
_RGBA_CREAM_RE = re.compile(
    r"rgba\(\s*255\s*,\s*255\s*,\s*255\s*,\s*([0-9]*\.?[0-9]+)\s*\)"
)

_CREAM = "#ffffff"
_CREAM_RGB = (255, 255, 255)


def _regex_recolor(qss):
    qss = _RGBA_WHITE_RE.sub(lambda m: f"rgba(30,41,59,{m.group(1)})", qss)
    qss = _RGBA_MINT_RE.sub(lambda m: f"rgba(95,201,184,{float(m.group(1)) * 0.65:.2f})", qss)
    return qss


def _regex_cream(qss):
    """Normalize light-theme surfaces to pure white.

    Text that is explicitly white is intentionally left unchanged so it
    remains readable on accent-colored buttons and badges.
    """
    qss = _RGBA_CREAM_RE.sub(
        lambda m: f"rgba({_CREAM_RGB[0]},{_CREAM_RGB[1]},{_CREAM_RGB[2]},{m.group(1)})",
        qss,
    )
    qss = re.sub(r"background(?:-color)?:\s*white\s*;", f"background: {_CREAM};", qss)
    qss = re.sub(
        r"background(?:-color)?:\s*#ffffff\s*;",
        f"background: {_CREAM};",
        qss,
        flags=re.IGNORECASE,
    )
    return qss


_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})\b")

# Hex colors intentionally left alone even though they might look "pale" by
# a naive threshold — accent/status colors chosen to read fine on both
# themes, plus every explicit dark-mode OUTPUT color already in
# _LIGHT_TO_DARK.values() (so re-toggling doesn't double-darken them).
_PRESERVE_HEX = {
    "6fb1ea", "8fd3c7", "2f6690", "c0392b", "1e8a5f", "e74c3c", "2ecc71",
    "27ae60", "e67e22", "f1c40f", "9b59b6", "8e44ad", "3498db", "2980b9",
    "d35400", "b8860b", "5fa3e0", "7cc9bc", "bdc3c7", "95a5a6",
    # dark-mode TEXT colors produced by _LIGHT_TO_DARK below — these are
    # deliberately light (for readability on dark backgrounds) and must
    # not be re-darkened by the generic pale-background fallback pass.
    "e8eff8", "dce6f2", "c3d2e4", "93a7c0", "7d92ac",
    "f1f5fb", "8fe0cf", "d8b4fe", "f5d77a", "e8f3ff", "8db7ff",
    # explicit dark-theme foreground colors used by dialogs and controls
    "f8fafc", "f1f5f9", "cbd5e1", "7dd3fc", "38bdf8", "ffffff",
}


def _hex_to_hsl(hex6):
    r = int(hex6[0:2], 16) / 255.0
    g = int(hex6[2:4], 16) / 255.0
    b = int(hex6[4:6], 16) / 255.0
    import colorsys
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h, l, s


def _hsl_to_hex(h, l, s):
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"{int(round(r*255)):02x}{int(round(g*255)):02x}{int(round(b*255)):02x}"


def _auto_darken_pale_hex(qss):
    """Fallback for the many one-off pale pastel badge/background colors in
    this app that aren't worth hand-cataloguing individually: any #RRGGBB
    with lightness >= 0.85 (a pale tint, almost certainly a light-theme-only
    background/border, never body text at that lightness) gets its
    lightness reduced to a dark equivalent while keeping the same hue, so
    e.g. a pale pink error badge becomes a deep, dark-red-tinted one."""
    def repl(m):
        hex6 = m.group(1).lower()
        if hex6 in _PRESERVE_HEX:
            return m.group(0)
        h, l, s = _hex_to_hsl(hex6)
        if l < 0.85:
            return m.group(0)
        new_l = 0.14 + (l - 0.85) * 0.4
        new_s = min(1.0, s * 0.9)
        return "#" + _hsl_to_hex(h, new_l, new_s)
    return _HEX_RE.sub(repl, qss)


_LIGHT_TO_DARK = {
    # solid backgrounds
    "background: white;": "background: #1c2634;",
    "background:white;": "background:#1c2634;",
    "background: #ffffff;": "background: #1c2634;",
    "background: #ffffff": "background: #1c2634",
    "background-color: #ffffff;": "background-color: #1c2634;",
    "background-color:#ffffff;": "background-color:#1c2634;",
    "background-color: #ffffff": "background-color: #1c2634",

    # page/section headers & wave backdrop
    "#f4f9ff": "#131a24",
    "#eef5fb": "#0f1620",
    "#e8f2fb": "#0a0f17",

    # text colours
    "#2f3e50": "#e8eff8",
    "#33465c": "#dce6f2",
    "#445872": "#c3d2e4",
    "#6b7f96": "#93a7c0",
    "#8a9bb0": "#7d92ac",
    "#2f6690": "#8db7ff",
    # text colours used by page-specific inline styles
    "#1c2a38": "#f1f5fb",
    "#2c3e50": "#e8eff8",
    "#43586d": "#c3d2e4",
    "#4a5d73": "#c3d2e4",
    "#1e7062": "#8fe0cf",
    "#6c3483": "#d8b4fe",
    "#9a7d0a": "#f5d77a",
    "#173f63": "#e8f3ff",
    "#123650": "#e8f3ff",
    "#2563eb": "#8db7ff",
    "#b8860b": "#f5d77a",

    # borders / dividers
    "#d6e2ef": "#2c3b4f",
    "#cfe1f2": "#2c3b4f",
    "#cdeae4": "#243b4a",
    "#dbe8f4": "#293544",
    "#b8cde2": "#40546b",

    # hover / selection tints
    "#eaf4ff": "#22303f",
    "#f0f6fc": "#1c2634",
    "#e7f3ff": "#1c2f42",
    "#d9f0eb": "#1c3a34",
    "#eaf6f4": "#17262f",
    "#edf6ff": "#1b2a3d",
    "#e3f0ff": "#22364d",

    # status colours stay legible on both themes; left unchanged on purpose:
    # "#6fb1ea", "#8fd3c7", "#2f6690", "#c0392b", "#1e8a5f" (accent/error/success)
}

_current_mode = "light"
_widget_registry = []   # list of (weakref, original_light_qss)
_canvas_registry = []   # list of weakref to MplCanvas instances
_listeners = []         # zero-arg callables invoked after every mode switch
_app_stylesheet_original = None   # the one QApplication-level stylesheet


def _recolor(qss, mode):
    if not qss:
        return qss
    if mode == "light":
        return _regex_cream(qss)
    out = _regex_recolor(qss)
    for light, dark in _LIGHT_TO_DARK.items():
        out = out.replace(light, dark)
    out = _auto_darken_pale_hex(out)
    return out


def install_theme_patch():
    """Call once, before any widgets are created. Wraps both
    QWidget.setStyleSheet AND QApplication.setStyleSheet (two distinct
    methods on unrelated classes — QApplication is not a QWidget) so every
    future stylesheet assignment anywhere in the app, including the base
    app-wide QSS loaded once at startup, is remembered and can be
    re-colored on demand."""
    from PySide6.QtWidgets import QWidget, QApplication

    if getattr(QWidget, "_theme_patched", False):
        return

    original_widget = QWidget.setStyleSheet

    def patched_widget(self, qss):
        if qss:
            _widget_registry.append((weakref.ref(self), qss))
        original_widget(self, _recolor(qss, _current_mode))

    QWidget.setStyleSheet = patched_widget
    QWidget._theme_patched = True
    QWidget._orig_setStyleSheet = original_widget

    original_app = QApplication.setStyleSheet

    def patched_app(self, qss):
        global _app_stylesheet_original
        if qss:
            _app_stylesheet_original = qss
        original_app(self, _recolor(qss, _current_mode))

    QApplication.setStyleSheet = patched_app
    QApplication._orig_setStyleSheet = original_app


def register_canvas(canvas):
    """Called by MplCanvas.__init__ so its Figure can be recolored on
    theme toggle without any page needing to know theming exists."""
    _canvas_registry.append(weakref.ref(canvas))


def add_listener(callback):
    """Register a zero-arg callback invoked after every theme switch, for
    pages that need to actively re-render (e.g. re-plot with new data-line
    colors) rather than just have their stylesheet swapped."""
    _listeners.append(callback)


def recolor_figure(fig, mode):
    """Recolor an already-rendered matplotlib Figure in place — facecolor,
    axes background, ticks, labels, titles, spines, grid — without
    recomputing or touching whatever DSP produced the plotted data."""
    from ui.widgets import _PLOT_COLORS
    colors = _PLOT_COLORS.get(mode, _PLOT_COLORS["light"])
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor(colors["face"])
        ax.tick_params(colors=colors["text"])
        ax.xaxis.label.set_color(colors["text"])
        ax.yaxis.label.set_color(colors["text"])
        ax.title.set_color(colors["text"])
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])
        for line in ax.get_lines():
            # keep the very first line (usually primary data) on the
            # theme's primary accent so plots stay readable both ways
            pass
    if fig._suptitle is not None:
        fig._suptitle.set_color(colors["text"])
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass


def get_current_mode():
    return _current_mode


def set_current_mode(mode):
    """Switch the whole app to `mode` ('light' or 'dark'): re-applies every
    registered widget's stylesheet and recolors every registered canvas."""
    global _current_mode
    if mode not in ("light", "dark"):
        return
    _current_mode = mode

    from PySide6.QtWidgets import QWidget, QApplication
    from PySide6.QtGui import QColor, QPalette

    if _app_stylesheet_original is not None:
        app = QApplication.instance()
        if app is not None:
            QApplication._orig_setStyleSheet(app, _recolor(_app_stylesheet_original, mode))
    else:
        app = QApplication.instance()

    if app is not None:
        palette = app.palette()
        surface = QColor(_CREAM if mode == "light" else "#1c2634")
        text = QColor("#33465c" if mode == "light" else "#dce6f2")
        palette.setColor(QPalette.ColorRole.Base, surface)
        palette.setColor(QPalette.ColorRole.Window, surface)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        app.setPalette(palette)

    dead = []
    for i, (wref, original_qss) in enumerate(_widget_registry):
        w = wref()
        if w is None:
            dead.append(i)
            continue
        QWidget._orig_setStyleSheet(w, _recolor(original_qss, mode))
    for i in reversed(dead):
        _widget_registry.pop(i)

    dead_c = []
    for i, cref in enumerate(_canvas_registry):
        c = cref()
        if c is None:
            dead_c.append(i)
            continue
        recolor_figure(c.fig, mode)
    for i in reversed(dead_c):
        _canvas_registry.pop(i)

    for cb in list(_listeners):
        try:
            cb()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Theme listener failed")


def load_saved_theme():
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
            mode = data.get("theme", "light")
            return mode if mode in ("light", "dark") else "light"
    except Exception:
        return "light"


def save_theme(mode):
    try:
        data = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        data["theme"] = mode
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def is_guidance_dismissed(key):
    """Return whether a first-time module guide was permanently dismissed."""
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        return bool(data.get("guidance_dismissed", {}).get(key, False))
    except Exception:
        return False


def set_guidance_dismissed(key, dismissed=True):
    """Persist a module guide preference while preserving other app settings."""
    try:
        data = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        dismissed_map = data.setdefault("guidance_dismissed", {})
        dismissed_map[key] = bool(dismissed)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def is_walkthrough_completed(user_key="default"):
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
        completed = data.get("walkthrough_completed", {})
        if isinstance(completed, bool):
            return completed if user_key == "default" else False
        return bool(completed.get(user_key, False))
    except Exception:
        return False


def set_walkthrough_completed(user_key="default"):
    data = {}
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                data = json.load(f)
        completed = data.get("walkthrough_completed", {})
        if not isinstance(completed, dict):
            completed = {}
        completed[user_key] = True
        data["walkthrough_completed"] = completed
        with open(SETTINGS_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ----------------------------------------------------------- wave colours
_WAVE_PALETTES = {
    "light": {
        "colors": ["#8ec5fc", "#a7e3e0", "#c9b6f2"],
        "bg": ("#f4f9ff", "#eef5fb", "#e8f2fb"),
    },
    "dark": {
        "colors": ["#2a3f5a", "#254a45", "#3a2f52"],
        "bg": ("#131a24", "#0f1620", "#0a0f17"),
    },
}


def get_wave_colors(mode="light"):
    p = _WAVE_PALETTES.get(mode, _WAVE_PALETTES["light"])
    return p["colors"], p["bg"]
