from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QListWidgetItem, QStackedWidget, QLabel, QPushButton
)
from PySide6.QtCore import Qt

from ui.wave_widget import WaveBackground
from ui import theme as theme_module
from ui.pages.home_page import HomePage
from ui.pages.noise_remover_page import NoiseRemoverPage
from ui.pages.equalizer_page import EqualizerPage
from ui.pages.editor_page import EditorPage
from ui.pages.morse_page import MorsePage
from ui.pages.matcher_page import MatcherPage
from ui.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self, username, logout_callback):
        super().__init__()
        self.setWindowTitle("Audio Signals Toolbox")
        self.resize(1180, 760)
        self._logout_callback = logout_callback

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        self.bg = WaveBackground(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---- sidebar ----
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)

        brand = QLabel("🌊 Toolbox")
        brand.setObjectName("SectionTitle")
        side_layout.addWidget(brand)

        user_label = QLabel(f"Signed in as <b>{username}</b>")
        user_label.setObjectName("Caption")
        side_layout.addWidget(user_label)
        side_layout.addSpacing(8)

        self.nav = QListWidget()
        self.nav.setObjectName("NavList")
        for label in ["Home", "Noise Remover", "Equalizer", "Editor",
                      "Morse Code Converter", "Audio Matcher", "Settings"]:
            QListWidgetItem(label, self.nav)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._on_nav)
        side_layout.addWidget(self.nav, 1)

        logout_btn = QPushButton("Log out")
        logout_btn.setObjectName("SecondaryButton")
        logout_btn.clicked.connect(self._on_logout)
        side_layout.addWidget(logout_btn)

        root.addWidget(sidebar)

        # ---- pages ----
        self.stack = QStackedWidget()
        self.pages = {}
        self._lazy_pages = {
            "Home": HomePage,
            "Noise Remover": NoiseRemoverPage,
            "Equalizer": EqualizerPage,
            "Editor": EditorPage,
            "Morse Code Converter": MorsePage,
            "Audio Matcher": MatcherPage,
            "Settings": SettingsPage,
        }
        # Home is built eagerly; the rest build lazily on first visit (faster startup)
        home = HomePage()
        self.pages["Home"] = home
        self.stack.addWidget(home)

        root.addWidget(self.stack, 1)

        theme_module.add_listener(self._on_theme_changed)

    def resizeEvent(self, event):
        self.bg.setGeometry(self.centralWidget().rect())
        super().resizeEvent(event)

    def _on_nav(self, row):
        label = self.nav.item(row).text()
        if label not in self.pages:
            page_cls = self._lazy_pages[label]
            page = page_cls()
            self.pages[label] = page
            self.stack.addWidget(page)
        self.stack.setCurrentWidget(self.pages[label])

    def _on_theme_changed(self):
        """Called after every theme switch (light<->dark). Widget
        stylesheets and plot canvases are already re-colored automatically
        by the theme system itself (see ui/theme.py); this hook exists for
        the rare page-specific extra some pages register via
        page.apply_theme(), for anything beyond generic stylesheet/canvas
        recoloring (e.g. re-running a page's own redraw method so its most
        recently plotted data reflects fresh chrome immediately rather
        than waiting for the next natural redraw)."""
        for page in self.pages.values():
            apply_fn = getattr(page, "apply_theme", None)
            if callable(apply_fn):
                try:
                    apply_fn()
                except Exception:
                    pass

    def _on_logout(self):
        self._logout_callback()
        self.close()
