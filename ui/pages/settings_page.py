from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton

from ui import theme as theme_module


class SettingsPage(QWidget):
    """Global app settings. Currently: dark/light theme toggle.

    Toggling here switches the ENTIRE running app immediately — every page
    already built, not just this one — via ui.theme.set_current_mode(),
    which re-applies every registered widget's stylesheet and recolors
    every registered plot canvas app-wide."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QFrame()
        header.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        h_layout = QVBoxLayout(header)
        title = QLabel("⚙️ Settings")
        title.setObjectName("SectionTitle")
        caption = QLabel("App-wide preferences.")
        caption.setObjectName("Caption")
        h_layout.addWidget(title)
        h_layout.addWidget(caption)
        layout.addWidget(header)

        theme_card = QFrame()
        theme_card.setStyleSheet("QFrame { background: rgba(255,255,255,0.78); border-radius: 14px; }")
        tc_layout = QVBoxLayout(theme_card)
        theme_title = QLabel("Appearance")
        theme_title.setObjectName("SectionTitle")
        tc_layout.addWidget(theme_title)

        theme_desc = QLabel("Choose Light or Dark. Applies immediately across the whole "
                             "application — every page, every plot, every panel.")
        theme_desc.setObjectName("Caption")
        theme_desc.setWordWrap(True)
        tc_layout.addWidget(theme_desc)

        btn_row = QHBoxLayout()
        self.light_btn = QPushButton("☀️  Light")
        self.dark_btn = QPushButton("🌙  Dark")
        for b in (self.light_btn, self.dark_btn):
            b.setObjectName("SecondaryButton")
            b.setCheckable(True)
            b.setMinimumWidth(120)
        self.light_btn.clicked.connect(lambda: self._set_theme("light"))
        self.dark_btn.clicked.connect(lambda: self._set_theme("dark"))
        btn_row.addWidget(self.light_btn)
        btn_row.addWidget(self.dark_btn)
        btn_row.addStretch()
        tc_layout.addLayout(btn_row)

        layout.addWidget(theme_card)
        layout.addStretch()

        theme_module.add_listener(self._sync_buttons)
        self._sync_buttons()

    def _sync_buttons(self):
        mode = theme_module.get_current_mode()
        self.light_btn.setChecked(mode == "light")
        self.dark_btn.setChecked(mode == "dark")
        # Re-polish so the :checked pseudo-state style actually repaints —
        # Qt doesn't always repaint checkable-button state changes made
        # from code without this nudge.
        for b in (self.light_btn, self.dark_btn):
            b.style().unpolish(b)
            b.style().polish(b)

    def _set_theme(self, mode):
        theme_module.set_current_mode(mode)
        theme_module.save_theme(mode)
