from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTabWidget, QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from ui.wave_widget import WaveBackground
import auth_db


class _Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        self.setObjectName("Card")
        self.setStyleSheet("QFrame#Card { background: rgba(255,255,255,0.85); border-radius: 22px; }")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(80, 120, 180, 60))
        self.setGraphicsEffect(shadow)


class LoginWindow(QWidget):
    """Emits login_successful(username) when a user authenticates."""
    login_successful = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReSonus — Sign in")
        self.resize(980, 640)

        # ---- layered background ----
        self.bg = WaveBackground(self)
        self.bg.setGeometry(self.rect())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = _Card()
        card.setFixedWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 36, 36, 28)
        card_layout.setSpacing(10)

        logo = QLabel("🌊")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet("font-size: 34px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                            "stop:0 #8ec5fc, stop:1 #a7e3e0); border-radius: 31px; "
                            "min-width: 62px; max-width: 62px; min-height: 62px; max-height: 62px;")
        logo_row = QHBoxLayout()
        logo_row.addStretch()
        logo_row.addWidget(logo)
        logo_row.addStretch()
        card_layout.addLayout(logo_row)

        title = QLabel("ReSonus")
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Signals & Linear Systems — Audio DSP Suite")
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_signin_tab(), "Sign in")
        self.tabs.addTab(self._build_signup_tab(), "Create account")
        card_layout.addWidget(self.tabs)

        hint = QLabel("Demo login — Username: <b>signals</b> &nbsp;|&nbsp; Password: <b>lti2026</b>")
        hint.setObjectName("HintLabel")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        card_layout.addSpacing(6)
        card_layout.addWidget(hint)

        center_row = QHBoxLayout()
        center_row.addStretch()
        center_row.addWidget(card)
        center_row.addStretch()

        outer.addStretch(1)
        outer.addLayout(center_row)
        outer.addStretch(2)

    def resizeEvent(self, event):
        self.bg.setGeometry(self.rect())
        super().resizeEvent(event)

    # -------------------------------------------------------------- Sign in
    def _build_signin_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        self.login_user = QLineEdit()
        self.login_user.setPlaceholderText("Username")
        self.login_pass = QLineEdit()
        self.login_pass.setPlaceholderText("Password")
        self.login_pass.setEchoMode(QLineEdit.Password)
        self.login_pass.returnPressed.connect(self._handle_login)

        self.login_error = QLabel("")
        self.login_error.setObjectName("ErrorLabel")
        self.login_error.setWordWrap(True)
        self.login_error.hide()

        btn = QPushButton("Sign in")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._handle_login)

        layout.addWidget(self.login_user)
        layout.addWidget(self.login_pass)
        layout.addWidget(self.login_error)
        layout.addWidget(btn)
        return w

    def _handle_login(self):
        username = self.login_user.text().strip()
        password = self.login_pass.text()
        if not username or not password:
            self._show_login_error("Enter both username and password.")
            return
        if auth_db.verify_login(username, password):
            self.login_error.hide()
            self.login_successful.emit(username)
        else:
            self._show_login_error("Username or password is incorrect.")

    def _show_login_error(self, msg):
        self.login_error.setText(msg)
        self.login_error.show()

    # ------------------------------------------------------------ Sign up
    def _build_signup_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        self.su_user = QLineEdit()
        self.su_user.setPlaceholderText("Choose a username")
        self.su_email = QLineEdit()
        self.su_email.setPlaceholderText("Email address")
        self.su_pass = QLineEdit()
        self.su_pass.setPlaceholderText("Password (min 6 characters)")
        self.su_pass.setEchoMode(QLineEdit.Password)
        self.su_pass2 = QLineEdit()
        self.su_pass2.setPlaceholderText("Confirm password")
        self.su_pass2.setEchoMode(QLineEdit.Password)
        self.su_pass2.returnPressed.connect(self._handle_signup)

        self.su_msg = QLabel("")
        self.su_msg.setWordWrap(True)
        self.su_msg.hide()

        btn = QPushButton("Create account")
        btn.setObjectName("PrimaryButton")
        btn.clicked.connect(self._handle_signup)

        layout.addWidget(self.su_user)
        layout.addWidget(self.su_email)
        layout.addWidget(self.su_pass)
        layout.addWidget(self.su_pass2)
        layout.addWidget(self.su_msg)
        layout.addWidget(btn)
        return w

    def _handle_signup(self):
        try:
            auth_db.register_user(
                self.su_user.text(), self.su_email.text(),
                self.su_pass.text(), self.su_pass2.text()
            )
            self.su_msg.setObjectName("SuccessLabel")
            self.su_msg.setStyleSheet("")  # re-apply stylesheet class
            self.su_msg.setText("Account created! Switch to the Sign in tab.")
            self.su_msg.show()
            self.su_user.clear(); self.su_email.clear()
            self.su_pass.clear(); self.su_pass2.clear()
        except auth_db.AuthError as e:
            self.su_msg.setObjectName("ErrorLabel")
            self.su_msg.setText(str(e))
            self.su_msg.show()
        self._repolish(self.su_msg)

    def _repolish(self, widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
