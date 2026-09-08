import sys
import logging
import traceback
from PySide6.QtWidgets import QApplication

from ui import theme as theme_module
theme_module.install_theme_patch()  # must happen before any QWidget exists

from ui.login_window import LoginWindow
from ui.main_window import MainWindow


def _install_crash_logging():
    logging.basicConfig(
        filename="app_crash.log",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.critical(
            "Unhandled application exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


class AppController:
    """Owns window references so login -> main-window handoff doesn't get
    garbage-collected, and handles logout -> back-to-login."""

    def __init__(self):
        self.login_window = None
        self.main_window = None
        self.show_login()

    def show_login(self):
        self.main_window = None
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.show_main_window)
        self.login_window.show()

    def show_main_window(self, username):
        self.login_window.close()
        self.login_window = None
        self.main_window = MainWindow(username, logout_callback=self.show_login)
        self.main_window.show()


def main():
    _install_crash_logging()
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # Load the previously chosen theme (defaults to light) before any
    # widget is created, so the very first frame already matches it.
    saved_mode = theme_module.load_saved_theme()
    theme_module.set_current_mode(saved_mode)

    with open("ui/theme.qss") as f:
        app.setStyleSheet(f.read())  # registered+recolored transparently if mode is dark

    controller = AppController()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
