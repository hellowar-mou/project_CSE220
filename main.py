import sys
from PySide6.QtWidgets import QApplication

from ui.login_window import LoginWindow
from ui.main_window import MainWindow


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
    app = QApplication(sys.argv)

    with open("ui/theme.qss") as f:
        app.setStyleSheet(f.read())

    controller = AppController()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
