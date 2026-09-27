import sys

from PySide6.QtWidgets import QApplication

from backend_manager import BackendStartupWorker
from screens.home_screen import HomeScreen


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MoodNews")
    app.setStyle("Fusion")

    window = HomeScreen()
    window.set_backend_status("Iniciando API e modelos...")
    window.set_backend_ready(False)

    backend = BackendStartupWorker()
    backend.status_changed.connect(window.set_backend_status)
    backend.ready.connect(window.set_backend_ready)
    backend.failed.connect(window.set_backend_error)
    app.aboutToQuit.connect(backend.stop_backend)

    window.show()
    backend.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()