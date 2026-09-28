import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend_manager import ModelStartupWorker
from screens.home_screen import HomeScreen


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MoodNews")
    app.setStyle("Fusion")

    window = HomeScreen()
    window.set_model_status("Carregando modelos...")

    model_worker = ModelStartupWorker()
    model_worker.ready.connect(window.set_model_service)
    model_worker.failed.connect(window.set_model_error)

    window.show()
    model_worker.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()