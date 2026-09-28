from PySide6.QtCore import QThread, Signal

from Backend.service import MoodNewsService


class ModelStartupWorker(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.ready.emit(MoodNewsService.load())
        except Exception as error:
            self.failed.emit(str(error))
