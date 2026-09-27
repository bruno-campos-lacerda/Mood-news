import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PySide6.QtCore import QThread, Signal


BACKEND_URL = os.getenv("MOODNEWS_BACKEND_URL", "http://127.0.0.1:8000/chat")
HEALTH_URL = BACKEND_URL.rsplit("/chat", 1)[0] + "/health"
BACKEND_DIR = Path(__file__).resolve().parents[1] / "Backend"


class BackendStartupWorker(QThread):
    status_changed = Signal(str)
    ready = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self._stopping = threading.Event()

    def run(self):
        hostname = urlparse(BACKEND_URL).hostname
        local_backend = hostname in ("localhost", "127.0.0.1", "::1")

        if local_backend and not self._api_responds():
            self.status_changed.emit("Iniciando API local...")
            try:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(urlparse(BACKEND_URL).port or 8000),
                    ],
                    cwd=BACKEND_DIR,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as error:
                self.failed.emit(f"Não foi possível iniciar a API: {error}")
                return

        deadline = time.monotonic() + 120
        while not self._stopping.is_set() and time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.failed.emit(
                    "A API foi encerrada ao iniciar. Confira as dependências do backend."
                )
                return

            try:
                response = requests.get(HEALTH_URL, timeout=2)
                if response.ok:
                    data = response.json()
                    if data.get("models_loaded"):
                        self.ready.emit()
                        return
                    if data.get("status") == "error":
                        self.failed.emit(
                            data.get("error") or "Não foi possível carregar os modelos de IA."
                        )
                        return
                    self.status_changed.emit("Carregando modelos de IA...")
            except (requests.RequestException, ValueError):
                pass

            self.msleep(500)

        if not self._stopping.is_set():
            self.failed.emit("A API ou os modelos não responderam dentro do tempo esperado.")

    @staticmethod
    def _api_responds():
        try:
            response = requests.get(HEALTH_URL, timeout=1)
            return response.ok
        except requests.RequestException:
            return False

    def stop_backend(self):
        self._stopping.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.isRunning():
            self.wait(5000)