import os
import time

import requests
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models.message import Message


BACKEND_URL = os.getenv("MOODNEWS_BACKEND_URL", "http://127.0.0.1:8000/chat")


class BackendWorker(QThread):
    response_ready = Signal(str)

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.text = text

    def run(self):
        payload = {"text": self.text}
        for attempt in range(2):
            try:
                response = requests.post(BACKEND_URL, json=payload, timeout=15)
                if response.status_code == 503 and attempt == 0:
                    time.sleep(2)
                    continue
                if response.ok:
                    data = response.json()
                    self.response_ready.emit(
                        str(data.get("response", "Sem resposta do servidor."))
                    )
                else:
                    try:
                        detail = response.json().get("detail", "")
                    except (ValueError, AttributeError):
                        detail = ""
                    self.response_ready.emit(
                        str(detail)
                        if detail
                        else f"Erro {response.status_code}: servidor não respondeu corretamente."
                    )
                return
            except requests.Timeout:
                self.response_ready.emit(
                    "O backend demorou mais que o esperado para responder. "
                    f"Verifique se a API está ativa em {BACKEND_URL}."
                )
                return
            except requests.RequestException as error:
                self.response_ready.emit(f"Erro ao conectar com o backend: {error}")
                return
            except (ValueError, TypeError) as error:
                self.response_ready.emit(f"Resposta inválida do backend: {error}")
                return


class MessageInput(QPlainTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not event.modifiers() & Qt.ShiftModifier:
            event.accept()
            self.send_requested.emit()
            return
        super().keyPressEvent(event)


class HomeScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoodNews")
        self.resize(920, 760)
        self.setMinimumSize(560, 520)
        self._messages = []
        self._worker = None
        self._build_ui()
        self._apply_styles()
        self._add_message(
            Message.create(
                "Olá! Eu sou o MoodNews, seu classificador de notícias. "
                "Envie uma notícia para descobrir seu impacto ambiental.",
                False,
            )
        )

    def set_backend_status(self, status: str):
        self.connection_label.setText(status)

    def set_backend_ready(self, ready: bool = True):
        self.input.setEnabled(ready)
        self.send_button.setEnabled(ready)
        self.connection_label.setText("API e modelos prontos" if ready else "API indisponível")

    def set_backend_error(self, error: str):
        self.set_backend_ready(False)
        self._add_message(Message.create(f"Não foi possível iniciar o MoodNews: {error}", False))

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 22, 28, 18)
        header_layout.setSpacing(14)

        mark = QLabel("M")
        mark.setObjectName("mark")
        mark.setAlignment(Qt.AlignCenter)
        mark.setFixedSize(46, 46)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(2)
        title = QLabel("MoodNews")
        title.setObjectName("title")
        subtitle = QLabel("Seu assistente de notícias ambientais")
        subtitle.setObjectName("subtitle")
        title_stack.addWidget(title)
        title_stack.addWidget(subtitle)
        header_layout.addWidget(mark)
        header_layout.addLayout(title_stack)
        header_layout.addStretch()
        self.connection_label = QLabel("API · localhost:8000")
        self.connection_label.setObjectName("connection")
        header_layout.addWidget(self.connection_label, alignment=Qt.AlignVCenter)
        layout.addWidget(header)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("conversation")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.message_container = QWidget()
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setContentsMargins(28, 20, 28, 20)
        self.message_layout.setSpacing(14)
        self.message_layout.addStretch()
        self.scroll_area.setWidget(self.message_container)
        layout.addWidget(self.scroll_area, 1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(28, 16, 28, 22)
        composer_layout.setSpacing(12)
        self.input = MessageInput()
        self.input.setObjectName("messageInput")
        self.input.setPlaceholderText("Cole ou digite uma notícia para classificar...")
        self.input.setFixedHeight(74)
        self.input.send_requested.connect(self._send_message)
        composer_layout.addWidget(self.input, 1)
        self.send_button = QPushButton("Analisar")
        self.send_button.setObjectName("sendButton")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setFixedSize(112, 48)
        self.send_button.clicked.connect(self._send_message)
        composer_layout.addWidget(self.send_button, alignment=Qt.AlignBottom)
        layout.addWidget(composer)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, #root { background: #101918; }
            #header { background: #172321; border-bottom: 1px solid #293834; }
            #mark { background: #d6f36a; color: #172321; border-radius: 14px;
                    font-size: 23px; font-weight: 800; }
            #title { color: #f0f4ec; font-size: 20px; font-weight: 700; }
            #subtitle { color: #9eada4; font-size: 12px; }
            #connection { color: #b8ca9d; background: #25332b; border-radius: 10px;
                          padding: 8px 11px; font-size: 11px; }
            #conversation { background: #101918; }
            #composer { background: #172321; border-top: 1px solid #293834; }
            #messageInput { background: #202d29; color: #edf2e9; border: 1px solid #34433c;
                            border-radius: 12px; padding: 13px 15px; font-size: 14px; }
            #messageInput:focus { border: 1px solid #d6f36a; }
            #messageInput::placeholder { color: #84948a; }
            #sendButton { background: #d6f36a; color: #172321; border: 0;
                          border-radius: 12px; font-size: 13px; font-weight: 700; }
            #sendButton:hover { background: #e2ff7e; }
            #sendButton:disabled { background: #3c4840; color: #89958a; }
            #assistantBubble { background: #202d29; border: 1px solid #34433c;
                               border-radius: 14px; }
            #userBubble { background: #34482f; border: 1px solid #526344;
                          border-radius: 14px; }
            #avatar { color: #d6f36a; font-size: 11px; font-weight: 700; }
            #messageText { color: #edf2e9; font-size: 14px; }
            #timestamp { color: #9eada4; font-size: 10px; }
            #copyButton { color: #b8ca9d; background: transparent; border: 0;
                          font-size: 10px; padding: 2px 5px; }
            #copyButton:hover { color: #e2ff7e; }
            #typing { color: #b8ca9d; font-size: 12px; padding: 6px 2px; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 3px; }
            QScrollBar::handle:vertical { background: #3b4a42; border-radius: 4px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def _add_message(self, message: Message):
        self._messages.append(message)
        row = QHBoxLayout()
        row.setSpacing(10)
        bubble = QFrame()
        bubble.setObjectName("userBubble" if message.is_user else "assistantBubble")
        bubble.setMaximumWidth(620)
        bubble.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(15, 12, 13, 9)
        bubble_layout.setSpacing(8)

        text = QLabel(message.content)
        text.setObjectName("messageText")
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text.setMaximumWidth(570)
        bubble_layout.addWidget(text)

        details = QHBoxLayout()
        timestamp = QLabel(message.timestamp.strftime("%H:%M"))
        timestamp.setObjectName("timestamp")
        details.addWidget(timestamp)
        details.addStretch()
        copy_button = QPushButton("Copiar")
        copy_button.setObjectName("copyButton")
        copy_button.setCursor(Qt.PointingHandCursor)
        copy_button.clicked.connect(
            lambda checked=False, value=message.content: self._copy_message(value)
        )
        details.addWidget(copy_button)
        bubble_layout.addLayout(details)

        if message.is_user:
            row.addStretch()
            row.addWidget(bubble)
            avatar = QLabel("EU")
            avatar.setObjectName("avatar")
            row.addWidget(avatar, alignment=Qt.AlignBottom)
        else:
            avatar = QLabel("MN")
            avatar.setObjectName("avatar")
            row.addWidget(avatar, alignment=Qt.AlignBottom)
            row.addWidget(bubble)
            row.addStretch()

        self.message_layout.insertLayout(self.message_layout.count() - 1, row)
        QApplication.processEvents()
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def _copy_message(self, text: str):
        QApplication.clipboard().setText(text)
        self.connection_label.setText("Mensagem copiada")

    def _send_message(self):
        text = self.input.toPlainText().strip()
        if not text or self._worker is not None:
            return
        self.input.clear()
        self._add_message(Message.create(text, True))
        self.send_button.setEnabled(False)
        self.send_button.setText("Aguarde")
        self.typing_label = QLabel("MoodNews está analisando...")
        self.typing_label.setObjectName("typing")
        self.message_layout.insertWidget(self.message_layout.count() - 1, self.typing_label)
        self._worker = BackendWorker(text, self)
        self._worker.response_ready.connect(self._show_response)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _show_response(self, response: str):
        if hasattr(self, "typing_label"):
            self.message_layout.removeWidget(self.typing_label)
            self.typing_label.deleteLater()
            del self.typing_label
        self._add_message(Message.create(response, False))

    def _worker_finished(self):
        self._worker = None
        self.send_button.setEnabled(True)
        self.send_button.setText("Analisar")
        self.input.setFocus()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        event.accept()