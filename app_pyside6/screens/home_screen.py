from pathlib import Path
import sqlite3

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent, QPixmap
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from Backend.service import MoodNewsService
from chat_history import ChatHistoryRepository
from models.message import Message
from screens.chat_history_chart import ChatHistoryChartWindow


CHART_PATH = (
    Path(__file__).resolve().parents[2]
    / "Backend"
    / "desempenho"
    / "resultados"
    / "noticias_por_dia.png"
)


class ClassificationWorker(QThread):
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, service: MoodNewsService, text: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.text = text

    def run(self):
        try:
            self.result_ready.emit((self.text, self.service.classify_result(self.text)))
        except Exception as error:
            self.failed.emit(str(error))


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
        self._model_service = None
        self._chart_pixmap = QPixmap()
        self._history_repository = None
        self._history_window = None
        self._history_error = None
        self._build_ui()
        self._apply_styles()
        try:
            self._history_repository = ChatHistoryRepository()
        except (OSError, sqlite3.Error) as error:
            self._history_error = str(error)
            self.connection_label.setText("Não foi possível abrir o histórico")
        self._add_message(
            Message.create(
                "Olá! Eu sou o MoodNews, seu classificador de notícias. "
                "Envie uma notícia para descobrir seu impacto ambiental.",
                False,
            )
        )

    def set_model_status(self, status: str):
        self.connection_label.setText(status)

    def set_model_service(self, service: MoodNewsService):
        self._model_service = service
        self.send_button.setEnabled(True)
        self.connection_label.setText(
            "Modelos prontos · Histórico indisponível"
            if self._history_error
            else "Modelos locais prontos"
        )

    def set_model_error(self, error: str):
        self._model_service = None
        self.send_button.setEnabled(True)
        self.connection_label.setText("Erro ao carregar modelos")
        self._add_message(Message.create(f"Não foi possível carregar os modelos: {error}", False))

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
        self.connection_label = QLabel("Carregando modelos locais...")
        self.connection_label.setObjectName("connection")
        header_layout.addWidget(self.connection_label, alignment=Qt.AlignVCenter)
        self.chat_button = QPushButton("Chat")
        self.chat_button.setObjectName("navigationButton")
        self.chat_button.clicked.connect(self._show_chat)
        self.chat_button.hide()
        self.chart_button = QPushButton("Historico de treinamento")
        self.chart_button.setObjectName("navigationButton")
        self.chart_button.clicked.connect(self._show_chart)
        header_layout.addWidget(self.chat_button)
        header_layout.addWidget(self.chart_button)
        self.history_button = QPushButton("Histórico do chat")
        self.history_button.setObjectName("navigationButton")
        self.history_button.clicked.connect(self._open_history_chart)
        header_layout.addWidget(self.history_button)
        layout.addWidget(header)

        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        chat_page = QWidget()
        chat_layout = QVBoxLayout(chat_page)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)
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
        chat_layout.addWidget(self.scroll_area, 1)

        composer = QFrame()
        composer.setObjectName("composer")
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(28, 16, 28, 22)
        composer_layout.setSpacing(12)
        self.input = MessageInput()
        self.input.setObjectName("messageInput")
        self.input.setPlaceholderText("Cole ou digite uma notícia para classificar...")
        self.input.setFixedHeight(96)
        self.input.send_requested.connect(self._send_message)
        composer_layout.addWidget(self.input, 1)
        self.send_button = QPushButton("Analisar")
        self.send_button.setObjectName("sendButton")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setFixedSize(112, 48)
        self.send_button.clicked.connect(self._send_message)
        composer_layout.addWidget(self.send_button, alignment=Qt.AlignBottom)
        chat_layout.addWidget(composer)
        self.pages.addWidget(chat_page)

        self.chart_scroll_area = QScrollArea()
        self.chart_scroll_area.setObjectName("conversation")
        self.chart_scroll_area.setWidgetResizable(True)
        self.chart_scroll_area.setFrameShape(QFrame.NoFrame)
        self.chart_image_label = QLabel("O gráfico será gerado ao abrir esta tela.")
        self.chart_image_label.setObjectName("chartImage")
        self.chart_image_label.setAlignment(Qt.AlignCenter)
        self.chart_scroll_area.setWidget(self.chart_image_label)
        self.pages.addWidget(self.chart_scroll_area)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, #root { background: #1a1b26; }
            #header { background: #1f2335; border-bottom: 1px solid #292e42; }
            #mark { background: #7aa2f7; color: #1a1b26; border-radius: 14px;
                    font-size: 23px; font-weight: 800; }
            #title { color: #c0caf5; font-size: 20px; font-weight: 700; }
            #subtitle { color: #a9b1d6; font-size: 12px; }
            #connection { color: #7dcfff; background: #24283b; border-radius: 10px;
                          padding: 8px 11px; font-size: 11px; }
            #conversation { background: #1a1b26; }
            #composer { background: #1f2335; border-top: 1px solid #292e42; }
            #messageInput { background: #24283b; color: #c0caf5; border: 1px solid #414868;
                            border-radius: 12px; padding: 13px 15px; font-size: 14px; }
            #messageInput:focus { border: 1px solid #7aa2f7; }
            #messageInput::placeholder { color: #565f89; }
            #sendButton { background: #7aa2f7; color: #1a1b26; border: 0;
                          border-radius: 12px; font-size: 13px; font-weight: 700; }
            #sendButton:hover { background: #89b4fa; }
            #sendButton:disabled { background: #414868; color: #a9b1d6; }
            #assistantBubble { background: #24283b; border: 1px solid #414868;
                               border-radius: 14px; }
            #userBubble { background: #283457; border: 1px solid #3d59a1;
                          border-radius: 14px; }
            #avatar { color: #bb9af7; font-size: 11px; font-weight: 700; }
            #messageText { color: #c0caf5; font-size: 14px; }
            #timestamp { color: #a9b1d6; font-size: 10px; }
            #copyButton { color: #7dcfff; background: transparent; border: 0;
                          font-size: 10px; padding: 2px 5px; }
            #copyButton:hover { color: #bb9af7; }
            #typing { color: #7dcfff; font-size: 12px; padding: 6px 2px; }
            #navigationButton { background: #24283b; color: #c0caf5;
                                border: 1px solid #414868; border-radius: 10px;
                                padding: 8px 12px; font-size: 12px; font-weight: 600; }
            #navigationButton:hover { background: #283457; border-color: #7aa2f7; }
            #chartImage { background: #1a1b26; color: #a9b1d6; font-size: 14px; }
            QScrollBar:vertical { background: #1a1b26; width: 8px; margin: 3px; }
            QScrollBar::handle:vertical { background: #414868; border-radius: 4px; min-height: 28px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)

    def _show_page(self, index: int):
        self.pages.setCurrentIndex(index)

    def _show_chat(self):
        self.pages.setCurrentIndex(0)
        self.chat_button.hide()
        self.chart_button.show()

    def _show_chart(self):
        self.pages.setCurrentIndex(1)
        self.chart_button.hide()
        self.chat_button.show()
        self._chart_pixmap = QPixmap(str(CHART_PATH))
        if self._chart_pixmap.isNull():
            self.chart_image_label.setText(
                f"Não foi possível carregar o gráfico pronto:\n{CHART_PATH}"
            )
            return
        self._scale_chart_image()

    def _open_history_chart(self):
        if self._history_repository is None:
            self.connection_label.setText(
                f"Histórico indisponível: {self._history_error or 'erro no banco de dados'}"
            )
            return
        if self._history_window is None:
            self._history_window = ChatHistoryChartWindow(
                self._history_repository,
                self,
            )
        else:
            self._history_window.refresh()
        self._history_window.show()
        self._history_window.raise_()
        self._history_window.activateWindow()

    def _scale_chart_image(self):
        if self._chart_pixmap.isNull():
            return
        size = self.chart_scroll_area.viewport().size()
        self.chart_image_label.setPixmap(
            self._chart_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pages.currentIndex() == 1:
            self._scale_chart_image()

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
        if self._model_service is None:
            self._add_message(
                Message.create(
                    "Os modelos de IA não estão disponíveis. Verifique os arquivos .pkl "
                    "na pasta Backend/models e reinicie o aplicativo.",
                    False,
                )
            )
            return
        self.send_button.setEnabled(False)
        self.send_button.setText("Aguarde")
        self.typing_label = QLabel("MoodNews está analisando...")
        self.typing_label.setObjectName("typing")
        self.message_layout.insertWidget(self.message_layout.count() - 1, self.typing_label)
        self._worker = ClassificationWorker(self._model_service, text, self)
        self._worker.result_ready.connect(self._show_response)
        self._worker.failed.connect(self._show_classification_error)
        self._worker.finished.connect(self._worker_finished)
        self._worker.start()

    def _show_response(self, classification):
        news, result = classification
        if hasattr(self, "typing_label"):
            self.message_layout.removeWidget(self.typing_label)
            self.typing_label.deleteLater()
            del self.typing_label
        self._add_message(Message.create(result.response, False))
        if self._history_repository is None:
            return
        try:
            self._history_repository.add(
                news=news,
                positive_probability=result.positive_probability,
                negative_probability=result.negative_probability,
            )
        except (OSError, sqlite3.Error) as error:
            self.connection_label.setText(f"Falha ao salvar histórico: {error}")
            self._add_message(
                Message.create(
                    "A análise foi concluída, mas não foi possível salvá-la no histórico.",
                    False,
                )
            )
            return
        if self._history_window is not None:
            self._history_window.refresh()

    def _show_classification_error(self, error: str):
        if hasattr(self, "typing_label"):
            self.message_layout.removeWidget(self.typing_label)
            self.typing_label.deleteLater()
            del self.typing_label
        self._add_message(Message.create(f"Não foi possível analisar a notícia: {error}", False))

    def _worker_finished(self):
        self._worker = None
        self.send_button.setEnabled(True)
        self.send_button.setText("Analisar")
        self.input.setFocus()

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
        event.accept()