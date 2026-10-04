import sqlite3
from datetime import datetime

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from chat_history import ChatHistoryRepository, ChatResult


BACKGROUND = QColor("#1a1b26")
TEXT_COLOR = QColor("#c0caf5")
MUTED_COLOR = QColor("#a9b1d6")
GRID_COLOR = QColor("#414868")
BAD_COLOR = QColor("#7aa2f7")
GOOD_COLOR = QColor("#9ece6a")


class ResultChart(QWidget):
    ITEM_WIDTH = 132
    CHART_TOP = 54
    BASELINE = 330
    BAR_WIDTH = 30
    SCALE_HEIGHT = 240

    def __init__(self, results: list[ChatResult], parent=None):
        super().__init__(parent)
        self.results = results
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMinimumHeight(440)
        self.setMouseTracking(True)

    def set_results(self, results: list[ChatResult]):
        self.results = results
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        return QSize(max(760, self.ITEM_WIDTH * len(self.results) + 56), 440)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        painter.setFont(self.font())

        for tick in (0, 25, 50, 75, 100):
            y = self.BASELINE - round(self.SCALE_HEIGHT * tick / 100)
            painter.setPen(QPen(GRID_COLOR, 1, Qt.DashLine))
            painter.drawLine(48, y, self.width() - 16, y)
            painter.setPen(MUTED_COLOR)
            painter.drawText(8, y + 5, f"{tick}%")

        if not self.results:
            painter.setPen(MUTED_COLOR)
            painter.drawText(self.rect(), Qt.AlignCenter, "Ainda não há resultados no histórico.")
            return

        for index, result in enumerate(self.results):
            center = 82 + index * self.ITEM_WIDTH
            bad_height = round(self.SCALE_HEIGHT * result.negative_probability)
            good_height = round(self.SCALE_HEIGHT * result.positive_probability)
            bad_x = center - self.BAR_WIDTH - 3
            good_x = center + 3

            painter.setPen(Qt.NoPen)
            painter.setBrush(BAD_COLOR)
            painter.drawRoundedRect(
                bad_x,
                self.BASELINE - bad_height,
                self.BAR_WIDTH,
                bad_height,
                5,
                5,
            )
            painter.setBrush(GOOD_COLOR)
            painter.drawRoundedRect(
                good_x,
                self.BASELINE - good_height,
                self.BAR_WIDTH,
                good_height,
                5,
                5,
            )

            painter.setPen(TEXT_COLOR)
            painter.drawText(
                bad_x - 4,
                self.BASELINE - bad_height - 8,
                self.BAR_WIDTH + 8,
                18,
                Qt.AlignCenter,
                f"{result.negative_probability:.0%}",
            )
            painter.drawText(
                good_x - 4,
                self.BASELINE - good_height - 8,
                self.BAR_WIDTH + 8,
                18,
                Qt.AlignCenter,
                f"{result.positive_probability:.0%}",
            )

            timestamp = datetime.fromisoformat(result.created_at).astimezone()
            painter.setPen(MUTED_COLOR)
            painter.drawText(
                center - 48,
                self.BASELINE + 27,
                96,
                18,
                Qt.AlignCenter,
                timestamp.strftime("%d/%m %H:%M"),
            )
            painter.setPen(TEXT_COLOR)
            painter.drawText(
                center - 50,
                self.BASELINE + 49,
                100,
                34,
                Qt.AlignHCenter | Qt.TextWordWrap,
                f"Notícia {index + 1}",
            )
    def mouseMoveEvent(self, event: QMouseEvent):
        index = round((event.position().x() - 82) / self.ITEM_WIDTH)
        if 0 <= index < len(self.results):
            result = self.results[index]
            timestamp = datetime.fromisoformat(result.created_at).astimezone()
            tooltip = (
                f"{result.news}\n\n"
                f"Ruim: {result.negative_probability:.0%} | "
                f"Boa: {result.positive_probability:.0%}\n"
                f"{timestamp.strftime('%d/%m/%Y %H:%M:%S')}"
            )
            QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
        else:
            QToolTip.hideText()


class ChatHistoryChartWindow(QMainWindow):
    def __init__(self, repository: ChatHistoryRepository, parent=None):
        super().__init__(parent)
        self.repository = repository
        self.setWindowTitle("Histórico de resultados do chatbot")
        self.resize(900, 540)
        self.setMinimumSize(560, 420)

        root = QWidget()
        root.setStyleSheet("background: #1a1b26; color: #c0caf5;")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)

        title = QLabel("Histórico de notícias analisadas")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        legend = QLabel(
            '<span style="color:#7aa2f7">■</span> Notícia ruim'
            '&nbsp;&nbsp;&nbsp;'
            '<span style="color:#9ece6a">■</span> Notícia boa'
        )
        legend.setStyleSheet("font-size: 13px;")
        layout.addWidget(legend)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #f7768e;")
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background: #1a1b26; }
            QScrollBar:horizontal { background: #1a1b26; height: 10px; margin: 2px; }
            QScrollBar::handle:horizontal { background: #414868; border-radius: 5px; min-width: 30px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)
        self.chart = ResultChart([])
        self.scroll_area.setWidget(self.chart)
        layout.addWidget(self.scroll_area, 1)
        self.setCentralWidget(root)
        self.refresh()

    def refresh(self):
        try:
            results = self.repository.list_all()
        except (OSError, sqlite3.Error) as error:
            self.status_label.setText(f"Não foi possível carregar o histórico: {error}")
            self.status_label.show()
            return
        self.status_label.hide()
        self.chart.set_results(results)
        self.chart.resize(self.chart.sizeHint())
