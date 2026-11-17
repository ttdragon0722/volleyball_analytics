"""Vertical-only scroll container for the side panels.

Without it a tall sidebar forces the window's minimum height (and so the window cannot fit
on a 1536x960 logical screen, e.g. 1920x1200 at 125% scaling). The container keeps the
panel's own width — including width changes the panel makes itself, like the left sidebar
resizing to the active tab — and only adds a scrollbar when the screen is too short.
"""
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QFrame, QScrollArea, QWidget


class SidebarScroll(QScrollArea):
    def __init__(self, panel: QWidget, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWidget(panel)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Let the panel's own themed background show through.
        self.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.viewport().setAutoFillBackground(False)
        self.setMinimumHeight(0)
        panel.installEventFilter(self)
        self.verticalScrollBar().installEventFilter(self)
        self._sync_width()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.Hide):
            self._sync_width()
        return super().eventFilter(obj, event)

    def _sync_width(self) -> None:
        panel = self.widget()
        if panel is None:
            return
        # The panels set a fixed width themselves (minimumWidth == their width).
        bar = self.verticalScrollBar()
        extra = bar.sizeHint().width() if bar.isVisible() else 0
        self.setFixedWidth(panel.minimumWidth() + extra)
