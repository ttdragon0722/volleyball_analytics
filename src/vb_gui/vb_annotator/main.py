import os
import sys
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow
from ui.theme import build_stylesheet

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(build_stylesheet())
    # <repo>/db/annotations.db by default; override with VB_ANNOTATIONS_DB
    default_db = Path(__file__).resolve().parents[3] / "db" / "annotations.db"
    db_path = Path(os.environ.get("VB_ANNOTATIONS_DB", default_db))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_save_path = db_path.as_posix()
    window = MainWindow(db_path=db_save_path)
    window.show()
    # Ask to resume once the window is on screen (the dialog needs a visible parent).
    QTimer.singleShot(0, window.offer_resume)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()