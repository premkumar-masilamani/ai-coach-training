import sys
import threading
import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from transcriber.preprocessing.audio_preprocessor import preprocess_audio
from transcriber.transcription.transcriber import Transcriber
from transcriber.utils.constants import AI_MODEL_WHISPER_CPP_DEFAULT_MODEL
from transcriber.utils.constants import WHISPER_CPP_LOCAL_BIN
from transcriber.utils.constants import WHISPER_CPP_PATH
from transcriber.utils.file_util import audio_extensions, save_transcript_as_text

logger = logging.getLogger(__name__)


class LogEmitter(QtCore.QObject):
    message = QtCore.Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: LogEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.emitter.message.emit(msg)


class AnimatedProgressBar(QtWidgets.QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._animated = False
        self._stripe_offset = 0

    def setAnimated(self, animated: bool):
        self._animated = animated
        self.update()

    def advancePattern(self):
        if not self._animated:
            return
        self._stripe_offset = (self._stripe_offset + 2) % 18
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._animated:
            return

        min_val = self.minimum()
        max_val = self.maximum()
        if max_val <= min_val:
            return

        ratio = (self.value() - min_val) / (max_val - min_val)
        if ratio <= 0:
            return

        inner = self.rect().adjusted(2, 2, -2, -2)
        chunk_width = int(inner.width() * ratio)
        if chunk_width <= 0:
            return

        chunk = QtCore.QRect(inner.left(), inner.top(), chunk_width, inner.height())
        painter = QtGui.QPainter(self)
        painter.setClipRect(chunk)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 85), 2)
        painter.setPen(pen)
        spacing = 12
        x = chunk.left() - chunk.height() + self._stripe_offset
        while x < chunk.right() + chunk.height():
            painter.drawLine(
                x,
                chunk.bottom(),
                x + chunk.height(),
                chunk.top(),
            )
            x += spacing


@dataclass
class ItemState:
    path: Path
    progress: int = 0
    status: str = "Queued"


class ItemWidget(QtWidgets.QWidget):
    def __init__(self, state: ItemState, parent=None):
        super().__init__(parent)
        self.state = state
        self._active = False
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        self.label = QtWidgets.QLabel(str(state.path))
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.progress = AnimatedProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(state.progress)
        self.progress.setFixedWidth(180)
        self.status = QtWidgets.QLabel(state.status)
        self.status.setFixedWidth(140)

        layout.addWidget(self.label, 1)
        layout.addWidget(self.progress, 0)
        layout.addWidget(self.status, 0)

    def update(self, progress: int, status: str):
        self.progress.setValue(progress)
        self.status.setText(status)

        active_states = {"Preprocessing", "Transcribing", "Saving"}
        self._active = status in active_states and progress < 100
        self.progress.setAnimated(self._active)

    def animate(self):
        if not self._active:
            return
        self.progress.advancePattern()


class DropArea(QtWidgets.QFrame):
    pathsDropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)
        self.setObjectName("dropArea")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.label = QtWidgets.QLabel("Drag and drop audio files here")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        if paths:
            self.pathsDropped.emit(paths)
        event.acceptProposedAction()


class Worker(QtCore.QThread):
    itemStatus = QtCore.Signal(Path, int, str)
    itemDone = QtCore.Signal(Path)
    allDone = QtCore.Signal()

    def __init__(self, items: list[Path], parent=None):
        super().__init__(parent)
        self.items = items
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        transcriber = Transcriber()
        for path in self.items:
            if self._stop_event.is_set():
                self.itemStatus.emit(path, 0, "Canceled")
                self.itemDone.emit(path)
                continue
            transcript_path = path.with_suffix(".txt")
            if transcript_path.exists():
                self.itemStatus.emit(path, 100, "Skipped (exists)")
                self.itemDone.emit(path)
                continue
            try:
                self.itemStatus.emit(path, 20, "Preprocessing")
                processed = preprocess_audio(path, stop_event=self._stop_event)
                self.itemStatus.emit(path, 65, "Transcribing")
                transcribed_json = transcriber.transcribe(
                    processed, stop_event=self._stop_event
                )
                if transcribed_json:
                    self.itemStatus.emit(path, 90, "Saving")
                    save_transcript_as_text(
                        transcript_path.parent, transcript_path, transcribed_json
                    )
                    self.itemStatus.emit(path, 100, "Done")
                else:
                    self.itemStatus.emit(path, 100, "Error")
            except InterruptedError:
                self.itemStatus.emit(path, 0, "Canceled")
                logger.info("Canceled: %s", path)
            except Exception as exc:
                self.itemStatus.emit(path, 100, f"Error: {exc}")
            finally:
                self.itemDone.emit(path)

        self.allDone.emit()


class SetupWorker(QtCore.QThread):
    statusUpdate = QtCore.Signal(str, str, str)
    setupDone = QtCore.Signal(bool, str)

    def run(self):
        try:
            transcriber = Transcriber(progress_cb=self._emit_progress)
            if not transcriber.binary_path:
                self.setupDone.emit(False, "whisper.cpp binary is unavailable.")
                return
            if not transcriber.model_path.is_file():
                self.setupDone.emit(False, "whisper.cpp model is unavailable.")
                return
            self.setupDone.emit(True, "Setup complete.")
        except Exception as exc:
            self.setupDone.emit(False, str(exc))

    def _emit_progress(self, item_id: str, status: str, path_text: str):
        self.statusUpdate.emit(item_id, status, path_text)


class TranscriberWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Transcriber")
        self.resize(980, 640)
        self.items: dict[Path, ItemState] = {}
        self.widgets: dict[Path, ItemWidget] = {}
        self.worker: Worker | None = None
        self.total_items = 0
        self.completed_items = 0
        self._log_emitter: LogEmitter | None = None
        self._log_handler: QtLogHandler | None = None
        self._close_requested = False
        self.setup_worker: SetupWorker | None = None
        self.setup_in_progress = True
        self._setup_rows: dict[str, QtWidgets.QListWidgetItem] = {}
        self._pulse_timer = QtCore.QTimer(self)
        self._pulse_timer.setInterval(120)
        self._pulse_timer.timeout.connect(self._animate_progress)

        self._build_ui()
        self._setup_logging()
        self._start_initial_setup()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Drop audio files to transcribe")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel("Transcripts are saved next to each audio file.")
        subtitle.setObjectName("subtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QtWidgets.QHBoxLayout()
        self.add_files_btn = QtWidgets.QPushButton("Add Files")
        self.add_folder_btn = QtWidgets.QPushButton("Add Folder")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.stop_btn.setEnabled(False)

        self.add_files_btn.clicked.connect(self._add_files)
        self.add_folder_btn.clicked.connect(self._add_folder)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.clear_btn.clicked.connect(self._clear)

        controls.addWidget(self.add_files_btn)
        controls.addWidget(self.add_folder_btn)
        controls.addStretch(1)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.clear_btn)
        layout.addLayout(controls)

        self.drop_area = DropArea()
        self.drop_area.pathsDropped.connect(self._add_paths)
        layout.addWidget(self.drop_area)

        overall_row = QtWidgets.QHBoxLayout()
        self.overall_label = QtWidgets.QLabel("Overall progress: 0/0")
        self.overall_progress = AnimatedProgressBar()
        self.overall_progress.setRange(0, 100)
        overall_row.addWidget(self.overall_label)
        overall_row.addStretch(1)
        layout.addLayout(overall_row)
        layout.addWidget(self.overall_progress)

        self.list_label = QtWidgets.QLabel("Items")
        layout.addWidget(self.list_label)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSpacing(6)
        layout.addWidget(self.list_widget, 1)

        log_label = QtWidgets.QLabel("Logs")
        layout.addWidget(log_label)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(150)
        layout.addWidget(self.log_view)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget {
                color: #161616;
            }
            QMainWindow {
                background: #f6f6f4;
            }
            QLabel {
                color: #161616;
            }
            QLabel#title {
                font-size: 20px;
                font-weight: 600;
                color: #111111;
            }
            QLabel#subtitle {
                color: #555555;
            }
            QPushButton {
                background: #111111;
                color: white;
                padding: 8px 14px;
                border-radius: 6px;
            }
            QPushButton:disabled {
                background: #777777;
                color: #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                text-align: center;
                height: 12px;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #111111,
                    stop: 0.5 #1f2933,
                    stop: 1 #111111
                );
                border-radius: 6px;
            }
            QFrame#dropArea {
                background: #ffffff;
                border: 1px dashed #c7c7c7;
                border-radius: 10px;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                color: #161616;
            }
            QPlainTextEdit {
                background: #0f1216;
                color: #c9f7d6;
                border: 1px solid #1f2933;
                border-radius: 10px;
                padding: 8px;
                font-family: "Menlo", "Consolas", monospace;
                font-size: 12px;
            }
            """
        )

    def _setup_logging(self):
        self._log_emitter = LogEmitter()
        self._log_emitter.message.connect(self._append_log)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        self._log_handler = QtLogHandler(self._log_emitter)
        self._log_handler.setFormatter(formatter)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(self._log_handler)

        has_stream = any(getattr(h, "_ui_stream_handler", False) for h in root.handlers)
        if not has_stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler._ui_stream_handler = True
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

        logger.info("UI initialized.")

    def _start_initial_setup(self):
        self.setup_in_progress = True
        self._set_controls_enabled(False)
        self.drop_area.setEnabled(False)
        self.list_label.setText("Initial setup downloads")
        self._setup_rows.clear()
        self.list_widget.clear()
        self.items.clear()
        self.widgets.clear()
        self._update_overall(0, 0)
        self._add_setup_row("tool.repo", str(WHISPER_CPP_PATH))
        self._add_setup_row("tool.binary", str(WHISPER_CPP_LOCAL_BIN))
        self._add_setup_row("model.default", str(AI_MODEL_WHISPER_CPP_DEFAULT_MODEL))
        logger.info("Preparing first-run tools and model. Controls are disabled until setup completes.")

        self.setup_worker = SetupWorker()
        self.setup_worker.statusUpdate.connect(self._on_setup_status)
        self.setup_worker.setupDone.connect(self._on_setup_done)
        self.setup_worker.finished.connect(self._on_setup_worker_finished)
        self.setup_worker.start()

    def _add_setup_row(self, item_id: str, path_text: str):
        item = QtWidgets.QListWidgetItem(f"Queued       {path_text}")
        self.list_widget.addItem(item)
        self._setup_rows[item_id] = item

    @QtCore.Slot(str, str, str)
    def _on_setup_status(self, item_id: str, status: str, path_text: str):
        row = self._setup_rows.get(item_id)
        if row:
            row.setText(f"{status:<12} {path_text}")

    @QtCore.Slot(bool, str)
    def _on_setup_done(self, success: bool, message: str):
        if success:
            self.setup_in_progress = False
            self.list_widget.clear()
            self._setup_rows.clear()
            self.list_label.setText("Items")
            self.drop_area.setEnabled(True)
            self._set_controls_enabled(True)
            logger.info("First-run setup complete. You can now add files.")
            return

        self.setup_in_progress = True
        self._set_controls_enabled(False)
        self.drop_area.setEnabled(False)
        logger.error("Setup failed: %s", message)
        QtWidgets.QMessageBox.critical(
            self,
            "Setup failed",
            "Failed to prepare whisper.cpp tools/model.\n"
            "Check logs, install `git` and `cmake`, then restart.",
        )

    @QtCore.Slot()
    def _on_setup_worker_finished(self):
        if self.setup_worker:
            self.setup_worker.deleteLater()
            self.setup_worker = None

    @QtCore.Slot(str)
    def _append_log(self, message: str):
        self.log_view.appendPlainText(message)
        scroll = self.log_view.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _add_files(self):
        if self.setup_in_progress:
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select audio files"
        )
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _add_folder(self):
        if self.setup_in_progress:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select folder"
        )
        if folder:
            self._add_paths([Path(folder)])

    def _add_paths(self, paths: list[Path]):
        if self.setup_in_progress:
            return
        added = 0
        for path in paths:
            if path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                        added += self._add_item(file_path)
            elif path.is_file():
                if path.suffix.lower() in audio_extensions:
                    added += self._add_item(path)

        if added == 0:
            QtWidgets.QMessageBox.information(
                self, "No audio files", "No supported audio files were found."
            )

    def _add_item(self, path: Path) -> int:
        if path in self.items:
            return 0
        state = ItemState(path=path)
        widget = ItemWidget(state)
        list_item = QtWidgets.QListWidgetItem(self.list_widget)
        list_item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(list_item)
        self.list_widget.setItemWidget(list_item, widget)
        self.items[path] = state
        self.widgets[path] = widget
        return 1

    def _clear(self):
        if self.setup_in_progress:
            return
        if self.worker and self.worker.isRunning():
            return
        self.items.clear()
        self.widgets.clear()
        self.list_widget.clear()
        self._update_overall(0, 0)

    def _start(self):
        if self.setup_in_progress:
            return
        if self.worker and self.worker.isRunning():
            return
        if not self.items:
            QtWidgets.QMessageBox.information(
                self, "No items", "Add at least one audio file first."
            )
            return
        self.total_items = len(self.items)
        self.completed_items = 0
        self._update_overall(0, self.total_items)
        self._set_controls_enabled(False)
        self.worker = Worker(list(self.items.keys()))
        self.worker.itemStatus.connect(self._on_item_status)
        self.worker.itemDone.connect(self._on_item_done)
        self.worker.allDone.connect(self._on_all_done)
        self.worker.finished.connect(self._on_worker_finished)
        logger.info("Starting processing for %s item(s).", self.total_items)
        self.overall_progress.setAnimated(True)
        self._pulse_timer.start()
        self.worker.start()

    def _stop(self):
        if self.setup_in_progress:
            return
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            logger.info("Stop requested. Pending items will be marked as canceled.")

    def _set_controls_enabled(self, enabled: bool):
        if self.setup_in_progress:
            self.add_files_btn.setEnabled(False)
            self.add_folder_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            return

        self.add_files_btn.setEnabled(enabled)
        self.add_folder_btn.setEnabled(enabled)
        self.start_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)

    @QtCore.Slot(Path, int, str)
    def _on_item_status(self, path: Path, progress: int, status: str):
        widget = self.widgets.get(path)
        if widget:
            widget.update(progress, status)

    @QtCore.Slot(Path)
    def _on_item_done(self, path: Path):
        self.completed_items += 1
        self._update_overall(self.completed_items, self.total_items)

    @QtCore.Slot()
    def _on_all_done(self):
        self._pulse_timer.stop()
        self.overall_progress.setAnimated(False)
        self._set_controls_enabled(True)
        logger.info("Processing finished. Completed: %s/%s", self.completed_items, self.total_items)
        self._update_overall(self.completed_items, self.total_items)

    @QtCore.Slot()
    def _on_worker_finished(self):
        # QThread::finished guarantees the thread is no longer running.
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self._close_requested:
            self._close_requested = False
            QtCore.QTimer.singleShot(0, self.close)

    def _update_overall(self, completed: int, total: int):
        self.overall_label.setText(f"Overall progress: {completed}/{total}")
        percent = int((completed / total) * 100) if total else 0
        self.overall_progress.setValue(percent)

    @QtCore.Slot()
    def _animate_progress(self):
        for widget in self.widgets.values():
            widget.animate()
        self.overall_progress.advancePattern()

    def closeEvent(self, event):
        if self.setup_worker and self.setup_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Setup in progress",
                "Initial setup is still running. Please wait for it to finish before closing.",
            )
            event.ignore()
            return

        if self.worker and self.worker.isRunning():
            if not self._close_requested:
                answer = QtWidgets.QMessageBox.question(
                    self,
                    "Processing in progress",
                    "Transcription is still running. Stop processing and close when safe?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    event.ignore()
                    return
                self._close_requested = True
                self._set_controls_enabled(False)
                self.worker.stop()
                logger.info("Close requested while processing. Waiting for worker to stop.")
            event.ignore()
            return

        root = logging.getLogger()
        if self._log_handler and self._log_handler in root.handlers:
            root.removeHandler(self._log_handler)
            self._log_handler = None
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f6f6f4"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f2f2f2"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#161616"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#161616"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#161616"))
    app.setPalette(palette)
    window = TranscriberWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
