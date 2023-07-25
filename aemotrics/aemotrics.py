import os
from dlc_generic_analysis import gui_utils, trimmer
from qtpy import QtCore, QtWidgets, QtGui, API as QT_API, QT_VERSION
from qtpy.QtCore import Qt
from .analysis import analyze
from .cropper import FaceCropper
from ._gui_utils import CloseListener, text
from .viewer import FacialViewer

try:
    from ._version import version
except ImportError:
    try:
        import subprocess

        version = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )
    except:
        version = "unknown"
except:
    version = "unknown"
from deeplabcut.version import VERSION as dlc_version
from tensorflow.version import VERSION as tf_version
import sys


def set_trace_in_qt():
    from _pydevd_bundle import pydevd_tracing
    from _pydevd_bundle.pydevd_comm import get_global_debugger

    debugger = get_global_debugger()
    if debugger is not None:
        pydevd_tracing.SetTrace(debugger.trace_dispatch)


class MainWidget(QtWidgets.QWidget):
    def __init__(self, window: QtWidgets.QMainWindow, model_path: str = None):
        """
        the main view for Aemotrics that allows users to select to analyze or view videos
        :param stack:
        :param model_path:
        """
        QtWidgets.QWidget.__init__(self)
        self.setLayout(QtWidgets.QGridLayout())
        analyze_button = QtWidgets.QPushButton("Analyze Videos")
        analyze_button.clicked.connect(self.on_click_analyze)
        self.layout().addWidget(analyze_button, 0, 0, 1, 2)
        view_video_button = QtWidgets.QPushButton("View Analyzed Video")
        view_video_button.clicked.connect(self.on_click_view)
        self.layout().addWidget(view_video_button, 1, 0, 1, 2)
        self.window = window
        self.model_path = model_path
        self.crop_checkbox = QtWidgets.QCheckBox("Crop Videos")
        self.layout().addWidget(self.crop_checkbox, 2, 0)
        self.crop_checkbox.setChecked(True)
        self.trim_checkbox = QtWidgets.QCheckBox("Trim Videos in time")

        self.layout().addWidget(self.trim_checkbox, 2, 1)
        self.label = text(" ")
        self.layout().addWidget(self.label, 3, 0, 1, 2)
        self.info = QtWidgets.QPushButton("info")
        self.layout().addWidget(self.info, 4, 0, 1, 2)
        self.info.clicked.connect(self.on_click_info)
        self.setWindowTitle("Aemotrics")

    def on_trimmer_done(self):
        files = self.trimmer.trimmed_videos
        aw = AnalyzeWorker(
            files,
            self.model_path,
            self.crop_checkbox.isChecked(),
        )
        aw.analysis_finished.connect(self.analysis_finished)
        aw.start()
        self.label.setText("Analyzing... please wait.")

    def on_click_analyze(self):
        files = gui_utils.open_files(self, "select videos to analyze")
        if len(files) > 0:
            if self.trim_checkbox.isChecked():
                self.trimmer = trimmer.Trimmer(files)
                cl = CloseListener(self.trimmer)
                cl.closed.connect(self.on_trimmer_done)
                self.trimmer.show()
                return
            self.aw = AnalyzeWorker(
                files,
                self.model_path,
                self.crop_checkbox.isChecked(),
            )
            self.aw.analysis_finished.connect(self.analysis_finished)
            self.aw.start()
            self.label.setText("Analyzing... please wait.")

    def analysis_finished(self, finished):
        if finished:
            self.label.setText("Analysis Finished!")

    def on_click_view(self):
        path = gui_utils.open_files(self, "select analyzed video to view")
        if len(path) > 0:
            viewer = FacialViewer(self.window.stack)
            self.window.stack.addWidget(viewer)
            self.window.stack.widget(1).load_video(path[0])
            self.window.stack.setCurrentIndex(1)
            self.window.setWindowState(Qt.WindowMaximized)

    def on_click_info(self):
        text = (
            f"**Aemotrics**\n\nVersion: {version}\n\nAuthor: Louis Adamian & Nat Adamian\n\n"
            f"Written at the Surgical Photonics & Engineering Lab\n\n"
            f"[surgicalphotonics.org](https://www.surgicalphotonics.org/)\n\n"
            f"Github: [github.com/surgicalphotonics/aemotrics](https://github.com/surgicalphotonics/aemotrics)\n\n"
            "Citation: \n\n"
            "DOI: "
        )
        detailed_text = (
            f"Python Version: {sys.version}\n"
            f"DeepLabCut Version: {dlc_version}\n"
            f"Tensorflow Version: {tf_version}\n"
            f"Qt API: {QT_API}\n"
            f"QT Version: {QT_VERSION}"
        )
        msg = QtWidgets.QMessageBox(None, "Aemotrics Info", QtWidgets.QMessageBox.Ok)
        msg.setText(text)
        msg.setTextFormat(QtCore.Qt.MarkdownText)
        msg.setDetailedText(detailed_text)
        msg.exec()


class MainWindow(QtWidgets.QMainWindow):
    """
    The main window that shows either the main view or the video viewer
    :param model_path: the path to the deeplabcut model being used
    """

    def __init__(self, model_path: str = None):
        QtWidgets.QMainWindow.__init__(self)
        self.stack = QtWidgets.QStackedWidget()
        main_widget = MainWidget(self, model_path)
        self.stack.addWidget(main_widget)
        self.stack.setCurrentWidget(main_widget)
        self.setCentralWidget(self.stack)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        return super(MainWindow, self).resizeEvent(event)


def main(model_path: str = None) -> None:
    """
    the main GUI application function that creates the QT gui and created the main MainWindow
    :param model_path: the path to the deeplabcut model
    :return: None
    """
    name = "Aemotrics"
    app = QtWidgets.QApplication.instance()
    if not app:
        if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        app = QtWidgets.QApplication([])
    app.setApplicationName(name)
    app.setApplicationDisplayName(name)
    app.setApplicationVersion(version)
    window = MainWindow(model_path)
    window.show()
    app.exec_()


class AnalyzeWorker(QtCore.QThread):
    analysis_finished = QtCore.Signal(bool)

    def __init__(self, files, model_path: str, crop: bool):
        QtCore.QThread.__init__(self)
        self.files = files
        self.model_path = model_path
        self.crop = crop

    def run(self):
        if len(self.files) > 0:
            if self.crop:
                cropper = FaceCropper()
                cropped_files = []
                for file in self.files:
                    try:
                        cropped_files.append(cropper.fast_crop(file))
                    except (IndexError, FileNotFoundError):
                        print(f"Failed to crop {os.path.split(file)[1]}")
                        cropped_files.append(file)
                self.files = cropped_files
            analyze(self.files, dlc_model_path=self.model_path)
            self.analysis_finished.emit(True)
