import os.path
import cv2
import numpy as np
import dlc_generic_analysis as dga
from dlc_generic_analysis.viewer import ViewerWidget
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from qtpy import QtCore, QtMultimedia, QtWidgets, QtGui
from qtpy.QtCore import Qt
from .reanalyze import ReanalyzePopup
import pandas as pd
from . import _utils
from ._utils import text

if os.name == "nt":
    os.environ["QT_MULTIMEDIA_PREFERRED_PLUGINS"] = "windowsmediafoundation"
r_poly_color = "#3f8e36"
l_poly_color = "#977fbe"


class RescalePopup(QtWidgets.QWidget):
    def __init__(self, radius_px, radis_mm, vid_path, stack):
        """
        Dialog that allows the user to recompute the pixel size of the video by changing the radius of the iris and
        creates a replaces the viewer when Recompute is run.
        :param radius_px: The radius of the iris in pixels
        :param radis_mm: the previously used iris radius in mm
        :param vid_path: the path to the analyzed video
        :param stack: the QStackWidget of the application to put the new viewer in
        """
        super(RescalePopup, self).__init__()
        self.radius_px = radius_px
        self.vid_path = vid_path
        self.stack = stack
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(text("Rescale Iris radius"))
        edit = QtWidgets.QHBoxLayout()
        self.scaling_edit = QtWidgets.QLineEdit()
        self.scaling_edit.setValidator(QtGui.QDoubleValidator())
        edit.addWidget(self.scaling_edit)
        self.scaling_edit.setText(round_str(radis_mm))
        edit.addWidget(text("mm"))
        layout.addLayout(edit)
        recompute_button = QtWidgets.QPushButton("Recompute")
        recompute_button.clicked.connect(self.on_recalculate)
        layout.addWidget(recompute_button)
        self.setLayout(layout)

    def on_recalculate(self):
        if self.scaling_edit.text() != "":
            if float(self.scaling_edit.text()) > 0:
                scale = float(self.scaling_edit.text()) / self.radius_px
                new_viewer = FacialViewer(self.stack)
                self.stack.addWidget(new_viewer)
                new_viewer.load_video(self.vid_path, scale)
                self.stack.setCurrentWidget(new_viewer)
                self.stack.removeWidget(self)
                self.hide()
                del self


def cross_corelate(data0: np.ndarray, data1: np.ndarray, mark_in, mark_out):
    return norm_xcor(data0[mark_in:mark_out], data1[mark_in:mark_out])


def difference(a: np.float_, b: np.float_):
    if a > b:
        maxv = a
        minv = b
    else:
        maxv = b
        minv = a
    return (maxv - minv) / maxv


class FacialViewer(ViewerWidget):
    def __init__(self, stack):
        """
        A window to view an analyzed video with a graph of metrics
        """
        super().__init__()
        self._video_player.setVideoOutput(self.video_viewer)
        self.video_viewer.setMinimumWidth(100)
        self.setMinimumSize(QtCore.QSize(1500, 840))
        self.video_loaded = False
        self.metric_panel = QtWidgets.QWidget(parent=None)
        self.mark_in = 0
        self.mark_out = 0
        back_button = QtWidgets.QPushButton("Back")
        back_button.clicked.connect(self.on_back)
        self.top_layout.addWidget(back_button)
        self.edit_px_size_button = QtWidgets.QPushButton("Recalculate Pixel Size")
        self.top_layout.setAlignment(self.edit_px_size_button, Qt.AlignRight)
        self.edit_px_size_button.clicked.connect(self.on_edit_px_scaling)
        self.top_layout.addWidget(self.edit_px_size_button)
        self.stack = stack

    def position_changed(self, position):
        frame = int(self.ms_to_frame(position)) - 1
        for i, line in enumerate(self.lines):
            line[0].set_xdata(frame / self.frame_rate)
            self.canvases[i].draw_idle()
        self.current_time_text.setText(round_str(position / 1000) + "s")
        self.symmetry_text.setText(round_str(self.symmetry[frame]))
        self.brow_height_l_text.setText(round_str(self.brow_height_l[frame]))
        self.brow_height_r_text.setText(round_str(self.brow_height_r[frame]))
        self.eye_area_l_text.setText(round_str(self.eye_area_r[frame]))
        self.eye_area_r_text.setText(round_str(self.eye_area_r[frame]))
        self.oral_excursion_l_text.setText(round_str(self.oral_excursion_l[frame]))
        self.oral_excursion_r_text.setText(round_str(self.oral_excursion_r[frame]))
        self.lower_lip_height_l_text.setText(round_str(self.lower_lip_height_l[frame]))
        self.lower_lip_height_r_text.setText(round_str(self.lower_lip_height_r[frame]))
        super(FacialViewer, self).position_changed(position)

    def load_video(self, path, pixel_scaling: float = None):
        self.vid_path = path
        vid = cv2.VideoCapture(path)
        self.frame_rate = vid.get(cv2.CAP_PROP_FPS)
        self._video_player.setNotifyInterval(1000 / self.frame_rate)
        self.frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.mark_out = int(self.frames / self.frame_rate)
        vid.release()
        self.mm = True
        data_path = os.path.splitext(path)[0]
        if not os.path.isfile(data_path + ".h5"):
            return
        data = pd.read_hdf(data_path + ".h5", key=os.path.split(data_path)[1])
        data = data[os.path.split(data_path)[1]]
        self.symmetry = data["scores", "score"].to_numpy()
        self.brow_height_l = data["eyes", "brow_dist_l"].to_numpy()
        self.brow_height_r = data["eyes", "brow_dist_r"].to_numpy()
        self.eye_area_r = data["eyes", "area_r"].to_numpy()
        self.eye_area_l = data["eyes", "area_l"].to_numpy()
        self.lower_lip_height_l = data["mouth", "lip_height_r"].to_numpy()
        self.lower_lip_height_l = data["mouth", "lip_height_r"].to_numpy()
        self.lower_lip_height_r = data["mouth", "lip_height_l"].to_numpy()
        self.oral_excursion_l = data["mouth", "excursions_l"].to_numpy()
        self.oral_excursion_r = data["mouth", "excursions_r"].to_numpy()
        self.iris_radei_r = data["scores", "iris_radei_r"]
        self.iris_radei_l = data["scores", "iris_radei_r"]
        self.glabella = data["glabella"].to_numpy()
        self.midline_bottom = data["midlineP2"].to_numpy()
        if self.mm:
            if pixel_scaling is not None:
                pix_size = pixel_scaling
            else:
                pix_size = data["scores", "pix_size"].to_numpy()[0]
            len_label = "mm"
            area_label = "mm^2"
            self.brow_height_l = self.brow_height_l * pix_size
            self.brow_height_r = self.brow_height_r * pix_size
            self.eye_area_l = self.eye_area_l * np.power(0.55, 2)
            self.eye_area_r = self.eye_area_r * np.power(0.55, 2)
            self.lower_lip_height_l = self.lower_lip_height_l * pix_size
            self.lower_lip_height_r = self.lower_lip_height_r * pix_size
            self.oral_excursion_l = self.oral_excursion_l * pix_size
            self.oral_excursion_r = self.oral_excursion_r * pix_size
        else:
            len_label = "px"
            area_label = "px^2"
        self.reanalyze_angles_frame = dga.utils.nan(self.symmetry.shape)
        metrics_layout = QtWidgets.QHBoxLayout()
        mark_buttons = QtWidgets.QGridLayout()
        adjust_midline_button = QtWidgets.QPushButton("Adjust Midline")
        # adjust_midline_button.setSizePolicy(QtWidgets.QSizePolicy.Minimum)
        self.top_layout.setAlignment(adjust_midline_button, Qt.AlignRight)
        adjust_midline_button.clicked.connect(self.on_adjust_midline)
        mark_in = QtWidgets.QPushButton("Mark In")
        mark_in.clicked.connect(self.on_mark_in)
        mark_out = QtWidgets.QPushButton("Mark Out")
        mark_out.clicked.connect(self.on_mark_out)
        Compute_button = QtWidgets.QPushButton("Compute")
        Compute_button.clicked.connect(self.on_compute)
        self.top_layout.addWidget(adjust_midline_button)
        self.mark_start_text = text("Start: " + round_str(self.mark_in) + "s")
        self.mark_stop_text = text("Stop: " + round_str(self.mark_out) + "s")
        mark_buttons.addWidget(self.mark_start_text, 0, 0)
        mark_buttons.addWidget(self.mark_stop_text, 0, 1)
        mark_buttons.addWidget(mark_in, 1, 0)
        mark_buttons.addWidget(mark_out, 1, 1)
        mark_buttons.addWidget(Compute_button, 2, 0, 1, 2)
        values_layout = QtWidgets.QGridLayout()
        values_layout.addWidget(text("**Current Values** @", "md"), 0, 0)
        values_layout.addWidget(text("Vermilion Symmetry"), 1, 0)
        values_layout.addWidget(text("Brow Height L"), 2, 0)
        values_layout.addWidget(text("Brow Height R"), 3, 0)
        values_layout.addWidget(text("Eye Area L"), 4, 0)
        values_layout.addWidget(text("Eye Area R"), 5, 0)
        values_layout.addWidget(text("Oral Comm Exc R"), 6, 0)
        values_layout.addWidget(text("OOral Comm Exc L"), 7, 0)
        values_layout.addWidget(text("Lower Lip height L"), 8, 0)
        values_layout.addWidget(text("Lower Lip height R"), 9, 0)
        self.current_time_text = text("0s")
        values_layout.addWidget(self.current_time_text, 0, 1)
        self.symmetry_text = text("")
        values_layout.addWidget(self.symmetry_text, 1, 1)
        self.brow_height_l_text = text("")
        values_layout.addWidget(self.brow_height_l_text, 2, 1)
        self.brow_height_r_text = text("")
        values_layout.addWidget(self.brow_height_r_text, 3, 1)
        self.eye_area_l_text = text("")
        values_layout.addWidget(self.eye_area_l_text, 4, 1)
        self.eye_area_r_text = text("")
        values_layout.addWidget(self.eye_area_r_text, 5, 1)
        self.oral_excursion_l_text = text("")
        values_layout.addWidget(self.oral_excursion_l_text, 6, 1)
        self.oral_excursion_r_text = text("")
        values_layout.addWidget(self.oral_excursion_r_text, 7, 1)
        self.lower_lip_height_r_text = text("")
        values_layout.addWidget(self.lower_lip_height_r_text, 8, 1)
        self.lower_lip_height_l_text = text("")
        values_layout.addWidget(self.lower_lip_height_l_text, 9, 1)
        ts_vals_layout = QtWidgets.QGridLayout()
        ts_vals_layout.addWidget(text("**Time Series Values**", "md"), 0, 0)
        ts_vals_layout.addWidget(text("**Means**", "md"), 1, 0)
        ts_vals_layout.addWidget(text("Smile Symmetry"), 2, 0)
        ts_vals_layout.addWidget(text("**Cross Corelations**", "md"), 3, 0)
        ts_vals_layout.addWidget(text("Brow Heights"), 4, 0)
        ts_vals_layout.addWidget(text("Eye Areas"), 5, 0)
        ts_vals_layout.addWidget(text("Oral Comm Exc"), 6, 0)
        ts_vals_layout.addWidget(text("Lower Lip Heights"), 7, 0)
        self.symmetry_mean = text(round_str(np.nanmean(self.symmetry)))
        ts_vals_layout.addWidget(self.symmetry_mean, 2, 1)
        self.brow_height_cc = text(round_str(norm_xcor(self.brow_height_r, self.brow_height_l)))
        ts_vals_layout.addWidget(self.brow_height_cc, 4, 1)
        self.eye_areas_cc = text(round_str(norm_xcor(self.eye_area_r, self.eye_area_l)))
        ts_vals_layout.addWidget(self.eye_areas_cc, 5, 1)
        self.oral_excursion_cc = text(
            round_str(norm_xcor(self.oral_excursion_r, self.oral_excursion_l))
        )
        self.lower_lip_height_cc = text(
            round_str(norm_xcor(self.lower_lip_height_l, self.lower_lip_height_r))
        )
        ts_vals_layout.addWidget(self.oral_excursion_cc, 6, 1)
        ts_vals_layout.addWidget(self.lower_lip_height_cc, 7, 1)
        ts_vals_layout.addLayout(mark_buttons, 10, 0, 10, 2)
        ts_vals = QtWidgets.QFrame(parent=None)
        ts_vals.setFrameStyle(QtWidgets.QFrame.Box)
        ts_vals.setLayout(ts_vals_layout)
        metrics_min_size = 160
        ts_vals.setMinimumWidth(metrics_min_size)
        values = QtWidgets.QFrame(parent=None)
        values.setLayout(values_layout)
        values.setFrameStyle(QtWidgets.QFrame.Panel)
        values.setMinimumWidth(metrics_min_size)
        metrics_layout.addWidget(values)
        metrics_layout.setStretchFactor(values, 1)
        metrics_layout.addWidget(ts_vals)
        metrics_layout.setStretchFactor(ts_vals, 1)
        self.metric_panel.setLayout(metrics_layout)
        plots_grid = QtWidgets.QGridLayout()
        plots_grid.setContentsMargins(0, 0, 0, 0)
        canvases = []
        numplots = 5
        for i in range(numplots):
            canvases.append(FigureCanvas(Figure()))
        self.canvases = [FigureCanvas(Figure()) for _ in range(numplots)]
        plots_grid.addWidget(canvases[0], 0, 0)
        plots_grid.addWidget(canvases[1], 1, 0)
        plots_grid.addWidget(canvases[2], 2, 0)
        plots_grid.addWidget(canvases[3], 0, 1)
        plots_grid.addWidget(canvases[4], 1, 1)
        plots_grid.addWidget(self.metric_panel, 2, 1)
        self.content_layout.addLayout(plots_grid, 4)
        xs = np.arange(self.symmetry.shape[0])
        xs = xs / self.frame_rate
        axs = [canvases[i].figure.subplots() for i in range(numplots)]
        self.lines = []
        self.lines.append(
            plot(
                axs[0],
                xs,
                [self.brow_height_l, self.brow_height_r],
                [l_poly_color, r_poly_color],
                ["left", "right"],
            )
        )
        axs[0].set_title("Brow Height")
        axs[0].set_ylabel(len_label)
        self.lines.append(plot(axs[1], xs, [self.symmetry]))
        axs[1].set_title("Vermilion Symmetry")
        axs[1].set_ylabel("Score")
        axs[2].set_title("Palpebral Fissure Areas")
        axs[2].set_ylabel(area_label)
        self.lines.append(
            plot(
                axs[2],
                xs,
                [self.eye_area_l, self.eye_area_r],
                [l_poly_color, r_poly_color],
                ["left", "right"],
            )
        )
        axs[3].set_title(" Lower Lip Heights")
        axs[3].set_ylabel(len_label)
        self.lines.append(
            plot(
                axs[3],
                xs,
                [self.lower_lip_height_l, self.lower_lip_height_r],
                [l_poly_color, r_poly_color],
                ["left", "right"],
            )
        )
        axs[4].set_title("Oral Commisure Excursions")
        axs[4].set_ylabel(len_label)
        self.lines.append(
            plot(
                axs[4],
                xs,
                [self.oral_excursion_l, self.oral_excursion_r],
                [l_poly_color, r_poly_color],
                ["left", "right"],
            )
        )
        for i in range(len(canvases)):
            canvases[i].figure.set_tight_layout(True)
        self.canvases = canvases
        self._video_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(path)))
        super(FacialViewer, self).load_video(path)
        self._play_pause.setEnabled(True)
        self._video_player.play()

    def on_back(self):
        self.stack.setCurrentIndex(0)
        self.stack.removeWidget(self)

    def on_mark_in(self, e):
        mark_in = int(self.ms_to_frame(self._video_player.position()))
        if self.mark_out > mark_in:
            self.mark_in = mark_in
            self.mark_start_text.setText(
                "Start: " + round_str(self._video_player.position() / 1000) + "s"
            )

    def on_mark_out(self, e):
        mark_out = int(self.ms_to_frame(self._video_player.position()))
        if mark_out > self.mark_in:
            self.mark_out = mark_out
            self.mark_stop_text.setText(
                "Stop: " + round_str(self._video_player.position() / 1000) + "s"
            )

    def on_compute(self):
        self.brow_height_cc.setText(
            round_str(
                norm_xcor(
                    self.brow_height_r[self.mark_in : self.mark_out],
                    self.brow_height_l[self.mark_in : self.mark_out],
                )
            )
        )
        self.eye_areas_cc.setText(
            round_str(
                norm_xcor(
                    self.eye_area_r[self.mark_in : self.mark_out],
                    self.eye_area_l[self.mark_in : self.mark_out],
                )
            )
        )
        self.oral_excursion_cc.setText(
            round_str(
                norm_xcor(
                    self.oral_excursion_r[self.mark_in : self.mark_out],
                    self.oral_excursion_l[self.mark_in : self.mark_out],
                )
            )
        )
        self.lower_lip_height_cc.setText(
            round_str(
                norm_xcor(
                    self.lower_lip_height_l[self.mark_in : self.mark_out],
                    self.lower_lip_height_r[self.mark_in : self.mark_out],
                )
            )
        )
        self.symmetry_mean.setText(
            round_str(np.nanmean(self.symmetry[self.mark_in : self.mark_out]))
        )

    def on_adjust_midline(self):
        frame = int(self.ms_to_frame(self._video_player.position()))
        self.edit_midline(self.vid_path, frame, self.glabella, self.midline_bottom)

    def edit_midline(
        self, vid_path: str, frame_num: int, glabella: np.ndarray, bottom: np.ndarray
    ) -> None:
        """
        Edit the midline on a video using a napari viewer to adjust the placement of the midline on specific frame
        :param vid_path: the path to the video to adjust
        :param frame_num: the number frame in the video to adjust
        :param glabella: the x,y coordinates glabella point on that frame
        :param bottom: the x,y coordinates bottom point on that frame
        """
        import napari

        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        s, im = cap.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if not s:
            raise AttributeError
        self.old_line = np.stack([np.flip(glabella[frame_num]), np.flip(bottom[frame_num])])
        self.vid_path = vid_path
        self.reanalyze_frame = frame_num
        self.viewer = napari.view_image(im)

        self.viewer.add_shapes(
            self.old_line,
            shape_type="line",
            edge_width=5,
            edge_color="white",
            opacity=1,
        )
        napari.run(force=True, max_loop_level=3)
        close_listener = _utils.CloseListener(self.viewer.window._qt_window)
        close_listener.closed.connect(self.recalc_midline)

    def recalc_midline(self):
        line = self.viewer.layers[1].data[0]
        print(line)
        self.popup = ReanalyzePopup(line, self.old_line, self.vid_path, self.reanalyze_frame)
        self.popup.show()

    def on_edit_px_scaling(self):
        iris_radius_px = np.nanmean([self.iris_radei_r, self.iris_radei_l])
        self.rescaler = RescalePopup(iris_radius_px, 5.9, self.vid_path, self.stack)
        self.rescaler.show()
        self.rescaler.raise_()


def plot(ax, xs, ys, colors=None, names=None):
    """
    :param ax: the matplotlib axs to plot on
    :param xs: the x values to plot
    :param ys: the y values to plot
    """
    if colors is not None and names is not None:
        for i in range(len(ys)):
            ax.plot(xs, ys[i], color=colors[i], label=names[i])
        ax.legend()
    else:
        for i in range(len(ys)):
            ax.plot(xs, ys[i])
    return ax.plot([0, 0], [np.nanmax(ys), np.nanmin(ys)])


def round_str(value: float, decimals: int = 2):
    return str(round(value, decimals))


def norm_xcor(a: np.ndarray, b: np.ndarray):
    not_nan = np.logical_and(np.logical_not(np.isnan(a)), np.logical_not(np.isnan(b)))
    a = a[not_nan]
    b = b[not_nan]
    return np.corrcoef(a, b)[0, 1]
