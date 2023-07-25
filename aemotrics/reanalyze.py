import os.path
import numpy as np
from qtpy import QtWidgets
import dlc_generic_analysis as dga
from aemotrics.analysis import Analysis
from glob import glob


class ReanalyzePopup(QtWidgets.QWidget):
    def __init__(self, new_line, old_line, video_path, frame):
        """
        reanalyze a video with a popup that allows the user to select to either reanalyze the whole video or just 1 frame
        :param new_line: the new midline for the manually corrected frame
        :param old_line: the old midline for the manually corrected frame
        :param video_path: the absolute path to the video being reanalyzed
        :param frame: the number of the frame manually corrected as a 0 indexed integer
        """
        super(ReanalyzePopup, self).__init__()
        self.angle = -dga.utils.angle_between_lines(
            dga.line.from_points_1d(old_line[0], old_line[1])[0],
            dga.line.from_points_1d(new_line[0], new_line[1])[0],
        )
        self.frame = frame
        self.video_path = video_path
        adjust_all = QtWidgets.QPushButton("Offset all frames")
        adjust_single = QtWidgets.QPushButton("Offset only this frame")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(adjust_all)
        layout.addWidget(adjust_single)
        self.setLayout(layout)
        adjust_all.clicked.connect(self.on_adjust_all)
        adjust_single.clicked.connect(self.on_adjust_single)
        self.adjust = ""

    def on_adjust_all(self):
        self.reanalyze(True)

    def on_adjust_single(self):
        self.reanalyze()

    def reanalyze(self, adjust_all=False):
        """
        reanalyzes the video with the new angle on the single frame or on all frames if adjust_all == True
        :param adjust_all: determines if all frames are re-analyzed
        """
        f = os.path.splitext(self.video_path)[0]
        parent_dir = os.path.split(os.path.split(f)[0])[0]
        h5s = glob(os.path.join(parent_dir, os.path.split(f)[1]) + "*_threshold.h5")
        video = glob(os.path.join(parent_dir, os.path.split(f)[1]) + ".*")
        if len(h5s) > 1:
            print("found multiple deeplabcut analyses. picking the first one")
        if adjust_all:
            analysis = Analysis(
                h5s[0],
                os.path.split(os.path.splitext(h5s[0])[0])[1]
                .replace(os.path.split(f)[1], "")
                .replace("_filtered", "")
                .replace("_threshold", ""),
                video[0],
                midline_adjustment=np.float_(self.angle),
            )
        else:
            analysis = Analysis(
                h5s[0],
                os.path.split(os.path.splitext(h5s[0])[0])[1]
                .replace(os.path.split(f)[1], "")
                .replace("_filtered", "")
                .replace("_threshold", ""),
                video[0],
                midline_adjustment=np.float_(self.angle),
                midline_adjustment_frame=self.frame,
            )
        analysis.write_csv()
        analysis.draw()
        print("Reanalysis Done!")
        return analysis
