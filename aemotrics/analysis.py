import os
from typing import List
import cv2
import dlc_generic_analysis as dga
import numpy as np
from matplotlib import patches, pyplot as plt
from . import _geometries, _utils, _draw
from ._points import IRIS_POINTS_R, IRIS_POINTS_L
from logging import info
import pandas as pd

dlc_model_dir_name = "Aemotrics_V3-Nate-2023-06-29"
model_dir = os.path.join(os.path.dirname(os.path.split(os.path.realpath(__file__))[0]), 'model', dlc_model_dir_name)

if not os.path.isfile(os.path.join(model_dir, "config.yaml")) and not os.path.isfile(
    os.path.join(model_dir, "config.yml")
):
    FileNotFoundError("No Model Downloaded")


def dlc_analyze(videos: List[str], dlc_model_path=None):
    """
    runs Deeplabcut analyze_videos on videos using the model in dlc_model_path
    :param videos: a list of paths to videos to analyze
    :param dlc_model_path: the path to the deeplabcut model directory if None it will look in the package base directory
    """
    if dlc_model_path is None:
        dlc_model_path = model_dir
    return dga.dlc_analyze(dlc_model_path, videos, gputouse=0)


class Analysis(dga.Analysis):
    def __init__(
        self,
        h5_path: str,
        dlc_model: str,
        video_path: str,
        start_frame: int = 0,
        end_frame: int = None,
        midline_adjustment: np.float_ = None,
        midline_adjustment_frame: int = None,
    ):
        """
        Analyze a video with the results from the deeplabcut prediction.

        calculates the midline of the face, area of the eyes, areas of the mouth on each side of the midline using the
        adjusted midline of one is provided. per frame midline adjustments override the global midline adjustment.
        :param h5_path: the path to the '.h5' file created by deeplabcut's prediction
        :param dlc_model: the name of the dlc model returned by deeplabcut.analyze_videos
        :param video_path: the raw video to draw on
        :param start_frame: the first frame to analyze
        :param end_frame: the last frame to analyze
        :param midline_adjustment: an offset angle to adjust the angle of the midline by for every frame.
        :param midline_adjustment_frame: an offset angle for each frame that offsets the frame only for that frame.
        no offset can be 0 or np.nan
        """
        self.video_path = video_path
        dir_path = os.path.split(video_path)[0]
        h5_path = os.path.abspath(h5_path)
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(h5_path)
        dga.Analysis.__init__(self, h5_path, dlc_model, startframe=start_frame, endframe=end_frame)
        eye_areas, eye_draw_r, eye_draw_l = _geometries.eyes(self.df)
        iris_r_points = dga.utils.point_array(self.df, IRIS_POINTS_R)
        iris_center_r, iris_radei_r = _utils.iris_circles(iris_r_points)
        self.iris_r = np.concatenate(
            (iris_center_r, iris_radei_r.reshape(iris_radei_r.shape[0], 1)), axis=1
        )
        iris_center_l, iris_radei_l = _utils.iris_circles(
            dga.utils.point_array(self.df, IRIS_POINTS_L)
        )
        self.iris_l = np.concatenate(
            (iris_center_l, iris_radei_l.reshape(iris_radei_l.shape[0], 1)), axis=1
        )
        self.iris_radius = np.nanmean([iris_radei_r, iris_radei_l])
        radei = np.concatenate([iris_radei_r, iris_radei_l])
        self.pix_size = _utils.eye_unit_convert(radei)

        vid = cv2.VideoCapture(video_path)
        self.frame_size = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vid.release()
        eye_lines, mid_x, midlines = _geometries.midline(self.df, self.frame_size[1])
        if midline_adjustment is not None:
            if midline_adjustment_frame is not None:
                (midlines[midline_adjustment_frame],) = _adjust_line(
                    [midlines[midline_adjustment_frame]], midline_adjustment
                )[0]
            else:
                midlines = _adjust_line(midlines, midline_adjustment)
        # Mouth
        frames = midlines.shape[0]
        mouth = _geometries.Mouth(self.df, midlines)

        mouth_overlaps = _geometries.mouth_poly_intersect(
            mouth.draw_pts_r, mouth.draw_pts_l, midlines
        )
        # Inner Mouth
        (
            inner_areas,
            mouth_area_inner_draw_r,
            mouth_area_inner_draw_l,
            tcl_l_inner,
            inner_mouth_x,
        ) = _geometries.inner_mouth_areas(self.df, midlines)
        inner_mouth_overlaps = _geometries.mouth_poly_intersect(
            mouth_area_inner_draw_r, mouth_area_inner_draw_l, midlines
        )
        # Brow
        (
            brow_height_lines_r,
            brow_height_lines_l,
            brow_height_r,
            brow_height_l,
        ) = _geometries.brow(self.df, eye_lines)
        # Lower Lips
        lip_heights_r, lip_heights_l = _geometries.lower_lip(
            eye_lines,
            mouth.bisect_points_r,
            mouth.bisect_points_l,
        )
        max_ind = 0
        right_mouth_lower_intersect = dga.utils.nan((midlines.shape[0], 2))
        left_mouth_lower_intersect = dga.utils.nan((midlines.shape[0], 2))
        for i in range(frames):
            r_line = mouth.perp_bisects_r[i]
            l_line = mouth.perp_bisects_l[i]
            tck = mouth.tck_ls[i]
            x = mouth.xs[i]
            if r_line is not None and not isinstance(tck, int):
                right_mouth_lower_intersect[i] = _utils.set_intersect(tck, x, r_line)
            else:
                right_mouth_lower_intersect.append(None)
            if l_line is not None and not isinstance(tck, int):
                left_mouth_lower_intersect[i] = _utils.set_intersect(tck, x, l_line)
        lower_lip_tangents_r = dga.utils.nan((midlines.shape[0], 6))
        lower_lip_tangents_l = dga.utils.nan((midlines.shape[0], 6))
        for i in range(frames):
            if not np.isnan(midlines[i]).any():
                slope = [i, 0]
                if isinstance(right_mouth_lower_intersect, tuple) and isinstance(
                    left_mouth_lower_intersect, tuple
                ):
                    r_y_int = (
                        right_mouth_lower_intersect[i][1]
                        - right_mouth_lower_intersect[i][0] * slope
                    )
                    l_y_int = (
                        left_mouth_lower_intersect[i][1] - left_mouth_lower_intersect[i][0] * slope
                    )
                    mid = mid_x[i]
                    right_tan_point = dga.utils.point_array(self.df, ["ROC"])
                    right_tan_line = dga.line.from_points(
                        np.stack(
                            (
                                right_tan_point[i, 0],
                                right_tan_point[i, 0] * slope + r_y_int,
                            )
                        ),
                        np.stack((mid, mid * slope + r_y_int)),
                    ).astype(int)
                    left_tan_point = dga.utils.point_array(self.df, ["LOC"])
                    left_tan_line = dga.line.from_points(
                        np.stack(
                            left_tan_point[i][0],
                            left_tan_point[i][0] * slope + l_y_int,
                        ),
                        np.stack((mid, mid * slope + l_y_int)),
                    ).astype(int)
                    lower_lip_tangents_r[i] = right_tan_line
                    lower_lip_tangents_l[i] = left_tan_line
        # Simple area ratios for mouth.
        mouth_area_ratios, mouth_ratio_max = _utils.get_ratios(mouth.areas)
        mouth_overlap_ratios = dga.utils.nan(frames)

        # Mouth Projection overlap ratios
        outer_larger = dga.utils.nan(frames)
        for i in range(frames):
            if mouth.areas[i, 0] > 0:
                larger = max(mouth.areas[i, 0], mouth.areas[i, 1])
                mouth_overlap_ratios[i] = mouth_overlaps[i] / larger
                outer_larger[i] = max((mouth.areas[i, 0], mouth.areas[i, 1]))

        # Mouth Projection Plot
        mouth_projection_pts_l = []
        mouth_projection_pts_r = []
        proj_pts_x = []
        proj_pts_y = []
        right_pts_x = []
        right_pts_y = []
        if mouth.draw_pts_l[max_ind] is not None:
            for point in mouth.draw_pts_l[max_ind]:
                if point is not None and not np.array_equal(point, np.array([np.nan, np.nan])):
                    pivot = (point[1] - midlines[max_ind, 1]) / midlines[max_ind, 0]
                    new_r = 2 * pivot - point[0]
                    new_l = point[1]
                    mouth_projection_pts_l.append([new_r, new_l])
                    proj_pts_x.append(new_r)
                    proj_pts_y.append(new_l)
                else:
                    mouth_projection_pts_l.append([np.nan, np.nan])

                    proj_pts_x.append(np.nan)
                    proj_pts_y.append(np.nan)
        self.mouth_projection_pts_l = np.array(mouth_projection_pts_l)
        if mouth.draw_pts_r[max_ind] is not None:
            for point in mouth.draw_pts_r[max_ind]:
                if point is not None and not np.array_equal(point, np.array([np.nan, np.nan])):
                    mouth_projection_pts_r.append([point[0], point[1]])
                    right_pts_x.append(point[0])
                    right_pts_y.append(point[1])
                else:
                    mouth_projection_pts_r.append([-np.nan, np.nan])
                    right_pts_x.append(np.nan)
                    right_pts_y.append(np.nan)
        self.mouth_projection_pts_r = np.array(mouth_projection_pts_r)
        eye_ratios = []
        if eye_areas is not None:
            for areas in eye_areas:
                if areas[0] != 0 and areas[0] is not None and areas[1] is not None:
                    num = max(abs(areas[0]), abs(areas[1]))
                    den = min(abs(areas[0]), abs(areas[1]))
                    if num / den < 5:
                        eye_ratios.append(num / den)
                    else:
                        eye_ratios.append(5)
                else:
                    eye_ratios.append(np.nan)

        # Inner Mouth simple area
        inner_mouth_area_ratios, _ = _utils.get_ratios(inner_areas)

        # Inner Mouth Projection Overlaps
        inner_larger = dga.utils.nan(frames)
        inner_mouth_overlap_ratios = dga.utils.nan(frames)
        for i in range(len(inner_mouth_overlaps)):
            if inner_areas[i][0] != 0:
                larger = max(inner_areas[i][0], inner_areas[i][1])
                inner_mouth_overlap_ratios[i] = inner_mouth_overlaps[i] / larger
                inner_larger[i] = larger
        # Inner Projection Plot
        if (
            mouth_area_inner_draw_l[max_ind] is not None
            and mouth_area_inner_draw_r[max_ind] is not None
        ):
            inner_proj_pts_x = []
            inner_proj_pts_y = []
            right_inner_x = []
            right_inner_y = []
            for point in mouth_area_inner_draw_l[max_ind]:
                pivot = (point[1] - midlines[max_ind, 1]) / midlines[max_ind, 0]
                new_r = 2 * pivot - point[0]
                new_l = point[1]
                inner_proj_pts_x.append(new_r)
                inner_proj_pts_y.append(new_l)
            for point in mouth_area_inner_draw_r[max_ind]:
                right_inner_x.append(point[0])
                right_inner_y.append(point[1])

        # Symmetry Scores
        symmetry = dga.utils.nan(len(mouth_overlaps))
        for i in range(len(mouth_overlaps)):
            if not np.isnan(mouth_overlaps[i]) and mouth_overlaps[i] > 0:
                score = mouth_overlaps[i] / outer_larger[i]
                symmetry[i] = score
            elif (
                inner_mouth_overlaps[i] is None
                or inner_mouth_overlaps[i] < 0
                and mouth_overlaps[i] is not None
                and mouth_overlaps[i] > 0
            ):
                score = mouth_overlaps[i] / outer_larger[i]
                symmetry[i] = score
        right_max = 0
        left_max = 0
        for i in range(len(mouth.areas)):
            if mouth.areas[i, 0] > right_max:
                right_max = mouth.areas[i, 0]
            if mouth.areas[i, 1] > left_max:
                left_max = mouth.areas[i, 1]
        symmetry = np.array(symmetry)
        # CSV return and output data
        compiled_data_list = []  # Returned by analyze with compiled data from video
        """compiled data indices: 
        0: video path
        1: symmetry score - 97th percentile difference frame for mouth, talk to nate about eyes.
        2: """
        temp_mouth = []
        for item in mouth_area_ratios:
            if item is not None:
                temp_mouth.append(item)
            else:
                temp_mouth.append(np.nan)
        np_mouth = np.array(temp_mouth)
        compiled_data_list.append(dir_path)
        compiled_data_list.append(np.percentile(np_mouth, 97))
        mouth_projection_pts_l = np.squeeze(mouth_projection_pts_l)
        # Corner mouth excursions
        excursion_ratios = []
        for i in range(len(mouth_area_ratios)):
            # List must be as long as mouth ratios for writer. Fill none if empty at end
            if i <= len(mouth.excursions_l) or i <= len(mouth.excursions_r):
                left_pix_dist = mouth.excursions_l[i]
                right_pix_dist = mouth.excursions_r[i]
                if left_pix_dist + right_pix_dist > 0:
                    excursion_ratios.append(
                        (right_pix_dist - left_pix_dist) / (right_pix_dist + left_pix_dist) / 2
                    )
                else:
                    excursion_ratios.append(np.nan)
            else:
                excursion_ratios.append(np.nan)
        self.mouth = mouth
        self.symmetry = symmetry
        self.mouth_area_ratios = mouth_area_ratios
        self.inner_mouth_area_ratios = inner_mouth_area_ratios
        self.mouth_excursion_ratios = excursion_ratios
        self.mouth_overlap_ratios = mouth_overlap_ratios
        self.lip_heights_r = lip_heights_r
        self.lip_heights_l = lip_heights_l
        self.mouth_projection_pts_r = np.array(mouth_projection_pts_r)
        self.mouth_projection_pts_l = np.array(mouth_projection_pts_l)
        self.brow_heights_r = brow_height_r
        self.brow_heights_l = brow_height_l
        self.brow_height_lines_r = brow_height_lines_r
        self.brow_height_lines_l = brow_height_lines_l
        self.eye_ratios = eye_ratios
        self.eye_areas = eye_areas
        self.sp_draw = np.stack([eye_draw_r, eye_draw_l])
        self.eye_lines = eye_lines
        self.midlines = midlines
        self.mouth_area_inner_draw_r = mouth_area_inner_draw_r
        self.mouth_area_inner_draw_l = mouth_area_inner_draw_l
        self.lower_lip_tangents_r = lower_lip_tangents_r
        self.lower_lip_tangents_l = lower_lip_tangents_l
        self._csv_path = ""
        self._draw_path = ""
        self.plot_path = ""
        symmetry_df = pd.DataFrame(
            {
                "symmetry": self.symmetry,
                "pix_size": self.pix_size,
            }
        )
        midp0 = self.midlines[:, 2:4]
        midline_point1 = pd.DataFrame({"x": midp0[:, 0], "y": midp0[:, 1]})
        glab = self.df["Glab"][["x", "y"]]
        midline = pd.concat([glab, midline_point1], keys=["glabella", "midlineP2"], axis=1)
        eyes = pd.DataFrame(
            {
                "area_r": self.eye_areas[:, 0],
                "area_l": self.eye_areas[:, 1],
                "area_ratio": self.eye_ratios,
                "brow_height_r": self.brow_heights_r,
                "brow_height_l": self.brow_heights_l,
                "iris_radei_r": self.iris_r[:, 2],
                "iris_radei_l": self.iris_l[:, 2],
            }
        )
        mouth = pd.DataFrame(
            {
                "area_r": self.mouth.areas[:, 0],
                "area_l": self.mouth.areas[:, 1],
                "excursions_r": self.mouth.excursions_r,
                "excursions_l": self.mouth.excursions_l,
                "lip_height_r": self.lip_heights_r,
                "lip_height_l": self.lip_heights_l,
                "excursion_ratio": self.mouth_excursion_ratios,
                "area_ratio": self.mouth_area_ratios,
                "area_overlap": self.mouth_overlap_ratios,
            }
        )
        df = pd.concat([symmetry_df, eyes, mouth], keys=["symmetry", "eyes", "mouth"], axis=1)
        df = df.join(midline)
        self.df = df

    def write_h5(self, filename: str = None) -> str:
        if filename is not None:
            data_file = os.path.splitext(filename)[0]
        else:
            dir_path, video = os.path.split(self.video_path)
            new_dir_path = os.path.join(dir_path, "analyzed_videos")
            if not os.path.isdir(new_dir_path):
                os.mkdir(new_dir_path)
            path = os.path.join(new_dir_path, video)
            data_file = os.path.splitext(path)[0]
        key = os.path.split(data_file)[1]
        df = pd.concat([self.df], keys=[key], axis=1)
        df.to_hdf(data_file + ".h5", key=key)
        return data_file

    def write_csv(self, filename: str = None) -> str:
        if filename is not None:
            data_file = os.path.splitext(filename)[0]
        else:
            dir_path, video = os.path.split(self.video_path)
            new_dir_path = os.path.join(dir_path, "analyzed_videos")
            if not os.path.isdir(new_dir_path):
                os.mkdir(new_dir_path)
            path = os.path.join(new_dir_path, video)
            data_file = os.path.splitext(path)[0]
        self.df.to_csv(data_file + ".csv")
        return data_file

    def plot(self, show: bool = False, save: bool = False, out_dir: str = "") -> str:
        """
        :param save: if true all figures will be saved as pngs in out_dir
        :param show: if true all figures will be shown
        :param out_dir: the directory the plots save to
        :return:
        """
        if out_dir != "":
            dir_path = out_dir
        else:
            dir_path = os.path.splitext(self.video_path)[0] + "_plots"
        # MPL Plotting
        # Area Overlaps
        video_name = os.path.split(os.path.splitext(self.video_path)[0])[1]
        plt.figure()
        plt.title("Area Overlap")
        plt.scatter(
            self.mouth_projection_pts_l[:, 0],
            self.mouth_projection_pts_l[:, 1],
            c="green",
        )
        plt.scatter(
            self.mouth_projection_pts_r[:, 0],
            self.mouth_projection_pts_r[:, 1],
            c="lightblue",
        )
        out_path = os.path.join(dir_path, video_name + "Area_overlap.png")
        if save:
            plt.savefig(out_path)
        if show:
            plt.show()
        # Ratio Symmetry
        plt.figure()
        plt.title("Symmetry Ratios")
        plt.plot(self.mouth_area_ratios, c="green")
        plt.plot(self.inner_mouth_area_ratios, c="yellow")
        plt.plot(self.eye_ratios, c="lightblue")
        out_path = os.path.join(dir_path, video_name + "Symmetry_ratios.png")
        if save:
            plt.savefig(out_path)
        if show:
            plt.show()
        plt.figure()

        # Mouth Overlap
        plt.title("Mouth Overlap")
        x = [i for i in range(len(self.mouth_overlap_ratios))]
        plt.scatter(x, self.mouth_overlap_ratios)
        out_path = os.path.join(dir_path, video_name + "Mouth_Overlap.png")
        if save:
            plt.savefig(out_path)
        if show:
            plt.show()
        # Symmetry Score
        scores = self.symmetry.astype(np.double)
        plot_scores = np.isfinite(scores)
        fig, ax1 = plt.subplots()

        # Symmetry
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Symmetry Score")
        ax1.plot(plot_scores, c="lightblue")
        ax1.tick_params(axis="y", labelcolor="lightblue")

        # Excursion
        ax2 = ax1.twinx()
        ax2.set_ylabel("Excursion distance (px)")
        ax2.plot(self.mouth.excursions_l, c="red")
        ax2.plot(self.mouth.excursions_r, c="yellow")

        # Legend
        blue_patch = patches.Patch(color="lightblue", label="Symmetry Score")
        red_patch = patches.Patch(color="red", label="Left Side Excursions")
        yellow_patch = patches.Patch(color="yellow", label="Right Side Excursions")
        plt.legend(handles=[blue_patch, yellow_patch, red_patch])

        fig.tight_layout()
        out_path = os.path.join(dir_path, video_name + "scores.png")
        if save:
            plt.savefig(out_path)
        if show:
            plt.show()
        self.plot_path = dir_path
        if save:
            return dir_path

    def draw(self) -> str:
        """
        draws areas and lines on the video and returns a new video
        :return:
        """
        out_path = self.video_path
        path = _draw.draw(
            self.video_path,
            out_path,
            self.mouth.draw_pts_r,
            self.mouth.draw_pts_l,
            self.sp_draw,
            self.midlines,
            self.eye_lines,
            self.brow_height_lines_r,
            self.brow_height_lines_l,
            self.mouth.excursion_lines_r,
            self.mouth.excursion_lines_l,
            self.lower_lip_tangents_r,
            self.lower_lip_tangents_l,
            self.mouth_area_inner_draw_r,
            self.mouth_area_inner_draw_l,
            self.iris_l,
            self.iris_r,
        )
        info("Video is here: " + self.video_path)
        self._draw_path = path
        return path


def analyze(
    videos: [str],
    write_csv: bool = True,
    draw: bool = True,
    plot: bool = False,
    dlc_model_path: str = None,
) -> List[Analysis]:
    """
    Runs deeplabcut inference on each video in videos then runs analysis on those results
    :param videos:
    :param write_csv:
    :param draw: If set true draws analysis on the videos
    :param plot: If set true plots the resultant values from the analysis
    :param dlc_model_path: the path to the deeplabcut model directory if None it will look in the package base directory
    :return: a list of Analysis, 1 for each video in videos
    """
    if dlc_model_path is None:
        dlc_model_path = model_dir
    print(f"dlc_model_path: {dlc_model_path}")
    h5s, model = dga.dlc_analyze(dlc_model_path, videos, gputouse=0)
    h5s = dga.filter.threshold_confidence(h5s, 0.85)
    return do_analysis(h5s, videos, model, write_csv, draw, plot)


def do_analysis(
    h5s,
    video_paths,
    dlc_scorer,
    write_csv: bool = True,
    draw: bool = True,
    show_plots=False,
    save_plots=False,
) -> List[Analysis]:
    """
    Runs an analysis for each h5.
    :param h5s: The h5 files to analyze
    :param video_paths: The videos that are being analyzed
    :param dlc_scorer: The name of the deeplabcut model used to analyze
    :param write_csv: If set true writes csv of analysis result
    :param draw: If set true draws analysis on the videos
    :param show_plots: If set true plots the resultant values from the analysis
    :param save_plots:
    :return: the list of analyses
    """
    analyses = []
    for i in range(len(h5s)):
        analyses.append(Analysis(h5s[i], dlc_scorer, video_paths[i]))
        analyses[i].write_h5()
    if write_csv:
        csvs = []
        for a in analyses:
            csvs.append(a.write_csv())
    if draw:
        annotated_videos = []
        for a in analyses:
            annotated_videos.append(a.draw())

    if show_plots or save_plots:
        for a in analyses:
            a.plot(show_plots, save_plots)
    print("Aemotrics Analysis Finished!")
    return analyses


def _adjust_line(lines, angle):
    slope = np.tan(np.arctan(lines[:, 0]) + angle)
    new_lines = dga.line.from_points(
        lines[:, 2:4],
        np.stack((lines[:, 2] - (lines[:, 3] - lines[:, 5]) * (-1 / slope), lines[:, 5]), axis=1),
    )
    return new_lines
