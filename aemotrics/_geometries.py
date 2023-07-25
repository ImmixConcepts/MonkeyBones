from typing import List
import dlc_generic_analysis as dga
import numpy as np
import scipy.stats
from pandas import DataFrame
from scipy import interpolate
from shapely.geometry import Polygon
from . import _points
from . import _utils


# Number of actions performed on each Riemann sum for spline integration.
SUM_TOTAL = 100
CORNEA_FRAMES = 100
# Number of tolerated pixels of deviation from cubic to linear mouth splines.
PIX_TOLERANCE = 10


class Mouth:
    def __init__(self, data_frame: DataFrame, mid_lines: np.ndarray):
        """
        Creates the areas and metrics for the mouth
        """
        points = dga.utils.point_array(data_frame, _points.MOUTH_PTS)
        """Calls functions related to mouth metrics. Called by analyze."""
        self.areas = dga.utils.nan((points.shape[1], 2))
        # Points passed to draw function
        self.draw_pts_r = []
        self.draw_pts_l = []
        # Excursion lines
        self.excursions_r = dga.utils.nan(points.shape[1])
        self.excursions_l = dga.utils.nan(points.shape[1])
        self.excursion_lines_r = dga.utils.nan((points.shape[1], 6))
        self.excursion_lines_l = dga.utils.nan((points.shape[1], 6))
        # Perpendicular bisecting excursion lines
        self.perp_excursions_r = dga.utils.nan(points.shape[1])
        self.perp_excursions_l = dga.utils.nan(points.shape[1])
        self.perp_bisects_r = dga.utils.nan((points.shape[1], 6))
        self.perp_bisects_l = dga.utils.nan((points.shape[1], 6))
        self.bisect_points_r = dga.utils.nan([points.shape[1], 2])
        self.bisect_points_l = dga.utils.nan([points.shape[1], 2])
        self.tck_ls = []
        self.xs = []
        for i in range(points.shape[1]):
            # Set up points
            pts = points[:, i]
            if not np.isnan(pts).any():
                # Split points into upper and lower lips x and y. Oral commissures appear in both
                # lists.
                upper = pts[0:11, :]
                lower = np.concatenate([np.atleast_2d(pts[0]), pts[11:20], np.atleast_2d(pts[10])])
                # Distance from oral commissure to midline.
                r_ex_y = pts[0, 1]
                l_ex_y = pts[8, 1]
                if mid_lines[i] is not None:
                    slope = mid_lines[i, 0]
                    y_int = mid_lines[i, 1]
                    r_mid_x = (r_ex_y - y_int) / slope
                    l_mid_x = (l_ex_y - y_int) / slope
                    self.perp_excursions_r[i] = dga.utils.scalar_dist(
                        pts[0], np.stack([r_mid_x, r_ex_y])
                    )
                    self.perp_excursions_l[i] = dga.utils.scalar_dist(
                        pts[8], np.stack([l_mid_x, l_ex_y])
                    )
                    r_y_int = r_ex_y + (1 / slope) * pts[0][0]
                    l_y_int = l_ex_y + (1 / slope) * pts[10][0]
                    # Corner excursion bisectors
                    r_mid = r_mid_x - (r_mid_x - pts[0, 0]) / 2
                    l_mid = l_mid_x - (l_mid_x - pts[10, 0]) / 2
                    r_ref_pt = (r_mid, r_mid * (-1 / slope) + r_y_int)
                    l_ref_pt = (l_mid, l_mid * (-1 / slope) + l_y_int)
                    self.perp_bisects_r[i] = dga.line.from_slope(
                        slope=slope, intercept=r_ref_pt[1] - r_mid * slope
                    )
                    self.perp_bisects_l[i] = dga.line.from_slope(
                        slope=slope, intercept=l_ref_pt[1] - l_mid * slope
                    )
                # Compute area and gather drawing information
                # Sort upx and lx in order of ascending value and corresponding ys with them.
                # upper[:, 0], upper[:, 1] = sort_pts(upper[:, 0], upper[:, 1])
                # lower[:, 0], lower[:, 1] = sort_pts(lower[:, 0], lower[:, 1])
                # Simpler n^2 sort; small data
                (
                    r_area,
                    l_area,
                    draw_r,
                    draw_l,
                    mid_lower,
                    bisect_r,
                    bisect_l,
                    tck_l,
                    x,
                ) = spline_area_mouth(
                    upper,
                    lower,
                    mid_lines[i],
                    self.perp_bisects_r[i],
                    self.perp_bisects_l[i],
                )
                if x is not None:
                    self.areas[i] = [r_area, l_area]
                    self.draw_pts_r.append(draw_r)
                    self.draw_pts_l.append(draw_l)
                    if bisect_r is not None and bisect_l is not None:
                        self.bisect_points_r[i] = bisect_r
                        self.bisect_points_l[i] = bisect_l
                    self.tck_ls.append(tck_l)
                    self.xs.append(x)
                    # Excursion from lower lip midpoint
                    self.excursions_r[i] = dga.utils.scalar_dist(pts[0], mid_lower)
                    self.excursions_l[i] = dga.utils.scalar_dist(pts[10], mid_lower)
                    int_mid_pt = np.stack((mid_lower[0], mid_lower[1])).astype(int)
                    self.excursion_lines_r[i] = dga.line.from_points_1d(
                        pts[0].astype(int), int_mid_pt
                    )
                    self.excursion_lines_l[i] = dga.line.from_points_1d(
                        pts[10].astype(int), int_mid_pt
                    )
                else:
                    self.draw_pts_r.append(None)
                    self.draw_pts_l.append(None)
                    self.tck_ls.append(np.nan)
                    self.xs.append(np.nan)
            else:
                # Empty returns in case of invalid data
                self.draw_pts_r.append(None)
                self.draw_pts_l.append(None)
                self.tck_ls.append(np.nan)
                self.xs.append(np.nan)
        self.areas = np.stack(self.areas)


def intersection(line0: np.ndarray, line1: np.ndarray):
    """
    Finds the intersection between 2 lines
    :param line0:
    :param line1:
    """
    x = (line0[:, 1] - line1[:, 1]) / (line1[:, 0] - line0[:, 0])
    y = line0[:, 0] * x + line0[:, 1]
    return np.stack([x, y], axis=1)


def _brow_compute(eye_lines, brow_pts):
    brow_lines = dga.utils.nan((brow_pts.shape[1], 6))
    for i in range(brow_pts.shape[1]):
        if not np.isnan(brow_pts[:, i]).any():
            linreg = scipy.stats.linregress(brow_pts[:, i, 0], brow_pts[:, i, 1])
            e0y = linreg.slope * brow_pts[0, i, 0] + linreg.intercept
            e1y = linreg.slope * brow_pts[2, i, 0] + linreg.intercept
            brow_lines[i] = dga.line.from_points_1d(
                np.array([brow_pts[0, i, 0], e0y]), np.array([brow_pts[2, i, 0], e1y])
            )
    brow_midpoints = np.stack(
        [
            np.where(
                np.greater(brow_lines[:, 4], brow_lines[:, 2]),
                (brow_lines[:, 4] - brow_lines[:, 2]) / 2 + brow_lines[:, 2],
                (brow_lines[:, 2] - brow_lines[:, 4]) / 2 + brow_lines[:, 4],
            ),
            np.where(
                np.greater(brow_lines[:, 5], brow_lines[:, 3]),
                (brow_lines[:, 5] - brow_lines[:, 3]) / 2 + brow_lines[:, 3],
                (brow_lines[:, 3] - brow_lines[:, 5]) / 2 + brow_lines[:, 5],
            ),
        ],
        axis=1,
    )
    brow_intersection_lines = dga.line.from_points(
        brow_midpoints,
        np.stack(
            [brow_midpoints[:, 0] + 1, brow_midpoints[:, 1] + -1 / eye_lines[:, 0]],
            axis=1,
        ),
    )
    brow_eye_intersect = intersection(brow_intersection_lines, eye_lines)
    brow_height_lines = dga.line.from_points(brow_midpoints, brow_eye_intersect)
    brow_heights = dga.utils.dist(
        np.stack([brow_height_lines[:, 2], brow_height_lines[:, 3]], axis=1),
        np.stack([brow_height_lines[:, 4], brow_height_lines[:, 5]], axis=1),
    )
    return brow_height_lines, brow_heights


def lower_lip(
    eye_lines: np.ndarray,
    r_bisects: np.ndarray,
    l_bisects: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    for each point calculates the distance between the lower lip point and the nearest point on the eye line that
    intercepts a line parallel to the midline through the lip point
    :param eye_lines: the line between the lateral canthi represented as a line ndarray
    :param r_bisects: the right mouth bisect points
    :param l_bisects: the left mouth bisect points
    :return: right_lower_lip_heights, left_lower_lip_heights
    """
    eye_y_max = np.amax(np.stack([eye_lines[:, 3], eye_lines[:, 5]]), axis=0) + 1
    right_vertical_lines = dga.line.from_points(
        r_bisects,
        np.stack(
            [
                eye_lines[:, 0] * (eye_y_max - r_bisects[:, 1]) + r_bisects[:, 0],
                eye_y_max,
            ],
            axis=1,
        ),
    )
    left_vertical_lines = dga.line.from_points(
        l_bisects,
        np.stack(
            [
                eye_lines[:, 0] * (eye_y_max - l_bisects[:, 1]) + l_bisects[:, 0],
                eye_y_max,
            ],
            axis=1,
        ),
    )
    right_top_points = intersection(eye_lines, right_vertical_lines)
    left_top_points = intersection(eye_lines, left_vertical_lines)
    right_lower_lip_heights = dga.utils.dist(right_top_points, r_bisects)
    left_lower_lip_heights = dga.utils.dist(left_top_points, l_bisects)
    return (
        right_lower_lip_heights,
        left_lower_lip_heights,
    )


def spline_area_mouth(
    upper: np.ndarray, lower: np.ndarray, mid_line: np.ndarray, r_bisector: np.ndarray, l_bisector
):
    """
    Computes areas using dga.utils.shoelace and approximating mouth as a 200 sided polygon. using bsplines
    :param upper: the upper mouth points
    :param lower: the lower mouth points
    :param mid_line: the facial midline
    :param r_bisector:
    :param l_bisector
    """
    if (
        upper[0] is None
        or upper[1] is None
        or np.isnan(upper).any()
        or lower[0] is None
        or lower[1] is None
        or np.isnan(lower).any()
        or np.isnan(mid_line).any()
    ):
        return -1, -1, None, None, None, None, None, None, None
    upper = upper[np.argsort(upper[:, 0], axis=0)]
    lower = lower[np.argsort(lower[:, 0], axis=0)]
    tck_u = interpolate.splrep(upper[:, 0], upper[:, 1], k=1)
    # Points need to be ordered by x value
    tck_l = interpolate.splrep(lower[:, 0], lower[:, 1], k=1)
    # Set cubic splines
    cubic_upper = interpolate.splrep(upper[:, 0], upper[:, 1], k=3)
    cubic_lower = interpolate.splrep(lower[:, 0], lower[:, 1], k=3)

    # Set up evenly spaced points along spline
    dist = (upper[-1, 0] - upper[0, 0]) / SUM_TOTAL
    draw_pts_r = []
    draw_pts_l = []
    x = upper[0, 0] + np.arange(SUM_TOTAL) * dist

    # Set up points along left and right side
    right_mouth_bisect = _utils.set_intersect(tck_l, x, r_bisector)
    left_mouth_bisect = _utils.set_intersect(tck_l, x, l_bisector)
    hu = interpolate.splev(x, tck_u)
    hl = interpolate.splev(x, tck_l)
    test_hu = interpolate.splev(x, cubic_upper)
    test_hl = interpolate.splev(x, cubic_lower)

    # Test for use of cubic or linear splines
    too_great_dif = False
    for i in range(len(hu)):
        if abs(test_hu[i] - hu[i]) > PIX_TOLERANCE:
            too_great_dif = True
            break
    for i in range(len(hl)):
        if abs(test_hl[i] - hl[i]) > PIX_TOLERANCE:
            too_great_dif = True
            break
    if not too_great_dif:
        hu = test_hu
        hl = test_hl
    rx = []
    ry = []
    lx = []
    ly = []
    mid_x_up = _utils.set_intersect(tck_u, x, mid_line)[0]
    mid_lower_pt = _utils.set_intersect(tck_l, x, mid_line)
    mid_x_l = mid_lower_pt[0]

    # Reorder points in case they appear out of order.
    for i in range(SUM_TOTAL):
        if x[i] < mid_x_up:
            rx.append(x[i])
            ry.append(hu[i])
            draw_pts_r.append([x[i], hu[i]])
        else:
            lx.append(x[i])
            ly.append(hu[i])
            draw_pts_l.append([x[i], hu[i]])
    stop_r = len(rx)
    stop_l = len(lx)
    for i in range(SUM_TOTAL):
        if x[i] < mid_x_l:
            rx.insert(stop_r, x[i])
            ry.insert(stop_r, hl[i])
            draw_pts_r.insert(stop_r, [x[i], hl[i]])
        else:
            lx.insert(stop_l, x[i])
            ly.insert(stop_l, hl[i])
            draw_pts_l.insert(stop_l, [x[i], hl[i]])
    # Calculate Areas
    r_area = _utils.shoelace(rx, ry)
    l_area = _utils.shoelace(lx, ly)
    draw_pts_r = np.array(draw_pts_r)
    draw_pts_l = np.array(draw_pts_l)
    return (
        r_area,
        l_area,
        draw_pts_r,
        draw_pts_l,
        mid_lower_pt,
        right_mouth_bisect,
        left_mouth_bisect,
        tck_l,
        x,
    )


def inner_mouth_areas(
    data_frame: DataFrame, mid_lines: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Places polygons and calculates areas of inner mouth.
    """
    points = dga.utils.point_array(data_frame, _points.INNER_MOUTH_PTS)
    areas = dga.utils.nan((len(points[0]), 2))
    left_draw_points = []
    right_draw_points = []
    tck_ls = []
    xs = []

    for i in range(len(points[0])):
        # Set up points
        pts = points[:, i]
        if np.isnan(pts).any():
            pass
        if not np.isnan(mid_lines[i]).any():
            # Split points into upper and lower lips x and y. Oral commissures appear in both lists.
            up = pts[0:8, :]  # points 0-7
            lower = np.array(
                [
                    pts[0, :],
                    pts[8, :],
                    pts[9, :],
                    pts[10, :],
                    pts[11, :],
                    pts[12, :],
                    pts[13, :],
                    pts[7, :],
                ]
            )
            # Corner excursion bisectors
            r_ex_y = pts[0, 1]
            l_ex_y = pts[7, 1]
            slope = mid_lines[i, 0]
            r_mid_x = (r_ex_y - mid_lines[i, 1]) / slope
            l_mid_x = (l_ex_y - mid_lines[i, 1]) / slope
            r_y_int = r_ex_y + (1 / slope) * pts[0][0]
            l_y_int = l_ex_y + (1 / slope) * pts[7][0]
            r_mid = r_mid_x - (r_mid_x - pts[0][0]) / 2
            l_mid = l_mid_x - (l_mid_x - pts[7][0]) / 2
            l_ref_pt = (l_mid, l_mid * (-1 / slope) + l_y_int)
            r_ref_pt = (r_mid, r_mid * (-1 / slope) + r_y_int)
            if not np.isnan(slope) and not np.isnan([r_ref_pt[1], r_mid]).any():
                r_bisector = dga.line.from_slope(slope=slope, intercept=r_ref_pt[1] - r_mid * slope)
                r_bisector[2] = int((900 - r_bisector[1]) / slope)
                r_bisector[3] = 900
                r_bisector[4] = int((600 - r_bisector[1]) / slope)
                r_bisector[5] = 600
            else:
                r_bisector = dga.line.nan_line()
            if not np.isnan(slope) and not np.isnan([l_ref_pt[1], l_mid]).any():
                l_bisector = dga.line.from_slope(slope=slope, intercept=l_ref_pt[1] - l_mid * slope)
                l_bisector[2] = int((900 - l_bisector[1]) / slope)
                l_bisector[3] = 900
                l_bisector[4] = int((600 - l_bisector[1]) / slope)
                l_bisector[5] = 600
            else:
                l_bisector = dga.line.nan_line()
            """
            Compute area and gather drawing information
            Sort upx and lx in order of ascending value and corresponding ys with them. 
            Simpler n^2 sort; small data
            """
            (
                r_area,
                l_area,
                draw_r,
                draw_l,
                mid_lower,
                right_bisect,
                left_bisect,
                tck_l,
                x,
            ) = spline_area_mouth(
                up,
                lower,
                mid_lines[i],
                r_bisector,
                l_bisector,
            )
            areas[i] = [r_area, l_area]
            right_draw_points.append(draw_r)
            left_draw_points.append(draw_l)
            tck_ls.append(tck_l)
            xs.append(x)
        else:
            # Empty returns in case of invalid data
            right_draw_points.append(None)
            left_draw_points.append(None)
            tck_ls.append(None)
            xs.append(None)
    return np.array(areas), right_draw_points, left_draw_points, tck_ls, xs


def mouth_poly_intersect(right_pts, left_pts, mid_lines) -> np.ndarray:
    """
    Calculates the overlapping area between a projection of the smaller side onto the larger.
    :param right_pts:
    :param left_pts:
    :param mid_lines:
    :return:
    """
    overlaps = dga.utils.nan(len(mid_lines))
    for i in range(len(right_pts)):
        if (
            isinstance(left_pts[i], np.ndarray)
            and isinstance(right_pts[i], np.ndarray)
            and not np.isnan(mid_lines[i]).any()
        ):
            pivot = (left_pts[i][:, 1] - mid_lines[i, 1]) / mid_lines[i, 0]
            new_x = 2 * pivot - left_pts[i][:, 0]
            new_y = left_pts[i][:, 1]
            projected_set = np.stack([new_x, new_y], axis=1)
            if isinstance(right_pts[i], np.ndarray) and len(right_pts[i].shape) == 2:
                if right_pts[i].shape[0] > 2 and projected_set.shape[0] > 2:
                    right_poly = Polygon(right_pts[i])
                    proj_poly = Polygon(projected_set)
                    if right_poly.is_valid and proj_poly.is_valid:
                        overlaps[i] = right_poly.intersection(proj_poly).area
    return overlaps


def brow(data_frame: DataFrame, eye_lines: np.ndarray):
    """

    :param data_frame: pandas dataframe
    :param eye_lines: the facial midline
    :return:
    """
    r_brow_pts = dga.utils.point_array(data_frame, _points.R_BROW_PTS)
    l_brow_pts = dga.utils.point_array(data_frame, _points.L_BROW_PTS)
    right_brow_lines, right_brow_heights = _brow_compute(eye_lines, r_brow_pts)
    left_brow_lines, left_brow_heights = _brow_compute(eye_lines, l_brow_pts)

    return right_brow_lines, left_brow_lines, right_brow_heights, left_brow_heights


def eyes(data_frame: DataFrame) -> (np.ndarray, List[float], List[float]):
    """
    Computes blink distances.
    :param data_frame: a Pandas DataFrame from the deeplabcut h5 file from inference
    :return: eye area, eye draw right, eye draw left
    """
    outer_eye_pts_r = dga.utils.point_array(data_frame, _points.OUTER_EYE_PTS_R)
    outer_eye_pts_l = dga.utils.point_array(data_frame, _points.OUTER_EYE_PTS_L)
    sp_draw_right = dga.utils.nan((outer_eye_pts_r.shape[1], 199, 2))
    sp_draw_left = dga.utils.nan((outer_eye_pts_r.shape[1], 199, 2))
    sp_areas = dga.utils.nan((outer_eye_pts_r.shape[1], 2))
    for i in range(outer_eye_pts_r.shape[1]):
        # Compute areas and append to list.
        if not np.isnan(outer_eye_pts_r[:, i]).any() and not np.isnan(outer_eye_pts_l[:, i]).any():
            area_r, spline_r = _utils.spline_area(outer_eye_pts_r[:, i])
            area_l, spline_l = _utils.spline_area(outer_eye_pts_l[:, i])
            sp_areas[i] = [area_r, area_l]
            if spline_r.shape == (199, 2):
                sp_draw_right[i] = spline_r
            if spline_l.shape == (199, 2):
                sp_draw_left[i] = spline_l
    sp_draw_left = np.stack(sp_draw_left)
    sp_draw_right = np.stack(sp_draw_right)
    return (
        np.array(sp_areas, dtype=np.float_),
        sp_draw_right,
        sp_draw_left,
    )


def midline(dataframe: DataFrame, f_height) -> (np.ndarray, List[float]):
    """
    Computes the midline as a line orthogonal to  the slopes of a line through the lateral canthi intercepting with the
    glabella point
    :param dataframe: the dataframe returned by DeepLabCuts prediction
    :param: f_height: the height in pixels of the frame
    """
    lateral_canthus_r = dga.utils.point_array(dataframe, ["RLC"])
    lateral_canthus_l = dga.utils.point_array(dataframe, ["LLC"])
    glabella = dga.utils.point_array(dataframe, ["Glab"])
    midpoint = np.stack(
        [
            (lateral_canthus_l[:, 0] + lateral_canthus_r[:, 0]) / 2,
            (lateral_canthus_l[:, 1] + lateral_canthus_r[:, 1]) / 2,
        ],
        axis=1,
    )
    lateral_slope = (lateral_canthus_r[:, 1] - lateral_canthus_l[:, 1]) / (
        lateral_canthus_r[:, 0] - lateral_canthus_l[:, 0]
    )
    p0x = lateral_slope * (glabella[:, 1]) + glabella[:, 0]
    p1x = -lateral_slope * (f_height - glabella[:, 1]) + glabella[:, 0]
    midx = glabella[
        :, 0
    ]  # calculate x value of point on the line in the fmiddle of the frame as a function of y
    eye_lines = dga.line.from_points(lateral_canthus_r, lateral_canthus_l)
    zeros = np.zeros(lateral_canthus_r.shape[0])
    heights = zeros.copy()
    heights[:] = f_height
    mid_lines = dga.line.from_points(
        np.stack((p0x, zeros), axis=1), np.stack((p1x, heights), axis=1)
    )
    return eye_lines, midx, mid_lines
