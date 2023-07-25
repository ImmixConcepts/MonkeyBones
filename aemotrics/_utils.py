from typing import Tuple
import dlc_generic_analysis as dga
import numpy as np
from circle_fit import least_squares_circle
from scipy import interpolate
import sys
import urllib.request
import tarfile
import os
from . import _geometries


def point_line_dist_1d(line, point):
    x1 = line[2]
    x2 = line[4]
    y1 = line[3]
    y2 = line[5]
    return np.abs((x2 - x1) * (y1 - point[1]) - (x1 - point[0]) * (y2 - y1)) / np.sqrt(
        np.power((x2 - x1), 2) + np.power((y2 - y1), 2)
    )


def point_line_dist(line, point):
    """
    calculates the least distance between a point and a line
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    :param line: a line as defined by dlc_generic_analysis.line
    :param point: a point where x is :,0 and y is :,1
    """
    x1 = line[:, 2]
    x2 = line[:, 4]
    y1 = line[:, 3]
    y2 = line[:, 5]
    return np.abs((x2 - x1) * (y1 - point[:, 1]) - (x1 - point[:, 0]) * (y2 - y1)) / np.sqrt(
        np.power((x2 - x1), 2) + np.power((y2 - y1), 2)
    )


def set_intersect(spline, x_vals: np.ndarray, line: np.ndarray) -> [np.ndarray]:
    """
    Approximates point of intersection between a set of points and midline.
    :param spline:
    :param x_vals:
    :param line:
    :return:
    """
    if not isinstance(x_vals, np.ndarray) or not isinstance(line, np.ndarray):
        return
    splines = [interpolate.splev([xval], spline)[0] for xval in x_vals]
    points = np.stack([x_vals, splines], axis=1)
    dists = dga.utils.dist(points, np.stack((x_vals, line[0] * x_vals + line[1]), axis=1))
    index = np.nanargmin(dists, axis=0)
    return points[index]


def shoelace(x: np.array, y: np.array) -> float:
    """
    Implementation of the shoelace formula to find area of irregular polygons.
    :param x: An ndarray of x coordinates
    :param y: An ndarray of y coordinates
    :return: area
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_ratios(values: np.ndarray) -> (np.ndarray, float):
    """
    gets the ratio of 2 values where the greater is divided by the smaller
    :param values: the values
    :return:
    """
    ratios = np.amax(values, axis=1) / np.amin(values, axis=1)
    ratios = np.array(ratios)
    return ratios, np.max(ratios)


def spline_area(points: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes areas by performing Riemann summation over area approximated by splines.

    :param points: An ndarray of shape(,2)
    :return: right area, left area right points, left points
    """
    upper = np.array(
        [points[0], points[4], points[2], points[6], points[1]], dtype=np.float_
    ).swapaxes(0, 1)
    lower = np.array(
        [points[0], points[5], points[3], points[7], points[1]], dtype=np.float_
    ).swapaxes(0, 1)
    right_area, draw_pts = sp_sum(upper, lower)
    return (
        right_area,
        np.array(draw_pts, dtype=np.float_),
    )


def sp_sum(upper: np.ndarray, lower: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Makes splines and computes Riemann sum.
    :param upper: upper
    :param lower:
    :return:
    """
    # Points need to be ordered by x value
    upper = upper[:, np.argsort(upper[0], axis=0)]
    lower = lower[:, np.argsort(lower[0], axis=0)]
    if not np.isnan(upper).any() and not np.isnan(lower).any():  # looks for nan values
        tck_u = interpolate.splrep(upper[0], upper[1])
        tck_l = interpolate.splrep(lower[0], lower[1])
        dist = (upper[0][4] - upper[0][0]) / _geometries.SUM_TOTAL
        x = [upper[0][0] + i * dist for i in range(_geometries.SUM_TOTAL)]
        hu = interpolate.splev(x, tck_u)
        hl = interpolate.splev(x, tck_l)
        points = np.concatenate(
            [
                np.array([x, hu]).swapaxes(0, 1),
                np.flip(np.array([x, hl]).swapaxes(0, 1), axis=0),
            ]
        )[:-1]
        points_x = np.array(x)
        points_x = np.concatenate([points_x, np.flip(points_x, axis=0)])[:-1]
        points_y = np.array(hu)
        points_y = np.concatenate([points_y, np.flip(np.array(hl)[1:], axis=0)])
        area = shoelace(points_x, points_y)
        return area, np.array(points, dtype=np.float_)
    return np.nan, np.array([np.nan])


def iris_circles(points):
    """
    Calculates circles from 4 points along the circumference of the iris
    :param: the points on the circumference of the iris
    :return: (centers, radei)
        centers: the centers of the circles
        radei: the radei of the circles
    """
    centers = dga.utils.nan((points.shape[1], 2))
    radei = np.squeeze(dga.utils.nan((points.shape[1], 1)))
    for i in range(points.shape[1]):
        if not np.isnan(points[:, i]).any():
            centers[i, 0], centers[i, 1], radei[i], _ = least_squares_circle(points[:, i])
    return centers, radei


def eye_unit_convert(radei: np.ndarray, iris_rad=5.9) -> float:
    """
    Converts pixels to millimeters based on diameter of iris.
    :param radei: the list of radei
    :param iris_rad: the radius of the iris uses 5.9 mm by default
    :return: the predicted pixel size
    """
    mean_rad = np.nanmean(radei)
    return iris_rad / mean_rad


def set_download():
    """
    downloads the Aemotrics model from the internet
    :return: the path to the newly downloaded model
    """
    model_name: str = "Aemotrics_V3-Nate-2021-01-20"
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.split(os.path.realpath(__file__))[0]
    dlc_model_path = os.path.join(base_dir, model_name)
    __download_model(dlc_model_path)
    return dlc_model_path


def __download_model(dlc_model_path) -> None:
    """
    downloads a model and places the base dir of it in dlc_model_path
    :param dlc_model_path: the base directory for the model on your system
    :return: None
    """
    if not os.path.isfile(os.path.join(dlc_model_path, "config.yaml")):
        location = os.path.split(os.path.realpath(__file__))[0]
        dl_url = ""
        model_name = os.path.split(dlc_model_path)[1]
        dest_f = os.path.join(location, "temp", model_name + ".tar.gz")
        dest = urllib.request.urlretrieve(dl_url, dest_f)[0]
        tar = tarfile.open(dest, mode="r")
        if not os.path.isdir(os.path.join(location, model_name)):
            os.mkdir(os.path.join(location, model_name))
        tar.extractall(model_name)
