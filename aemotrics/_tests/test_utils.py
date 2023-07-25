import numpy as np
from numpy import testing
from aemotrics import _utils


def test_point_line_dist_1d():
    line = np.array([1, 0, 0, 0, 10, 10])
    point = np.array([0, 1])
    testing.assert_almost_equal(_utils.point_line_dist_1d(line, point), np.sqrt(0.5))


def test_point_line_dist():
    line = np.array([[1, 0, 0, 0, 10, 10], [2, 0, 0, 0, 2, 1]])
    point = np.array([[0, 1], [0, 0]])
    testing.assert_array_almost_equal(
        _utils.point_line_dist(line, point), np.array([np.sqrt(0.5), 0])
    )


def test_shoelace_trapezoid():
    xs = np.array([0, 0, 1, 1])
    ys = np.array([3, 1, 0, 4])
    area = _utils.shoelace(xs, ys)
    assert area == 3


def test_shoelace_hex():
    xs = [1, 2, 3, 3, 2, 1, 0, 0]
    ys = [0, 0, 1, 2, 3, 3, 2, 1]
    assert _utils.shoelace(xs, ys) == 7


def test_get_ratios():
    values = np.array([[200, 5], [4, 12], [2.5, 5]])
    ratios, max_ratio = _utils.get_ratios(values)
    assert max_ratio == 40
    np.testing.assert_array_equal(ratios, np.array([40, 3, 2], dtype=float))


def test_iris_circles():
    points = np.array([[[-1, 1], [7, 5]], [[-1, -1], [3, 1]], [[1, -1], [7, 1]], [[1, 1], [3, 5]]])
    center, radei = _utils.iris_circles(points)
    np.testing.assert_array_equal(center, np.array([[0, 0], [5, 3]]))
    np.testing.assert_array_equal(radei, np.array([np.sqrt(2), np.sqrt(2**2 * 2)]))
