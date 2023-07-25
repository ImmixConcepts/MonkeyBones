from dlc_generic_analysis import line
import numpy as np
from aemotrics import _geometries
from numpy import testing

eye_lines = np.array(
    [
        [1e-10, 1500, 200, 1500, 880, 1500],
        [5 / 68, 1450 - ((5 / 68) * 200), 200, 1450, 880, 1500],
    ]
)


def test_intersection():
    """
    finds the intersection between y=x and y=-x+1 defined at [0,0],[1,1] and [1,0][0,1]
    """
    l0 = np.array([[1, 0, 0, 0, 1, 1]])
    l1 = np.array([[-1, 1, 1, 0, 0, 1]])
    testing.assert_array_equal(_geometries.intersection(l0, l1), np.array([[0.5, 0.5]]))
    l1 = np.array([[100000, -110000000, 0, 1100, 1100.0185, 1850]])
    l2 = np.array([[0.1, 10, 0, 10, 2000, 210]])
    testing.assert_array_almost_equal(
        np.array([[1100.001200001, 120.0001200001]]), _geometries.intersection(l1, l2)
    )


def test_lower_lip():
    r_bisect = np.array([[300, 200], [200, 210]])
    l_bisect = np.array([[780, 200], [785, 190]])
    right_heights, left_heights = _geometries.lower_lip(eye_lines, r_bisect, l_bisect)
    testing.assert_array_almost_equal(np.array([1300, 1250.1063505042]), right_heights)
    testing.assert_array_almost_equal([1300, 1313.6346440515], left_heights)


def test_brow():
    r_brow_pts = np.array([[300, 1850], [290, 1800]])
    l_brow_pts = np.array([[780, 1820], [770, 1900]])
    dist_r = np.array([1850 - 1500, 346.1810161])
    dist_l = np.array([1820 - 1500, 411.4142697])
    _, _, test_dist_r, test_dist_l = _geometries._brow_compute(eye_lines, r_brow_pts, l_brow_pts)
    testing.assert_array_almost_equal(dist_r, test_dist_r)
    testing.assert_array_almost_equal(dist_l, test_dist_l)
