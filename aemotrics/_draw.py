import os
from logging import info
import cv2
import numpy as np


def draw(
    path: str,
    out_path: str,
    mouth_pts_r,
    mouth_pts_l,
    spline_pts,
    mid_lines,
    eye_lines,
    r_brows,
    l_brows,
    rx_lines,
    lx_lines,
    right_lower_lip_tangents,
    left_lower_lip_tangents,
    right_inner_draw,
    left_inner_draw,
    iris_l,
    iris_r,
    vid_type: str = ".mp4",
):
    """
    Takes each frame from video and stitches it back into new video with
    line drawn on.
    """
    cap = cv2.VideoCapture(path)
    frames = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s, im = cap.read()
    out_dir, f_name = os.path.split(out_path)
    f_name = os.path.splitext(f_name)[0]
    vid_type = vid_type.lower()
    if vid_type == ".mp4" or vid_type == "mp4":
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    elif vid_type == ".avi" or vid_type == "avi":
        fourcc = cv2.VideoWriter_fourcc("x", "v", "i", "d")
    else:
        raise AttributeError("video can only be an mp4 or avi")
    out_dir = os.path.join(out_dir, "analyzed_videos")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    out = os.path.join(out_dir, f_name + vid_type)
    w = cv2.VideoWriter(out, fourcc, frames, (width, height))
    # Writing Loop
    info("Filling areas on your video.")
    i = 0
    r_brows = r_brows.astype(int)
    l_brows = l_brows.astype(int)
    while s:
        blk = np.zeros(im.shape, np.uint8)
        # Setting up points and filtering out low confidence and NoneTypes
        if not np.isnan(iris_l[i]).any():
            cv2.circle(
                im,
                (int(iris_l[i, 0]), int(iris_l[i, 1])),
                int(iris_l[i, 2]),
                (255, 255, 255),
                2,
            )
        if not np.isnan(iris_r[i]).any():
            cv2.circle(
                im,
                (int(iris_r[i, 0]), int(iris_r[i, 1])),
                int(iris_r[i, 2]),
                (255, 255, 255),
                2,
            )
        if right_inner_draw[i] is not None:
            r_in_pts = right_inner_draw[i]
        else:
            r_in_pts = []
        if left_inner_draw[i] is not None:
            l_in_pts = left_inner_draw[i]
        else:
            l_in_pts = []
        r_psp = spline_pts[0, i, :199]
        l_psp = spline_pts[1, i, :199]
        # Printing midlines
        if not np.isnan(mid_lines[i]).any():
            end1 = (int(-mid_lines[i, 1] / mid_lines[i, 0]), 0)
            end2 = (
                int((height - mid_lines[i, 1]) / mid_lines[i, 0]),
                height,
            )
            cv2.line(im, end1, end2, (51, 119, 17), 2)
        if not np.isnan(eye_lines[i]).any():
            end1 = (0, int(eye_lines[i, 1]))
            end2 = (
                width,
                int(eye_lines[i, 0] * width + eye_lines[i, 1]),
            )
            cv2.line(im, end1, end2, (51, 119, 17), 2)
        # Printing brow lines
        if not np.isnan(r_brows[i, 2:]).any() and np.greater(r_brows[i, 2:], 0).all():
            cv2.line(im, r_brows[i, 2:4], r_brows[i, 4:6], (153, 170, 68), 2)
        if not np.isnan(l_brows[i, 2:]).any() and np.greater(l_brows[i, 2:], 0).all():
            cv2.line(im, l_brows[i, 2:4], l_brows[i, 4:6], (136, 34, 51), 2)
        # Printing excursion lines
        if isinstance(rx_lines[i], np.ndarray):
            cv2.line(
                im, rx_lines[i, 2:4].astype(int), rx_lines[i, 4:6].astype(int), (51, 119, 17), 2
            )
        if isinstance(rx_lines[i], np.ndarray):
            cv2.line(
                im, lx_lines[i, 2:4].astype(int), lx_lines[i, 4:6].astype(int), (136, 34, 51), 2
            )

        if not np.isnan(right_lower_lip_tangents[i]).any():
            cv2.line(
                im,
                right_lower_lip_tangents[i, 2:4],
                right_lower_lip_tangents[i, 4:6],
                (136, 34, 51),
                2,
            )
        if not np.isnan(left_lower_lip_tangents[i]).any():
            cv2.line(
                im,
                left_lower_lip_tangents[i, 2:4],
                left_lower_lip_tangents[i, 4:6],
                (51, 119, 17),
                2,
            )
        # Filtered mouth polygons
        if len(r_psp) > 0:
            cv2.fillPoly(blk, [np.int32(r_psp)], (51, 119, 17))
            cv2.fillPoly(blk, [np.int32(l_psp)], (136, 34, 51))
        if mouth_pts_r[i] is not None and mouth_pts_l[i] is not None:
            r_pts = mouth_pts_r[i][: len(mouth_pts_r[i])]
            l_pts = mouth_pts_l[i][: len(mouth_pts_l[i])]
            if len(r_pts) > 0 and len(l_pts) > 0:
                cv2.fillPoly(blk, [np.int32(r_pts)], (51, 119, 17))
                cv2.fillPoly(blk, [np.int32(l_pts)], (136, 34, 51))
        if len(r_in_pts) > 0:
            cv2.fillPoly(blk, [np.int32(r_in_pts)], (136, 34, 51))
            cv2.fillPoly(blk, [np.int32(l_in_pts)], (51, 119, 17))

        # Add filled polygons to image, write, and acquire next
        out = cv2.addWeighted(im, 1.0, blk, 0.75, 1)
        w.write(out)
        i += 1
        s, im = cap.read()
    cap.release()
    w.release()
    return out
