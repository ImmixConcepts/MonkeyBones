import os
from logging import info, warning
import cv2
from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CascadeClassifier,
    VideoCapture,
    VideoWriter,
)


class FaceCropper(object):
    HEIGHT_SCALAR = 1.20

    def __init__(self):
        cv2_dir, _ = os.path.split(cv2.__file__)
        cacade = cv2_dir + "/data/haarcascade_frontalface_default.xml"
        self.face_cascade = CascadeClassifier(cacade)

    def fast_crop(self, vidpath, videotype=".mp4") -> str:
        """
        Crops video around subject's face.
        :param vidpath: the path to the video to crop
        :param videotype: the filetype of the video
        :return: the path of the cropped file
        """
        f_path, ext = os.path.splitext(vidpath)
        name = f_path + "_cropped"
        out = name + ext
        if not os.path.isfile(out):
            cap = VideoCapture(vidpath)
            frames = cap.get(CAP_PROP_FPS)
            s, im = cap.read()
            width = int(cap.get(CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(CAP_PROP_FRAME_HEIGHT))
            f_count = int(cap.get(CAP_PROP_FRAME_COUNT))
            if videotype == ".mp4":
                fourcc = VideoWriter.fourcc("m", "p", "4", "v")
            elif videotype == ".avi":
                fourcc = VideoWriter.fourcc("x", "v", "i", "d")
            else:
                fourcc = 0
            info("Cropping Video " + name)
            count = 0
            xl, xu, yl, yu, w, h = self.generate(im, width, height)
            w = VideoWriter(out, fourcc, frames, (w, h))
            while s:
                crop_im = im[yl:yu, xl:xu]
                w.write(crop_im)
                count += 1
                s, im = cap.read()
                pct = count / f_count
                info("%d%%" % (pct * 100))
            cap.release()
            w.release()
            info("Cropped video available here:" + out)
        else:
            info("cropped video already exists")
        return out

    def generate(self, img, width=None, height=None):
        """Generates face cropping using cv2.face_cascade"""
        if img is None:
            info("Can't open image file")
            raise FileNotFoundError
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))[0]
        if faces is None:
            warning("Failed to detect face")
            raise AttributeError

        x, y, w, h = faces
        h *= self.HEIGHT_SCALAR
        center_x = x + w / 2
        center_y = y + h / 2
        x_lower = int(center_x - w / 2)  # Remember width is our own width and not equal to w
        if x_lower < 0:
            x_lower = 0
        x_upper = int(x_lower + w)
        y_lower = int(center_y - h / self.HEIGHT_SCALAR / 2)
        if y_lower < 0:
            y_lower = 0
        y_upper = int(y_lower + h)
        return x_lower, x_upper, y_lower, y_upper, w, int(h)
