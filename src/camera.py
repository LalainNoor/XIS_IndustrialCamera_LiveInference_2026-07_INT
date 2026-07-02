"""
Video source module that simulates an industrial camera using a video file.
"""

import cv2
from config import FRAME_WIDTH, FRAME_HEIGHT

class Camera:

    def __init__(self, source):

        self.cap = cv2.VideoCapture(source)

        # Camera configuration
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

    def read(self):
        """
        Read one frame from the video source.
        """
        return self.cap.read()

    def get_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self.cap.release()