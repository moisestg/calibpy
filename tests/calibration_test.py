import unittest
import os
import pathlib

from calibpy import calibration as calib


class CalibrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.filepath = pathlib.Path(__file__).parent.absolute()
        print(self.filepath)

    def test_video_to_frames(self):
        out = calib.video_to_frames(
            os.path.join(self.filepath, '..', 'res/checkerboard.mp4'),
            frame_rate=5., save_imgs=False)
        assert len(out) == 2
