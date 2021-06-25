import unittest
import os
import pathlib

import calibpy.calibration as calib


class CalibrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.filepath = pathlib.Path(__file__).parent.absolute()

    def test_convert_video_to_frames(self):
        out = calib.convert_video_to_frames(
            os.path.join(self.filepath, '..', 'res/checkerboard.mp4'),
            frame_rate=5., save_imgs=False)
        assert len(out) == 2
