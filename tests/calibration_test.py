import unittest
import os
import pathlib

import numpy as np

import calibpy.calibration as calib

import numpy as np


class CalibrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.filepath = pathlib.Path(__file__).parent.absolute()

    def test_convert_video_to_frames(self):
        out = calib.convert_video_to_frames(
            os.path.join(self.filepath, '..', 'res/checkerboard.mp4'),
            frame_rate=50., save_imgs=False)
        points_frames = calib.track_points_seq(out, (1920, 1080))
        num_frames = points_frames.shape[0]
        board_points = calib.generate_homogeneous_board_points()
        H_frames = np.zeros((num_frames, 3, 3))
        for idx in range(num_frames):
            H_frames[idx] = calib.estimate_homography(points_frames[idx], board_points)
        calib.estimate_camera_intrinsic_matrix(H_frames)
        assert len(out) == 2


if __name__ == '__main__':
    unittest.main()
