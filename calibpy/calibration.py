import os
from typing import Optional, Tuple, List
import cv2 as cv
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from matplotlib.path import Path


LK_PARAMS = dict(
    winSize=(31, 31), maxLevel=3,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))


def read_image(image_path: str) -> np.ndarray:
    """
    """
    image = cv.imread(image_path)
    return image


def get_frame(vidcap: cv.VideoCapture, msec: Optional[float] = None):
    """
    """
    if msec is not None:
        vidcap.set(cv.CAP_PROP_POS_MSEC, msec)
    has_frames, image = vidcap.read()
    return has_frames, image


def convert_video_to_frames(
        video_path: str, frame_rate: float = 500, save_imgs: bool = False
        ) -> np.ndarray:
    """
    """
    vidcap = cv.VideoCapture(video_path)
    video_dir, video_filename = os.path.split(video_path)
    video_name = os.path.splitext(video_filename)[0]
    if save_imgs:
        frames_dir = os.path.join(video_dir, video_name)
        os.makedirs(frames_dir, exist_ok=True)
    msec = -frame_rate
    success = True
    count = 0
    frames = []

    while success:
        msec = round(msec + frame_rate, 2)
        success, image = get_frame(vidcap, msec)
        if success:
            frames.append(image)
            count += 1
            if save_imgs:
                cv.imwrite(
                    os.path.join(frames_dir, 'frame_'+str(count)+'.jpg'),
                    image)
    return np.stack(frames)


def convert_frames_to_video(
        out_path: str, images: np.ndarray, points: np.ndarray = None,
        frame_rate: int = 20):
    """
    """
    wh_tuple = images[0].shape[0:2][::-1]
    out = cv.VideoWriter(
        out_path, cv.VideoWriter_fourcc(*'mp4v'), frame_rate, wh_tuple)
    for idx in range(images.shape[0]):
        image = images[idx].copy()
        if points is not None:
            curr_points = points[idx]
            for point_idx in range(curr_points.shape[0]):
                cv.circle(
                    image, tuple(curr_points[point_idx]), 10, (0, 0, 255), -1)
        out.write(image)
    out.release()


def display_image(image: np.ndarray, wh: Tuple[int, int],
                  points: Optional[np.ndarray] = None, image_time: float = 0):
    """ frame_time: in miliseconds
    """
    image = image.copy()
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', *wh)
    if points is not None:
        for point_idx in range(points.shape[0]):
            cv.circle(image, tuple(points[point_idx]), 10, (0, 0, 255), -1)
    cv.imshow('image', image)
    cv.waitKey(image_time)
    cv.destroyAllWindows()


def display_video(in_path: str, wh: Tuple[int, int]):
    """
    """
    vidcap = cv.VideoCapture(in_path)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', *wh)
    while vidcap.isOpened():
        success, image = get_frame(vidcap, msec=None)
        if not success:
            break
        cv.imshow('image', image)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break
    vidcap.release()
    cv.destroyAllWindows()


def click_point(event, x, y, flags, param):  # param
    """ Mouse left click callback function
    """
    if event == cv.EVENT_LBUTTONDOWN:
        image, points = param
        cv.circle(image, (x, y), 10, (0, 0, 255), -1)
        points.append(np.array([x, y]))


def click_tracking_points(
        image: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    """ Click points up to down, left to right (column wise)
    """
    image = image.copy()
    points: List[np.ndarray] = []
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', *wh)
    cv.setMouseCallback('image', click_point, [image, points])

    while True:
        # display the image and wait for a keypress
        cv.imshow('image', image)
        key = cv.waitKey(1) & 0xFF
        # if the 'q' key is pressed, break from the loop
        if key == ord('q'):
            cv.destroyAllWindows()
            break
    return np.stack(points)


def track_points_image(
    image1: np.ndarray, points1: np.ndarray, image2: np.ndarray
        ) -> np.ndarray:
    """ cv.calcOpticalFlowPyrLK requires points to be of type np.float32!
    """
    points2, st, err = cv.calcOpticalFlowPyrLK(
        image1, image2, points1, None, **LK_PARAMS)
    return points2


def track_points_seq(images: np.ndarray, wh: Tuple[int, int]):
    """
    """
    initial_points: np.ndarray = click_tracking_points(images[0], wh)
    initial_points = np.float32(initial_points)  # type: ignore
    points = [initial_points]
    prev_points = initial_points
    prev_image = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)
    for image_idx in range(1, len(images)):
        curr_image = cv.cvtColor(images[image_idx], cv.COLOR_BGR2GRAY)
        next_points = track_points_image(prev_image, prev_points, curr_image)
        points.append(next_points)
        prev_points = next_points
        prev_image = curr_image
    return np.int_(np.stack(points))


def generate_homogeneous_board_points(
        board_side: float = .025, board_rows: int = 4, board_cols: int = 4
        ) -> np.ndarray:
    """ Column wise
    """
    board_points = []
    for c in range(board_cols):
        for r in range(board_rows):
            board_points.append(np.array([c*board_side, r*board_side, 1]))
    return np.stack(board_points)


def generate_eqspaced_image_points(
        max_x: int, max_y: int, board_rows: int = 4, board_cols: int = 4
        ) -> np.ndarray:
    """
    """
    image_points = []
    for x in np.linspace(0, max_x, board_rows):
        for y in np.linspace(0, max_y, board_cols):
            image_points.append(np.array([x, y]))
    return np.stack(image_points)


def estimate_homography(
        image_points: np.ndarray, board_points: np.ndarray) -> np.ndarray:
    """ Direct Linear Transform (DLT) Homography estimation
    """
    num_points = image_points.shape[0]
    assert num_points == board_points.shape[0]
    A = np.zeros((3*num_points, 9))
    for i in range(num_points):
        X = board_points[i]
        x = image_points[i]
        u, v = x
        A[3*i, 3:6] = -X
        A[3*i, 6:9] = v*X
        A[3*i+1, 0:3] = X
        A[3*i+1, 6:9] = -u*X
        A[3*i+2, 0:3] = -v*X
        A[3*i+2, 3:6] = u*X
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape((3, 3))
    return H


def calculate_interior_points(
        points: np.ndarray, plot_hull: bool = False) -> np.ndarray:
    """ Compute convex hull of the points and check if an image point is
        inside the poligon and return those as a numpy array.

        Clicked points have increasing Y in downward direction so we have to
        invert so the shape is plot as seen in the image!

        image_dims: Specified as (image_max_x, image_max_y)
    """
    if plot_hull:
        plot_points = points.copy()
        plot_points[:, 1] = -plot_points[:, 1]
        p_hull = ConvexHull(plot_points)
        _ = convex_hull_plot_2d(p_hull)
        plt.show()
    int_points: List[np.ndarray] = []
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            curr_p = np.array([x, y])
            if hull_path.contains_point(curr_p):
                int_points.append(curr_p)
    return np.stack(int_points)


def project_image(
        image: np.ndarray, proj_image: np.ndarray, int_points: np.ndarray,
        H: np.ndarray):
    """
    """
    return
