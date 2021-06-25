import os
from typing import Optional, Tuple
import cv2 as cv
import numpy as np


LK_PARAMS = dict(
    winSize=(31, 31), maxLevel=3,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.03))


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
    """
    """
    image = image.copy()
    points = []
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
    initial_points = click_tracking_points(images[0], wh)
    initial_points = np.float32(initial_points)
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
