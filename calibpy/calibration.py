import os
from typing import List
import cv2 as cv


def get_frame(vidcap: cv.VideoCapture, sec: float):
    """
    """
    vidcap.set(cv.CAP_PROP_POS_MSEC, sec*1000)
    has_frames, image = vidcap.read()
    return has_frames, image


def video_to_frames(
        video_path: str, frame_rate: float = 0.5, save_imgs: bool = False
        ) -> List:
    """
    """
    vidcap = cv.VideoCapture(video_path)
    video_dir, video_filename = os.path.split(video_path)
    video_name = os.path.splitext(video_filename)[0]
    if save_imgs:
        frames_dir = os.path.join(video_dir, video_name)
        os.makedirs(frames_dir, exist_ok=True)
    sec = -frame_rate
    success = True
    count = 0
    frames = []

    while success:
        sec = round(sec + frame_rate, 2)
        success, image = get_frame(vidcap, sec)
        if success:
            frames.append(image)
            count += 1
            if save_imgs:
                cv.imwrite(
                    os.path.join(frames_dir, "frame_"+str(count)+".jpg"),
                    image)
    return frames
