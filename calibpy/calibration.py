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
        image: np.ndarray, proj_image: np.ndarray, proj_points: np.ndarray,
        H: np.ndarray) -> np.ndarray:
    """
    """
    image = image.copy()
    homogeneous_proj_points = np.hstack(
        (proj_points, np.ones((proj_points.shape[0], 1))))
    proj_image_points = np.matmul(H, homogeneous_proj_points.T).T
    proj_image_points = proj_image_points[:, 0:2] / proj_image_points[:, 2:3]
    proj_image_points = np.rint(proj_image_points).astype(np.int_)
    proj_image_dims = proj_image.shape[0:2][::-1]
    np.clip(proj_image_points[:, 0], 0, proj_image_dims[0]-1,
            proj_image_points[:, 0])
    np.clip(proj_image_points[:, 1], 0, proj_image_dims[1]-1,
            proj_image_points[:, 1])
    image[proj_points[:, 1], proj_points[:, 0]] = proj_image[
        proj_image_points[:, 1], proj_image_points[:, 0]]
    return image


def _build_v_vector(hi: np.ndarray, hj: np.ndarray):
    """ Helper function
    """
    return np.array([
        hi[0]*hj[0],
        hi[0]*hj[1]+hi[1]*hj[0],
        hi[0]*hj[2]+hi[2]*hj[0],
        hi[1]*hj[1],
        hi[1]*hj[2]+hi[2]*hj[1],
        hi[2]*hj[2]
    ])


def estimate_camera_intrinsic_matrix(
        H_frames: np.ndarray) -> np.ndarray:
    """
    """
    num_frames = H_frames.shape[0]
    A = np.zeros((2*num_frames, 6))
    for i in range(num_frames):
        h1 = H_frames[i][:, 0]
        h2 = H_frames[i][:, 1]
        A[2*i, :] = _build_v_vector(h1, h2)
        A[2*i+1, :] = _build_v_vector(h1, h1) - _build_v_vector(h2, h2)
    _, _, Vt = np.linalg.svd(A)
    b = Vt[-1, :]
    B = np.zeros((3, 3))
    B[np.triu_indices(3)] = b
    B += B.T - np.diag(np.diag(B))
    K = np.linalg.cholesky(B)
    return K


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

