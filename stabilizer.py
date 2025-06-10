from math import cos, sin
import cv2
import numpy as np


class VideoStabilizer:
    """
    Real-time video stabilizer using feature tracking and trajectory smoothing.

    This class tracks visual features between frames using the Lucas-Kanade method,
    estimates frame-to-frame transforms, and applies temporal smoothing to reduce jitter.
    A correction transform is computed from the predicted trajectory to generate a stabilized output.
    """

    def __init__(
        self,
        first_frame,
        prediction_window_size,
        polynomial_prediction_order,
        border_mode,
        gftt_config,
    ):
        """Initialize the stabilizer with the first frame and config settings."""
        self.first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.gftt = gftt_config
        self.first_pts = cv2.goodFeaturesToTrack(
            self.first_gray,
            maxCorners=self.gftt["max_corners"],
            qualityLevel=self.gftt["quality_level"],
            minDistance=self.gftt["min_distance"],
        )

        self.transforms = []  #  raw frame-to-frame transforms (dx, dy, da)
        self.trajectory = [(0, 0, 0)]  # cumulative Cᵢ
        self.predicted = []  # predicted Ĉᵢ
        self.polyfit_order = (
            polynomial_prediction_order  # degree of polynomial for smoothing
        )
        self.prediction_window_size = (
            prediction_window_size  # num of frames before to average
        )

        self.warp_matrices = []  # full 3x3 transformation matrices

        # how to fill empty pixels after warp
        self.border_mode = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
        }.get(border_mode, cv2.BORDER_CONSTANT)

    def build_affine(self, dx, dy, da):
        """Build an affine transformation matrix from dx, dy, and rotation angle."""
        cos_a = cos(da)
        sin_a = sin(da)
        return np.array(
            [[cos_a, -sin_a, dx], [sin_a, cos_a, dy], [0, 0, 1]], dtype=np.float32
        )

    def track_features(self, curr_frame_grayscale, min_tracked_features):
        """Track features from prev frame to current using optical flow."""

        # using Lucas-Kanade optical flow to track the good features
        # ref: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga2950b3200f1b7aa9ef6b600c15d3f88e
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.first_gray, curr_frame_grayscale, self.first_pts, None
        )

        # filter out points that were not tracked
        tracked_pts_prev = self.first_pts[status[:, 0] == 1]
        tracked_pts_curr = curr_pts[status[:, 0] == 1]

        if len(tracked_pts_prev) < min_tracked_features:
            # redetect if too many points are lost
            tracked_pts_curr = cv2.goodFeaturesToTrack(
                curr_frame_grayscale,
                maxCorners=self.gftt["max_corners"],
                qualityLevel=self.gftt["quality_level"],
                minDistance=self.gftt["min_distance"],
            )
            self.first_pts = tracked_pts_curr  # reinitialize tracking to avoid  crash. may produce one unstable frame

        return tracked_pts_prev, tracked_pts_curr

    def update_predicted_trajectory(self, dx, dy, da):
        """
        Predicts a smoothed trajectory by fitting a polynomial curve
        to the recent history of raw transforms, then applies exponential smoothing.

        Args:
            dx (float): Translation in x direction.
            dy (float): Translation in y direction.
            da (float): Rotation angle in radians.
        """
        self.trajectory.append((dx, dy, da))

        n = len(self.trajectory)
        window_size = min(2 * self.prediction_window_size + 1, n)
        start_idx = n - window_size
        window = self.trajectory[start_idx:]

        frame_indices = np.arange(start_idx, n)

        # extract each motion component
        dx_vals = np.array([t[0] for t in window])
        dy_vals = np.array([t[1] for t in window])
        da_vals = np.array([t[2] for t in window])

        # fit 2nd degree polynomial to each motion component
        fit_dx = np.polyfit(frame_indices, dx_vals, deg=self.polyfit_order)
        fit_dy = np.polyfit(frame_indices, dy_vals, deg=self.polyfit_order)
        fit_da = np.polyfit(frame_indices, da_vals, deg=self.polyfit_order)

        # predict motion for the current frame index
        cx = np.polyval(fit_dx, n - 1)
        cy = np.polyval(fit_dy, n - 1)
        ca = np.polyval(fit_da, n - 1)

        self.predicted.append((cx, cy, ca))

    def apply_correction(self, sx, sy, sa):
        """
        Calculates a transformation matrix to correct the camera motion.

        It compares the current predicted trajectory with the raw estimated motion,
        calculates the difference, and uses it to construct an affine correction matrix
        to stabilize the current frame.

        """
        raw_x, raw_y, raw_a = self.trajectory[-1]
        diff_dx = sx - raw_x
        diff_dy = sy - raw_y
        diff_da = sa - raw_a

        # build correction matrix
        correction_transform = self.build_affine(diff_dx, diff_dy, diff_da)
        return correction_transform

    def stabilize(self, current_frame, min_tracked_features):
        """
        Takes the current frame and returns a stabilized version of it.

        It works by tracking feature points from the previous frame using optical flow,
        estimating how the camera moved, and then applying a predicted correction to
        cancel out the jitter. This helps keep the video looking steady, even if the
        original footage is a bit shaky.

        args:
            current_frame (np.ndarray): The new video frame to stabilize
            min_tracked_features (int): if too few features are tracked, re-detect them

        Returns:
            np.ndarray: The stabilized frame
        """

        curr_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # short-baseline optical flow  (prev -> curr)
        tracked_pts_prev, tracked_pts_curr = self.track_features(
            curr_frame_grayscale, min_tracked_features
        )

        # incremental transform between the 2frames
        M, _ = cv2.estimateAffinePartial2D(tracked_pts_prev, tracked_pts_curr)
        if M is None:
            return current_frame

        dx, dy = M[0, 2], M[1, 2]
        da = np.arctan2(M[1, 0], M[0, 0])
        self.transforms.append((dx, dy, da))

        # accumulate
        self.update_predicted_trajectory(dx, dy, da)

        # difference between predicted and raw trajectory
        sx, sy, sa = self.predicted[-1]
        final_M = self.apply_correction(sx, sy, sa)

        # final warp
        h, w = current_frame.shape[:2]
        stabilized = cv2.warpPerspective(
            current_frame,
            final_M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )

        return stabilized
