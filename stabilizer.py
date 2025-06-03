import cv2
import numpy as np
from math import atan2, cos, sin


class VideoStabilizer:
    """
    Real-time video stabilizer using feature tracking and trajectory smoothing.

    This class tracks visual features between frames using the Lucas-Kanade method,
    estimates frame-to-frame transforms, and applies temporal smoothing to reduce jitter.
    A correction transform is computed from the smoothed trajectory to generate a stabilized output.
    """
    def __init__(self, first_frame, first_pts, smoothing_radius, correction_boost, angle_smoothing, border_mode):
        """Initialize the stabilizer with the first frame and config settings."""
        self.first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # convert  to grayscale
        self.first_pts = first_pts
        self.prev_gray  = self.first_gray.copy()
        self.prev_pts   = self.first_pts.copy()

        self.transforms = []  #  raw frame-to-frame transforms (dx, dy, da)
        self.trajectory = [(0,0,0)]   # cumulative Cᵢ
        self.smoothed   = []          # smoothed Ĉᵢ
        self.smoothing_radius = smoothing_radius  # num of frames before/after to average
        self.angle_smoothing = angle_smoothing  # toggle smoothing of rotation
        self.warp_matrices = []  # full 3x3 transformation matrices

        # how to fill empty pixels after warp
        self.border_mode = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT
        }.get(border_mode, cv2.BORDER_CONSTANT)

    def build_affine(self, dx, dy, da):
        """Build an affine transformation matrix from dx, dy, and rotation angle."""
        cos_a = cos(da)
        sin_a = sin(da)
        return np.array([
            [cos_a, -sin_a, dx],
            [sin_a,  cos_a, dy],
            [0, 0, 1]
        ], dtype=np.float32)

    def track_features(self, curr_gray, bidirectional_max_error, min_tracked_features):
        """Track features from prev frame to current using optical flow."""

        # using Lucas-Kanade optical flow to track the good features
        # ref: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga2950b3200f1b7aa9ef6b600c15d3f88e
        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None)

        good_prev = self.prev_pts[st[:, 0] == 1]
        good_curr = curr_pts[st[:, 0] == 1]

        if len(good_prev) < min_tracked_features:
            # redetect if too many points are lost
            good_curr = cv2.goodFeaturesToTrack(curr_gray, 200, 0.01, 30)
            good_prev = good_curr # reinitialize tracking to avoid  crash. may produce one unstable frame


        return good_prev, good_curr

    def update_trajectory(self, dx, dy, da):
        """Accumulate and smooth trajectory."""
        prev_cx, prev_cy, prev_ca = self.trajectory[-1]
        self.trajectory.append((prev_cx + dx, prev_cy + dy, prev_ca + da))

        if len(self.trajectory) > 2 * self.smoothing_radius:
            window = self.trajectory[-(2 * self.smoothing_radius + 1):]
            cx = np.mean([t[0] for t in window])
            cy = np.mean([t[1] for t in window])
            ca = np.mean([t[2] for t in window])
            self.smoothed.append((cx, cy, ca))
        else:
            self.smoothed.append(self.trajectory[-1])

    def apply_correction(self, M, sx, sy, sa):
        """
        apply correction between smoothed and original trajectory:

        This function adjusts the estimated motion matrix `M` by comparing the smoothed 
        trajectory values (sx, sy, sa) with the current raw trajectory.
        The difference between them is used to build a correction transform, which is then 
        applied on top of the original motion estimate.

        Returns:
            np.ndarray: A full 3x3 warp matrix to apply with cv2.warpPerspective.
        """
        tx, ty = M[0, 2], M[1, 2]
        ta     = np.arctan2(M[1, 0], M[0, 0])

        raw_x, raw_y, raw_a = self.trajectory[-1]
        diff_dx = sx - raw_x
        diff_dy = sy - raw_y
        diff_da = sa - raw_a

        correction = self.build_affine(diff_dx, diff_dy, diff_da)
        return correction @ np.vstack([M, [0, 0, 1]])


    def stabilize(self, current_frame, bidirectional_max_error, min_tracked_features):
        """
        Takes the current frame and returns a stabilized version of it.

        It works by tracking feature points from the previous frame using optical flow,
        estimating how the camera moved, and then applying a smoothed correction to
        cancel out the jitter. This helps keep the video looking steady, even if the
        original footage is a bit shaky.

        args:
            current_frame (np.ndarray): The new video frame to stabilize
            bidirectional_max_error (float): Used to filter out bad feature matches
            min_tracked_features (int): if too few features are tracked, re-detect them

        Returns:
            np.ndarray: The stabilized frame
        """

        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        #short-baseline optical flow  (prev -> curr)
        good_prev, good_curr = self.track_features(curr_gray, bidirectional_max_error, min_tracked_features)

        # incremental transform between the 2frames
        M, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
        if M is None:
            return current_frame

        dx, dy = M[0,2], M[1,2]
        da     = np.arctan2(M[1,0], M[0,0])
        self.transforms.append((dx, dy, da))

        #accumulate
        self.update_trajectory(dx, dy, da)

        # difference between smoothed and raw trajectory
        sx, sy, sa = self.smoothed[-1]
        final_M = self.apply_correction(M, sx, sy, sa)

        # final warp
        h, w = current_frame.shape[:2]
        stabilized = cv2.warpPerspective(
            current_frame, final_M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode)

        # keeping for next iteration
        self.prev_gray, self.prev_pts = curr_gray, good_curr.reshape(-1,1,2)

        return stabilized
