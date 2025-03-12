import cv2
import numpy as np


class VideoStabilizer:
    def __init__(self, frame:np.ndarray, damping:float=0.5, border:int=100, alpha:float=0.05):
        """
        Initializes the Stabilizer with the given parameters.
        Args:
            frame (np.ndarray): The initial frame to initialize the stabilizer.
            smoothing_radius (int, optional): The radius for smoothing the motion vectors. Defaults to 30.
            damping (float, optional): The damping factor to reduce the intensity of the motion. Defaults to 0.5.
            border (int, optional): The border size to avoid border artifacts. Defaults to 100.
            alpha (float, optional): The weight for the moving average of the transformed frame. Defaults to 0.9.
        """
        self.damping = damping
        self.feature_params = dict(
        maxCorners=1000,       # Increased to detect more features for accuracy
        qualityLevel=0.001,    # Retain sensitivity for feature detection
        minDistance=30,        # Lowered for denser feature points
        blockSize=7            # Reduced block size for finer features
        ) 
    
        # Lucas-Kanade optical flow parameters adjusted for higher accuracy
        self.lk_params = dict(
        winSize=(21, 21),     # Reduced window size for precise matching
        maxLevel=4,           # Fewer pyramid levels for refined estimation
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 105, 0.000001)
        )
        height, width = frame.shape[:2]
        self.height = height
        self.width = width
        self.border = border
        self.prev_frame = frame
        self.prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        self.moving_avg = None
        self.history = []
        self.total_history = []
        self.alpha = alpha

    def smooth_trajectory(self, trajectory):
        """
        Smooth the trajectory using a moving average filter.
        
        Args:
            trajectory: The original trajectory (Nx3 array of x, y, angle)
        
        Returns:
            The smoothed trajectory
        """
        if self.moving_avg is None:
            self.moving_avg = trajectory
        self.moving_avg = self.alpha * trajectory + (1 - self.alpha) * self.moving_avg
        return self.moving_avg

    @staticmethod
    def fix_border(frame, border_size):
        """
        Fix border artifacts after warping by cropping and resizing.
        
        Args:
            frame: The frame to fix
            border_size: Border size to crop
            
        Returns:
            Frame with fixed borders
        """
        h, w = frame.shape[:2]
        
        # Crop the border
        cropped = frame[border_size:h-border_size, border_size:w-border_size]
        
        # Resize to original size
        resized = cv2.resize(cropped, (w, h))
        
        return resized

    def __call__(self, curr_frame):
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None, **self.lk_params)
        
        # Filter out invalid points
        idx = np.where(status == 1)[0]
        assert len(idx) >= 4, "Need at least 4 points for perspective transform"
            
        prev_pts_valid = prev_pts[idx]
        curr_pts_valid = curr_pts[idx]
        
        # Estimate affine transformation
        m, _ = cv2.estimateAffinePartial2D(prev_pts_valid, curr_pts_valid)
        
        if m is None:  # If transformation estimation fails
            m = np.eye(2, 3, dtype=np.float32)
            
        
        trajectory = np.zeros((1, 3), dtype=np.float32)
        trajectory[0, 0] = m[0, 2]  # x translation
        trajectory[0, 1] = m[1, 2]  # y translation
        trajectory[0, 2] = np.arctan2(m[1, 0], m[0, 0])
        
        # Check if the trajectory is out of distribution
        if len(self.history) > 30:
            recent_trajectories = np.array(self.history[-30:])
            mean_trajectory = np.mean(recent_trajectories, axis=0)
            std_trajectory = np.std(recent_trajectories, axis=0)
            
            z_score = np.abs((trajectory - mean_trajectory) / std_trajectory)
            
            if np.any(z_score > 3):  # reject the trajectory if it 3 stds from the mean
                trajectory = mean_trajectory  # Use the mean trajectory instead
            else:
                self.history.append(trajectory)
        else:
            self.history.append(trajectory)

        self.total_history.append(trajectory)
        smoothed_trajectory = self.smooth_trajectory(trajectory)

        dx = smoothed_trajectory[0, 0]
        dy = smoothed_trajectory[0, 1]
        da = smoothed_trajectory[0, 2]

        rot_matrix = cv2.getRotationMatrix2D((0, 0), np.degrees(da), 1.0)

        corrected_transform = np.zeros((2, 3), dtype=np.float32)
        # Add rotation
        corrected_transform = rot_matrix
        
        # add translations 
        corrected_transform[0, 2] = dx
        corrected_transform[1, 2] = dy      
        
        # Apply the transformation to the current frame
        stabilized_frame = cv2.warpAffine(curr_frame, corrected_transform, (self.width, self.height))

        
        # stabilized_frame = self.fix_border(stabilized_frame, self.border)
        return stabilized_frame

