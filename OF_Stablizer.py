import cv2
import numpy as np

class OpticalFlowStabilizer:
    def __init__(self, winSize=(19, 19), maxLevel=2, criteria_count=30, criteria_eps=0.01, maxCorners=150, minDistance=30):
        """
        Initialize the Optical Flow Stabilizer with configurable parameters.
        """
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=winSize, 
                              maxLevel=maxLevel, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, criteria_count, criteria_eps))
        
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=maxCorners,
                                   qualityLevel=0.01,
                                   minDistance=minDistance,
                                   blockSize=3)

        self.prev_gray = None
        self.h, self.w = None, None
        
        # Global accumulation matrix (Identity)
        self.H_global = np.eye(3, dtype=np.float32)

    def _reset_tracking(self, frame_gray, frame_bgr):
        """Resets tracking state and returns the original frame."""
        self.prev_gray = frame_gray.copy()
        return frame_bgr

    def _initialize_reference(self, gray, frame):
        """Initialize the reference frame if not set."""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.h, self.w = frame.shape[:2]
            return True
        return False

    def _track_features(self, gray):
        """
        Track features from previous frame to current frame using Optical Flow
        with Forward-Backward Error Check.
        Returns: (good_prev, good_curr) points or (None, None) if tracking fails.
        """
        # Track features: previous frame -> current frame
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        if p0 is None:
            return None, None

        # Calculate Optical Flow (Forward: Prev -> Curr)
        p1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        
        # Calculate Optical Flow (Backward: Curr -> Prev) for error check
        if p1 is not None:
            p0r, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.lk_params)
            
            # Keep only robust points (Forward-Backward Error < 1.0 pixel)
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)
            good = d < 1.0
            
            # Combine status and error check
            good_curr = p1[good]
            good_prev = p0[good]
            
            # Need at least 4 points for homography
            if len(good_curr) < 4:
                return None, None
                
            return good_prev, good_curr
            
        return None, None

    def stabilize(self, frame):
        """
        Stabilizes the current frame by aligning it to the first frame using Sequential Optical Flow.
        Method: Tracks Frame(t-1) -> Frame(t), estimates transform, accumulates it, and warps back to Frame 0.
        """
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialization (Frame 0)
        if self._initialize_reference(gray, frame):
            return frame

        # Track Features
        good_prev, good_curr = self._track_features(gray)
        if good_prev is None:
            return self._reset_tracking(gray, frame)

        # Estimate Motion: Previous -> Current
        M_curr_to_prev, _ = cv2.findHomography(good_curr, good_prev, cv2.RANSAC, 5.0)
        
        if M_curr_to_prev is None:
            return self._reset_tracking(gray, frame)

        # Accumulate global transform
        self.H_global = np.dot(self.H_global, M_curr_to_prev)

        # Warp current frame -> frame 0
        stabilized_frame = cv2.warpPerspective(frame, self.H_global, (self.w, self.h))

        # Update previous frame
        self.prev_gray = gray.copy()
        
        return stabilized_frame
