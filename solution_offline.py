import cv2
import numpy as np
from tqdm import tqdm


def stabilize_video(input_path, output_path, smoothing_radius=30, damping=0.5):
    """
    Stabilizes a video using feature detection and motion tracking.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save stabilized video
        smoothing_radius: Radius for smoothing transformation parameters
        damping: Damping factor to reduce abrupt corrections
    """
    # Read input video
    cap = cv2.VideoCapture(input_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read the first frame
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize transformation matrix storage
    transforms = []
    
    # Adjusted feature detection parameters for high resolution (1080x1920)
    feature_params = dict(
        maxCorners=1000,       # Increased to detect more features for accuracy
        qualityLevel=0.001,    # Retain sensitivity for feature detection
        minDistance=30,        # Lowered for denser feature points
        blockSize=7            # Reduced block size for finer features
    )
    
    # Lucas-Kanade optical flow parameters adjusted for higher accuracy
    lk_params = dict(
        winSize=(21, 21),     # Reduced window size for precise matching
        maxLevel=4,           # Fewer pyramid levels for refined estimation
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 105, 0.000001)
    )
    
    # Detect initial features
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    
    # Process all frames
    for i in tqdm(range(n_frames - 1)):
        # Read next frame
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        
        # Filter out invalid points
        idx = np.where(status == 1)[0]
        if len(idx) < 4:  # Need at least 4 points for perspective transform
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
            continue
            
        prev_pts_valid = prev_pts[idx]
        curr_pts_valid = curr_pts[idx]
        
        # Estimate affine transformation
        m, _ = cv2.estimateAffinePartial2D(prev_pts_valid, curr_pts_valid)
        
        if m is None:  # If transformation estimation fails
            m = np.eye(2, 3, dtype=np.float32)
            
        # Store transformation
        transforms.append(m)
        
        # Update previous frame and points
        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
        
    # Release input video capture
    cap.release()
    
    # Compute trajectory
    trajectory = np.zeros((n_frames - 1, 3), dtype=np.float32)
    for i in range(len(transforms)):
        trajectory[i, 0] = transforms[i][0, 2]  # x translation
        trajectory[i, 1] = transforms[i][1, 2]  # y translation
        trajectory[i, 2] = np.arctan2(transforms[i][1, 0], transforms[i][0, 0])  # rotation angle
    
    # Smooth trajectory
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius)
    
    # Create smoothed transforms
    smoothed_transforms = []
    for i in range(len(transforms)):
        dx = damping * (smoothed_trajectory[i, 0] - trajectory[i, 0])
        dy = damping * (smoothed_trajectory[i, 1] - trajectory[i, 1])
        da = damping * (smoothed_trajectory[i, 2] - trajectory[i, 2])
        
        m = transforms[i].copy()
        
        # Create rotation matrix for the angle difference
        rot_matrix = cv2.getRotationMatrix2D((0, 0), np.degrees(da), 1.0)
        
        # Apply rotation correction first, then add translation correction
        corrected_transform = np.zeros((2, 3), dtype=np.float32)
        # Apply rotation
        corrected_transform[0, 0] = rot_matrix[0, 0] * m[0, 0] + rot_matrix[0, 1] * m[1, 0]
        corrected_transform[0, 1] = rot_matrix[0, 0] * m[0, 1] + rot_matrix[0, 1] * m[1, 1]
        corrected_transform[1, 0] = rot_matrix[1, 0] * m[0, 0] + rot_matrix[1, 1] * m[1, 0]
        corrected_transform[1, 1] = rot_matrix[1, 0] * m[0, 1] + rot_matrix[1, 1] * m[1, 1]
        
        # Apply translation (original + correction)
        corrected_transform[0, 2] = rot_matrix[0, 0] * m[0, 2] + rot_matrix[0, 1] * m[1, 2] + rot_matrix[0, 2] + dx
        corrected_transform[1, 2] = rot_matrix[1, 0] * m[0, 2] + rot_matrix[1, 1] * m[1, 2] + rot_matrix[1, 2] + dy
        
        smoothed_transforms.append(corrected_transform)
    
    # Apply smoothed transformations and write output video
    cap = cv2.VideoCapture(input_path)
    for i in range(n_frames - 1):
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply transformation
        stabilized_frame = cv2.warpAffine(frame, smoothed_transforms[i], (width, height))
        
        # Fix border artifacts
        border = 100
        stabilized_frame = fix_border(stabilized_frame, border)
        
        # Write frame to output video
        out.write(stabilized_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path


def smooth_trajectory(trajectory, radius):
    """
    Smooth the trajectory using a moving average filter.
    
    Args:
        trajectory: The original trajectory (Nx3 array of x, y, angle)
        radius: The radius of the smoothing window
    
    Returns:
        The smoothed trajectory
    """
    smoothed = np.copy(trajectory)
    for i in range(len(trajectory)):
        start = max(0, i - radius)
        end = min(len(trajectory), i + radius + 1)
        
        # Average window
        count = end - start
        smoothed[i] = np.sum(trajectory[start:end], axis=0) / count
        
    return smoothed


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


def main():
    input_path = "data/car.mp4"
    output_path = "data/car_stabilized_video_3.mp4"
    
    print("Stabilizing video...")
    border_size = 100  # Unchanged
    smoothing_radius = 10  # Tighter smoothing
    damping = 0.3  # New damping factor to reduce shaking
    
    # Call stabilize_video with updated parameters (include damping)
    stabilize_video(input_path, output_path, smoothing_radius, damping)
    print(f"Stabilized video saved to {output_path}")


if __name__ == '__main__':
    main()