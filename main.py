import os
import cv2
import numpy as np
import yaml
from OF_Stablizer import OpticalFlowStabilizer
from utils import get_video_path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class FrameLoader:
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(video_path)

    def get_frame(self):
        return self.camera.read()

    def release(self):
        self.camera.release()

def display_frames(original_frame, aligned_frame):
    cv2.putText(original_frame, f"original video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    orig = original_frame[::3, ::3]

    cv2.putText(aligned_frame, f"aligned video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    aligned = aligned_frame[::3, ::3]

    display_frame = np.concatenate((orig, aligned), axis=0)
    cv2.imshow('frame', display_frame)
    cv2.waitKey(1)


def main():
    # Setup
    video_path = get_video_path(ROOT_DIR)
    if video_path is None:
        return

    frame_loader = FrameLoader(video_path)
    stabilizer = OpticalFlowStabilizer()

    while True:
        ret, frame = frame_loader.get_frame()
        if not ret:
            break

        aligned_frame = stabilizer.stabilize(frame)
        display_frames(frame, aligned_frame)

    frame_loader.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
