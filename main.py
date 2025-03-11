import os

import cv2
import numpy as np
import yaml
from solution_online import VideoStabilizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class FrameLoader:
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(video_path)

    def get_frame(self):
        return self.camera.read()


def display_frames(original_frame, aligned_frame):
    cv2.putText(original_frame, f"original video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    orig = original_frame[::3, ::3]

    cv2.putText(aligned_frame, f"aligned video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    aligned = aligned_frame[::3, ::3]

    display_frame = np.concatenate((orig, aligned), axis=0)
    cv2.imshow('frame', display_frame)
    cv2.waitKey(1)


def main():
    config = yaml.safe_load(open(os.path.join(ROOT_DIR, "config.yml")))
    frame_loader = FrameLoader(os.path.join(ROOT_DIR, config['video_dir_relative'], config['video_name']))

    ret, frame = frame_loader.get_frame()
    video_stabilizer = VideoStabilizer(frame)
    display_frames(frame, frame)
    while True:
        ret, frame = frame_loader.get_frame()
        stabilized_frame = video_stabilizer(frame)
        display_frames(frame, stabilized_frame)


if __name__ == '__main__':
    main()
