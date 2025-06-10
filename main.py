import os

import cv2
import numpy as np
import yaml

from stabilizer import VideoStabilizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class FrameLoader:
    def __init__(self, video_path):
        """Initialize video capture from the given path."""
        self.camera = cv2.VideoCapture(video_path)

    def get_frame(self):
        """Read and return the next frame from the video."""
        return self.camera.read()  # Returns (ret, frame)


def display_frames(original_frame, aligned_frame, labels):
    """Display the original and stabilized frames stacked vertically."""
    if aligned_frame is None or aligned_frame.size == 0:
        return

    orig = original_frame.copy()
    aligned = aligned_frame.copy()

    # crop to smallest dimensions for stacking
    min_h = min(orig.shape[0], aligned.shape[0])
    min_w = min(orig.shape[1], aligned.shape[1])
    orig = orig[:min_h, :min_w]
    aligned = aligned[:min_h, :min_w]

    cv2.putText(
        orig,
        labels.get("original", "original"),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        6,
    )
    cv2.putText(
        aligned,
        labels.get("stabilized", "stabilized"),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        6,
    )

    display_frame = np.concatenate((orig, aligned), axis=0)
    cv2.imshow("frame", display_frame)
    cv2.waitKey(1)


def main():
    """Load config, process the video frame-by-frame, and save stabilized output."""

    config_data = yaml.safe_load(open(os.path.join(ROOT_DIR, "config.yml")))
    config = config_data["stabilization"]
    labels = config.get("frame_labels", {})

    video_dir = os.path.join(ROOT_DIR, config_data["video_dir_relative"])
    os.makedirs(video_dir, exist_ok=True)

    output_name = config_data.get("output_name", "stabilized_output.mp4")
    output_path = os.path.join(video_dir, output_name)

    # load first frame to init
    video_path = os.path.join(
        ROOT_DIR, config_data["video_dir_relative"], config_data["video_name"]
    )
    frame_loader = FrameLoader(video_path)
    ret, first_frame = frame_loader.get_frame()
    if not ret:
        print("Error: Could not read video file.")
        return

    # Create stabilizer object using only required parameters
    stabilizer = VideoStabilizer(
        first_frame,
        config["prediction_window_size"],
        config["polynomial_prediction_order"],
        config.get("border_mode", "constant"),
        config["good_features_to_track"],
    )

    # setup video writer
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*config.get("output_codec", "mp4v"))
    out_writer = cv2.VideoWriter(
        output_path, fourcc, config.get("output_fps", 30.0), (width, height)
    )

    # process each frame
    while True:
        ret, frame = frame_loader.get_frame()
        if not ret:
            break

        # stabilization to current frame
        # stabilized_frame = stabilizer.stabilize(frame, config["min_tracked_features"], config["scale_beta"])
        stabilized_frame = stabilizer.stabilize(frame, config["min_tracked_features"])

        display_frames(frame, stabilized_frame, labels)

        # write stabiliz frame to output video
        out_writer.write(stabilized_frame)

    out_writer.release()


if __name__ == "__main__":
    main()
