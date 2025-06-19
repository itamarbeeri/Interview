#!/usr/bin/env python3

import argparse
import cv2
import numpy as np

def stabilize(video_in: str, video_out: str, roi: tuple[int, int, int, int]) -> None:
    
    x, y, w, h = roi
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_in}")
    ret, frame0 = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    ref_cx = x + w / 2
    ref_cy = y + h / 2

    # Kalman filter setup
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kf.statePre = np.array([[ref_cx], [ref_cy], [0], [0]], dtype=np.float32)

    # Tracker init
    try:
        tracker = cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(gray0, (x, y, w, h))

    height, width = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError(f"Cannot open video writer: {video_out}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, box = tracker.update(gray)
        if ok:
            bx, by, bw, bh = map(int, box)
            meas = np.array([[bx + bw/2], [by + bh/2]], dtype=np.float32)
            kf.correct(meas)
        pred = kf.predict()
        cx, cy = float(pred[0]), float(pred[1])

        tx = -(cx - ref_cx)
        ty = -(cy - ref_cy)
        M = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        stab = cv2.warpAffine(frame, M, (width, height))
        out.write(stab)

    cap.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video stabilization using Kalman + MOSSE/KCF")
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('output', help='Path to save stabilized output')
    parser.add_argument('--roi', nargs=4, type=int, default=[100, 100, 440, 280],
                        metavar=('X', 'Y', 'W', 'H'),
                        help='Region of interest for tracking')
    args = parser.parse_args()
    stabilize(args.input, args.output, tuple(args.roi))
    print(f"Stabilized video saved to {args.output}")
