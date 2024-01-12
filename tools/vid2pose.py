from src.dwpose import DWposeDetector
import os
from pathlib import Path

from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    dir_path, video_name = (
        os.path.dirname(args.video_path),
        os.path.splitext(os.path.basename(args.video_path))[0],
    )
    out_path = os.path.join(dir_path, video_name + "_kps.mp4")

    detector = DWposeDetector()
    detector = detector.to(f"cuda")

    fps = get_fps(args.video_path)
    frames = read_frames(args.video_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)

        kps_results.append(result)

    print(out_path)
    save_videos_from_pil(kps_results, out_path, fps=fps)
