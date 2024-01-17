import argparse
import json
import os

# -----
# [{'vid': , 'kps': , 'other':},
#  {'vid': , 'kps': , 'other':}]
# -----
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
# -----
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

pose_dir = args.root_path + "_dwpose"

# collect all video_folder paths
video_mp4_paths = set()
for root, dirs, files in os.walk(args.root_path):
    for name in files:
        if name.endswith(".mp4"):
            video_mp4_paths.add(os.path.join(root, name))
video_mp4_paths = list(video_mp4_paths)

meta_infos = []
for video_mp4_path in video_mp4_paths:
    relative_video_name = os.path.relpath(video_mp4_path, args.root_path)
    kps_path = os.path.join(pose_dir, relative_video_name)
    meta_infos.append({"video_path": video_mp4_path, "kps_path": kps_path})

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
