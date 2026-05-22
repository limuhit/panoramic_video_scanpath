#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from spath.paths import get_work_dir, resolve_path
from spath.serialization import save_json


def count_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if nframes <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")
    return nframes


def normalize_name(value):
    name = Path(value.strip()).name
    if not name:
        return ""
    if Path(name).suffix != ".npy":
        name = f"{Path(name).stem}.npy"
    return name


def read_name_list(path):
    names = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.split("#", 1)[0].strip()
            if item:
                names.append(normalize_name(item))
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate sample names found in {path}.")
    return names


def video_id_from_name(name):
    return Path(name).stem.split("_")[0]


def numeric_video_id(name):
    video_id = video_id_from_name(name)
    try:
        return int(video_id)
    except ValueError as exc:
        raise ValueError(f"VRW23 split expects numeric video ids, got {name!r}.") from exc


def validate_names(requested, available, label):
    missing = sorted(set(requested) - available)
    if missing:
        shown = ", ".join(missing[:10])
        more = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise RuntimeError(f"{label} list contains files without matching .mp4/.npy pairs: {shown}{more}")
    return sorted(requested)


def build_split(args, names):
    available = set(names)
    if args.split_mode == "list":
        if not args.train_list or not args.test_list:
            raise RuntimeError("--split-mode list requires --train-list and --test-list.")
        train = validate_names(read_name_list(args.train_list), available, "train")
        test = validate_names(read_name_list(args.test_list), available, "test")
        overlap = sorted(set(train) & set(test))
        if overlap:
            shown = ", ".join(overlap[:10])
            raise RuntimeError(f"train/test lists overlap: {shown}")
        return {"train": train, "test": test}

    if args.split_mode == "vrw23":
        train = sorted(name for name in names if numeric_video_id(name) < 400)
        test = sorted(name for name in names if numeric_video_id(name) >= 400)
        if not train or not test:
            raise RuntimeError("VRW23 split produced an empty train or test set. Check video ids 000-501.")
        return {"train": train, "test": test}

    rng = np.random.default_rng(args.seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    ntest = max(1, int(len(shuffled) * args.test_ratio))
    train = sorted(shuffled[ntest:])
    test = sorted(shuffled[:ntest])
    if not train or not test:
        raise RuntimeError("Random split produced an empty train or test set. Adjust --test-ratio.")
    return {"train": train, "test": test}


def main():
    parser = argparse.ArgumentParser(description="Create videos_frame.json and path_dict3.json for SPath training.")
    parser.add_argument("--work-dir", default=None, help="Base directory for relative paths. Defaults to SPATH_WORK_DIR or the repository root.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--path-dir", default="HM")
    parser.add_argument("--split-mode", choices=["random", "vrw23", "list"], default="random")
    parser.add_argument("--train-list", default=None, help="Text file for --split-mode list.")
    parser.add_argument("--test-list", default=None, help="Text file for --split-mode list.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    base_dir = get_work_dir(args.work_dir)
    args.data_root = resolve_path(args.data_root, base_dir)
    args.train_list = resolve_path(args.train_list, base_dir)
    args.test_list = resolve_path(args.test_list, base_dir)
    if not 0 < args.test_ratio < 1:
        raise RuntimeError("--test-ratio must be between 0 and 1.")

    root = Path(args.data_root)
    video_dir = root / args.video_dir
    path_dir = root / args.path_dir
    path_files = sorted(path_dir.glob("*.npy"))
    names = [path.name for path in path_files if (video_dir / (path.stem + ".mp4")).exists()]
    if not names:
        raise RuntimeError(f"No matching .mp4/.npy pairs found under {video_dir} and {path_dir}.")

    split = build_split(args, names)

    frame_counts = {}
    for name in names:
        video_id = video_id_from_name(name)
        if video_id not in frame_counts:
            frame_counts[video_id] = count_frames(video_dir / (Path(name).stem + ".mp4"))

    save_json(split, root / "path_dict3.json")
    save_json(frame_counts, root / "videos_frame.json")
    print(f"wrote {root / 'path_dict3.json'} and {root / 'videos_frame.json'}")
    print(f"train={len(split['train'])} test={len(split['test'])} videos={len(frame_counts)}")


if __name__ == "__main__":
    main()
