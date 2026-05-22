#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from spath.paths import get_work_dir, resolve_path
from spath.sampler import ScanpathSampler
from spath.visualization import (
    combine_frames,
    compute_plcc,
    export_comparison_frames,
    load_display_scanpath,
)


def main():
    parser = argparse.ArgumentParser(description="Run SPath sampling on one panoramic video.")
    parser.add_argument("--work-dir", default=None, help="Base directory for relative paths. Defaults to SPATH_WORK_DIR or the repository root.")
    parser.add_argument("--video", required=True, help="Input panoramic video.")
    parser.add_argument("--checkpoint", default="checkpoints/spath_vrw23.pt", help="Trained SPath checkpoint.")
    parser.add_argument("--init-scanpath", required=True, help="Initial/ground-truth scanpath .npy used to seed the first frames.")
    parser.add_argument("--gt-scanpath", default=None, help="Optional ground-truth scanpath for visualization and PLCC selection.")
    parser.add_argument("--output-dir", default="outputs/inference")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sample-num", type=int, default=3)
    parser.add_argument("--stride", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--npred", type=int, default=5)
    parser.add_argument("--start-idx", type=int, default=5)
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true", help="Allow legacy torch.load checkpoint deserialization. Use only for trusted files.")
    args = parser.parse_args()
    base_dir = get_work_dir(args.work_dir)
    args.video = resolve_path(args.video, base_dir)
    args.checkpoint = resolve_path(args.checkpoint, base_dir)
    args.init_scanpath = resolve_path(args.init_scanpath, base_dir)
    args.gt_scanpath = resolve_path(args.gt_scanpath, base_dir)
    args.output_dir = resolve_path(args.output_dir, base_dir)

    output_dir = Path(args.output_dir)
    scanpath_dir = output_dir / "scanpaths"
    video_dir = output_dir / "videos"
    fig_dir = output_dir / "figures"

    sampler = ScanpathSampler(
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        sample_num=args.sample_num,
        stride=args.stride,
        window_size=args.window_size,
        npred=args.npred,
        device_id=args.gpu_id,
        allow_unsafe_checkpoint=args.allow_unsafe_checkpoint,
    )
    paths = sampler.sample(
        video_path=args.video,
        init_scanpath_path=args.init_scanpath,
        start_idx=args.start_idx,
        save_scanpath_dir=scanpath_dir,
        save_video_dir=video_dir,
        output_prefix="sample",
    )
    print(f"saved sampled scanpaths to {scanpath_dir}")

    if args.gt_scanpath:
        gt = load_display_scanpath(args.gt_scanpath, max_points=75)
        best_score = -1
        best_idx = 0
        for idx in range(paths.shape[0]):
            pred_path = scanpath_dir / f"sample_f{idx}.npy"
            pred = load_display_scanpath(pred_path, max_points=75)
            score = compute_plcc(gt, pred)
            if score > best_score:
                best_score = score
                best_idx = idx
        pred = load_display_scanpath(scanpath_dir / f"sample_f{best_idx}.npy", max_points=75)
        export_comparison_frames(args.video, gt, pred, fig_dir)
        combined = combine_frames(fig_dir)
        print(f"best_path=sample_f{best_idx}.npy plcc={best_score:.4f} figure={combined}")


if __name__ == "__main__":
    main()
