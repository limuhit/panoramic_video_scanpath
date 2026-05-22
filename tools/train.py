#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(str(Path(__file__).resolve().parents[1]))

from spath.data import create_train_test_loaders
from spath.model import ScanpathPredictor
from spath.paths import get_work_dir, resolve_path
from spath.serialization import safe_torch_load
from spath.training import (
    CheckpointSaver,
    cleanup_distributed,
    load_partial_state_dict,
    setup_distributed,
    train_one_epoch,
    unwrap_model,
    validate_nll,
)


def worker(rank, world_size, args):
    gpu_id = args.gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    if world_size > 1:
        setup_distributed(rank, world_size, args.master_addr, args.master_port)

    train_loader, val_loader = create_train_test_loaders(
        root_dir=args.data_root,
        samples_per_video=args.samples_per_video,
        window_size=args.window_size,
        npred=args.npred,
        stride=args.data_stride,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        device_id=gpu_id,
        seed=args.seed,
        train_crop_len=args.train_crop_len,
        eval_crop_len=args.eval_crop_len,
        video_dir=args.video_dir,
        path_dir=args.path_dir,
        frames_file=args.frames_file,
        split_file=args.split_file,
        num_workers=args.num_workers,
    )

    model = ScanpathPredictor(args.window_size, args.npred, args.sample_num, args.sample_stride, args.num_gaussians, gpu_id)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        checkpoint = safe_torch_load(args.resume, map_location=device, allow_unsafe=args.allow_unsafe_checkpoint)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1 if isinstance(checkpoint, dict) else 1
    elif args.init_weights:
        loaded, skipped = load_partial_state_dict(model, args.init_weights, device, allow_unsafe=args.allow_unsafe_checkpoint)
        if rank == 0:
            print(f"Loaded {loaded} compatible tensors from {args.init_weights}; skipped {len(skipped)}.", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[gpu_id])

    saver = CheckpointSaver(args.output_dir, args.prefix) if rank == 0 else None
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.clip, args.alpha, args.log_every)
        val_loss = validate_nll(model, val_loader, device)
        if rank == 0:
            ckpt_path = saver.save(model, epoch, val_loss, optimizer)
            print(
                f"epoch={epoch} train_nll={train_loss:.6f} val_nll={val_loss:.6f} "
                f"time={time.time() - t0:.1f}s checkpoint={ckpt_path}",
                flush=True,
            )

    if rank == 0:
        final_path = Path(args.output_dir) / f"{args.prefix}_final.pt"
        torch.save({"epoch": args.epochs, "model": unwrap_model(model).state_dict()}, final_path)
    cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SPath on panoramic video scanpath data.")
    parser.add_argument("--work-dir", default=None, help="Base directory for relative paths. Defaults to SPATH_WORK_DIR or the repository root.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing videos, HM, videos_frame.json, and path_dict3.json.")
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--prefix", default="spath")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-video", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--npred", type=int, default=5)
    parser.add_argument("--data-stride", type=int, default=5)
    parser.add_argument("--sample-stride", type=float, default=0.2)
    parser.add_argument("--sample-num", type=int, default=1)
    parser.add_argument("--num-gaussians", type=int, default=3)
    parser.add_argument("--train-crop-len", type=int, default=45)
    parser.add_argument("--eval-crop-len", type=int, default=900)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--path-dir", default="HM")
    parser.add_argument("--frames-file", default="videos_frame.json")
    parser.add_argument("--split-file", default="path_dict3.json")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--init-weights", default=None, help="Optional checkpoint to partially initialize matching model tensors.")
    parser.add_argument("--resume", default=None, help="Resume from a full training checkpoint.")
    parser.add_argument("--allow-unsafe-checkpoint", action="store_true", help="Allow legacy torch.load checkpoint deserialization. Use only for trusted files.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", default="12355")
    args = parser.parse_args()
    base_dir = get_work_dir(args.work_dir)
    args.data_root = resolve_path(args.data_root, base_dir)
    args.output_dir = resolve_path(args.output_dir, base_dir)
    args.init_weights = resolve_path(args.init_weights, base_dir)
    args.resume = resolve_path(args.resume, base_dir)
    return args


def main():
    args = parse_args()
    world_size = len(args.gpu_ids)
    if world_size > 1:
        mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        worker(0, 1, args)


if __name__ == "__main__":
    main()
