import tempfile
from pathlib import Path

import numpy as np
import torch

from SPath_operator import PreData
from spath.serialization import load_json

try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.backend import TensorListGPU
    from nvidia.dali.plugin.pytorch import feed_ndarray
except ImportError as exc:  # pragma: no cover - only hit on non-DALI machines.
    pipeline_def = None
    fn = None
    types = None
    TensorListGPU = None
    feed_ndarray = None
    _DALI_IMPORT_ERROR = exc
else:
    _DALI_IMPORT_ERROR = None


def _require_dali():
    if _DALI_IMPORT_ERROR is not None:
        raise ImportError(
            "Training data loading requires NVIDIA DALI. Install the DALI build "
            "matching your CUDA version, for example nvidia-dali-cuda120."
        ) from _DALI_IMPORT_ERROR


if pipeline_def is not None:

    @pipeline_def
    def _video_pipe(file_list, sequence_length, rank, world_size, pad_sequences):
        video, label, start_frame_num = fn.readers.video(
            device="gpu",
            file_list=file_list,
            sequence_length=sequence_length,
            file_list_include_preceding_frame=False,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=False,
            initial_fill=10,
            prefetch_queue_depth=2,
            image_type=types.RGB,
            dtype=types.FLOAT,
            file_list_frame_num=True,
            stride=1,
            pad_sequences=pad_sequences,
            enable_frame_num=True,
        )
        return video, label, start_frame_num

    @pipeline_def
    def _path_pipe(file_list, rank, world_size):
        path = fn.readers.numpy(device="cpu", file_list=file_list, shard_id=rank, num_shards=world_size, random_shuffle=False)
        return path

else:
    _video_pipe = None
    _path_pipe = None


def _validate_sample_name(name):
    if not isinstance(name, str):
        raise ValueError(f"Metadata sample name must be a string, got {type(name).__name__}.")
    path = Path(name)
    if path.name != name or path.suffix != ".npy":
        raise ValueError(f"Metadata sample name must be a plain .npy filename, got {name!r}.")
    return name


def _load_split(path):
    split = load_json(path)
    if not isinstance(split, dict) or "train" not in split or "test" not in split:
        raise ValueError(f"{path} must contain train and test lists.")
    for key in ("train", "test"):
        if not isinstance(split[key], list):
            raise ValueError(f"{path}:{key} must be a list.")
        split[key] = [_validate_sample_name(name) for name in split[key]]
    return split


def _load_frame_counts(path):
    frame_counts = load_json(path)
    if not isinstance(frame_counts, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    cleaned = {}
    for key, value in frame_counts.items():
        if not isinstance(key, str) or "/" in key or "\\" in key:
            raise ValueError(f"{path} contains an invalid video id {key!r}.")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path}:{key} must be a positive integer frame count.")
        cleaned[key] = value
    return cleaned


class ScanPathWindowDataset:
    """DALI-backed iterable dataset for SPath training and validation.

    Expected files under ``root_dir``:
    - ``videos/*.mp4``: viewport videos, normally 252 x 448.
    - ``HM/*.npy``: processed local scanpaths with shape ``T x 2 x 61``.
    - ``videos_frame.json``: dict mapping video id to frame count.
    - ``path_dict3.json``: dict with ``train`` and ``test`` filename lists.
    """

    def __init__(
        self,
        root_dir,
        crop_len,
        samples_per_video,
        window_size,
        npred,
        stride,
        batch_size=1,
        rank=0,
        world_size=1,
        train=True,
        seed=0,
        device_id=0,
        num_workers=4,
        video_dir="videos",
        path_dir="HM",
        frames_file="videos_frame.json",
        split_file="path_dict3.json",
        whole=False,
    ):
        _require_dali()
        self.root_dir = Path(root_dir)
        self.video_dir = self.root_dir / video_dir
        self.path_dir = self.root_dir / path_dir
        self.crop_len = crop_len
        self.samples_per_video = samples_per_video
        self.window_size = window_size
        self.npred = npred
        self.stride = stride
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.device_id = device_id
        self.num_workers = num_workers

        split = _load_split(self.root_dir / split_file)
        if whole:
            self.path_list = split["train"] + split["test"]
        elif train:
            self.path_list = split["train"]
        else:
            self.path_list = split["test"]
        self.frame_counts = _load_frame_counts(self.root_dir / frames_file)
        self.samples = len(self.path_list)
        self.nbatch = self.samples // (world_size * batch_size)
        self.idx = 0
        self.video_file = None
        self.path_file = None
        self.pre_data = PreData(samples_per_video, window_size, npred, stride, train=train, device=device_id)

    def __len__(self):
        return self.samples

    def num_batch(self):
        return self.nbatch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _make_file_lists(self):
        rng = np.random.default_rng(self.epoch + self.seed)
        indices = list(range(self.samples))
        if self.train:
            rng.shuffle(indices)

        start_frames = rng.integers(0, 10000, size=self.samples, dtype=np.uint32)
        video_rows = []
        path_rows = []
        for sample_idx in indices:
            stem = self.path_list[sample_idx].split(".")[0]
            video_id = stem.split("_")[0]
            nframes = self.frame_counts[video_id]
            if self.train:
                if nframes < self.crop_len:
                    raise ValueError(f"{stem}.mp4 has {nframes} frames, shorter than train crop length {self.crop_len}.")
                start = int(start_frames[sample_idx] % (nframes - self.crop_len + 1))
                end = start + self.crop_len
            else:
                start = 0
                end = nframes
            video_rows.append(f"{self.video_dir / (stem + '.mp4')} {sample_idx} {start} {end}\n")
            path_rows.append(f"{self.path_dir / (stem + '.npy')}\n")

        self.video_file = tempfile.NamedTemporaryFile(mode="w")
        self.video_file.writelines(video_rows)
        self.video_file.flush()
        self.path_file = tempfile.NamedTemporaryFile(mode="w")
        self.path_file.writelines(path_rows)
        self.path_file.flush()

    def __iter__(self):
        self.idx = 0
        self._make_file_lists()
        pad_sequences = not self.train
        self.video_pipe = _video_pipe(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            device_id=self.device_id,
            file_list=self.video_file.name,
            sequence_length=self.crop_len,
            rank=self.rank,
            world_size=self.world_size,
            pad_sequences=pad_sequences,
            exec_async=True,
        )
        self.video_pipe.build()
        self.path_pipe = _path_pipe(
            batch_size=self.batch_size,
            num_threads=self.num_workers,
            device_id=self.device_id,
            file_list=self.path_file.name,
            rank=self.rank,
            world_size=self.world_size,
            exec_async=True,
        )
        self.path_pipe.build()
        self.video_pipe.schedule_run()
        self.path_pipe.schedule_run()
        return self

    def _cast_video_tensor(self, sequence_out: TensorListGPU, video_len):
        tensor = sequence_out.as_tensor()
        out = torch.empty(tuple(tensor.shape()), device=f"cuda:{self.device_id}", dtype=torch.float32)
        feed_ndarray(tensor, out)
        out = out.permute(0, 1, 4, 2, 3)
        out = out[:, : video_len[0]]
        return out.contiguous()

    def _cast_path_tensor(self, sequence_out: TensorListGPU, start_frames):
        size = sequence_out[0].shape()
        if len(size) != 3:
            raise ValueError(f"Expected processed scanpaths with shape T x 2 x 61, got {size}.")
        path_len = self.crop_len if self.train else size[0]
        out = torch.empty((self.batch_size, path_len, size[1], size[2]), device=f"cuda:{self.device_id}", dtype=torch.float32)
        for idx, seq in enumerate(sequence_out):
            start = int(start_frames[idx])
            tmp = torch.empty(tuple(seq.shape()), device=f"cuda:{self.device_id}", dtype=torch.float32)
            feed_ndarray(seq._as_gpu(), tmp)
            out[idx] = tmp[start : start + path_len]
        return out.contiguous()

    def __next__(self):
        if self.idx >= self.nbatch:
            self.video_file.close()
            self.path_file.close()
            raise StopIteration

        video_out, label, start_frame_num = self.video_pipe.share_outputs()
        path_out = self.path_pipe.share_outputs()
        start_frame_num = start_frame_num.as_cpu().as_array().reshape(-1)
        path_tensor = self._cast_path_tensor(path_out[0], start_frame_num)
        video_tensor = self._cast_video_tensor(video_out, [path_tensor.shape[1]])
        labels = label.as_cpu().as_array().reshape(-1).tolist()
        self.video_pipe.release_outputs()
        self.path_pipe.release_outputs()
        self.video_pipe.schedule_run()
        self.path_pipe.schedule_run()
        self.idx += 1
        video_windows, path_windows, targets = self.pre_data(video_tensor, path_tensor)
        return video_windows, path_windows, targets


def create_train_test_loaders(
    root_dir,
    samples_per_video=6,
    window_size=5,
    npred=5,
    stride=5,
    batch_size=2,
    rank=0,
    world_size=1,
    device_id=0,
    seed=0,
    train_crop_len=45,
    eval_crop_len=900,
    **kwargs,
):
    train_loader = ScanPathWindowDataset(
        root_dir=root_dir,
        crop_len=train_crop_len,
        samples_per_video=samples_per_video,
        window_size=window_size,
        npred=npred,
        stride=stride,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        train=True,
        device_id=device_id,
        seed=seed,
        **kwargs,
    )
    test_loader = ScanPathWindowDataset(
        root_dir=root_dir,
        crop_len=eval_crop_len,
        samples_per_video=samples_per_video,
        window_size=window_size,
        npred=npred,
        stride=stride,
        batch_size=1,
        rank=0,
        world_size=1,
        train=False,
        device_id=device_id,
        seed=seed,
        **kwargs,
    )
    return train_loader, test_loader
