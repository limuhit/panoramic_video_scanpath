from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

import SPath
from spath.model import ScanpathPredictor, load_checkpoint


def read_video_rgb(video_path):
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = [frame for frame in reader]
    reader.close()
    arr = np.asarray(frames, dtype=np.float32)
    return arr.transpose(0, 3, 1, 2)


def load_scanpath(path):
    return np.load(path).astype(np.float32)


def tensor_to_video_frames(x):
    x = torch.clip(x * 255, 0, 255)
    x = x.detach().cpu().numpy()
    x = x.transpose(0, 1, 3, 4, 2).astype(np.uint8)
    return x[:, :, :, :, ::-1].copy()


class VideoPathHolder:
    def __init__(self, top_k=20, window_size=5, npred=5, device_id=0):
        self.device = f"cuda:{device_id}"
        self.top_k = top_k
        self.window_size = window_size
        self.npred = npred
        self.op = SPath.ViewportBatchOp(112.0, 252, 448, window_size, npred, top_k, device_id, False)

    def init_video_path(self, video_path, init_path, init_len=0):
        self.video = read_video_rgb(video_path) / 255.0
        self.video_len = self.video.shape[0]
        h, w = self.video.shape[2:]
        self.path = np.zeros((self.top_k, self.video_len, 2), dtype=np.float32)
        if init_len > 0:
            init_scanpath = load_scanpath(init_path)
            self.path[:, :init_len] = init_scanpath[:init_len]
            self.start_idx = init_len
        else:
            self.start_idx = 1
        self.path = torch.from_numpy(self.path).to(self.device).contiguous()
        start_video = np.zeros((self.window_size, 3, h, w), dtype=np.float32)
        start_video[-init_len:] = self.video[:init_len]
        self.start_video = torch.from_numpy(start_video).to(self.device).contiguous()
        self.op.set_video(self.start_video, init_len, self.video_len)
        self.start = True
        self.vend = False

    def get_video_slice(self):
        if self.start:
            self.start = False
            return self.start_video
        if self.start_idx + self.window_size > self.video_len:
            self.vend = True
            return None
        video_slice = torch.from_numpy(self.video[self.start_idx : self.start_idx + self.window_size]).to(self.device).contiguous()
        self.start_idx += self.window_size
        return video_slice

    def __iter__(self):
        return self

    def __next__(self):
        video_slice = self.get_video_slice()
        if self.vend:
            raise StopIteration
        return self.op.forward(video_slice, self.path)

    def set_path(self, vp_path, fork_ids):
        self.op.set_path(self.path, vp_path, fork_ids, True)

    def output_path(self):
        return self.path.detach().cpu().numpy()


class ScanpathSampler:
    def __init__(
        self,
        checkpoint_path,
        top_k=20,
        sample_num=3,
        stride=0.2,
        window_size=5,
        npred=5,
        device_id=0,
        allow_unsafe_checkpoint=False,
    ):
        self.device = f"cuda:{device_id}"
        self.holder = VideoPathHolder(top_k, window_size, npred, device_id)
        self.top_k = top_k
        self.npred = npred
        self.model = ScanpathPredictor(window_size, npred, sample_num, stride, 3, device_id).to(self.device)
        load_checkpoint(self.model, checkpoint_path, self.device, allow_unsafe=allow_unsafe_checkpoint)
        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        video_path,
        init_scanpath_path,
        start_idx=5,
        pid_ku=96,
        pid_pu=0.29,
        save_scanpath_dir=None,
        save_video_dir=None,
        output_prefix="sample",
    ):
        self.holder.init_video_path(video_path, init_scanpath_path, start_idx)
        prob = torch.zeros((self.top_k,), dtype=torch.float32, device=self.device)
        self.model.nvigator.Ziegler_Nichols(pid_ku, pid_pu)
        writers = self._open_writers(save_video_dir, output_prefix)
        nframes = 0

        first_batch = True
        for video_batch, path_batch, target in self.holder:
            if writers:
                nframes += self._write_video_batch(writers, video_batch)
            self.model.forward_base(video_batch, path_batch, prob)
            abs_err = 1e6
            self.model.lbx = None
            loop = 0
            while abs_err > 100:
                if loop > 2:
                    self.model.clear_stack(100)
                    break
                if first_batch:
                    self.model.nvigator.start_naviagor(path_batch[:, -1, self.npred - 1])
                    first_batch = False
                else:
                    self.model.nvigator.start_naviagor(-target)
                for _ in range(self.npred):
                    prob = self.model.forward_pred()
                abs_err = self.model.check_error(100)
                loop += 1
            fork_ids = self.model.history.type(torch.int32).contiguous()
            self.holder.set_path(self.model.tmp_path, fork_ids)

        for writer in writers:
            writer.release()

        paths = self.holder.output_path()
        if save_scanpath_dir is not None:
            out_dir = Path(save_scanpath_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            end = nframes if nframes else paths.shape[1]
            for idx in range(paths.shape[0]):
                np.save(out_dir / f"{output_prefix}_f{idx}.npy", paths[idx, :end])
        return paths

    def _open_writers(self, save_video_dir, output_prefix):
        if save_video_dir is None:
            return []
        out_dir = Path(save_video_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        writers = []
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        for idx in range(self.top_k):
            writers.append(cv2.VideoWriter(str(out_dir / f"{output_prefix}_f{idx}.mp4"), fourcc, 5, (448, 252)))
        return writers

    def _write_video_batch(self, writers, video_batch):
        frames = tensor_to_video_frames(video_batch)
        for sample_idx in range(video_batch.shape[0]):
            for frame_idx in range(video_batch.shape[1]):
                writers[sample_idx].write(frames[sample_idx][frame_idx])
        return int(video_batch.shape[1])
