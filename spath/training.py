import os
from pathlib import Path

import torch
import torch.distributed as dist

from spath.serialization import safe_torch_load


def setup_distributed(rank, world_size, master_addr="localhost", master_port="12355"):
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", str(master_port))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def negative_log_likelihood(prob):
    return torch.mean(-torch.log(prob + 1e-16))


def train_one_epoch(model, loader, optimizer, device, epoch, clip=1.0, alpha=1.0, log_every=20):
    model.train()
    loader.set_epoch(epoch)
    total_loss = 0.0
    nstep = 0
    for batch_idx, (video, path, target) in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        video = video.to(device)
        path = path.to(device)
        target = target.to(device)
        prob = model(video, path, target)
        loss = alpha * negative_log_likelihood(prob)
        loss.backward()
        parameters = model.module.parameters() if hasattr(model, "module") else model.parameters()
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        total_loss += loss.item()
        nstep += 1
        if log_every and batch_idx % log_every == 0:
            print(f"epoch={epoch} step={batch_idx}/{loader.num_batch()} loss={loss.item():.6f}", flush=True)
    return total_loss / max(nstep, 1)


@torch.no_grad()
def validate_nll(model, loader, device):
    model.eval()
    total_loss = 0.0
    nstep = 0
    for video, path, target in loader:
        video = video.to(device)
        path = path.to(device)
        target = target.to(device)
        prob = model(video, path, target)
        loss = negative_log_likelihood(prob)
        total_loss += loss.item()
        nstep += 1
    return total_loss / max(nstep, 1)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_partial_state_dict(model, checkpoint_path, device, allow_unsafe=False):
    state = safe_torch_load(checkpoint_path, map_location=device, allow_unsafe=allow_unsafe)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            loaded[key] = value
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return len(loaded), skipped


class CheckpointSaver:
    def __init__(self, output_dir, prefix="spath"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.best_loss = None

    def save(self, model, epoch, val_loss, optimizer=None):
        payload = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model": unwrap_model(model).state_dict(),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        latest_path = self.output_dir / f"{self.prefix}_latest.pt"
        torch.save(payload, latest_path)
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = self.output_dir / f"{self.prefix}_best.pt"
            torch.save(payload, best_path)
            return best_path
        return latest_path
