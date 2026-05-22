# Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization

[![arXiv](https://img.shields.io/badge/arXiv-2305.02536-b31b1b.svg)](https://arxiv.org/abs/2305.02536)

Code for our IEEE TPAMI 2026 paper.

## ⚙️ Installation

```bash
conda create -n spath python=3.9 -y
conda activate spath

# Install a CUDA-enabled PyTorch build that matches your driver/toolkit.
# Example only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Needed for DALI video loading.
# Choose the DALI package matching your CUDA version.
pip install nvidia-dali-cuda120

pip install -e .
```

If you need to force GPU architectures during compilation, set `TORCH_CUDA_ARCH_LIST`, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install -e .
```

## 📁 Working Directory

All command-line tools resolve relative paths from `SPATH_WORK_DIR`. Define it once per shell:

```bash
export SPATH_WORK_DIR="$(pwd)"
mkdir -p data/raw data/processed checkpoints outputs splits
```

If `SPATH_WORK_DIR` is not set, the repository root is used.
You can also override the base directory per command with `--work-dir`.

## 📦 Data and Checkpoints

### Dataset Downloads

The raw datasets are not redistributed in this repository. Download them from the dataset authors:

- CVPR18 / VR-EyeTracking: [project page](https://github.com/xuyanyu-shh/VR-EyeTracking), [Baidu Pan](https://pan.baidu.com/s/1_YBi0GY1TQb6n3hJzQBKoQ), extraction code `kdcj`.
- VRW23 / VRVQW: [project page](https://github.com/Yao-Yiru/VR-Video-Database), [Baidu Pan](https://pan.baidu.com/s/18XEUNjMrOlIaqA2kQXEZfw?), extraction code `Fang`.

See [docs/datasets.md](docs/datasets.md) for dataset details, split rules, and preprocessing notes.

### Preprocessing

Training expects processed viewport-video data:

```text
data/processed/vrw23/
  videos/
    000_000.mp4
    000_001.mp4
  HM/
    000_000.npy
    000_001.npy
  path_dict3.json
  videos_frame.json
```

The preprocessing contract is:

1. Downsample panoramic videos and scanpaths to 5 fps.
2. Convert raw scanpaths to longitude/latitude degrees and save them as `HM_raw/{video_id}_{subject_id}.npy`.
3. Render gaze-centered viewport videos with FoV `112` degrees and size `448 x 252`; save them under `videos/`.
4. Project each temporal scanpath neighborhood into the local viewport coordinate system and save `HM/*.npy` arrays with shape `T x 2 x 61`.

See [docs/data_format.md](docs/data_format.md) for the tensor contract.

Create basic metadata from matching `videos/*.mp4` and `HM/*.npy` files:

```bash
python tools/prepare_metadata.py --data-root data/processed/vrw23
```

For the two paper datasets:

```bash
# VRW23 / VRVQW: first 400 videos for training, remaining 102 for testing.
python tools/prepare_metadata.py \
  --data-root data/processed/vrw23 \
  --split-mode vrw23

# CVPR18 / VR-EyeTracking: pass the train/test split used by the CVPR18 paper.
python tools/prepare_metadata.py \
  --data-root data/processed/cvpr18 \
  --split-mode list \
  --train-list splits/cvpr18_train.txt \
  --test-list splits/cvpr18_test.txt
```

### Pretrained Models

Download the released pretrained weights and place them as:

- [spath_vrw23.pt](https://drive.google.com/file/d/1IwzIUrDriTewD2qdHietvYtH4i0kqFrM/view?usp=drivesdk): trained on VRW23/VRVQW.
- [spath_cvpr18.pt](https://drive.google.com/file/d/1wv58V_AJdAuEjx-Yay4Xukdj-Ib3IcuG/view?usp=drivesdk): trained on CVPR18/VR-EyeTracking.

```text
checkpoints/spath_vrw23.pt
checkpoints/spath_cvpr18.pt
```

## 🚀 Training

Single GPU:

```bash
python tools/train.py \
  --data-root data/processed/vrw23 \
  --gpu-ids 0 \
  --output-dir outputs/checkpoints
```

Multi-GPU:

```bash
python tools/train.py \
  --data-root data/processed/vrw23 \
  --gpu-ids 0 1 2 3 \
  --batch-size 2 \
  --epochs 100 \
  --lr 1e-4 \
  --output-dir outputs/checkpoints
```

Resume training:

```bash
python tools/train.py \
  --data-root data/processed/vrw23 \
  --gpu-ids 0 1 2 3 \
  --resume outputs/checkpoints/spath_latest.pt
```

The best checkpoint is written to:

```text
outputs/checkpoints/spath_best.pt
```

## 🔍 Inference

```bash
python tools/infer.py \
  --video data/processed/vrw23/videos/000_000.mp4 \
  --init-scanpath data/processed/vrw23/HM_raw/000_000.npy \
  --gt-scanpath data/processed/vrw23/HM_raw/000_000.npy \
  --output-dir outputs/inference \
  --gpu-id 0
```

By default this uses `checkpoints/spath_vrw23.pt`. To use the CVPR18 model or your own trained checkpoint, add:

```bash
--checkpoint checkpoints/spath_cvpr18.pt

# or
--checkpoint outputs/checkpoints/spath_best.pt
```

The inference script writes sampled scanpaths and optional comparison figures under `outputs/inference`.

## 📚 Citation

```bibtex
@article{li2026scanpath,
  title={Scanpath Prediction in Panoramic Videos via Expected Code Length Minimization},
  author={Li, Mu and Fan, Kanglong and Ma, Kede},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
