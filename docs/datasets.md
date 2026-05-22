# Datasets and Preprocessing

This project does not redistribute the raw datasets. Download the raw videos and annotations from the dataset authors, then convert them into the SPath training layout described in [data_format.md](data_format.md).

## Download Links

| Name in this repo | Dataset | Source / paper | Download |
| --- | --- | --- | --- |
| `cvpr18` | VR-EyeTracking, used by Xu et al. CVPR 2018 | [project page](https://github.com/xuyanyu-shh/VR-EyeTracking), [CVF paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_Gaze_Prediction_in_CVPR_2018_paper.html) | [Baidu Pan](https://pan.baidu.com/s/1_YBi0GY1TQb6n3hJzQBKoQ), code `kdcj` |
| `vrw23` | VR Video Quality in the Wild / VRVQW, used by Fang et al. VRW 2023 and the extended TCSVT version | [project page](https://github.com/Yao-Yiru/VR-Video-Database), [arXiv paper](https://arxiv.org/abs/2206.08751) | [Baidu Pan](https://pan.baidu.com/s/18XEUNjMrOlIaqA2kQXEZfw?), code `Fang` |

Check the original dataset terms before redistribution. The released SPath checkpoints are provided under `checkpoints/`; place raw datasets under `data/raw/` or another directory below `SPATH_WORK_DIR`.

## Target Layout

After preprocessing, create one processed root per dataset:

```text
data/processed/vrw23/
  videos/
    000_000.mp4
    000_001.mp4
  HM/
    000_000.npy
    000_001.npy
  HM_raw/
    000_000.npy
    000_001.npy
  path_dict3.json
  videos_frame.json
```

`videos/` contains viewport videos centered at each observer's gaze. `HM/` contains local scanpath tensors with shape `T x 2 x 61`. `HM_raw/` is optional for training but useful for inference and debugging.

## Coordinate Convention

Each raw scanpath should be a `float32` array of shape `T x 2`:

```text
scanpath[t] = [longitude_degrees, latitude_degrees]
```

The expected longitude range is `[0, 360)` and the latitude range is `[-90, 90]`. If a dataset gives equirectangular pixel coordinates `(x, y)` for a video of width `W` and height `H`, convert them with:

```python
longitude = x / W * 360.0
latitude = 90.0 - y / H * 180.0
```

Keep the time order unchanged and make sure each scanpath has the same frame count as its corresponding downsampled video.

## Temporal Preprocessing

The paper setting uses 5 fps. For each dataset:

1. Decode the panoramic equirectangular videos.
2. Downsample videos to 5 fps.
3. Resample the gaze records to the same 5 fps timeline.
4. Save one raw scanpath per `(video, subject)` pair as `HM_raw/{video_id}_{subject_id}.npy`.

For high-rate gaze logs, interpolate longitude/latitude to the target timestamps. For frame-aligned annotations, keep the matching frames after downsampling. For longitude interpolation near the wrap boundary, unwrap longitude first, interpolate, then wrap back to `[0, 360)`.

## Spatial Preprocessing

SPath trains on viewports, not on full panoramas. For every raw panoramic video and every observer scanpath:

1. Ensure the equirectangular video has a 2:1 aspect ratio. The original experiments used resized panoramas for efficiency; the exact ERP resolution is less important than preserving the 2:1 geometry.
2. For each frame `t`, use the gaze point `HM_raw[t]` as the viewport center.
3. Render a rectilinear viewport with field of view `112` degrees and output size `448 x 252` using `SPath_operator.Viewport(112, 252, 448, ...)`.
4. Save the result as `videos/{video_id}_{subject_id}.mp4`.

The model and CUDA preprocessing use center bias `(223.5, 125.5)`, so keep the viewport resolution at `448 x 252` unless you also update the operator settings and retrain.

## Local Scanpath Tensor

For every frame `t`, build a local temporal neighborhood of 61 points:

```text
raw[t - 30], ..., raw[t], ..., raw[t + 30]
```

Pad the beginning and end with the edge gaze point. Project the 61 equirectangular gaze points into the viewport coordinate system centered at frame `t` using `SPath_operator.Erp2vp(112, 252, 448, ...)`. Save the projected tensor as:

```text
HM/{video_id}_{subject_id}.npy  # shape T x 2 x 61, dtype float32
```

Points created only by temporal padding should be filled with the viewport center `(223.5, 125.5)`. This matches the original `ws = 30` preprocessing and the CUDA `PreData` operator.

## Splits and Metadata

Create metadata after all `videos/*.mp4` and `HM/*.npy` pairs are ready.

For VRW23/VRVQW, use the paper split: videos `000` to `399` for training and `400` to `501` for testing.

```bash
python tools/prepare_metadata.py \
  --data-root data/processed/vrw23 \
  --split-mode vrw23
```

For CVPR18/VR-EyeTracking, use the train/test split from Xu et al. for reproduction. Put one sample name per line; both `000_000.npy` and `000_000` are accepted.

```bash
python tools/prepare_metadata.py \
  --data-root data/processed/cvpr18 \
  --split-mode list \
  --train-list splits/cvpr18_train.txt \
  --test-list splits/cvpr18_test.txt
```

For a smoke test on a new folder, a random split is also available:

```bash
python tools/prepare_metadata.py \
  --data-root data/processed/smoke \
  --split-mode random \
  --test-ratio 0.2 \
  --seed 0
```

## Sanity Check

Before training, verify that the processed files agree:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import cv2
import numpy as np

root = Path(os.environ.get("SPATH_WORK_DIR", ".")) / "data/processed/vrw23"
with open(root / "path_dict3.json", "r", encoding="utf-8") as f:
    split = json.load(f)
with open(root / "videos_frame.json", "r", encoding="utf-8") as f:
    frames = json.load(f)

for name in split["train"][:5] + split["test"][:5]:
    hm = np.load(root / "HM" / name)
    cap = cv2.VideoCapture(str(root / "videos" / (Path(name).stem + ".mp4")))
    nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(name, hm.shape, nframe, frames[Path(name).stem.split("_")[0]])
    assert hm.ndim == 3 and hm.shape[1:] == (2, 61)
    assert hm.shape[0] == nframe
PY
```
