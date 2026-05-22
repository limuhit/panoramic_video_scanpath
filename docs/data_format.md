# Data Format

SPath training uses viewport-video samples and processed local scanpath tensors. Dataset download links and preprocessing steps are described in [datasets.md](datasets.md).

## Directory

```text
data/processed/vrw23/
  videos/
    000_000.mp4
  HM/
    000_000.npy
  path_dict3.json
  videos_frame.json
```

`path_dict3.json` stores the train/test scanpath lists:

```json
{
  "test": ["400_000.npy"],
  "train": ["000_000.npy", "000_001.npy"]
}
```

`videos_frame.json` maps the video id prefix to frame count:

```json
{
  "000": 300,
  "400": 300
}
```

The helper below creates both files from matching `videos/*.mp4` and `HM/*.npy` pairs:

```bash
python tools/prepare_metadata.py --data-root data/processed/vrw23
```

Use `--split-mode vrw23` for the VRW23/VRVQW 400/102 split, or `--split-mode list --train-list ... --test-list ...` for a paper-provided CVPR18 split.

## Video Files

Each training video should be a viewport video, not the original full equirectangular panorama. The default model expects frames around `252 x 448`; this produces the `112`-length visual feature vector used by the prediction head.

## Processed Scanpaths

Each `HM/*.npy` file should contain a `float32` array with shape:

```text
T x 2 x 61
```

- `T`: number of frames.
- axis `1`: x/y coordinates.
- axis `2`: local temporal neighborhood centered at the current frame.

The values are absolute viewport pixel coordinates before the CUDA `PreData` operator subtracts the default center bias `(223.5, 125.5)`. This follows the original preprocessing convention used by the training code.

For inference-only sampling, raw scanpaths in `HM_raw/*.npy` can be `T x 2` arrays. They are used by `tools/infer.py` to seed the first frames and optionally visualize predictions.
