from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


def plane_to_sphere(y, x):
    lat = (y - 0.5) * 180
    lon = (x - 0.5) * 360
    return lat, lon


def sphere_to_image(lon, lat, width, height):
    x = (lon + 180) / 360 * width
    y = (90 - lat) / 180 * height
    return int(x), int(y)


def load_display_scanpath(npy_file, max_points=None):
    data = np.load(npy_file)
    if max_points is not None:
        data = data[:max_points]
    sphere_data = np.zeros_like(data)
    for idx in range(data.shape[0]):
        plane_lon, plane_lat = data[idx, 0], data[idx, 1]
        if 0 <= plane_lon <= 1 and 0 <= plane_lat <= 1:
            sphere_lat, sphere_lon = plane_to_sphere(plane_lat, plane_lon)
            sphere_data[idx, 0] = sphere_lon
            sphere_data[idx, 1] = sphere_lat
        else:
            sphere_data[idx, 0] = plane_lon
            sphere_data[idx, 1] = plane_lat
    return sphere_data


def compute_plcc(true_position, pred_position):
    lon_true = true_position[:, 0]
    lat_true = true_position[:, 1]
    lon_pred = pred_position[:, 0]
    lat_pred = pred_position[:, 1]
    return 0.5 * (
        abs(np.corrcoef(lon_true, lon_pred)[0, 1]) +
        abs(np.corrcoef(lat_true, lat_pred)[0, 1])
    )


def draw_scanpath(frame, scanpath_data, current_point_idx, label):
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    boundary_threshold = w * 0.8
    points = []
    for lon, lat in scanpath_data:
        x, y = sphere_to_image(lon, lat, w, h)
        points.append((max(0, min(x, w - 1)), max(0, min(y, h - 1))))

    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        color = (227, 91, 0) if idx < current_point_idx else (2, 175, 85)
        thickness = 10
        if abs(end[0] - start[0]) > boundary_threshold:
            x1, y1 = start
            x2, y2 = end
            if x1 < x2:
                cv2.line(frame_copy, (x1, y1), (0, y1), color, thickness)
                cv2.line(frame_copy, (w - 1, y2), (x2, y2), color, thickness)
            else:
                cv2.line(frame_copy, (x1, y1), (w - 1, y1), color, thickness)
                cv2.line(frame_copy, (0, y2), (x2, y2), color, thickness)
        else:
            cv2.line(frame_copy, start, end, color, thickness)

    if 0 <= current_point_idx < len(points):
        x, y = points[current_point_idx]
        cv2.circle(frame_copy, (x, y), 25, (65, 113, 197), 10)
        cv2.circle(frame_copy, (x, y), 10, (65, 113, 197), -1)
    cv2.putText(frame_copy, f"{label} Frame {current_point_idx + 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    return frame_copy


def export_comparison_frames(video_path, gt_scanpath, pred_scanpath, output_dir, frame_indices=(0, 24, 49, 74)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path, "ffmpeg")
    for frame_idx in frame_indices:
        frame = reader.get_data(frame_idx)
        pred = draw_scanpath(frame, pred_scanpath, frame_idx, "Pred")
        gt = draw_scanpath(frame, gt_scanpath, frame_idx, "GT")
        combined = np.zeros((frame.shape[0] * 2, frame.shape[1], 3), dtype=np.uint8)
        combined[: frame.shape[0]] = pred
        combined[frame.shape[0] :] = gt
        imageio.imwrite(output_dir / f"frame_{frame_idx:04d}.jpg", combined)
    reader.close()


def combine_frames(output_dir, frame_indices=(0, 24, 49, 74), gap_width=30):
    output_dir = Path(output_dir)
    images = []
    for idx in frame_indices:
        img = cv2.imread(str(output_dir / f"frame_{idx:04d}.jpg"))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not images:
        return None
    h, w, c = images[0].shape
    combined = np.ones((h, len(images) * w + (len(images) - 1) * gap_width, c), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        start = idx * (w + gap_width)
        combined[:, start : start + w] = img
    fig, ax = plt.subplots(1, 1, figsize=(combined.shape[1] / 100, combined.shape[0] / 100), dpi=100)
    ax.imshow(combined)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = output_dir / "comparison.png"
    fig.savefig(out_path, format="png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    return out_path
