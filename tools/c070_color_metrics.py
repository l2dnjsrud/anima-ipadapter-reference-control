from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def c070_color_metrics(image_path: Path) -> tuple[float, float, float, float, float]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    spread = arr.max(axis=2) - arr.min(axis=2)
    green_mask = (green > 50.0) & (spread > 25.0) & (green > red * 1.08) & (green > blue * 1.03)
    strong_mask = (green > 70.0) & (spread > 35.0) & (green > np.maximum(red, blue) * 1.15)
    red_mask = (red > 90.0) & (spread > 50.0) & (red > np.maximum(green, blue) * 1.25)
    center = np.zeros(green_mask.shape, dtype=bool)
    center[32:96, 32:96] = True
    return (
        float(green_mask.mean()),
        float(strong_mask.mean()),
        float(red_mask.mean()),
        float(green_mask[center].mean()),
        float(green_mask[~center].mean()),
    )
