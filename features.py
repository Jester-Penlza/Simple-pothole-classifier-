"""Small, explainable image feature extractor for the pothole classifier."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SIZE = (16, 16)
HISTOGRAM_BINS = 12


def extract_features(image_path: str | Path) -> list[float]:
    """Convert one image into a compact numeric feature vector.

    The model uses three easy-to-explain groups of features:
    1. a 16 x 16 grayscale thumbnail (overall shapes and dark areas),
    2. horizontal/vertical edge strengths (road texture and boundaries), and
    3. simple RGB color histograms plus channel means and standard deviations.
    """

    with Image.open(image_path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        small_rgb = rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    pixels = [tuple(value / 255.0 for value in pixel) for pixel in small_rgb.getdata()]
    channels = [[pixel[channel] for pixel in pixels] for channel in range(3)]
    gray = [
        0.299 * red + 0.587 * green + 0.114 * blue
        for red, green, blue in pixels
    ]

    width, height = IMAGE_SIZE
    horizontal_edges: list[float] = []
    vertical_edges: list[float] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            right = index if x == width - 1 else index + 1
            below = index if y == height - 1 else index + width
            horizontal_edges.append(abs(gray[right] - gray[index]))
            vertical_edges.append(abs(gray[below] - gray[index]))

    histograms: list[float] = []
    for values in channels:
        counts = [0] * HISTOGRAM_BINS
        for value in values:
            bin_index = min(HISTOGRAM_BINS - 1, int(value * HISTOGRAM_BINS))
            counts[bin_index] += 1
        histograms.extend(count / len(values) for count in counts)

    channel_means = [sum(values) / len(values) for values in channels]
    channel_stds = [
        sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        for values, mean in zip(channels, channel_means)
    ]

    return gray + horizontal_edges + vertical_edges + histograms + channel_means + channel_stds

