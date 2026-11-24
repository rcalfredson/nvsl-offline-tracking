"""Per-frame body cores and bounded foreground reconstruction; no chamber priors."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CoreRegion:
    mask: np.ndarray
    core: np.ndarray
    origin: tuple[int, int]
    truncated: bool


def body_core_regions(foreground, radius=2., growth=2):
    """Keep thick seeds, then expand by at most `growth` 8-neighbor steps.

    Padding treats the crop edge as background for the Euclidean distance map.
    Each seed is a connected component of distance > radius (saved-image pixels).
    Expansion stays inside the input foreground. Competing labels never overlap:
    an unassigned pixel with neighbors from different seeds remains unassigned.
    This is bounded growth, not reconstruction to convergence, which would
    recover the same unwanted thin attachments. Inherit parent truncation so
    erosion cannot disguise a foreground object cut off by the chamber ROI.
    """
    foreground = np.asarray(foreground)
    if foreground.dtype != np.uint8:
        foreground = (foreground > 0).astype(np.uint8)
    height, width = foreground.shape
    x0, y0, w, h = cv2.boundingRect(foreground)
    x1, y1 = x0 + w, y0 + h
    if radius < 0:
        # A negative threshold includes zero-distance background in the seeds.
        x0, y0, x1, y1 = 0, 0, width, height
    elif not w:
        return []
    # OpenCV labels 8-connected seeds in 2x2 blocks. Keep the original block
    # grid so equal-area candidates retain their original tie order.
    x0 -= x0 % 2
    y0 -= y0 % 2
    foreground = foreground[y0:y1, x0:x1]
    padded = cv2.copyMakeBorder(foreground, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    count, seeds, stats, _ = cv2.connectedComponentsWithStats(
        (distance > radius).astype(np.uint8), connectivity=8)
    if count <= 1:
        return []

    # A new crop edge inside the ROI cannot truncate a parent. Label parents
    # only when there is an original ROI edge to check.
    parents = None
    roi_edges = []
    if y0 == 0:
        roi_edges.append((0, slice(None)))
    if y1 == height:
        roi_edges.append((-1, slice(None)))
    if x0 == 0:
        roi_edges.append((slice(None), 0))
    if x1 == width:
        roi_edges.append((slice(None), -1))
    if any(foreground[edge].any() for edge in roi_edges):
        _, parents = cv2.connectedComponents(foreground, connectivity=8)
        truncated = set(np.concatenate([parents[edge] for edge in roi_edges])) - {0}

    owners = seeds.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(growth):
        high = cv2.dilate(owners, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        low = cv2.erode(np.where(owners > 0, owners, count).astype(np.float32), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=count)
        assign = (foreground > 0) & (owners == 0) & (high > 0) & (high == low)
        owners[assign] = high[assign]

    regions = []
    reach = max(0, growth)
    for label, (sx, sy, sw, sh, _) in enumerate(stats[1:].tolist(), start=1):
        # A label can grow by at most one pixel per axis in each iteration.
        left, top = max(0, sx - reach), max(0, sy - reach)
        right, bottom = min(seeds.shape[1], sx + sw + reach), min(seeds.shape[0], sy + sh + reach)
        mask = (owners[top:bottom, left:right] == label).astype(np.uint8)
        dx, dy, w, h = cv2.boundingRect(mask)
        left, top = left + dx, top + dy
        core = seeds[top:top + h, left:left + w] == label
        touches_roi = False
        if parents is not None:
            first_seed_x = sx + np.argmax(seeds[sy, sx:sx + sw] == label)
            touches_roi = parents[sy, first_seed_x] in truncated
        regions.append(CoreRegion(mask[dy:dy + h, dx:dx + w], core,
                                  (x0 + left, y0 + top), touches_roi))
    return regions
