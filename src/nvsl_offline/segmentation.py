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
    foreground = (np.asarray(foreground) > 0).astype(np.uint8)
    distance = cv2.distanceTransform(np.pad(foreground, 1), cv2.DIST_L2,
                                    cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    count, seeds = cv2.connectedComponents((distance > radius).astype(np.uint8), connectivity=8)
    if count <= 1:
        return []
    _, parents = cv2.connectedComponents(foreground, connectivity=8)
    truncated = set(np.r_[parents[0], parents[-1], parents[:, 0], parents[:, -1]]) - {0}
    owners = seeds.astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(growth):
        high = cv2.dilate(owners, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        low = cv2.erode(np.where(owners > 0, owners, count).astype(np.float32), kernel,
                        borderType=cv2.BORDER_CONSTANT, borderValue=count)
        assign = (foreground > 0) & (owners == 0) & (high > 0) & (high == low)
        owners[assign] = high[assign]
    regions = []
    for label in range(1, count):
        yy, xx = np.nonzero(owners == label)
        x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
        core = seeds[y0:y1, x0:x1] == label
        seed_y, seed_x = np.nonzero(seeds == label)
        parent = parents[seed_y[0], seed_x[0]]
        regions.append(CoreRegion((owners[y0:y1, x0:x1] == label).astype(np.uint8),
                                  core, (x0, y0), parent in truncated))
    return regions
