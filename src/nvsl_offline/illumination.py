"""Robust, spatially distributed additive lighting correction for one frame."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IlluminationEstimate:
    offset: float
    tile_mad: float
    tiles: int


def global_offset(raw_residual, rois):
    """Estimate current-image minus background brightness in gray levels.

    Split each chamber ROI into 4x4 tiles; take the median on a stride-four
    grid inside each tile, then the median of those equally weighted tiles.
    A small foreground object therefore cannot dominate by being very dark.
    No candidate, original trajectory, time index or neighboring frame is used.
    Positive offset means brighten the background; negative means darken it.
    Assumes an additive shift shared by most tiles; tile MAD measures disagreement,
    not calibrated uncertainty. Fixed sampling is part of this method's definition.
    """
    differences = []
    for x0, y0, x1, y1 in rois:
        xs = np.linspace(x0, x1, 5, dtype=int)
        ys = np.linspace(y0, y1, 5, dtype=int)
        for top, bottom in zip(ys[:-1], ys[1:]):
            for left, right in zip(xs[:-1], xs[1:]):
                values = raw_residual[top:bottom:4, left:right:4]
                if values.size:
                    differences.append(-float(np.median(values)))
    if not differences:
        raise ValueError("No chamber pixels available for illumination correction")
    values = np.asarray(differences)
    offset = float(np.median(values))
    return IlluminationEstimate(offset, float(np.median(np.abs(values-offset))), len(values))
