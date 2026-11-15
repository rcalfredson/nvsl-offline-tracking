"""Conservative per-frame detections; no motion prediction or interpolation."""

from dataclasses import dataclass

import cv2
import numpy as np

from .background import Background
from .geometry import weighted_moment_ellipse
from .illumination import global_offset
from .recording import Calibration


@dataclass(frozen=True)
class DetectionConfig:
    mode: str = "absolute"
    threshold: float = 25.0
    signal_floor: float = 6.0
    score_threshold: float = 4.0
    area_min: float = 40.0
    area_max: float = 1000.0
    ambiguity_ratio: float = 0.5
    reject_ambiguous: bool = True
    reject_truncated: bool = True
    shape_fit: str = "weighted_moments"
    illumination: str = "none"

    def validate(self):
        if self.mode not in ("absolute", "normalized"):
            raise ValueError("Unknown detector mode")
        if self.shape_fit not in ("box", "weighted_moments"):
            raise ValueError("Unknown body shape method")
        if self.illumination not in ("none", "global_offset"):
            raise ValueError("Unknown illumination correction")
        vals = [self.threshold, self.signal_floor, self.score_threshold, self.area_min,
                self.area_max, self.ambiguity_ratio]
        if (not np.isfinite(vals).all() or min(vals[:4]) <= 0 or self.area_max < self.area_min
                or not 0 < self.ambiguity_ratio <= 1):
            raise ValueError("Invalid detector configuration")


@dataclass
class Candidate:
    ellipse: tuple
    area: float
    contour: np.ndarray
    truncated: bool


@dataclass
class Detection:
    slot: int
    ellipse: tuple | None
    candidates: list[Candidate]
    rejected_counts: dict[str, int]
    reason: str
    runner_up_ratio: float | None
    shape_method: str = "missing"
    illumination_offset: float = 0.0
    illumination_tile_mad: float = 0.0


def detect(frame, background: Background, calibration: Calibration, config: DetectionConfig):
    config.validate()
    residual = background.residual(frame)
    offset = tile_mad = 0.0
    if config.illumination == "global_offset":
        lighting = global_offset(residual, [calibration.roi(slot) for slot in range(4)])
        offset, tile_mad = lighting.offset, lighting.tile_mad
        residual = residual + offset
    if config.mode == "absolute":
        mask = residual > config.threshold
    else:
        mask = ((residual > config.signal_floor)
                & (residual/background.noise > config.score_threshold))
    mask = mask.astype(np.uint8)*255
    results = []
    for slot in range(4):
        x0, y0, x1, y1 = calibration.roi(slot)
        crop = mask[y0:y1, x0:x1]
        contours, hierarchy = cv2.findContours(crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        counts = dict(small=0, large=0, few_points=0, truncated=0)
        candidates = []
        for j, contour in enumerate(contours):
            if hierarchy[0][j][3] != -1:
                continue
            area = cv2.contourArea(contour)
            if area < config.area_min:
                counts["small"] += 1
                continue
            if area > config.area_max:
                counts["large"] += 1
                continue
            if len(contour) < 5:
                counts["few_points"] += 1
                continue
            points = contour[:, 0, :]
            truncated = bool(np.any((points[:, 0] == 0) | (points[:, 0] == crop.shape[1]-1)
                                   | (points[:, 1] == 0) | (points[:, 1] == crop.shape[0]-1)))
            full_contour = contour + np.array([[[x0, y0]]], dtype=contour.dtype)
            candidates.append(Candidate(cv2.minAreaRect(full_contour), area, full_contour, truncated))
            counts["truncated"] += int(truncated)
        candidates.sort(key=lambda c: c.area, reverse=True)
        ratio = candidates[1].area/candidates[0].area if len(candidates) > 1 else None
        reason, ellipse = "no_candidate", None
        if candidates:
            if config.reject_truncated and candidates[0].truncated:
                reason = "truncated"
            elif config.reject_ambiguous and ratio is not None and ratio >= config.ambiguity_ratio:
                reason = "ambiguous"
            else:
                reason, ellipse = "accepted", candidates[0].ellipse
        shape_method = "missing"
        if ellipse is not None:
            shape_method = "box"
            if config.shape_fit == "weighted_moments":
                refined = weighted_moment_ellipse(candidates[0].contour, residual, mask, ellipse[0])
                if refined is not None:
                    ellipse, shape_method = refined, "weighted_moments"
                else:
                    shape_method = "box_fallback"
        results.append(Detection(slot, ellipse, candidates, counts, reason, ratio, shape_method,
                                 offset, tile_mad))
    return results, residual, mask
