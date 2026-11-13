"""Optional image-based body geometry with a fixed detection center."""

import cv2
import numpy as np


def canonical_ellipse(ellipse):
    """Equivalent OpenCV geometry with w <= h, theta modulo 180 degrees."""
    center, (w, h), theta = ellipse
    if w > h:
        w, h, theta = h, w, theta+90
    return center, (w, h), theta % 180


def weighted_moment_ellipse(contour, residual, mask, center):
    """Use D²-weighted second moments of accepted foreground about fixed center.

    For a uniform filled ellipse, covariance eigenvalues are squared semiaxes / 4;
    full diameters are therefore 4*sqrt(eigenvalue). Squared positive darkness
    weights reduce the influence of weak peripheral pixels. This is a moment
    estimate, not a boundary fit or a guaranteed anatomical body segmentation.
    None means numerically unresolved geometry; caller retains the box estimate.
    """
    x0, y0, width, height = cv2.boundingRect(contour)
    component = np.zeros((height, width), np.uint8)
    offset = np.array([[[x0, y0]]], dtype=contour.dtype)
    cv2.drawContours(component, [contour-offset], -1, 1, -1)
    selected = (component > 0) & (mask[y0:y0+height, x0:x0+width] > 0)
    ys, xs = np.nonzero(selected)
    if len(xs) < 5:
        return None
    weights = np.maximum(residual[ys+y0, xs+x0].astype(np.float64), 0)**2
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return None
    delta = np.column_stack((xs+x0-center[0], ys+y0-center[1])).astype(np.float64)
    moments = (delta.T*weights)@delta/weights.sum()
    try:
        eigenvalues, axes = np.linalg.eigh(moments)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 1e-8:
        return None
    minor, major = 4*np.sqrt(eigenvalues)
    # OpenCV theta rotates the width axis; height is the major axis here.
    theta = (np.degrees(np.arctan2(axes[1, 1], axes[0, 1]))-90) % 180
    return center, (float(minor), float(major)), float(theta)
