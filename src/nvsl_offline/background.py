"""Recording-wide, tiled robust estimates from a bounded sample stack."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BackgroundConfig:
    quantile: float = 0.8
    noise_floor: float = 3.0
    noise_smoothing: float = 3.0
    tile_rows: int = 32

    def validate(self):
        if not (0.5 <= self.quantile < 1):
            raise ValueError("Background quantile must be in [0.5, 1)")
        if (not np.isfinite([self.noise_floor, self.noise_smoothing]).all()
                or self.noise_floor <= 0 or self.noise_smoothing < 0 or self.tile_rows < 1):
            raise ValueError("Invalid background noise/tile configuration")


@dataclass
class Background:
    median: np.ndarray
    estimate: np.ndarray
    noise: np.ndarray
    config: BackgroundConfig

    @classmethod
    def fit(cls, samples, config=BackgroundConfig()):
        config.validate()
        if samples.ndim != 3 or samples.shape[0] < 3 or samples.dtype != np.uint8:
            raise ValueError("Background needs at least three uint8 grayscale samples")
        _, height, width = samples.shape
        median, estimate, noise = [np.empty((height, width), np.float32) for _ in range(3)]
        for y in range(0, height, config.tile_rows):
            block = np.asarray(samples[:, y:y+config.tile_rows], dtype=np.float32)
            med = np.median(block, axis=0)
            median[y:y+config.tile_rows] = med
            estimate[y:y+config.tile_rows] = np.quantile(block, config.quantile, axis=0)
            noise[y:y+config.tile_rows] = 1.4826*np.median(np.abs(block-med), axis=0)
        if config.noise_smoothing:
            noise = cv2.GaussianBlur(noise, (0, 0), config.noise_smoothing)
        noise = np.maximum(noise, config.noise_floor)
        return cls(median, estimate, noise, config)

    def save(self, path):
        # Caller owns a newly created run directory; never writes reference assets.
        np.savez_compressed(path, median=self.median, estimate=self.estimate, noise=self.noise,
                            **{f"config_{k}": v for k, v in asdict(self.config).items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            config = BackgroundConfig(**{k: z[f"config_{k}"].item() for k in asdict(BackgroundConfig())})
            return cls(z["median"], z["estimate"], z["noise"], config)

    def residual(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return self.estimate-gray
