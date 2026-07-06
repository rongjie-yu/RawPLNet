from dataclasses import dataclass
import os

import numpy as np


@dataclass
class RawSynthesisConfig:
    invisp_checkpoint: str
    enable_eld_noise: bool = True
    noise_maxstep: int = 10000
    output_mode: str = "gray_replicated"

    def __post_init__(self):
        if self.output_mode not in {"gray_replicated", "gray_first_channel"}:
            raise ValueError("RawPLNet training does not support RGB fallback output modes")
        if self.noise_maxstep <= 0:
            raise ValueError("noise_maxstep must be positive")

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            invisp_checkpoint=str(cfg.INVISP_CHECKPOINT),
            enable_eld_noise=bool(cfg.ENABLE_ELD_NOISE),
            noise_maxstep=int(cfg.NOISE_MAXSTEP),
            output_mode=str(cfg.OUTPUT_MODE),
        )


class RawSynthesizer:
    def __init__(self, config, simulator=None, device="cuda"):
        self.config = config
        if simulator is None:
            if not config.invisp_checkpoint:
                raise FileNotFoundError("DATASETS.RAW.INVISP_CHECKPOINT is empty")
            if not os.path.isfile(config.invisp_checkpoint):
                raise FileNotFoundError(config.invisp_checkpoint)
            from .noise_simulator import NoiseSimulator

            simulator = NoiseSimulator(device=device, ckpt_path=config.invisp_checkpoint)
        self.simulator = simulator

    def synthesize_rgb(self, rgb, iter_idx=0):
        rgb_float = self._as_float_rgb(rgb)
        raw = self.simulator.rgb2raw(rgb_float, batched=False)
        if self.config.enable_eld_noise:
            ratio_dec = min(self.config.noise_maxstep, int(iter_idx)) / float(self.config.noise_maxstep)
            raw = self.simulator.raw2noisyRaw(raw, ratio_dec=ratio_dec, batched=False)
        demosaiced = self.simulator.raw2demosaicRaw(raw, batched=False)
        demosaiced = np.asarray(demosaiced, dtype=np.float32)
        self._validate_demosaiced_raw(demosaiced)
        gray = self._to_gray(demosaiced)
        if self.config.output_mode == "gray_replicated":
            return np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
        output = demosaiced.copy()
        output[..., 0] = gray
        return output.astype(np.float32)

    def _as_float_rgb(self, rgb):
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("RGB input must have shape H x W x 3")
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _validate_demosaiced_raw(self, raw):
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError("Demosaiced Raw output must have shape H x W x 3")
        if not np.isfinite(raw).all():
            raise ValueError("Demosaiced Raw output contains NaN or Inf")
        if raw.min(initial=0.0) < -1e-6 or raw.max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError("Demosaiced Raw output must be in [0, 1]")

    def _to_gray(self, raw):
        return np.clip(
            0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2],
            0.0,
            1.0,
        ).astype(np.float32)
