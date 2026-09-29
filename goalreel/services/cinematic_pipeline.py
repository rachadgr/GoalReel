"""Pipeline cinématique — interpolation (RIFE) et super-résolution.

Ces étapes opèrent sur des frames (pas sur l'analyse) et produisent un clip
via FFmpeg. Elles sont **optionnelles** et n'utilisent RIFE/SR que si les
modèles sont disponibles ; sinon elles appliquent un fallback explicite
(blend / Lanczos) sans prétendre à une qualité neuronale.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2

from ..models.manager import ModelManager
from .ai import InterpolationService, SuperResolutionService


def _ffmpeg(args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *map(str, args)],
                          check=True, capture_output=True, text=True)


class CinematicPipeline:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.interp = InterpolationService(manager)
        self.sr = SuperResolutionService(manager)

    def interpolate_clip(self, frames, out_path: str, factor: int = 2,
                         fps: int = 30) -> dict:
        """Interpole ``frames`` (liste BGR) et écrit un clip à ``out_path``."""
        slow = self.interp.slowdown_frames(frames, factor=factor)
        mode = "rife" if self.interp.available else "fallback"
        self._write_frames(slow, out_path, fps=fps)
        return {"status": "OK", "mode": mode, "in_frames": len(frames),
                "out_frames": len(slow), "fps": fps, "out": str(out_path)}

    def upscale_clip(self, frames, out_path: str, fps: int = 30) -> dict:
        up = [self.sr.upscale(f)[0] for f in frames]
        mode = "model" if self.sr.available else "fallback"
        self._write_frames(up, out_path, fps=fps)
        return {"status": "OK", "mode": mode, "scale": self.sr.scale,
                "in_frames": len(frames), "out": str(out_path)}

    @staticmethod
    def _write_frames(frames, out_path: str, fps: int = 30):
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(frames):
                cv2.imwrite(f"{tmp}/f_{i:06d}.png", f)
            _ffmpeg(["-y", "-framerate", fps, "-i", f"{tmp}/f_%06d.png",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", out_path])
