"""Provider novel-view réel — Stable Virtual Camera (SEVA).

Réutilise l'entrée officielle SEVA (``demo.py`` CLI) plutôt que de
réimplémenter sa boucle de diffusion. Le provider :

  * vérifie la présence du code SEVA et des poids (HF) ;
  * exécute SEVA dans un sous-processus avec la tâche demandée ;
  * collecte les images de synthèse produites ;
  * renvoie un résultat structuré consommable par le pipeline.

Un usage réel nécessite un GPU (flash-attn / VRAM). En l'absence de ces
conditions, ``generate`` renvoie ``MODEL_UNAVAILABLE`` avec la raison exacte.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


class SevaProvider:
    name = "seva"

    def __init__(self, root: Path, version: str = "1.1"):
        self.root = Path(root)
        self.version = version
        self.demo = self.root / "demo.py"

    # -- utilitaires ------------------------------------------------------
    def _weights_present(self) -> bool:
        """Vérifie la présence locale des poids SEVA (sinon HF télécharge)."""
        # HF cache : ~/.cache/huggingface/hub/models--stabilityai--stable-virtual-camera
        hub = Path.home() / ".cache" / "huggingface" / "hub"
        pat = "models--stabilityai--stable-virtual-camera"
        for base in (hub,):
            d = base / pat
            if d.is_dir():
                return True
        local = self.root / ("modelv%s.safetensors" % self.version if float(self.version) > 1 else "model.safetensors")
        return local.is_file()

    def status(self) -> dict[str, Any]:
        return {
            "status": "OK" if self.demo.is_file() else "MODEL_UNAVAILABLE",
            "provider": self.name,
            "version": self.version,
            "root": str(self.root),
            "weights_cached": self._weights_present(),
        }

    # -- génération -------------------------------------------------------
    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Exécute SEVA sur un dossier d'images d'entrée.

        ``request`` (dict) :
          * ``input_dir`` (str) : dossier d'images d'entrée (obligatoire)
          * ``output_dir`` (str) : dossier de sortie
          * ``task`` (str) : tâche SEVA (ex. "basic", "advance", "spiral")
          * ``H``, ``W``, ``T`` : dimensions/trames (optionnels)
          * ``data_items`` (str) : sous-sélection (optionnel)
        """
        if not self.demo.is_file():
            return {"status": "MODEL_UNAVAILABLE",
                    "reason": f"SEVA demo.py introuvable: {self.demo}"}

        input_dir = request.get("input_dir")
        if not input_dir or not Path(input_dir).exists():
            return {"status": "UNKNOWN", "reason": "input_dir requis et doit exister"}

        output_dir = Path(request.get("output_dir", self.root / "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        task = request.get("task", "basic")

        cmd = [
            sys.executable, str(self.demo),
            "--data_path", str(input_dir),
            "--task", str(task),
            "--output_dir", str(output_dir),
            "--version", str(self.version),
        ]
        for key in ("H", "W", "T", "data_items"):
            if request.get(key) is not None:
                cmd += [f"--{key}", str(request[key])]

        env = dict(os.environ)
        # Les poids sont récupérés depuis HF à la volée si non cachés.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(self.root))
        except FileNotFoundError as exc:
            return {"status": "ERROR", "reason": f"SEVA exec failed: {exc}"}

        if proc.returncode != 0:
            return {
                "status": "ERROR",
                "reason": "SEVA process failed",
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-1500:],
                "weights_cached": self._weights_present(),
            }

        generated = sorted(str(p) for p in output_dir.rglob("*.png"))
        return {"status": "OK", "provider": self.name, "task": task,
                "generated": generated, "count": len(generated)}
