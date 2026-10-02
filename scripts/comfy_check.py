#!/usr/bin/env python3
"""Diagnostic de connectivité ComfyUI et de présence des modèles Wan 2.2.

Usage :

    python scripts/comfy_check.py
    COMFY_URL=http://127.0.0.1:8188 python scripts/comfy_check.py

Sortie : statut du serveur + vérification des 3 fichiers de modèles requis
(diffusion, VAE, text encoder). Aucun secret n'est affiché.

Code de sortie :
    0  ComfyUI joignable et modèles requis présents ;
    1  ComfyUI joignable mais modèle(s) manquant(s) ;
    2  ComfyUI injoignable.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from goalreel.studio.studio_config import load_studio_settings  # noqa: E402


def _fetch_object_info(base_url: str, timeout: float = 10.0) -> dict:
    resp = httpx.get(f"{base_url}/object_info", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _available_names(info: dict, node: str, field: str) -> list[str]:
    try:
        return list(info[node]["input"]["required"][field][0])
    except (KeyError, TypeError, IndexError):
        return []


def main() -> int:
    settings = load_studio_settings()
    url = settings.comfy_url
    print(f"ComfyUI URL: {url}")

    try:
        info = _fetch_object_info(url)
    except Exception as exc:  # noqa: BLE001
        print(f"[X] ComfyUI injoignable: {exc}")
        print("    Vérifiez que ComfyUI tourne et que COMFY_URL est correct.")
        return 2

    print("[OK] ComfyUI joignable.")

    checks = {
        "UNETLoader.unet_name": (info, "UNETLoader", "unet_name", settings.models.diffusion),
        "VAELoader.vae_name": (info, "VAELoader", "vae_name", settings.models.vae),
        "CLIPLoader.clip_name": (info, "CLIPLoader", "clip_name", settings.models.text_encoder),
    }

    missing = []
    for label, (obj_info, node, field, expected) in checks.items():
        names = _available_names(obj_info, node, field)
        if not names:
            print(f"[?] {label}: impossible de lire la liste ({node}/{field}).")
            continue
        if expected in names:
            print(f"[OK] {label}: {expected}")
        else:
            print(f"[X] {label}: {expected} ABSENT.")
            missing.append(expected)

    if missing:
        print("\nModèles manquants — placez-les dans ComfyUI :")
        print("  models/diffusion_models/", settings.models.diffusion)
        print("  models/vae/", settings.models.vae)
        print("  models/text_encoders/", settings.models.text_encoder)
        return 1

    print("\nTous les modèles requis sont présents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
