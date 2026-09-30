"""Étalonnage cinématique discret (grade + vignette) — traitement par pixels.

Style visé
----------
Étalonnage « football cinématique » **retenu**, pas un look gaming :

  * courbe de contraste douce type filmique (S-curve sur la luminance) ;
  * préservation de la peau, des maillots et de la pelouse (pas de dominante) ;
  * léger *teal-shadow / warm-highlight* (nuit de stade) très contenu ;
  * saturation bornée, contraste borné, gamma borné ;
  * vignette très douce, uniquement en option (`vignette=True`).

Interdits : contours gaming, flou excessif, effets glitch, HDR surtraité,
sur-accentuation. Aucun de ces effets n'est produit par ce module.

Le traitement est **déterministe** et purement déterministe/vectorisé (numpy),
donc testable et reproductible. Il agit sur les pixels source réels — il ne
génère ni joueur, ni stade, ni angle.
"""
from __future__ import annotations

import numpy as np

# Paramètres documentés (bornes conservatrices).
CONTRAST_GAIN = 1.057      # <= ~6 % : contraste filmique doux
SAT_GAIN = 1.047           # <= ~5 % : saturation légère
TEAL_SHADOW = 0.022        # teinte froide dans les ombres (très faible)
WARM_HIGHLIGHT = 0.020     # teinte chaude dans les hautes lumières (très faible)
SHADOW_SOFTNESS = 0.22     # mélange conservé du signal original dans les ombres
VIGNETTE_STRENGTH = 0.16   # atténuation max. aux coins
GRAIN_STRENGTH = 0.0       # grain DÉSACTIVÉ par défaut (jamais « pour le style »)


def _vignette_mask(h: int, w: int, strength: float) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = (xs * xs + ys * ys) / 2.0
    return (1.0 - strength * (r2 ** 1.4)).astype(np.float32)


def grade_frame(frame: np.ndarray, vignette: bool = False,
                filmic: bool = True) -> np.ndarray:
    """Applique l'étalonnage discret à une frame BGR uint8.

    Renvoie une nouvelle frame uint8 de mêmes dimensions. Aucun effet n'est
    jamais revendiqué au-delà de ce que le code applique réellement.
    """
    if frame is None or frame.size == 0:
        return frame
    src = frame.astype(np.float32)
    h, w = src.shape[:2]

    # 1) Contraste doux + teinte filmique par canal (BGR).
    if filmic:
        luma = (0.114 * src[..., 0] + 0.587 * src[..., 1] + 0.299 * src[..., 2])
        luma_n = np.clip(luma / 255.0, 0.0, 1.0)
        # S-curve douce : contraste filmique retenu, sans écrêtage dur.
        curve = np.clip(
            luma_n + CONTRAST_GAIN * (luma_n - 0.5) * luma_n * (1.0 - luma_n) * 2.0,
            0.0, 1.0)
        gain_l = (curve / np.maximum(luma_n, 1e-4))[..., None]
        out = src * gain_l
        # teal-shadow / warm-highlight, bornés par le signal réel (BGR).
        shadow = (1.0 - luma_n) * TEAL_SHADOW * 255.0
        highlight = luma_n * WARM_HIGHLIGHT * 255.0
        out[..., 0] = out[..., 0] + shadow * 0.9 + highlight * 0.4   # B
        out[..., 1] = out[..., 1] + shadow * 0.6 + highlight * 0.6   # G
        out[..., 2] = out[..., 2] + shadow * 0.4 + highlight * 0.9   # R
        # conservation partielle du signal original dans les ombres (pas de crash).
        out = out * (1.0 - SHADOW_SOFTNESS * (1.0 - luma_n)[..., None]) + \
            src * (SHADOW_SOFTNESS * (1.0 - luma_n)[..., None])
    else:
        luma = (0.114 * src[..., 0] + 0.587 * src[..., 1] + 0.299 * src[..., 2])[..., None]
        out = (src - luma) * CONTRAST_GAIN + luma

    # 2) Saturation bornée (autour de la luminance).
    luma2 = (0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2])[..., None]
    out = (out - luma2) * SAT_GAIN + luma2

    # 3) Vignette très douce (optionnelle).
    if vignette:
        out = out * _vignette_mask(h, w, VIGNETTE_STRENGTH)[..., None]

    # 4) Grain optionnel : désactivé (GRAIN_STRENGTH = 0) — aucun bruit ajouté.
    return np.clip(out, 0, 255).astype(np.uint8)


def grade_summary() -> dict[str, object]:
    """Description honnête et testable du traitement appliqué."""
    return {
        "style": "restrained_football_cinematic",
        "contrast_gain": CONTRAST_GAIN,
        "saturation_gain": SAT_GAIN,
        "teal_shadow": TEAL_SHADOW,
        "warm_highlight": WARM_HIGHLIGHT,
        "vignette_strength": VIGNETTE_STRENGTH,
        "grain_strength": GRAIN_STRENGTH,
        "grain_enabled": GRAIN_STRENGTH > 0.0,
        "sharpening": "none",
        "outlines": "none",
        "glitch": "none",
        "hdr_overprocess": "none",
    }


def vignette_frame(frame: np.ndarray, strength: float = VIGNETTE_STRENGTH) -> np.ndarray:
    """Vignette seule (utilisée pour le fondu OUTRO, documentée)."""
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    mask = _vignette_mask(h, w, strength)[..., None]
    return np.clip(frame.astype(np.float32) * mask, 0, 255).astype(np.uint8)
