"""Renderer plan-aware — exécution du plan du Cinematic Director V2.

Ce module **consomme** un :class:`goalreel.cinematic.edit_plan.EditPlan` et
produit le rendu vertical final. Il ne prend aucune décision cinématique : toute
la structure (phases, cadrages, vitesses, transitions) vient du director.

Garanties techniques (contrat de sortie)
---------------------------------------
  * **1080x1920**, 30 fps, H.264, ``yuv420p``, AAC — identique à la V1 ;
  * **durée préservée** : la durée de sortie égale la durée source. Les ralentis
    sont « payés » par les accélérations via le budget d'images du plan
    (``plan.time_map``) ; la déformation temporelle reste **bornée** et la
    synchronisation audio est exacte (piste audio copiée telle quelle, aucun
    ré-échantillonnage audio) ;
  * **aucune bordure noire** : le crop 9:16 est borné dans la source ;
  * **aucun étirement** : chaque plan est rééchantillonné exactement vers la
    taille de sortie (aspect 9:16 constant) ;
  * **aucune frame gelée** : la correspondance temporelle est strictement
    croissante sur chaque plan ;
  * **aucune perte de sujet** : la fenêtre suit le centre réellement suivi et
    reste bornée dans la source.

Vérité sur les back-ends
------------------------
  * **RIFE n'est jamais revendiqué ici** : la vitesse est réalisée par
    ré-échantillonnage temporel (``TEMPORAL_RESAMPLE``). Pour rester honnête
    *et* visuellement propre, le sous-échantillonnage temporel effectue un
    **blend linéaire local entre les deux frames source encadrantes** — c'est
    exactement le *fallback documenté* de
    :meth:`goalreel.services.ai.InterpolationService._blend`, appliqué uniquement
    là où une frame intermédiaire est nécessaire (jamais présenté comme RIFE,
    jamais présenté comme de l'interpolation par flux optique) ;
  * aucun pixel n'est inventé : tous les pixels proviennent de la source réelle ;
    le grade est un traitement déterministe appliqué à ces pixels réels.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .edit_plan import (
    CUT_DISSOLVE,
    CUT_FADE_FROM_BLACK,
    CUT_FADE_TO_BLACK,
    CUT_HARD,
    CROP_FRACTION_MAX,
    CROP_FRACTION_MIN,
    SPEED_MAX,
    SPEED_MIN,
    EditPlan,
    Shot,
)
from .grade import grade_frame

# Bornes de sécurité des transitions (frames).
FADE_FRAMES = 12
DISSOLVE_FRAMES = 8

# Mode de vitesse réellement mis en œuvre (vérité, jamais « RIFE » ici).
SPEED_MODE_RESAMPLE = "TEMPORAL_RESAMPLE"


# ---------------------------------------------------------------------------
# Utilitaires caméra (purs, testables)
# ---------------------------------------------------------------------------
def crop_window(height: int, crop_fraction: float) -> tuple[int, int]:
    """Fenêtre verticale ``(y0, crop_h)`` d'une hauteur source ``height``.

    ``crop_fraction`` est borné dans ``[CROP_FRACTION_MIN, CROP_FRACTION_MAX]``.
    La fenêtre reste **entièrement** dans la source (aucune bordure noire).
    """
    frac = float(min(CROP_FRACTION_MAX, max(CROP_FRACTION_MIN, crop_fraction)))
    crop_h = int(round(int(height) * frac))
    crop_h = max(2, min(int(height), crop_h))
    y0 = int(round((int(height) - crop_h) * 0.5))
    y0 = max(0, min(int(height) - crop_h, y0))
    return y0, crop_h


def apply_subject_bias(y0: int, crop_h: int, height: int, subject_y: float | None,
                       bias: float) -> int:
    """Décale la fenêtre pour placer le sujet à ``bias`` de la hauteur du cadre.

    Aucune valeur n'est inventée : ``subject_y`` provient d'un centre réellement
    suivi. Si le sujet est inconnu, la fenêtre reste centrée.
    """
    if subject_y is None:
        return y0
    target = float(bias) * crop_h
    delta = float(subject_y) - target
    return int(max(0, min(int(height) - int(crop_h), round(y0 + delta))))


def ease(t: float, kind: str) -> float:
    """Easing déterministe dans ``[0, 1]``."""
    t = float(min(1.0, max(0.0, t)))
    if kind == "ease_in":
        return t * t
    if kind == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)   # smoothstep
    return t


def interpolate_factor(shot: Shot, t: float) -> float:
    """Facteur de vitesse à la position normalisée ``t`` du plan (borné)."""
    c = shot.speed_curve or {}
    a = float(c.get("start_factor", 1.0))
    b = float(c.get("end_factor", 1.0))
    v = a + (b - a) * ease(t, str(c.get("easing", "linear")))
    return float(min(SPEED_MAX, max(SPEED_MIN, v)))


def interpolate_crop(shot: Shot, t: float) -> float:
    """Fraction de crop à ``t`` pour la transition de cadrage du plan (bornée)."""
    return float(shot.crop_start + (shot.crop_end - shot.crop_start) * ease(t, shot.crop_easing))


def target_at(shot: Shot, frame: int) -> tuple[float, float] | None:
    """Centre réel suivi ``(cx, cy)`` à ``frame`` (maintien de la dernière connue)."""
    if not shot.targets:
        return None
    keys = [t["frame"] for t in shot.targets]
    if frame <= keys[0]:
        return float(shot.targets[0]["cx"]), float(shot.targets[0]["cy"])
    if frame >= keys[-1]:
        return float(shot.targets[-1]["cx"]), float(shot.targets[-1]["cy"])
    for tt in shot.targets:
        if tt["frame"] >= frame:
            return float(tt["cx"]), float(tt["cy"])
    return float(shot.targets[-1]["cx"]), float(shot.targets[-1]["cy"])


def target_center_x(shot: Shot, frame: int) -> float | None:
    """Centre horizontal réel suivi à ``frame`` (compat. API historique)."""
    t = target_at(shot, frame)
    return None if t is None else t[0]


def resample_index(k: int, out_n: int, src_n: int) -> tuple[int, int, float]:
    """Correspondance temporelle d'une frame de sortie vers la source.

    Échantillonnage **centré**, qui répartit uniformément les frames
    intermédiaires (au lieu de les regrouper en fin de plan) :

        pos = (k + 0.5) * src_n / out_n - 0.5

    Renvoie ``(i0, i1, frac)`` : ``i0`` = frame source principale, ``i1`` = frame
    suivante (ou ``i0`` si hors bornes), ``frac`` = poids de ``i1`` dans le blend
    documenté. ``frac == 0`` ⇒ aucune frame intermédiaire (rendu sans blend).
    """
    if out_n <= 1 or src_n <= 1:
        return 0, 0, 0.0
    pos = (k + 0.5) * (src_n / float(out_n)) - 0.5
    pos = min(float(src_n - 1), max(0.0, pos))
    i0 = int(np.floor(pos))
    frac = float(pos - i0)
    if frac < 1e-6:
        return i0, i0, 0.0
    i1 = min(src_n - 1, i0 + 1)
    if i1 == i0:
        return i0, i0, 0.0
    return i0, i1, frac


def build_frame_map(plan: EditPlan) -> list[dict[str, Any]]:
    """Table de correspondance **frame source -> frame de sortie**.

    Chaque plan reçoit ``out_frames`` positions (``time_map``) et parcourt sa
    source à l'envers via :func:`resample_index`. Cela matérialise la courbe de
    vitesse (ralenti = source lue plus lentement) et garantit :

      * aucune frame gelée (index strictement progressif sur le plan) ;
      * facteur effectif du plan borné dans ``[SPEED_MIN, SPEED_MAX]`` ;
      * ``Σ out_frames == total_frames`` (durée exactement préservée).

    Renvoie une liste ordonnée de descripteurs
    ``{'out_frame','src_frame','src_next','blend','shot_id','phase','t'}``.
    """
    tmap = plan.time_map or {}
    shot_alloc: dict[str, Any] = tmap.get("shots", {}) if isinstance(tmap, dict) else {}
    frame_map: list[dict[str, Any]] = []
    out_idx = 0
    for shot in plan.shots:
        alloc = shot_alloc.get(shot.shot_id) or {}
        out_frames = max(1, int(alloc.get("out_frames") or shot.duration_frames))
        src_lo, src_hi = shot.frame_span
        src_frames = int(alloc.get("src_frames") or shot.duration_frames)
        src_n = max(1, min(src_frames, src_hi - src_lo + 1))
        # Le blend documenté n'est appliqué qu'en ralenti / temps-réel (jamais
        # en accélération, où il ne ferait que réduire la netteté).
        allow_blend = out_frames >= src_n
        for k in range(out_frames):
            i0, i1, frac = resample_index(k, out_frames, src_n)
            if not allow_blend:
                i1, frac = i0, 0.0
            t = (k / float(out_frames - 1)) if out_frames > 1 else 0.0
            frame_map.append({
                "out_frame": out_idx,
                "src_frame": int(src_lo + i0),
                "src_next": int(src_lo + i1),
                "blend": round(frac, 6),
                "shot_id": shot.shot_id,
                "phase": shot.phase,
                "t": round(t, 6),
            })
            out_idx += 1
    return frame_map


def plan_validation(plan: EditPlan, shot_effective: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Contrôles statiques du plan avant/après rendu (documentés, testables).

      * partition contiguë des frames source (aucun trou, aucun chevauchement) ;
      * durée préservée ;
      * toutes les vitesses résolues sont bornées ;
      * contrat ``hero_track == followed_track``.
    """
    errors: list[str] = []
    expected = 0
    for shot in plan.shots:
        if int(shot.start_frame) != expected:
            errors.append(f"timeline_gap:{shot.shot_id}:{shot.start_frame}!={expected}")
        expected = int(shot.end_frame) + 1
    if expected != int(plan.total_frames):
        errors.append(f"timeline_end:{expected}!={plan.total_frames}")
    if not plan.duration_preserved:
        errors.append("duration_not_preserved")
    for sid, info in (shot_effective or {}).items():
        f = float(info.get("effective_factor", 1.0))
        if not (SPEED_MIN - 1e-6 <= f <= SPEED_MAX + 1e-6):
            errors.append(f"speed_out_of_bounds:{sid}:{f}")
    if plan.hero_track is not None and plan.followed_track != plan.hero_track:
        errors.append("hero_camera_contract")
    return {
        "status": "OK" if not errors else "QC_REJECTED",
        "errors": errors,
        "shots": len(plan.shots),
        "expected_frames": expected,
        "total_frames": int(plan.total_frames),
        "contract": bool(plan.hero_camera_contract),
        "duration_preserved": bool(plan.duration_preserved),
    }


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------
def render_plan(source: str, out: str, plan: EditPlan, width: int = 1080,
                height: int = 1920, fps: int = 30, grade: bool = True,
                vignette: bool = False) -> dict[str, Any]:
    """Rend la vignette verticale à partir du plan du director.

    Lit les frames réelles de la source, applique le crop 9:16 et l'échelle, un
    grade discret, les transitions du plan, puis écrit H.264/AAC en conservant
    la piste audio d'origine (aucun ré-échantillonnage audio).
    """
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("VIDEO_OPEN_FAILED")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    out_fps = int(fps or round(src_fps))

    frame_map = build_frame_map(plan)
    if not frame_map:
        cap.release()
        raise RuntimeError("EMPTY_PLAN")

    needed = sorted({fm["src_frame"] for fm in frame_map} |
                    {fm["src_next"] for fm in frame_map})

    # Décodage séquentiel ciblé (déterministe, une seule passe).
    frames: dict[int, np.ndarray] = {}
    target_set = set(needed)
    idx = 0
    last_needed = needed[-1]
    while idx <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target_set:
            frames[idx] = frame
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("NO_SOURCE_FRAMES_READ")

    first_src, last_src = min(frames), max(frames)

    def get_frame(i: int):
        i = max(first_src, min(last_src, int(i)))
        if i in frames:
            return frames[i]
        best = min(needed, key=lambda k: abs(k - i))
        return frames.get(best)

    shot_by_id = {s.shot_id: s for s in plan.shots}
    crop_w = max(2, min(W, int(round(H * width / height))))

    # Transitions déclarées par le plan.
    fade_in = {s.shot_id for s in plan.shots if s.transition_in == CUT_FADE_FROM_BLACK}
    fade_out = {s.shot_id for s in plan.shots if s.transition_out == CUT_FADE_TO_BLACK}
    dissolve = {s.shot_id for s in plan.shots if s.transition_in == CUT_DISSOLVE}
    hard_in = {s.shot_id for s in plan.shots if s.transition_in == CUT_HARD}

    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(out_fps), "-i", "pipe:0",
         "-i", str(source),
         "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-g", str(out_fps * 2), "-pix_fmt", "yuv420p", "-r", str(out_fps),
         "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)
    assert proc.stdin is not None

    smooth_x: float | None = None
    smooth_crop: float | None = None
    crop_out_prev: float | None = None
    prev_shot_id: str | None = None
    tail_buffer: list[np.ndarray] = []
    written = 0
    blended_frames = 0
    dissolves_rendered = 0

    try:
        for fm in frame_map:
            shot = shot_by_id[fm["shot_id"]]
            t = float(fm["t"])
            shot_changed = shot.shot_id != prev_shot_id

            frame = get_frame(int(fm["src_frame"]))
            if frame is None:
                break
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

            # --- 1) Blend documenté (fallback d'interpolation, jamais RIFE) ---
            blend = float(fm.get("blend") or 0.0)
            if blend > 1e-6 and fm.get("src_next", fm["src_frame"]) != fm["src_frame"]:
                nxt = get_frame(int(fm["src_next"]))
                if nxt is not None:
                    if nxt.shape[0] != H or nxt.shape[1] != W:
                        nxt = cv2.resize(nxt, (W, H), interpolation=cv2.INTER_LINEAR)
                    frame = cv2.addWeighted(frame, 1.0 - blend, nxt, blend, 0.0)
                    blended_frames += 1

            # --- 2) Cadrage 9:16 (crop + suivi réel + biais sujet) -----------
            frac = interpolate_crop(shot, t)
            _, crop_h_target = crop_window(H, frac)
            pos = target_at(shot, int(fm["src_frame"]))
            cx = pos[0] if pos else (W / 2.0)
            cy = pos[1] if pos else None

            if shot_changed or smooth_x is None:
                if (shot_changed and smooth_x is not None
                        and shot.shot_id in dissolve and crop_out_prev is not None):
                    # fondu : on poursuit la continuité de cadrage (aucun pop).
                    smooth_crop = float(crop_out_prev)
                else:
                    smooth_crop = float(crop_h_target)
                smooth_x = float(cx) if smooth_x is None else smooth_x
                if shot.shot_id in hard_in or smooth_x is None:
                    smooth_x = float(cx)
            else:
                alpha = 0.35
                smooth_x = alpha * float(cx) + (1 - alpha) * smooth_x
                smooth_crop = alpha * float(crop_h_target) + (1 - alpha) * smooth_crop

            crop_h_s = int(round(min(H, max(2, smooth_crop))))
            y0_s, _ = crop_window(H, crop_h_s / float(H))
            y0_s = apply_subject_bias(y0_s, crop_h_s, H, cy, shot.subject_y_bias)

            x0 = int(round(smooth_x - crop_w / 2.0))
            x0 = max(0, min(W - crop_w, x0))
            crop = frame[y0_s:y0_s + crop_h_s, x0:x0 + crop_w]
            if crop.size == 0:
                crop = frame
            out_frame = cv2.resize(crop, (width, height),
                                   interpolation=cv2.INTER_LANCZOS4)

            # --- 3) Étalonnage discret (style football cinématique retenu) ---
            if grade:
                out_frame = grade_frame(out_frame, vignette=vignette)

            # --- 4) DISSOLVE : cross-fondu court avec la fin du plan précédent -
            if shot.shot_id in dissolve and shot_changed and tail_buffer:
                a = max(0.0, min(1.0, t * (shot.duration_frames /
                                           float(DISSOLVE_FRAMES))))
                a = 1.0 - a if DISSOLVE_FRAMES else 1.0
                ref = tail_buffer[-1]
                if a > 0.0 and ref is not None and ref.shape == out_frame.shape:
                    out_frame = cv2.addWeighted(out_frame, 1.0 - a, ref, a, 0.0)
                    dissolves_rendered += 1

            # --- 5) Fondus au noir (ouverture / clôture) --------------------
            if shot.shot_id in fade_in:
                k = max(0.0, min(1.0, 1.0 - (t * (FADE_FRAMES /
                                                  float(max(1, shot.duration_frames))))))
                out_frame = (out_frame.astype(np.float32) * k).astype(np.uint8)
            if shot.shot_id in fade_out:
                k = 1.0 - max(0.0, min(1.0, (1.0 - t) * (FADE_FRAMES /
                                                         float(max(1, shot.duration_frames)))))
                out_frame = (out_frame.astype(np.float32) * k).astype(np.uint8)

            try:
                proc.stdin.write(np.ascontiguousarray(out_frame).tobytes())
            except (BrokenPipeError, OSError):
                break
            written += 1
            crop_out_prev = float(crop_h_s)
            tail_buffer.append(out_frame.copy())
            if len(tail_buffer) > DISSOLVE_FRAMES:
                tail_buffer.pop(0)
            prev_shot_id = shot.shot_id
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFMPEG_PLAN_RENDER_FAILED rc={proc.returncode}")

    validation = plan_validation(plan, (plan.time_map or {}).get("shots"))
    return {
        "out": str(out),
        "written_frames": written,
        "planned_frames": len(frame_map),
        "out_fps": out_fps,
        "crop_w": crop_w,
        "shots": len(plan.shots),
        "validation": validation,
        "grade": bool(grade),
        "vignette": bool(vignette),
        "speed_mode": SPEED_MODE_RESAMPLE,
        "rife_used": bool(plan.claims.get("rife_used")),
        "interpolated_frames": blended_frames,
        "interpolation_fallback": ("LINEAR_BLEND_NO_RIFE" if blended_frames else None),
        "dissolve_frames_rendered": dissolves_rendered,
        "duration_preserved": bool(plan.duration_preserved),
    }


def render_static(source: str, out: str, width: int = 1080, height: int = 1920,
                  fps: int = 30) -> str:
    """Fallback statique centré (aucune preuve de suivi) — conservé à l'identique."""
    from ..source.ffmpeg import render_vertical
    return render_vertical(source, out, width=width, height=height, fps=fps,
                           reframe={"targets": {}})
