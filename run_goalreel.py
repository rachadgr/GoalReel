#!/usr/bin/env python3
# =============================================================================
# GoalReel — Virtual Cinema Engine
# Orchestrateur racine : exécute le pipeline complet et écrit les sorties
# dans outputs/.
#
#   python run_goalreel.py --video assets/SOURCE_MASTER_1000006866.mp4
#   python run_goalreel.py --video chemin.mp4 --out outputs --models models \
#                          --checkpoints checkpoints
#
# Fichiers produits dans outputs/ :
#   source_manifest.json   métadonnées de la source (ffprobe + analyse)
#   backend_status.json    état réel de chaque modèle (OK / MODEL_UNAVAILABLE)
#   event_timeline.json    chronologie d'événements (preuve contrainte)
#   hero_moment.json       moment "héro" sélectionné
#   final_reel.mp4         rendu vertical final 1080x1920
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Rendre le package importable depuis la racine du dépôt.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from goalreel.core.ffprobe import probe                      # noqa: E402
from goalreel.core.io import write_json                      # noqa: E402
from goalreel.source.analysis import analyze_video           # noqa: E402
from goalreel.scene.camera import CameraEstimator            # noqa: E402
from goalreel.events.football import FootballEventEngine     # noqa: E402
from goalreel.events.hero import score_hero, track_stats     # noqa: E402
from goalreel.source.ffmpeg import render_vertical           # noqa: E402
from goalreel.qc.final import final_qc                       # noqa: E402
from goalreel.models.manager import ModelManager             # noqa: E402
from goalreel.services.analysis_pipeline import AnalysisPipeline  # noqa: E402
from goalreel.tracking.continuity import analyze_continuity  # noqa: E402

# Bornes de coût pour les embeddings OSNet de continuité (Subject Continuity V2.1).
CONTINUITY_MAX_TRACKS = 80      # nombre max de fragments embeddés
CONTINUITY_CROPS_PER_TRACK = 3  # crops échantillonnés par fragment
CONTINUITY_MAX_TOTAL = 320      # plafond dur d'inférences Re-ID


def _track_embeddings(manager: ModelManager, video: str, tracks: dict) -> tuple[dict, str]:
    """Embeddings OSNet réels par fragment (moyennés, L2-normalisés).

    Renvoie ``(embeddings, truth)`` où ``truth`` ∈ {``osnet``, ``MODEL_UNAVAILABLE``,
    ``NO_TRACKS``, ``NO_EMBEDDINGS``}. Aucun embedding n'est simulé : si OSNet
    est indisponible, on renvoie ``{}`` et la continuité continue avec les
    autres preuves mesurées (échec en douceur).
    """
    if not tracks:
        return {}, "NO_TRACKS"
    reid = manager.load("reid")
    if reid is None:
        return {}, "MODEL_UNAVAILABLE"

    import cv2
    import numpy as np

    # Fragments triés par persistance décroissante (déterministe), plafonnés.
    order = sorted(tracks, key=lambda tid: (-len(tracks[tid]), int(tid)))
    order = order[:CONTINUITY_MAX_TRACKS]

    needed: dict[int, list] = {}
    for tid in order:
        items = [it for it in tracks[tid] if isinstance(it, dict) and it.get("bbox")]
        if not items:
            continue
        n = len(items)
        idxs = sorted({0, n // 2, n - 1})
        idxs = idxs[:CONTINUITY_CROPS_PER_TRACK]
        for i in idxs:
            it = items[i]
            fr = int(it.get("frame", i))
            needed.setdefault(fr, []).append((int(tid), it["bbox"]))

    # Décodage séquentiel unique ; collecte des embeddings disponibles.
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}, "NO_EMBEDDINGS"
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    total = 0
    idx = 0
    try:
        while total < CONTINUITY_MAX_TOTAL:
            ok, frame = cap.read()
            if not ok:
                break
            todo = needed.get(idx)
            if todo:
                for tid, bbox in todo:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    with manager.time_it("reid"):
                        emb = reid.embed(crop)
                    if emb is None:
                        continue
                    sums[tid] = (sums[tid] + emb) if tid in sums else emb.copy()
                    counts[tid] = counts.get(tid, 0) + 1
                    total += 1
                    if total >= CONTINUITY_MAX_TOTAL:
                        break
            idx += 1
    finally:
        cap.release()

    embeddings: dict[int, np.ndarray] = {}
    for tid, s in sums.items():
        v = s / float(counts.get(tid, 1))
        n = float(np.linalg.norm(v))
        if n > 0:
            embeddings[tid] = (v / n).astype("float32")
    return embeddings, ("osnet" if embeddings else "NO_EMBEDDINGS")



def _stage_to_dict(stage):
    return stage.to_dict() if hasattr(stage, "to_dict") else dict(stage)


def build_backend_status(models: Path, checkpoints: Path,
                         manager: ModelManager | None = None) -> dict:
    """Interroge chaque back-end via le ModelManager et renvoie son état RÉEL.

    Le ModelManager gère le chargement paresseux, le cache, le choix du
    périphérique et les états. La sortie reste compatible ``backends.v1``.
    """
    manager = manager or ModelManager()
    # Chargement best-effort de chaque modèle (aucune exception ne remonte).
    # ``try_stage`` classe honnêtement l'indisponibilité (truth).
    for name in manager.settings.models:
        manager.try_stage(name)
    statuses = manager.statuses()
    backends = [
        {
            "name": name,
            "stage": st["stage"],
            "status": st["status"],
            "truth": st.get("truth"),
            "device": st["device"],
            "path": st["path"],
            "message": st["message"],
            "metrics": {"avg_infer_ms": st["avg_infer_ms"], "load_time_s": st["load_time_s"],
                        "margin": st.get("meta", {})},
        }
        for name, st in statuses.items()
    ]
    return {
        "schema": "goalreel.backends.v2",
        "device": manager.health()["device"],
        "cuda": manager.health()["cuda"],
        "torch": manager.health()["torch"],
        "backends": backends,
        "summary": {name: st["status"] for name, st in statuses.items()},
        "truth_summary": {name: st.get("truth") for name, st in statuses.items()},
    }


def run(video: str, out: Path, models: Path, checkpoints: Path, every: int = 15,
        manager: ModelManager | None = None, analyze: bool = True,
        max_frames: int = 0, stages: list[str] | None = None,
        continuity_embeddings: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    video = str(video)
    manager = manager or ModelManager()

    # 1) Sonde + analyse de la source --------------------------------------
    info = probe(video)
    analysis = analyze_video(video, out / "_frames", every)

    source_manifest = {
        "schema": "goalreel.source_manifest.v1",
        "core_invariant": "THE EVENT IS IMMUTABLE. ONLY THE CAMERA VIEWPOINT MAY CHANGE.",
        "video": info.to_dict(),
        "analysis": {k: v for k, v in analysis.items() if k != "sampled_frames"},
        "sampled_frame_count": len(analysis.get("sampled_frames", [])),
    }
    write_json(out / "source_manifest.json", source_manifest)

    # 2) État réel des back-ends -------------------------------------------
    backend_status = build_backend_status(models, checkpoints, manager)
    write_json(out / "backend_status.json", backend_status)

    # 2a) Vérité du runtime RÉEL des modèles (Phase 10) --------------------
    # Chargement authentique + inférence réelle, classés honnêtement
    # (CODE_INTEGRATED / CHECKPOINT_PRESENT / MODEL_LOADS / INFERENCE_WORKS /
    #  GPU_VALIDATED / END_TO_END_VALIDATED). Aucune fausse revendication.
    from goalreel.models.verify import verify_all
    try:
        model_truth = verify_all(video, manager=manager,
                                 every=max(1, min(every, 30)))
    except Exception as exc:  # la vérification ne doit jamais casser le pipeline
        model_truth = {"schema": "goalreel.model_truth.v1", "error": str(exc)}
    write_json(out / "model_truth.json", model_truth)

    # 2b) Analyse réelle par modèles (détection, suivi, reid, depth, ...) ---
    analysis_report = None
    if analyze:
        pipeline = AnalysisPipeline(manager)
        analysis_report = pipeline.run(video, every=every, max_frames=max_frames,
                                       stages=stages)
        write_json(out / "analysis_report.json", {
            "schema": "goalreel.analysis.v1",
            "stages": analysis_report["stages"],
            "model_status": manager.statuses(),
        })

    # 3) Estimation caméra (preuve optique réelle) -------------------------
    camera = _stage_to_dict(CameraEstimator().estimate(video))

    # 4) Événements (contrainte par la preuve) -----------------------------
    # Les trajectoires issues du suivi réel alimentent l'engine. En leur
    # absence, aucun événement n'est inventé (comportement voulu).
    tracks = {}
    records = (analysis_report or {}).get("records", {}) or {}
    if records.get("tracks"):
        by_id: dict[int, list] = {}
        for frame_idx, tracked in records["tracks"].items():
            for t in tracked:
                by_id.setdefault(t["track_id"], []).append({"frame": frame_idx, **t})
        tracks = by_id

    # Preuve ballon RÉELLE (classe COCO ``sports ball``) : utilisée pour la
    # proximité ballon du classement héros, uniquement quand elle existe.
    # Aucune position de ballon n'est inventée. Itération triée => déterminisme.
    ball_detections: list[dict] = []
    dets = records.get("detections") or {}
    for frame_idx in sorted(dets, key=lambda k: int(k)):
        for d in dets[frame_idx]:
            if d.get("class_name") == "sports ball" and d.get("bbox") is not None:
                ball_detections.append({
                    "frame": int(frame_idx),
                    "bbox": [float(v) for v in d["bbox"]],
                    "confidence": float(d.get("confidence", 0.0)),
                })

    events = FootballEventEngine().infer(tracks=tracks)
    event_timeline = {
        "schema": "goalreel.event_timeline.v1",
        "camera_estimation": camera,
        "events": events,
        "count": len(events),
        "policy": "NO_EVENT_WITHOUT_EVIDENCE",
        "evidence_summary": {
            "tracked_subjects": len(tracks),
            "detected_persons": sum(
                1 for items in dets.values() for d in items
                if d.get("class_name") == "person") if dets else 0,
            "detected_balls": len(ball_detections),
            "ball_class": "sports ball (COCO)",
            "note": ("YOLOv8s is generic COCO: 'person' and 'sports ball' are real "
                     "detections; football events are inferred from tracked motion. "
                     "'referee'/'goal' classes do not exist and are never invented."),
        },
    }
    write_json(out / "event_timeline.json", event_timeline)

    # 5) Moment héro (intelligence multi-preuves V2) ------------------------
    # Le héros est choisi à partir d'un score explicable combinant des preuves
    # RÉELLES : amplitude + persistance + continuité des trajectoires suivies,
    # proximité temporelle au pic d'action réel, pertinence spatiale, et
    # proximité au ballon UNIQUEMENT si une détection de ballon réelle existe.
    # Aucun événement/ballon/identité n'est inventé. Le ``track_id`` retenu
    # pilote ensuite la caméra 9:16 afin que celle-ci suive RÉELLEMENT le sujet
    # du moment héros (hero_track == followed_track).
    tstats = track_stats(tracks)
    hero_moment = {
        "schema": "goalreel.hero_moment.v1",
        **score_hero(events, track_stats=tstats, tracks=tracks,
                     ball_detections=ball_detections,
                     width=info.width, height=info.height),
    }
    write_json(out / "hero_moment.json", hero_moment)

    # 5a) SUBJECT CONTINUITY V2.1 -----------------------------------------
    # Relie les fragments de suivi bruts d'un MÊME joueur en une identité
    # canonique lorsque des preuves mesurées suffisent (trou borné + continuité
    # spatiale/vitesse/géométrie + apparence OSNet si disponible). Ne fabrique
    # aucune identité, ne remplace aucun traqueur. Le résultat
    # (``track_continuity.json``) alimente le Director V2 et le reframe.
    continuity_embeddings_map: dict = {}
    embeddings_truth = "DISABLED"
    if continuity_embeddings and tracks:
        try:
            continuity_embeddings_map, embeddings_truth = _track_embeddings(
                manager, video, tracks)
        except Exception as exc:  # la continuité ne doit jamais casser le pipeline
            continuity_embeddings_map, embeddings_truth = {}, f"ERROR:{type(exc).__name__}"
    track_continuity = analyze_continuity(
        tracks, embeddings=continuity_embeddings_map,
        ball_detections=ball_detections, hero_track=hero_moment.get("track_id"),
        width=info.width, height=info.height)
    track_continuity["appearance"]["source"] = (
        "osnet" if embeddings_truth == "osnet" else None)
    track_continuity["appearance"]["pipeline_truth"] = embeddings_truth
    write_json(out / "track_continuity.json", track_continuity)

    # 5b) Novel-view / caméra virtuelle (état réel du backend) -------------
    from goalreel.generation.novel_view.planner import NovelViewPlanner
    novel_view_status = NovelViewPlanner().status()
    write_json(out / "novel_view_status.json", {
        "schema": "goalreel.novel_view.v1",
        **novel_view_status,
    })

    # 5c) Reframe 9:16 piloté par la PREUVE de suivi + timing cinématique ----
    # Si de vraies trajectoires existent, la caméra 9:16 SUIT réellement le
    # sujet suivi (pan horizontal) et sa ligne temporelle est structurée en
    # phases autour de l'événement héros (anticipation → hero_moment → réaction
    # → célébration). Le sujet suivi est celui du MOMENT HÉROS lorsqu'il est
    # disponible (cohérence événement <-> caméra) ; sinon le plus persistant
    # (fallback documenté). Sinon => fallback statique (jamais un faux suivi).
    # Aucune coordonnée inventée : centres issus du suivi ByteTrack, les frames
    # sans suivi maintiennent la dernière position réelle connue.
    from goalreel.cinematic.reframe import build_reframe_targets
    reframe_targets, reframe_meta = build_reframe_targets(
        tracks, preferred_track=hero_moment.get("track_id"),
        hero_event=hero_moment.get("event"), width=info.width,
        identity=track_continuity.get("hero_identity"))
    write_json(out / "camera_reframe.json", {
        "schema": "goalreel.camera_reframe.v1",
        **reframe_meta,
        "hero_track": hero_moment.get("track_id"),
        "targets_count": len(reframe_targets),
    })

    # 5d) CINEMATIC DIRECTOR V2 ------------------------------------------
    # Couche de décision déterministe : transforme la preuve réelle (suivi,
    # événements, héros multi-preuves, ballon COCO, coupes/mouvement réels
    # mesurés sur la source, identité canonique de continuité) en un PLAN DE
    # MONTAGE explicite (phases, cadrages, vitesses, transitions). Le plan est
    # enregistré (``cinematic_edit_plan.json``) puis exécuté par le renderer
    # plan-aware. Aucun événement n'est inventé : les phases non soutenues par
    # la preuve sont omises et listées avec leur raison.
    from goalreel.cinematic.director import build_edit_plan
    from goalreel.cinematic.evidence_model import scan_scene
    from goalreel.cinematic.render_plan import render_plan, plan_validation

    scene = scan_scene(video)
    interp_state = manager.state("interpolation")
    rife_available = bool(getattr(interp_state, "loaded", False))
    edit_plan = build_edit_plan(
        video, tracks=tracks, hero_moment=hero_moment, events=events,
        ball_detections=ball_detections,
        camera_transforms=(camera.get("metrics", {}) or {}).get("transforms"),
        width=info.width, height=info.height, fps=info.fps,
        total_frames=info.frames, scene=scene, rife_available=rife_available,
        continuity=track_continuity)
    write_json(out / "cinematic_edit_plan.json", edit_plan.to_dict())
    director_validation = plan_validation(edit_plan,
                                          (edit_plan.time_map or {}).get("shots"))

    # 6) Rendu vertical final ----------------------------------------------
    # Le renderer plan-aware exécute le plan du director. Si le plan ne
    # contient aucun plan exploitable, on retombe honnêtement sur le reframe
    # de suivi historique (jamais un rendu fabriqué).
    final_path = out / "final_reel.mp4"
    render_info = None
    if edit_plan.shots:
        render_info = render_plan(video, str(final_path), edit_plan)
    else:
        render_vertical(video, str(final_path),
                        reframe={"targets": reframe_targets, **reframe_meta})
    qc = final_qc(str(final_path))
    write_json(out / "final_qc.json", {
        "schema": "goalreel.final_qc.v1",
        **qc,
        "director": director_validation,
        "render": render_info,
        "continuity": {
            "schema": track_continuity.get("schema"),
            "raw_tracks": track_continuity.get("stats", {}).get("raw_tracks"),
            "identity_count": track_continuity.get("stats", {}).get("identity_count"),
            "links_accepted": track_continuity.get("stats", {}).get("links_accepted"),
            "hero_identity": {
                "canonical_track_id": track_continuity.get("hero_identity", {}).get("canonical_track_id"),
                "source_track_ids": track_continuity.get("hero_identity", {}).get("source_track_ids"),
                "confidence": track_continuity.get("hero_identity", {}).get("confidence"),
                "fallback": track_continuity.get("hero_identity", {}).get("fallback"),
            },
            "camera_follow_mode": edit_plan.camera_follow_mode,
        },
        "reframe": {**reframe_meta, "hero_track": hero_moment.get("track_id"),
                    "targets_count": len(reframe_targets)},
    })

    outputs = {
        "source_manifest": str(out / "source_manifest.json"),
        "backend_status": str(out / "backend_status.json"),
        "model_truth": str(out / "model_truth.json"),
        "event_timeline": str(out / "event_timeline.json"),
        "hero_moment": str(out / "hero_moment.json"),
        "track_continuity": str(out / "track_continuity.json"),
        "novel_view_status": str(out / "novel_view_status.json"),
        "camera_reframe": str(out / "camera_reframe.json"),
        "cinematic_edit_plan": str(out / "cinematic_edit_plan.json"),
        "final_qc": str(out / "final_qc.json"),
        "final_reel": str(final_path),
    }
    if analysis_report is not None:
        outputs["analysis_report"] = str(out / "analysis_report.json")
    return {
        "outputs": outputs,
        "backend_summary": backend_status["summary"],
        "model_truth_summary": model_truth.get("summary"),
        "continuity": {
            "raw_tracks": track_continuity.get("stats", {}).get("raw_tracks"),
            "identity_count": track_continuity.get("stats", {}).get("identity_count"),
            "links_accepted": track_continuity.get("stats", {}).get("links_accepted"),
            "hero_identity": track_continuity.get("hero_identity", {}),
            "appearance": track_continuity.get("appearance", {}),
        },
        "cinematic_edit_plan": {
            "shots": edit_plan.shot_count,
            "phases_present": edit_plan.phases_present,
            "hero_track": edit_plan.hero_track,
            "followed_track": edit_plan.followed_track,
            "hero_camera_contract": edit_plan.hero_camera_contract,
            "hero_lineage": edit_plan.hero_lineage,
            "camera_follow_mode": edit_plan.camera_follow_mode,
            "duration_preserved": edit_plan.duration_preserved,
            "director_validation": director_validation,
        },
        "final_qc": qc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="run_goalreel",
                                 description="GoalReel — orchestre le pipeline complet")
    ap.add_argument("--video", required=True, help="vidéo source (football)")
    ap.add_argument("--out", default="outputs", help="dossier de sortie (défaut: outputs)")
    ap.add_argument("--models", default="models", help="dossier des modèles (défaut: models)")
    ap.add_argument("--checkpoints", default="checkpoints",
                    help="dossier des checkpoints (défaut: checkpoints)")
    ap.add_argument("--every", type=int, default=15, help="échantillonnage 1 frame sur N")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="plafond de frames analysées (0 = illimité)")
    ap.add_argument("--stages", default=None,
                    help="étapes à exécuter (ex: detection,tracking,reid,depth)")
    ap.add_argument("--no-analyze", action="store_true",
                    help="désactiver l'analyse par modèles (manifest + rendu seulement)")
    ap.add_argument("--no-continuity-embeddings", action="store_true",
                    help="désactiver les embeddings OSNet pour la continuité de sujet")
    args = ap.parse_args()
    stages = args.stages.split(",") if args.stages else None

    report = run(args.video, Path(args.out), Path(args.models),
                 Path(args.checkpoints), args.every, analyze=not args.no_analyze,
                 max_frames=args.max_frames, stages=stages,
                 continuity_embeddings=not args.no_continuity_embeddings)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
