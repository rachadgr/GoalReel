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
from goalreel.events.hero import score_hero                  # noqa: E402
from goalreel.source.ffmpeg import render_vertical           # noqa: E402
from goalreel.qc.final import final_qc                       # noqa: E402
from goalreel.models.manager import ModelManager             # noqa: E402
from goalreel.services.analysis_pipeline import AnalysisPipeline  # noqa: E402


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
        max_frames: int = 0, stages: list[str] | None = None) -> dict:
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
    if analysis_report and analysis_report.get("records", {}).get("tracks"):
        by_id: dict[int, list] = {}
        for frame_idx, tracked in analysis_report["records"]["tracks"].items():
            for t in tracked:
                by_id.setdefault(t["track_id"], []).append({"frame": frame_idx, **t})
        tracks = by_id
    events = FootballEventEngine().infer(tracks=tracks)
    event_timeline = {
        "schema": "goalreel.event_timeline.v1",
        "camera_estimation": camera,
        "events": events,
        "count": len(events),
        "policy": "NO_EVENT_WITHOUT_EVIDENCE",
    }
    write_json(out / "event_timeline.json", event_timeline)

    # 5) Moment héro --------------------------------------------------------
    hero_moment = {
        "schema": "goalreel.hero_moment.v1",
        **score_hero(events),
    }
    write_json(out / "hero_moment.json", hero_moment)

    # 5b) Novel-view / caméra virtuelle (état réel du backend) -------------
    from goalreel.generation.novel_view.planner import NovelViewPlanner
    novel_view_status = NovelViewPlanner().status()
    write_json(out / "novel_view_status.json", {
        "schema": "goalreel.novel_view.v1",
        **novel_view_status,
    })

    # 5c) Reframe 9:16 piloté par la PREUVE de suivi -----------------------
    # Si de vraies trajectoires existent, la caméra 9:16 SUIT réellement le
    # sujet suivi (pan horizontal). Sinon => fallback statique (jamais un faux
    # suivi). On ne réinvente rien : coordonnées issues du suivi ByteTrack.
    from goalreel.cinematic.reframe import build_reframe_targets
    reframe_targets, reframe_meta = build_reframe_targets(tracks)
    write_json(out / "camera_reframe.json", {
        "schema": "goalreel.camera_reframe.v1",
        **reframe_meta,
        "targets_count": len(reframe_targets),
    })

    # 6) Rendu vertical final ----------------------------------------------
    final_path = out / "final_reel.mp4"
    render_vertical(video, str(final_path),
                    reframe={"targets": reframe_targets, **reframe_meta})
    qc = final_qc(str(final_path))
    write_json(out / "final_qc.json", {
        "schema": "goalreel.final_qc.v1",
        **qc,
        "reframe": {**reframe_meta, "targets_count": len(reframe_targets)},
    })

    outputs = {
        "source_manifest": str(out / "source_manifest.json"),
        "backend_status": str(out / "backend_status.json"),
        "event_timeline": str(out / "event_timeline.json"),
        "hero_moment": str(out / "hero_moment.json"),
        "novel_view_status": str(out / "novel_view_status.json"),
        "camera_reframe": str(out / "camera_reframe.json"),
        "final_qc": str(out / "final_qc.json"),
        "final_reel": str(final_path),
    }
    if analysis_report is not None:
        outputs["analysis_report"] = str(out / "analysis_report.json")
    return {
        "outputs": outputs,
        "backend_summary": backend_status["summary"],
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
    args = ap.parse_args()
    stages = args.stages.split(",") if args.stages else None

    report = run(args.video, Path(args.out), Path(args.models),
                 Path(args.checkpoints), args.every, analyze=not args.no_analyze,
                 max_frames=args.max_frames, stages=stages)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
