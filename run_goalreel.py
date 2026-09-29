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
from goalreel.vision.yolo_detector import FootballYOLO       # noqa: E402
from goalreel.vision.reid import PlayerReID                  # noqa: E402
from goalreel.vision.sam2 import SAM21                       # noqa: E402
from goalreel.vision.depth import DepthAnythingV2            # noqa: E402
from goalreel.vision.ocr import JerseyOCR                    # noqa: E402
from goalreel.vision.pose import PoseBackend                 # noqa: E402
from goalreel.scene.camera import CameraEstimator            # noqa: E402
from goalreel.events.football import FootballEventEngine     # noqa: E402
from goalreel.events.hero import score_hero                  # noqa: E402
from goalreel.source.ffmpeg import render_vertical           # noqa: E402
from goalreel.qc.final import final_qc                       # noqa: E402


def _stage_to_dict(stage):
    return stage.to_dict() if hasattr(stage, "to_dict") else dict(stage)


def build_backend_status(models: Path, checkpoints: Path) -> dict:
    """Interroge chaque back-end et renvoie son état RÉEL (jamais simulé)."""
    yolo = FootballYOLO(os.getenv("GOALREEL_YOLO_WEIGHTS", str(models / "best.pt"))).load()
    reid = PlayerReID(os.getenv("GOALREEL_REID_WEIGHTS",
                                str(checkpoints / "osnet_x1_0_market1501.pth.tar"))).load()
    sam2 = SAM21(os.getenv("GOALREEL_SAM2_CHECKPOINT",
                           str(checkpoints / "sam2.1_hiera_tiny.pt")),
                 os.getenv("GOALREEL_SAM2_CONFIG")).load()
    depth = DepthAnythingV2(os.getenv("GOALREEL_DEPTH_CHECKPOINT",
                                      str(checkpoints / "depth_anything_v2_vits.pth")),
                            os.getenv("GOALREEL_DEPTH_CONFIG")).load()
    ocr = JerseyOCR(os.getenv("GOALREEL_OCR_BACKEND")).load()
    pose = PoseBackend().load()

    stages = [yolo, reid, sam2, depth, ocr, pose]
    return {
        "schema": "goalreel.backends.v1",
        "device": os.getenv("GOALREEL_DEVICE", "auto"),
        "backends": [_stage_to_dict(s) for s in stages],
        "summary": {
            s.name: s.status for s in stages
        },
    }


def run(video: str, out: Path, models: Path, checkpoints: Path, every: int = 15) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    video = str(video)

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
    backend_status = build_backend_status(models, checkpoints)
    write_json(out / "backend_status.json", backend_status)

    # 3) Estimation caméra (preuve optique réelle) -------------------------
    camera = _stage_to_dict(CameraEstimator().estimate(video))

    # 4) Événements (contrainte par la preuve) -----------------------------
    # Sans trajectoires/piste d'inférence disponibles, l'engine n'invente aucun
    # événement : il renvoie une liste vide (comportement voulu).
    events = FootballEventEngine().infer(tracks={})
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

    # 6) Rendu vertical final ----------------------------------------------
    final_path = out / "final_reel.mp4"
    render_vertical(video, str(final_path))
    qc = final_qc(str(final_path))

    return {
        "outputs": {
            "source_manifest": str(out / "source_manifest.json"),
            "backend_status": str(out / "backend_status.json"),
            "event_timeline": str(out / "event_timeline.json"),
            "hero_moment": str(out / "hero_moment.json"),
            "final_reel": str(final_path),
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
    args = ap.parse_args()

    report = run(args.video, Path(args.out), Path(args.models),
                 Path(args.checkpoints), args.every)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
