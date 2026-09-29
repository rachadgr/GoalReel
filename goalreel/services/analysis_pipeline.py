"""Pipeline d'analyse par étapes, configurable et robuste.

Enchaîne les étapes du pipeline GoalReel en utilisant réellement les modèles
via la couche ``services.ai``. Optimisations :
  * échantillonnage de frames (``frame_stride`` / ``every``) ;
  * suivi par ByteTrack au lieu d'une re-détection pour chaque frame logique ;
  * plafond de frames analysées (``max_frames``) ;
  * exécution *lazy* des modèles (un seul chargement).

Robustesse : une étape qui échoue (ou dont le modèle est indisponible) est
consignée comme ``MODEL_UNAVAILABLE``/``ERROR`` sans interrompre les autres.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from ..core.types import StageResult
from ..models.manager import ModelManager
from .ai import (
    DepthService,
    DetectorService,
    InterpolationService,
    PoseService,
    ReIDService,
    SegmenterService,
    TrackerService,
)

DEFAULT_STAGES = ("detection", "tracking", "reid", "depth", "segmentation", "pose")


class AnalysisPipeline:
    def __init__(self, manager: ModelManager):
        self.manager = manager
        self.detector = DetectorService(manager)
        self.depth = DepthService(manager)
        self.reid = ReIDService(manager)
        self.segmenter = SegmenterService(manager)
        self.pose = PoseService(manager)
        self.interp = InterpolationService(manager)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _crop(frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        return frame[y1:y2, x1:x2]

    def _sample_frames(self, video: str, every: int, max_frames: int):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError("VIDEO_OPEN_FAILED")
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                yield idx, frame
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            idx += 1
        cap.release()

    # -- étapes -----------------------------------------------------------
    def _stage_detection(self, video, every, max_frames, records) -> StageResult:
        if not self.detector.available:
            return StageResult("detection", "MODEL_UNAVAILABLE",
                               "Detector unavailable", metrics={"required": self.manager.state("detector").path})
        total = persons = balls = 0
        for idx, frame in self._sample_frames(video, every, max_frames):
            dets = self.detector.detect(frame)
            records.setdefault("detections", {})[idx] = dets
            total += len(dets)
            persons += sum(1 for d in dets if d["class_name"] == "person")
            balls += sum(1 for d in dets if d["class_name"] == "sports ball")
        return StageResult("detection", "OK", "Detected players/objects (sampled)",
                           evidence="VISIBLE_FROM_SOURCE",
                           metrics={"sampled_frames": len(records.get("detections", {})),
                                    "objects": total, "persons": persons, "balls": balls})

    def _stage_tracking(self, video, every, max_frames, records) -> StageResult:
        detections = records.get("detections")
        if not detections:
            return StageResult("tracking", "UNKNOWN", "No detections to track")
        tracker = TrackerService()
        per_frame = {}
        for idx in sorted(detections):
            per_frame[idx] = tracker.update(detections[idx])
        records["tracks"] = per_frame
        ids = {t["track_id"] for tracks in per_frame.values() for t in tracks}
        return StageResult("tracking", "OK", "ByteTrack association",
                           evidence="TEMPORALLY_INFERRED",
                           metrics={"unique_tracks": len(ids),
                                    "frames": len(per_frame)})

    def _stage_reid(self, video, every, max_frames, records) -> StageResult:
        if not self.reid.available:
            return StageResult("reid", "MODEL_UNAVAILABLE", "Re-ID unavailable")
        detections = records.get("detections") or {}
        embedded, identities = 0, {}
        # Échantillonne quelques crops de joueurs (perf.)
        for idx, frame in self._sample_frames(video, every, min(max_frames or 3, 3)):
            for d in detections.get(idx, []):
                if d["class_name"] != "person":
                    continue
                emb = self.reid.embed(self._crop(frame, d["bbox"]))
                if emb is None:
                    continue
                embedded += 1
                # regroupement naïf glouton (1ère passe) : identité la + proche
                res = self.reid.match(emb, identities, threshold=0.6)
                if res["identity"] is None:
                    identities[len(identities)] = emb
                else:
                    identities[res["identity"]] = emb
        records["identities"] = len(identities)
        return StageResult("reid", "OK", "Player embeddings + greedy identity clustering",
                           evidence="TEMPORALLY_INFERRED",
                           metrics={"embeddings": embedded, "identities": len(identities)})

    def _stage_depth(self, video, every, max_frames, records) -> StageResult:
        if not self.depth.available:
            return StageResult("depth", "MODEL_UNAVAILABLE", "Depth unavailable")
        import numpy as np

        stats = []
        for idx, frame in self._sample_frames(video, every, min(max_frames or 2, 2)):
            d = self.depth.infer(frame)
            if d is not None:
                stats.append({"frame": idx, "min": float(d.min()), "max": float(d.max()),
                              "mean": float(d.mean())})
        records["depth"] = stats
        if not stats:
            return StageResult("depth", "ERROR", "Depth produced no output")
        return StageResult("depth", "OK", "Depth maps computed (sampled)",
                           evidence="DEPTH_INFERRED", metrics={"samples": len(stats),
                                                              "first": stats[0]})

    def _stage_segmentation(self, video, every, max_frames, records) -> StageResult:
        if not self.segmenter.available:
            return StageResult("segmentation", "MODEL_UNAVAILABLE", "SAM2 unavailable")
        detections = records.get("detections") or {}
        segs = []
        for idx, frame in self._sample_frames(video, every, min(max_frames or 2, 2)):
            persons = [d for d in detections.get(idx, []) if d["class_name"] == "person"]
            if not persons:
                continue
            m = self.segmenter.segment_bbox(frame, persons[0]["bbox"])
            if m is not None:
                segs.append({"frame": idx, "score": m["score"]})
        records["segmentation"] = segs
        if not segs:
            return StageResult("segmentation", "UNKNOWN", "No person to segment in samples")
        return StageResult("segmentation", "OK", "SAM2 masks computed",
                           evidence="VISIBLE_FROM_SOURCE",
                           metrics={"masks": len(segs), "first_score": segs[0]["score"]})

    def _stage_pose(self, video, every, max_frames, records) -> StageResult:
        if not self.pose.available:
            return StageResult("pose", "MODEL_UNAVAILABLE", "Pose unavailable")
        total = 0
        for idx, frame in self._sample_frames(video, every, min(max_frames or 2, 2)):
            total += len(self.pose.estimate(frame))
        return StageResult("pose", "OK", "Pose estimated", evidence="VISIBLE_FROM_SOURCE",
                           metrics={"persons": total})

    # -- orchestration ----------------------------------------------------
    def run(self, video: str, every: int = 15, max_frames: int = 0,
            stages: list[str] | None = None) -> dict[str, Any]:
        stages = stages or list(DEFAULT_STAGES)
        records: dict[str, Any] = {}
        step_map = {
            "detection": self._stage_detection,
            "tracking": self._stage_tracking,
            "reid": self._stage_reid,
            "depth": self._stage_depth,
            "segmentation": self._stage_segmentation,
            "pose": self._stage_pose,
        }
        results: list[StageResult] = []
        for name in stages:
            fn = step_map.get(name)
            if fn is None:
                results.append(StageResult(name, "UNKNOWN", f"Unknown stage '{name}'"))
                continue
            try:
                results.append(fn(video, every, max_frames, records))
            except Exception as exc:  # robustesse : on n'interrompt pas le pipeline
                results.append(StageResult(name, "ERROR", f"{type(exc).__name__}: {exc}"))
        return {"stages": [r.to_dict() for r in results], "records": records}
