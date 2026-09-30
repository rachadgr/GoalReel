"""Serveur HTTP minimal GoalReel — expose l'état réel des modèles.

Endpoints :
  * ``GET /health``         : statut du service
  * ``GET /models``         : liste des modèles, état, device, temps d'inf.
  * ``GET /pipeline``       : description des étapes du pipeline
  * ``GET /pipeline/status``: état des modèles mappé sur le pipeline
  * ``GET /models/<name>``  : détail d'un modèle (charge à la demande)
  * ``POST /models/<name>/load``   : charge un modèle
  * ``POST /models/<name>/unload`` : décharge un modèle

Aucune route existante n'est supprimée (``/health`` conservé).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from goalreel.models.manager import ModelManager

_manager = ModelManager()

PIPELINE_STAGES = [
    ("video_input", "Video Input"),
    ("video_analysis", "Video Analysis"),
    ("scene_detection", "Scene/Event Detection"),
    ("player_detection", "Player Detection"),
    ("ball_detection", "Ball Detection"),
    ("player_tracking", "Player Tracking"),
    ("reid", "Player Identity / Re-ID"),
    ("pose", "Pose Estimation"),
    ("segmentation", "Segmentation / Matting"),
    ("subject_selection", "Subject Selection"),
    ("cinematic_camera", "Cinematic Camera / Crop"),
    ("depth", "Depth / Background Separation"),
    ("interpolation", "Slow Motion / Frame Interpolation"),
    ("super_resolution", "Super Resolution"),
    ("color_grading", "Color Grading"),
    ("cinematic_effects", "Cinematic Effects"),
    ("ffmpeg_render", "FFmpeg Render"),
    ("final_reel", "Final Reel"),
]

# Mapping étape pipeline -> modèle(s) responsable(s)
STAGE_MODEL_MAP = {
    "player_detection": "detector",
    "ball_detection": "detector",
    "reid": "reid",
    "pose": "pose",
    "segmentation": "segmenter",
    "cinematic_camera": "virtual_camera",
    "depth": "depth",
    "interpolation": "interpolation",
    "super_resolution": "super_resolution",
}


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/health":
            return self._json({"status": "ok", "service": "goalreel"})
        if path == "/models":
            return self._json({"schema": "goalreel.models_api.v1",
                               "device": _manager.health()["device"],
                               "cuda": _manager.health()["cuda"],
                               "torch": _manager.health()["torch"],
                               "models": _manager.health()["models"]})
        if path == "/pipeline":
            return self._json({"stages": [{"id": i, "label": l} for i, l in PIPELINE_STAGES]})
        if path == "/pipeline/status":
            statuses = _manager.statuses()
            out = []
            for sid, label in PIPELINE_STAGES:
                model = STAGE_MODEL_MAP.get(sid)
                st = statuses.get(model) if model else None
                out.append({"id": sid, "label": label, "model": model,
                            "status": st["status"] if st else "OK",
                            "truth": st.get("truth") if st else None,
                            "device": st["device"] if st else None,
                            "avg_infer_ms": st["avg_infer_ms"] if st else 0})
            return self._json({"stages": out})
        if path.startswith("/models/"):
            name = path.split("/")[2]
            if name not in _manager.settings.models:
                return self._json({"status": "ERROR", "message": "unknown model"}, 404)
            inst, st = _manager.try_stage(name)
            return self._json({"model": _manager.state(name).to_dict(),
                               "stage_result": st.to_dict()})
        return self._json({"status": "ERROR", "message": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "models":
            name, action = parts[2], parts[3]
            if name not in _manager.settings.models:
                return self._json({"status": "ERROR", "message": "unknown model"}, 404)
            if action == "load":
                try:
                    _manager.load(name, force=True)
                except Exception as exc:
                    return self._json({"status": "ERROR", "message": str(exc)}, 500)
                return self._json({"model": _manager.state(name).to_dict()})
            if action == "unload":
                _manager.unload(name)
                return self._json({"model": _manager.state(name).to_dict()})
        return self._json({"status": "ERROR", "message": "not found"}, 404)


def main(host="0.0.0.0", port=8000):
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
