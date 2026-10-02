#!/usr/bin/env python3
"""Point d'entrée du GoalReel AI Football Reel Studio (FastAPI + uvicorn).

Usage :

    python studio_server.py
    # ou, avec rechargement automatique en développement :
    python studio_server.py --reload

Variables d'environnement principales (voir ``.env.example``) :

    COMFY_URL       URL de l'API ComfyUI          (défaut: http://127.0.0.1:8188)
    HOST            interface d'écoute            (défaut: 0.0.0.0)
    PORT            port d'écoute                 (défaut: 7860)
    MAX_UPLOAD_MB   taille max d'upload image     (défaut: 20)
    CORS_ORIGINS    origines CORS séparées par ,  (défaut: *)

Le serveur démarre **même si ComfyUI est hors ligne** : ``/api/status`` renvoie
alors ``available: false`` avec un message clair, sans crash.

Ce serveur est distinct du ``server.py`` historique (dashboard modèles /
pipeline) : les deux peuvent coexister.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from goalreel.studio.api import create_app  # noqa: E402
from goalreel.studio.studio_config import load_studio_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GoalReel Reel Studio server")
    parser.add_argument("--host", default=None, help="interface d'écoute")
    parser.add_argument("--port", type=int, default=None, help="port d'écoute")
    parser.add_argument("--reload", action="store_true", help="auto-reload (dev)")
    parser.add_argument(
        "--log-level",
        default=None,
        help="niveau de log (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_studio_settings()

    host = args.host or settings.host
    port = args.port or settings.port
    log_level = (args.log_level or "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print(
            "uvicorn n'est pas installé. Installez les dépendances studio :\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    app = create_app(settings)
    logging.getLogger("goalreel.studio").info(
        "GoalReel Reel Studio → http://%s:%s  (ComfyUI: %s)",
        host,
        port,
        settings.comfy_url,
    )
    uvicorn.run(app, host=host, port=port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
