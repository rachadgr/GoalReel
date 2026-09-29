# GoalReel — Virtual Cinema Engine

**Principe fondamental : L'ÉVÉNEMENT EST IMMUABLE. SEUL LE POINT DE VUE CAMÉRA PEUT CHANGER.**

GoalReel analyse de vraies séquences de football et construit un pipeline contraint
par la preuve : identité joueur, suivi, compréhension d'événements, profondeur /
segmentation, caméras virtuelles, reconstruction novel-view, QC et rendu vertical final.

## Politique de vérité
Aucune fausse détection, fausse profondeur, faux masque, fausse identité, faux
horodatage d'événement ou faux score de QC. Un checkpoint / back-end manquant est
signalé comme `MODEL_UNAVAILABLE`.

## Sans Gemini
Ce dépôt ne contient volontairement **aucun** runtime Gemini / Google GenAI / Veo.

## Arborescence du dépôt

```
GoalReel/
├── README.md
├── MODELS.md
├── requirements.txt
├── run_goalreel.py                     # orchestrateur racine → outputs/
├── pyproject.toml
├── server.py
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── goalreel-ci.yml             # CI (analyse, no-gemini scan, tests, rendu)
├── docs/
│   ├── ARCHITECTURE.md
│   └── NO_GEMINI.md
├── goalreel/                           # package Python
│   ├── __init__.py
│   ├── pipeline.py
│   ├── cli/main.py
│   ├── core/                           # ffprobe, io, types
│   ├── source/                         # analyse, ffmpeg
│   ├── vision/                         # yolo, bytetrack, reid, ocr, sam2, depth, ball, pose
│   ├── scene/                          # camera, evidence, representation
│   ├── events/                         # football, hero
│   ├── generation/                     # shot planner, reference pack, novel-view providers
│   ├── cinematic/                      # assembler, grading, reframe, rife, speed, virtual_camera
│   ├── qc/                             # event, final, identity, temporal
│   ├── audio/
│   └── jobs/
├── models/
│   ├── README.md
│   ├── registry.json
│   └── best.pt                         # (fourni par l'utilisateur — non versionné)
├── checkpoints/
│   ├── sam2.1_hiera_tiny.pt            # (téléchargé — non versionné)
│   ├── depth_anything_v2_vits.pth      # (téléchargé — non versionné)
│   ├── osnet_x1_0_market1501.pth.tar   # (téléchargé — non versionné)
│   └── RIFE/
│       └── flownet.pkl                 # (téléchargé — non versionné)
├── assets/
│   └── SOURCE_MASTER_1000006866.mp4
├── scripts/
│   ├── check_no_gemini.py
│   ├── run_source.sh
│   └── download_models.sh
├── tests/
│   ├── test_core.py
│   └── test_ffmpeg.py
├── web/                                # UI (Vite + React)
└── outputs/                            # sorties du pipeline
    ├── source_manifest.json
    ├── backend_status.json
    ├── event_timeline.json
    ├── hero_moment.json
    └── final_reel.mp4
```

## Démarrage rapide

```bash
# 1) Dépendances
python -m pip install -e '.[dev]'
# ou : python -m pip install -r requirements.txt

# 2) Checkpoints (poids publics)
bash scripts/download_models.sh
#    → placer votre modèle YOLO football custom dans models/best.pt

# 3) Pipeline complet → remplit outputs/
python run_goalreel.py --video assets/SOURCE_MASTER_1000006866.mp4

# 4) CLI détaillée
python -m goalreel.cli.main analyze assets/SOURCE_MASTER_1000006866.mp4 --out runs --name source
python -m goalreel.cli.main render-vertical assets/SOURCE_MASTER_1000006866.mp4 --out runs/source/render/baseline.mp4

# 5) Tests
python -m pytest -q
```

## Modèles optionnels
Placez les checkpoints dans `checkpoints/` (ou `models/`) et configurez `.env`
depuis `.env.example`. Les poids volumineux ne sont pas committés.

Checkpoints de production requis selon les étapes utilisées :
- `models/best.pt` — YOLO football custom
- Re-ID — `checkpoints/osnet_x1_0_market1501.pth.tar`
- SAM 2.1 — `checkpoints/sam2.1_hiera_tiny.pt`
- Depth Anything V2 — `checkpoints/depth_anything_v2_vits.pth`
- RIFE — `checkpoints/RIFE/flownet.pkl`

Voir **[MODELS.md](MODELS.md)** pour les sources et tailles exactes.

## Sémantique des états
`OK`, `MODEL_UNAVAILABLE`, `UNKNOWN`, `ERROR`, `QC_REJECTED`.

## Pipeline
SOURCE → REF TEMPORAL → YOLO → BYTETRACK → RE-ID → OCR → SAM2.1 → DEPTH V2 →
BALL/POSE → EVENT TIMELINE → HERO → SCENE → VIRTUAL CAMERA → NOVEL VIEW →
IDENTITY/EVENT/TEMPORAL QC → RIFE/SPEED → GRADE → AUDIO → FFMPEG → FINAL QC → REPORT.
