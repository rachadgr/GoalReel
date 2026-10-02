# GoalReel — AI Football Reel Studio

**Phone/Web UI → FastAPI → ComfyUI API → Wan 2.2 TI2V 5B → MP4 → Preview/Download**

GoalReel transforme une **image de football** en **clip vidéo cinématographique**
généré par le modèle **Wan 2.2 TI2V 5B** (via ComfyUI), piloté depuis le
téléphone ou le navigateur. Le dépôt contient aussi le moteur historique
« Virtual Cinema Engine » (analyse de vraies séquences vidéo), qui reste
**intact**.

> Principe de vérité : aucune fausse détection, aucune fausse génération.
> Un composant indisponible (ComfyUI hors ligne, modèle manquant) est signalé
> clairement — jamais simulé.

---

## Table des matières

1. [Architecture](#architecture)
2. [Prérequis](#prérequis)
3. [Installation](#installation)
4. [Variables d'environnement](#variables-denvironnement)
5. [Démarrer ComfyUI + modèles Wan 2.2](#démarrer-comfyui--modèles-wan-22)
6. [Lancer le backend (Reel Studio)](#lancer-le-backend-reel-studio)
7. [Lancer le frontend](#lancer-le-frontend)
8. [API endpoints](#api-endpoints)
9. [Workflow Wan 2.2](#workflow-wan-22)
10. [Presets football](#presets-football)
11. [Cycle de vie d'un job](#cycle-de-vie-dun-job)
12. [Tests](#tests)
13. [Kaggle (GPU T4)](#kaggle-gpu-t4)
14. [Accès depuis le téléphone (Cloudflare Tunnel)](#accès-depuis-le-téléphone-cloudflare-tunnel)
15. [Troubleshooting](#troubleshooting)
16. [Moteur historique (Virtual Cinema Engine)](#moteur-historique-virtual-cinema-engine)

---

## Architecture

Deux services **indépendants** coexistent dans le dépôt :

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  Reel Studio (nouveau)      │     │  Virtual Cinema Engine       │
│  studio_server.py (FastAPI) │     │  server.py (stdlib HTTP)     │
│  goalreel/studio/*          │     │  goalreel/* (vision pipeline)│
│  → ComfyUI + Wan 2.2 TI2V 5B│     │  → analyse vidéo football    │
│  Port par défaut : 7860     │     │  Port par défaut : 8000      │
└─────────────────────────────┘     └──────────────────────────────┘
        ▲                    ▲
        │                    │
   web/ (React/Vite, mobile-first)
```

Flux du Reel Studio :

```
Image (téléphone)
   │  POST /api/upload
   ▼
FastAPI (studio_server.py)
   │  POST /upload/image
   ▼
ComfyUI  ──►  Wan 2.2 TI2V 5B  ──►  MP4
   ▲                                   │
   │  GET /history, /view              │
   └───────────────────────────────────┘
   │
   ▼
GET /api/status/{job_id}  →  progression
GET /api/result/{job_id}  →  preview + download
```

---

## Prérequis

* **Python** ≥ 3.10
* **FFmpeg** (rendu / post-traitement) — `ffmpeg -version`
* **Node.js** ≥ 18 (frontend)
* **ComfyUI** avec GPU (CUDA recommandé ; T4 16 Go validé pour 832×480)
* Les **3 modèles Wan 2.2** placés dans ComfyUI (voir plus bas)

---

## Installation

```bash
git clone https://github.com/rachadgr/GoalReel.git
cd GoalReel

# (optionnel mais recommandé) environnement virtuel
python -m venv .venv && source .venv/bin/activate

# Dépendances Python (inclut FastAPI, uvicorn, httpx, Pillow)
python -m pip install -r requirements.txt
# ou : python -m pip install -e '.[dev]'

# Frontend
cd web && npm install && npm run build && cd ..
```

---

## Variables d'environnement

Copiez `.env.example` vers `.env` et ajustez si besoin.

| Variable | Défaut | Description |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | URL de l'API ComfyUI |
| `HOST` | `0.0.0.0` | Interface d'écoute FastAPI |
| `PORT` | `7860` | Port FastAPI |
| `MAX_UPLOAD_MB` | `20` | Taille max d'upload image |
| `CORS_ORIGINS` | `*` | Origines CORS (séparées par virgule) |
| `GOALREEL_STUDIO_DIFFUSION_MODEL` | `wan2.2_ti2v_5B_fp16.safetensors` | Modèle diffusion |
| `GOALREEL_STUDIO_VAE_MODEL` | `wan2.2_vae.safetensors` | VAE |
| `GOALREEL_STUDIO_TEXT_ENCODER` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | Text encoder |
| `GOALREEL_STUDIO_WIDTH` | `832` | Largeur |
| `GOALREEL_STUDIO_HEIGHT` | `480` | Hauteur |
| `GOALREEL_STUDIO_LENGTH` | `49` | Nombre de frames |
| `GOALREEL_STUDIO_FPS` | `16` | Images/seconde |
| `GOALREEL_STUDIO_STEPS` | `20` | Steps du sampler |
| `GOALREEL_STUDIO_CFG` | `5.0` | CFG |
| `GOALREEL_STUDIO_JOB_TIMEOUT` | `1800` | Timeout génération (s) |
| `GOALREEL_STUDIO_POLL_INTERVAL` | `2.0` | Intervalle de polling (s) |

> Toutes les variables studio acceptent aussi le préfixe `GOALREEL_STUDIO_*`.
> Aucun secret n'est stocké dans le dépôt : tout passe par l'environnement.

---

## Démarrer ComfyUI + modèles Wan 2.2

### 1. Modèles à placer dans ComfyUI (jamais versionnés ici)

```
ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
ComfyUI/models/vae/wan2.2_vae.safetensors
ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

### 2. Lancer ComfyUI

```bash
python main.py --listen 0.0.0.0 --port 8188
```

### 3. Vérifier la connectivité + les modèles

```bash
python scripts/comfy_check.py
# [OK] ComfyUI joignable.
# [OK] UNETLoader.unet_name: wan2.2_ti2v_5B_fp16.safetensors
# [OK] VAELoader.vae_name: wan2.2_vae.safetensors
# [OK] CLIPLoader.clip_name: umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

---

## Lancer le backend (Reel Studio)

```bash
python studio_server.py
# → http://0.0.0.0:7860

# avec auto-reload en développement
python studio_server.py --reload --port 7860
```

Le serveur démarre **même si ComfyUI est hors ligne** : `/api/status` renvoie
alors `available: false` avec un message explicite, sans crash.

Le backend sert aussi le frontend buildé (`web/dist`) sur `/` si présent.

---

## Lancer le frontend

**Développement** (Vite proxifie `/api` vers `http://127.0.0.1:7860`) :

```bash
cd web
npm run dev      # → http://localhost:5173
```

**Production** :

```bash
cd web
npm run build    # génère web/dist, servi par studio_server.py sur /
```

Pour pointer vers une API distante :

```bash
VITE_API_BASE=https://mon-api.example.com npm run build
```

---

## API endpoints

Tous les endpoints sont disponibles avec le préfixe `/api` **et** en alias
racine (compatibilité) : `/health`, `/status`, `/upload`, `/generate`,
`/result/{job_id}`.

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Santé du service |
| `GET` | `/api/status` | Statut ComfyUI (`available`) |
| `GET` | `/api/config` | Configuration publique (sans secret) |
| `GET` | `/api/presets` | Presets football disponibles |
| `POST` | `/api/upload` | Upload image (multipart, champ `image`) |
| `POST` | `/api/generate` | Démarre une génération (form-data) |
| `GET` | `/api/status/{job_id}` | Progression d'un job |
| `GET` | `/api/jobs` | Jobs récents |
| `GET` | `/api/result/{job_id}` | Preview / download MP4 |

### Exemple complet (curl)

```bash
# 1. Upload
curl -s -F "image=@match.png" http://localhost:7860/api/upload
# {"upload_id":"ab12cd34ef56","filename":"ab12cd34ef56.png", ...}

# 2. Génération (avec preset)
curl -s -F "upload_id=ab12cd34ef56" -F "preset=goal_celebration" \
     http://localhost:7860/api/generate
# {"job_id":"9f8e...","state":"QUEUED","status_url":"/api/status/9f8e...", ...}

# 3. Progression
curl -s http://localhost:7860/api/status/9f8e...
# {"job_id":"9f8e...","state":"GENERATING","progress":55,"message":"...", ...}

# 4. Résultat
curl -o reel.mp4 http://localhost:7860/api/result/9f8e...
```

### Format de réponse (job)

```json
{
  "job_id": "9f8e7d6c5b4a",
  "state": "GENERATING",
  "progress": 55,
  "message": "Génération en cours (Wan 2.2 TI2V 5B)…",
  "created_at": "2025-01-01T12:00:00+00:00",
  "updated_at": "2025-01-01T12:00:12+00:00",
  "result_url": null,
  "error": null,
  "request": { "preset_id": "goal_celebration", "width": 832, "height": 480 }
}
```

---

## Workflow Wan 2.2

Le graphe est construit par `goalreel/studio/wan22.py` :

```
LoadImage → UNETLoader → ModelSamplingSD3
          → CLIPLoader → VAELoader
          → CLIPTextEncode (positif / négatif)
          → Wan22ImageToVideoLatent
          → KSampler → VAEDecode → CreateVideo → SaveVideo
```

* `SaveVideo` utilise le format **plat** `{"format": "mp4", "codec": "h264"}`
  (pas d'ancien `format` imbriqué) ;
* les paramètres (`width`, `height`, `length`, `fps`, `steps`, `cfg`, `seed`,
  prompts) sont **modifiables** via l'API / l'UI ;
* défauts T4 : `832×480 · 49 frames · 16 fps · 20 steps · cfg 5.0` ;
* un rendu **9:16** final s'obtient par **post-traitement** (ne pas demander
  1080×1920 directement à Wan, sous risque de dépassement mémoire).

---

## Presets football

| ID | Label | Usage |
|---|---|---|
| `cinematic_football` | Cinematic Football | Plan général cinématographique |
| `goal_celebration` | Goal Celebration | Célébration de but |
| `dribble` | Dribble | Dribble serré |
| `sprint` | Sprint | Course rapide |
| `shot_on_goal` | Shot on Goal | Frappe au but |
| `goalkeeper_save` | Goalkeeper Save | Arrêt du gardien |
| `player_introduction` | Player Introduction | Présentation de joueur |
| `slow_motion_hero` | Slow Motion Hero | Plan héroïque au ralenti |

Chaque preset fusionne une action spécifique avec un **style réaliste**
(stade pro, anatomie naturelle, kit cohérent, caméra cinématographique,
éclairage naturel) et une **liste négative** anti-artefacts (membres en trop,
joueurs dupliqués, visage/maillot déformés, scintillement, fond instable,
aspect CGI, texte, watermark).

---

## Cycle de vie d'un job

```
QUEUED → UPLOADING → SUBMITTING → GENERATING → PROCESSING → COMPLETED
                                                           ↘ FAILED
```

* `progress` : 0 → 100 ;
* `message` : description lisible à chaque étape ;
* `prompt_id` (ComfyUI) est **distinct** du `job_id` (jamais confondus) ;
* en cas d'erreur, `error` contient un message clair, **sans stack trace**.

---

## Tests

Les tests n'exigent **ni GPU ni ComfyUI** (le client HTTP est mocké).

```bash
# Tous les tests
python -m pytest -q

# Tests du Reel Studio uniquement
python -m pytest -q \
  tests/test_studio_config.py \
  tests/test_studio_wan22.py \
  tests/test_studio_presets.py \
  tests/test_studio_jobs.py \
  tests/test_studio_comfyui.py \
  tests/test_studio_service.py \
  tests/test_studio_api.py

# Contrôles qualité
python -m compileall -q goalreel studio_server.py server.py
python scripts/check_no_gemini.py
```

---

## Kaggle (GPU T4)

1. **Activer le GPU** : *Settings → Accelerator → GPU T4 ×2*.
2. **Installer ComfyUI** et y placer les 3 modèles Wan 2.2 (chemins ci-dessus).
3. **Lancer ComfyUI** : `python main.py --listen 0.0.0.0 --port 8188`.
4. **Installer GoalReel** : `pip install -r requirements.txt`.
5. **Lancer le studio** : `python studio_server.py --port 7860`.
6. **Exposer** les ports via un tunnel (voir ci-dessous) ou l'API Kaggle.

> Astuce : sous Kaggle, `COMFY_URL` peut rester `http://127.0.0.1:8188` si
> ComfyUI tourne dans la même session.

---

## Accès depuis le téléphone (Cloudflare Tunnel)

Le plus simple pour ouvrir l'app depuis un mobile sans déployer :

```bash
# 1. Backend
python studio_server.py --host 0.0.0.0 --port 7860

# 2. Tunnel (nécessite cloudflared)
cloudflared tunnel --url http://localhost:7860
# → https://<random>.trycloudflare.com
```

Ouvrez l'URL générée sur le téléphone. Comme le frontend buildé est servi par
FastAPI sur `/`, **une seule URL suffit** (UI + API).

Variante `ngrok` :

```bash
ngrok http 7860
```

---

## Troubleshooting

| Symptôme | Cause probable | Solution |
|---|---|---|
| `/api/status` → `available: false` | ComfyUI arrêté / mauvaise URL | Démarrer ComfyUI, vérifier `COMFY_URL` |
| `Workflow rejeté par ComfyUI` | Nom de modèle erroné / absent | `python scripts/comfy_check.py` |
| Job bloqué en `GENERATING` | Génération longue ou ComfyUI figé | Augmenter `GOALREEL_STUDIO_JOB_TIMEOUT` |
| `Fichier trop volumineux` | Image > `MAX_UPLOAD_MB` | Compresser ou augmenter la limite |
| `Le fichier n'est pas une image valide` | Contenu ≠ image (MIME falsifié) | Fournir PNG/JPEG/WEBP réel |
| Out of memory côté ComfyUI | Résolution trop élevée | Rester en 832×480, post-traiter le 9:16 |
| Frontend vide | `web/dist` absent | `cd web && npm run build` |
| CORS bloqué | Origine non autorisée | Ajuster `CORS_ORIGINS` |

---

## Moteur historique (Virtual Cinema Engine)

Le dépôt conserve le pipeline d'analyse vidéo original (YOLO, ByteTrack,
Re-ID, SAM 2.1, Depth Anything V2, RIFE, caméras virtuelles, QC, rendu
vertical). Il est **inchangé** et indépendant du Reel Studio.

**Principe fondamental : L'ÉVÉNEMENT EST IMMUABLE. SEUL LE POINT DE VUE CAMÉRA
PEUT CHANGER.**

### Sans Gemini

Ce dépôt ne contient volontairement **aucun** runtime Gemini / Google GenAI /
Veo (voir `docs/NO_GEMINI.md`, vérifié par la CI via
`scripts/check_no_gemini.py`).

### Démarrage rapide (moteur historique)

```bash
# Checkpoints (poids publics)
bash scripts/download_models.sh
# placer votre modèle YOLO football custom dans models/best.pt

# Pipeline complet → remplit outputs/
python run_goalreel.py --video assets/SOURCE_MASTER_1000006866.mp4

# Dashboard modèles / pipeline (serveur historique, port 8000)
python server.py
```

### Sémantique des états

`OK`, `MODEL_UNAVAILABLE`, `UNKNOWN`, `ERROR`, `QC_REJECTED`.

### Pipeline historique

SOURCE → REF TEMPORAL → YOLO → BYTETRACK → RE-ID → OCR → SAM2.1 → DEPTH V2 →
BALL/POSE → EVENT TIMELINE → HERO → SCENE → VIRTUAL CAMERA → NOVEL VIEW →
IDENTITY/EVENT/TEMPORAL QC → RIFE/SPEED → GRADE → AUDIO → FFMPEG → FINAL QC →
REPORT.

---

## Arborescence (extraits)

```
GoalReel/
├── studio_server.py                # ⭐ entrée Reel Studio (FastAPI)
├── server.py                       # serveur historique (dashboard modèles)
├── run_goalreel.py                 # orchestrateur pipeline vision
├── goalreel/
│   ├── studio/                     # ⭐ Reel Studio
│   │   ├── studio_config.py        #   configuration
│   │   ├── presets.py              #   presets football
│   │   ├── jobs.py                 #   cycle de vie / store
│   │   ├── wan22.py                #   générateur de workflow
│   │   ├── comfyui.py              #   client HTTP ComfyUI
│   │   ├── service.py              #   orchestration (background)
│   │   └── api.py                  #   API FastAPI
│   └── ...                         # pipeline vision historique
├── web/                            # ⭐ frontend React/Vite mobile-first
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── api.js
│       ├── style.css
│       └── components/
│           ├── ReelStudio.jsx      #   écran de génération
│           ├── ModelDashboard.jsx  #   dashboard historique
│           └── StatusPill.jsx      #   pastille statut ComfyUI
├── scripts/
│   ├── comfy_check.py              # ⭐ diagnostic ComfyUI + modèles
│   └── check_no_gemini.py          #    politique « No-GenAI »
├── tests/
│   └── test_studio_*.py            # ⭐ tests Reel Studio (mockés)
└── .env.example
```
