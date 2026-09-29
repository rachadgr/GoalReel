# GoalReel — Registre des modèles (MODELS)

> Politique de vérité : **aucun faux poids, aucune fausse détection**.
> Un checkpoint absent doit produire `MODEL_UNAVAILABLE`, jamais une sortie inventée.

Ce document décrit les checkpoints attendus par le pipeline GoalReel et leur
emplacement exact, tel que requis par l'architecture cible.

## Arborescence attendue

```
models/
  best.pt                              # YOLO football custom

checkpoints/
  sam2.1_hiera_tiny.pt                 # SAM 2.1 (Hiera Tiny)
  depth_anything_v2_vits.pth           # Depth Anything V2 (ViT-S)
  osnet_x1_0_market1501.pth.tar        # Re-ID OSNet x1.0 (Market-1501)
  RIFE/
    flownet.pkl                        # RIFE (interpolation de frames)
```

## Téléchargement automatique

Les quatre checkpoints **publics** s'obtiennent en une commande :

```bash
bash scripts/download_models.sh          # tout ce qui est public
bash scripts/download_models.sh sam2 reid  # sous-ensemble
```

## Détail des modèles

| Nom | Fichier cible | Source | Taille | Rôle |
|-----|---------------|--------|--------|------|
| Football YOLO | `models/best.pt` | **custom** (voir ci-dessous) | ~6–50 Mo | Détection joueurs/ballon |
| SAM 2.1 (Tiny) | `checkpoints/sam2.1_hiera_tiny.pt` | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) | ~149 Mo | Segmentation objet |
| Depth Anything V2 | `checkpoints/depth_anything_v2_vits.pth` | [HF: depth-anything](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) | ~95 Mo | Estimation de profondeur |
| OSNet x1.0 | `checkpoints/osnet_x1_0_market1501.pth.tar` | [HF: retail-heat-osnet](https://huggingface.co/MYerassyl/retail-heat-osnet) | ~10 Mo | Ré-identification joueur |
| RIFE | `checkpoints/RIFE/flownet.pkl` | [HF: sonic/RIFE](https://huggingface.co/yuvraj108c/sonic) | ~61 Mo | Interpolation temporelle |

## À propos de `best.pt` (modèle football custom)

`best.pt` est un modèle **YOLO entraîné sur mesure** pour le football. Il n'existe
pas de poids public équivalent générique. Deux options :

1. **Fournir votre checkpoint** : placez le fichier dans `models/best.pt`.
2. **En entraîner un** : entraînez un détecteur (Ultralytics YOLO) sur un jeu de
   données football (joueurs, ballon, arbitres, buts) puis exportez `best.pt`.

Tant que `models/best.pt` est absent, l'étape YOLO renvoie proprement
`MODEL_UNAVAILABLE` et le reste du pipeline reste exécutable.

## Configuration par variables d'environnement

Copiez `.env.example` vers `.env` et ajustez les chemins absolus si besoin :

```dotenv
GOALREEL_YOLO_WEIGHTS=models/best.pt
GOALREEL_REID_WEIGHTS=checkpoints/osnet_x1_0_market1501.pth.tar
GOALREEL_SAM2_CHECKPOINT=checkpoints/sam2.1_hiera_tiny.pt
GOALREEL_DEPTH_CHECKPOINT=checkpoints/depth_anything_v2_vits.pth
GOALREEL_OCR_BACKEND=
GOALREEL_NOVEL_VIEW_BACKEND=unavailable
GOALREEL_DEVICE=auto
```

## Politique de commit Git

Les poids volumineux **ne sont pas** versionnés (limite GitHub : 100 Mo/fichier).
`checkpoints/` et `models/*.pt` sont ignorés par `.gitignore`. Utilisez
`scripts/download_models.sh` (ou Git LFS) pour les reconstituer après clonage.

## États renvoyés par le pipeline

- `OK` — stage exécuté avec des preuves réelles.
- `MODEL_UNAVAILABLE` — checkpoint/back-end manquant (jamais de sortie simulée).
- `UNKNOWN` — preuve insuffisante pour conclure.
- `ERROR` — erreur d'exécution.
- `QC_REJECTED` — sortie rejetée par le contrôle qualité.
