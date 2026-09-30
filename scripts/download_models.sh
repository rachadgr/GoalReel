#!/usr/bin/env bash
# =============================================================================
# GoalReel — Téléchargement des checkpoints officiels
# =============================================================================
# Télécharge les poids publics vers les chemins EXACTS attendus par le registre
# (models/registry.json) et par goalreel/config.py. Les poids ne sont PAS
# commités sur GitHub (voir .gitignore) car leur taille dépasse la limite de
# 100 Mo/fichier.
#
# Usage :
#   bash scripts/download_models.sh            # tout télécharger
#   bash scripts/download_models.sh sam2 reid  # sous-ensemble
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/checkpoints"
MODELS="$ROOT/models"

# Arborescence EXACTE du registre (models/registry.json).
mkdir -p "$CKPT/RIFE" \
         "$MODELS/detection" "$MODELS/segmentation" "$MODELS/depth" \
         "$MODELS/reid" "$MODELS/interpolation" "$MODELS/super_resolution" "$MODELS/pose"

dl () {  # dl <url> <dest>
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then echo "[skip] $dest déjà présent"; return 0; fi
  echo "[get ] $url"
  curl -fL --retry 3 --retry-delay 2 -C - -o "$dest" "$url"
  echo "[ ok ] $dest"
}

want () { [[ $# -eq 0 || " ${TARGETS[*]} " == *" $1 "* ]]; }
TARGETS=("$@")

# --- Détection (YOLOv8s, COCO générique) -----------------------------------
if want yolo; then
  dl "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt" \
     "$MODELS/detection/yolov8s.pt"
fi

# --- SAM 2.1 (Meta, Hiera Tiny) --------------------------------------------
if want sam2; then
  dl "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
     "$MODELS/segmentation/sam2.1_hiera_tiny.pt"
fi

# --- Depth Anything V2 (Small / ViT-S) -------------------------------------
if want depth; then
  dl "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
     "$MODELS/depth/depth_anything_v2_vits.pth"
fi

# --- OSNet x1.0 (backbone ImageNet) ----------------------------------------
# Chemin EXACT du registre (models/reid/osnet_x1_0_imagenet.pth) : il s'agit des
# poids ImageNet fournis par l'auteur de torchreid (torchreid.models.osnet).
#   doc : https://kaiyangzhou.github.io/deep-person-reid/
if want reid; then
  if [[ -f "$MODELS/reid/osnet_x1_0_imagenet.pth" ]]; then
    echo "[skip] $MODELS/reid/osnet_x1_0_imagenet.pth déjà présent"
  else
    echo "[warn] osnet_x1_0_imagenet.pth non téléchargeable automatiquement ici"
    echo "       (poids torchreid ImageNet ; l'URL historique est un lien Google Drive)."
    echo "       Déposez le fichier dans models/reid/ ou utilisez gdown avec l'ID 1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY."
    echo "       Variante publique équivalente : osnet_x1_0 Market-1501 (mêmes backbones, "
    echo "       tête 751 classes) — le checkpoint ImageNet 1000 classes est celui documenté."
  fi
fi

# --- RIFE flownet ----------------------------------------------------------
if want rife; then
  dl "https://huggingface.co/yuvraj108c/sonic/resolve/main/RIFE/flownet.pkl" \
     "$CKPT/RIFE/flownet.pkl"
fi

# --- YOLO football custom (best.pt) ----------------------------------------
# models/best.pt est un PLACEHOLDER (marqueur texte) : il n'est PAS un vrai
# modèle. Pour un détecteur football entraîné sur mesure, fournissez votre
# propre checkpoint et pointez GOALREEL_DETECTOR_WEIGHTS dessus
# (ou déposez-le sur models/detection/yolov8s.pt pour remplacer le COCO).
if want yolo-custom; then
  echo "[info] Pour un détecteur football : export GOALREEL_DETECTOR_WEIGHTS=/chemin/votre_best.pt"
fi

echo "Terminé."
