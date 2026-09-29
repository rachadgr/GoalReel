#!/usr/bin/env bash
# =============================================================================
# GoalReel — Téléchargement des checkpoints officiels
# =============================================================================
# Télécharge les poids publics vers checkpoints/ dans le but de reproduire
# exactement l'arborescence requise. Les poids ne sont PAS commités sur GitHub
# (voir .gitignore) car leur taille dépasse la limite de 100 Mo/fichier.
#
# Usage :
#   bash scripts/download_models.sh            # tout télécharger
#   bash scripts/download_models.sh sam2 reid  # sous-ensemble
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="$ROOT/checkpoints"
MODELS="$ROOT/models"
mkdir -p "$CKPT/RIFE" "$MODELS"

dl () {  # dl <url> <dest>
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then echo "[skip] $dest déjà présent"; return 0; fi
  echo "[get ] $url"
  curl -fL --retry 3 --retry-delay 2 -C - -o "$dest" "$url"
  echo "[ ok ] $dest"
}

want () { [[ $# -eq 0 || " ${TARGETS[*]} " == *" $1 "* ]]; }
TARGETS=("$@")

# --- SAM 2.1 (Meta, Hiera Tiny) --------------------------------------------
if want sam2; then
  dl "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
     "$CKPT/sam2.1_hiera_tiny.pt"
fi

# --- Depth Anything V2 (Small / ViT-S) -------------------------------------
if want depth; then
  dl "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
     "$CKPT/depth_anything_v2_vits.pth"
fi

# --- OSNet x1.0 Market-1501 (Re-ID) ----------------------------------------
if want reid; then
  dl "https://huggingface.co/MYerassyl/retail-heat-osnet/resolve/main/osnet_x1_0_market1501.pth" \
     "$CKPT/osnet_x1_0_market1501.pth.tar"
fi

# --- RIFE flownet ----------------------------------------------------------
if want rife; then
  dl "https://huggingface.co/yuvraj108c/sonic/resolve/main/RIFE/flownet.pkl" \
     "$CKPT/RIFE/flownet.pkl"
fi

# --- YOLO football custom (best.pt) ----------------------------------------
# best.pt est un modèle YOLO football ENTRAÎNÉ SUR MESURE : il n'existe pas de
# poids public équivalent. Fournissez votre propre checkpoint ici, ou
# entraînez-en un (voir MODELS.md → section "best.pt").
if want yolo; then
  if [[ -f "$MODELS/best.pt" ]]; then
    echo "[skip] models/best.pt déjà présent"
  else
    echo "[warn] models/best.pt absent : modèle custom football requis."
    echo "       Le pipeline le signalera proprement comme MODEL_UNAVAILABLE."
    echo "       Placez votre best.pt dans models/ avant l'inférence."
  fi
fi

echo "Terminé."
