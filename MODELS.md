# GoalReel — Registre des modèles (MODELS)

> Politique de vérité : **aucun faux poids, aucune fausse détection**.
> Un checkpoint absent doit produire `MODEL_UNAVAILABLE`, jamais une sortie inventée.

Cet environnement intègre réellement quatre checkpoints fournis. Les états
*véridiques* distingus sont : `READY`, `CHECKPOINT_MISSING`, `DEPENDENCY_MISSING`,
`INCOMPATIBLE`, `GPU_REQUIRED`, `LOAD_ERROR`, `MODEL_UNAVAILABLE`, `DISABLED`.

## Arborescence réelle

```
models/
  detection/yolov8s.pt                # YOLOv8s (COCO générique) — FOURNI  → READY
  segmentation/sam2.1_hiera_tiny.pt   # SAM 2.1 Hiera Tiny  — FOURNI  → READY (runtime sam2)
  depth/depth_anything_v2_vits.pth    # Depth Anything V2 ViT-S — FOURNI → READY
  reid/osnet_x1_0_imagenet.pth        # OSNet x1.0 (ImageNet) — FOURNI → READY
checkpoints/
  RIFE/flownet.pkl                    # RIFE (ECCV2022) — ABSENT → CHECKPOINT_MISSING
```

## Détail des modèles

| Nom | Fichier | Fourni | Rôle | Statut réel |
|-----|---------|--------|------|-------------|
| YOLO détection | `models/detection/yolov8s.pt` | oui | Joueurs / ballon (COCO) | READY |
| SAM 2.1 | `models/segmentation/sam2.1_hiera_tiny.pt` | oui | Segmentation | READY (nécessite `sam2`) |
| Depth Anything V2 | `models/depth/depth_anything_v2_vits.pth` | oui | Profondeur | READY |
| OSNet x1.0 | `models/reid/osnet_x1_0_imagenet.pth` | oui | Ré-identification | READY |
| RIFE | `checkpoints/RIFE/flownet.pkl` | non | Interpolation | CHECKPOINT_MISSING |

## Vérité sur `yolov8s.pt`
`yolov8s.pt` est un détecteur **COCO générique** (80 classes). Il n'est **pas**
entraîné football. Les classes `person` et `sports ball` existent ; `referee` et
`goal` n'existent pas et ne sont jamais inventés. Pour un vrai détecteur football,
fournir un `best.pt` custom via `GOALREEL_DETECTOR_WEIGHTS`.

## Vérité sur OSNet
Les poids sont des backbones **ImageNet** (pas Market-1501, pas football-specific).
Ils produisent un embedding L2-normalisé 512-d exploitable pour la similarité cosinus.

## Vérité sur RIFE
Le code (ECCV2022-RIFE) est *vendorisé* dans `vendor/rife` : `IFNet` se construit et
s'exécute (CPU). Seuls les poids `flownet.pkl` manquent → `CHECKPOINT_MISSING`, et
l'interpolation utilise un fallback de blend explicite (jamais présenté comme RIFE).

## Vérité sur SEVA (caméra virtuelle)
`third_party/stable-virtual-camera` est intégré (**CODE_INTEGRATED**) mais les poids
HF `stabilityai/stable-virtual-camera` ne sont pas fournis et un GPU est requis.
L'état reste `MODEL_UNAVAILABLE` (`GPU_REQUIRED` / `CHECKPOINT_MISSING`).

## Configuration par variables d'environnement

Copiez `.env.example` vers `.env` et ajustez les chemins relatifs si besoin.

