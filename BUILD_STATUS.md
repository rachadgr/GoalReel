# GoalReel Build Status

Build target: GOALREEL VIRTUAL CINEMA ENGINE
Core invariant: THE EVENT IS IMMUTABLE. ONLY THE CAMERA VIEWPOINT MAY CHANGE.

## Real integration of the uploaded models (verified in this environment)

| Model | Checkpoint | Load | Inference | GPU | Status |
|-------|-----------|------|-----------|-----|--------|
| Detector (YOLOv8s) | `models/detection/yolov8s.pt` | PASS (ultralytics) | PASS (CPU) | NOT AVAILABLE | READY |
| Depth Anything V2 (ViT-S) | `models/depth/depth_anything_v2_vits.pth` | PASS (0 missing / 0 unexpected) | PASS (HxW depth) | NOT AVAILABLE | READY |
| Re-ID (OSNet x1.0) | `models/reid/osnet_x1_0_imagenet.pth` | PASS (0 missing / 0 unexpected) | PASS (512-d embedding) | NOT AVAILABLE | READY |
| SAM 2.1 (Hiera Tiny) | `models/segmentation/sam2.1_hiera_tiny.pt` | PASS (sam2 runtime) | PASS (box mask) | NOT AVAILABLE | READY |
| RIFE (IFNet) | `checkpoints/RIFE/flownet.pkl` | code builds; weights absent | fallback (blend) only | NOT AVAILABLE | CHECKPOINT_MISSING |
| SEVA (virtual camera) | HF weights not provided | CODE_INTEGRATED | not run | GPU REQUIRED | MODEL_UNAVAILABLE |
| Pose | no checkpoint | — | — | — | CHECKPOINT_MISSING |
| Super-resolution | no checkpoint | — | — | — | CHECKPOINT_MISSING / DISABLED |

**Truthfulness notes**
- `yolov8s.pt` is a **COCO-generic** detector (80 classes). It is **not** football-trained.
  `person` and `sports ball` classes exist; `referee` / `goal` classes do **not** exist and
  are never invented.
- OSNet weights are **ImageNet** backbones (not Market-1501, not football-specific); 512-d
  L2-normalised embeddings are extracted for cosine re-identification.
- RIFE code (ECCV2022-RIFE) is vendored and `IFNet` builds + runs on CPU; only the
  `flownet.pkl` weights are missing, so the interpolation stage uses an explicit
  linear-blend fallback (never presented as RIFE).
- SEVA is **code-integrated only**; its diffusion weights are not provided.

## Verified in this environment
- Source FFprobe: PASS (1024x576 / 30 FPS / 16.13s / 484 frames)
- OpenCV temporal source analysis: PASS
- Real model initialisation for detector / depth / reid / sam2: PASS (CPU)
- Real inference for detector / depth / reid / sam2 on source frames: PASS (CPU)
- Real FFmpeg vertical render: PASS (1080x1920 / 30 FPS / H.264 / yuv420p / AAC)
- Final baseline QC: PASS
- `python -m compileall -q goalreel run_goalreel.py`: PASS
- `python -m pytest -q`: PASS
- Forbidden Gemini/Google GenAI/Veo runtime scan: PASS

## GPU validation
GPU VALIDATION: NOT AVAILABLE (no CUDA GPU in this environment). All model validation above
was performed on CPU. No GPU results are claimed.

## Not falsely claimed as complete
The following remain unavailable and are reported honestly (never fabricated):
- RIFE real interpolation (weights absent)
- production novel-view generation (SEVA weights + GPU required)
- OCR backend
- pose estimation
- super-resolution
