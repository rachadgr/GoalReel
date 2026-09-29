# Model registry

Do not commit large weights. Put local checkpoints here or configure absolute paths.

Required production assets are documented by the master architecture:
- custom football YOLO
- player Re-ID (OSNet/Torchreid or equivalent)
- SAM 2.1
- Depth Anything V2
- RIFE
- optional OCR backend
- optional novel-view backend

A missing asset must produce MODEL_UNAVAILABLE, never a fake output.
