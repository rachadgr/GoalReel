# GoalReel Build Status

Build target: GOALREEL VIRTUAL CINEMA ENGINE
Core invariant: THE EVENT IS IMMUTABLE. ONLY THE CAMERA VIEWPOINT MAY CHANGE.

## Verified in this environment
- Source FFprobe: PASS
- OpenCV temporal source analysis: PASS
- 484 source frames read
- 30 FPS / 1024x576 / 16.133333s source verified
- 2D camera-motion estimation from optical flow: PASS
- ByteTrack association unit test: PASS
- Real FFmpeg vertical render: PASS
- Final baseline: 1080x1920 / 30 FPS / H.264 / yuv420p / AAC
- Final baseline QC: PASS
- pytest: PASS
- forbidden Gemini/Google GenAI/Veo runtime scan: PASS

## Not falsely claimed as complete
The current environment does not contain the production checkpoints/runtime for:
- custom football YOLO
- Player Re-ID
- SAM 2.1
- Depth Anything V2
- OCR backend
- RIFE
- production novel-view generation

These stages return MODEL_UNAVAILABLE rather than fabricated outputs.

## Important source-media note
FFmpeg/decoder reports malformed H.264 NAL units in the supplied source at some packets, but the source remains probeable and a valid 484-frame baseline render was produced. A future ingest-normalization stage should optionally transcode the source to a clean mezzanine before heavy CV inference.
