import os, json
from pathlib import Path
from .core.ffprobe import probe
from .core.io import write_json
from .core.types import StageResult
from .source.analysis import analyze_video
from .vision.yolo_detector import FootballYOLO
from .vision.reid import PlayerReID
from .vision.sam2 import SAM21
from .vision.depth import DepthAnythingV2
from .vision.ocr import JerseyOCR
from .vision.pose import PoseBackend
from .scene.camera import CameraEstimator
from .events.football import FootballEventEngine
from .events.hero import score_hero
from .source.ffmpeg import render_vertical, validate_output
from .qc.final import final_qc

def run_analysis(video,out,models='models',every=15):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    info=probe(video); analysis=analyze_video(video,out/'frames',every)
    stages=[]
    stages.append(StageResult('source_analysis','OK','Source analyzed',evidence='VISIBLE_FROM_SOURCE',metrics={k:v for k,v in analysis.items() if k!='sampled_frames'},artifacts=analysis['sampled_frames'][:10]))
    stages += [
      FootballYOLO(os.getenv('GOALREEL_YOLO_WEIGHTS',str(Path(models)/'best.pt'))).load(),
      PlayerReID(os.getenv('GOALREEL_REID_WEIGHTS',str(Path(models)/'reid.pth'))).load(),
      SAM21(os.getenv('GOALREEL_SAM2_CHECKPOINT',str(Path(models)/'sam2.1.pt')),os.getenv('GOALREEL_SAM2_CONFIG')).load(),
      DepthAnythingV2(os.getenv('GOALREEL_DEPTH_CHECKPOINT',str(Path(models)/'depth_anything_v2.pth')),os.getenv('GOALREEL_DEPTH_CONFIG')).load(),
      JerseyOCR(os.getenv('GOALREEL_OCR_BACKEND')).load(), PoseBackend().load(),
    ]
    stages.append(CameraEstimator().estimate(video))
    events=[]; hero=score_hero(events)
    report={'schema':'goalreel.report.v2','mode':'FOOTAGE','source':info.to_dict(),'stages':[s.to_dict() for s in stages],'event_timeline':events,'hero_moment':hero,'novel_view':{'status':'MODEL_UNAVAILABLE','reason':'No production novel-view backend configured'},'identity':{'status':'UNKNOWN','reason':'No Re-ID inference available'},'warnings':['Novel-view generation requires a configured production backend and compatible checkpoints.','YOLO/Re-ID/SAM2.1/Depth/OCR statuses reflect actual environment.']}
    write_json(out/'report.json',report); return report

def render_baseline(video,out):
    p=render_vertical(video,out); return final_qc(p)
