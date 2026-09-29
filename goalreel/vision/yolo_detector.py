from pathlib import Path
from ..core.types import StageResult

PLACEHOLDER_MARKER = 'GOALREEL_PLACEHOLDER_CHECKPOINT'

def _is_placeholder(path: Path) -> bool:
    """Vrai si le fichier est un marqueur de structure et non un vrai checkpoint."""
    try:
        with path.open('rb') as fh:
            return PLACEHOLDER_MARKER.encode() in fh.read(256)
    except OSError:
        return False

class FootballYOLO:
    def __init__(self,weights): self.weights=Path(weights) if weights else None; self.model=None
    def load(self):
        if not self.weights or not self.weights.is_file(): return StageResult('yolo','MODEL_UNAVAILABLE','Custom football YOLO checkpoint is missing',metrics={'required':str(self.weights or 'models/best.pt')})
        if _is_placeholder(self.weights): return StageResult('yolo','MODEL_UNAVAILABLE','Custom football YOLO checkpoint is a placeholder (not a trained model)',metrics={'required':str(self.weights),'placeholder':True})
        try: from ultralytics import YOLO
        except Exception as e: return StageResult('yolo','MODEL_UNAVAILABLE',f'ultralytics unavailable: {e}')
        try: self.model=YOLO(str(self.weights)); return StageResult('yolo','OK','Custom football YOLO loaded',evidence='VISIBLE_FROM_SOURCE',metrics={'weights':str(self.weights)})
        except Exception as e: return StageResult('yolo','ERROR',f'YOLO load failed: {e}')
    def detect(self,frame):
        if self.model is None: raise RuntimeError('YOLO_NOT_LOADED')
        r=self.model.predict(frame,verbose=False)[0]
        return [{'bbox':[float(x) for x in b],'class_id':int(c),'confidence':float(s)} for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy())]
