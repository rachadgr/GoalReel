from pathlib import Path
from ..core.types import StageResult
class FootballYOLO:
    def __init__(self,weights): self.weights=Path(weights) if weights else None; self.model=None
    def load(self):
        if not self.weights or not self.weights.is_file(): return StageResult('yolo','MODEL_UNAVAILABLE','Custom football YOLO checkpoint is missing',metrics={'required':str(self.weights or 'models/best.pt')})
        try: from ultralytics import YOLO
        except Exception as e: return StageResult('yolo','MODEL_UNAVAILABLE',f'ultralytics unavailable: {e}')
        try: self.model=YOLO(str(self.weights)); return StageResult('yolo','OK','Custom football YOLO loaded',evidence='VISIBLE_FROM_SOURCE',metrics={'weights':str(self.weights)})
        except Exception as e: return StageResult('yolo','ERROR',f'YOLO load failed: {e}')
    def detect(self,frame):
        if self.model is None: raise RuntimeError('YOLO_NOT_LOADED')
        r=self.model.predict(frame,verbose=False)[0]
        return [{'bbox':[float(x) for x in b],'class_id':int(c),'confidence':float(s)} for b,c,s in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy())]
