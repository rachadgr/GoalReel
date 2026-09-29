from pathlib import Path
from ..core.types import StageResult
class DepthAnythingV2:
    def __init__(self,checkpoint=None,config=None): self.checkpoint=Path(checkpoint) if checkpoint else None; self.config=config
    def load(self):
        if not self.checkpoint or not self.checkpoint.is_file(): return StageResult('depth_anything_v2','MODEL_UNAVAILABLE','Depth Anything V2 checkpoint missing',metrics={'required':str(self.checkpoint or 'models/depth_anything_v2.pth')})
        try: import torch
        except Exception as e:return StageResult('depth_anything_v2','MODEL_UNAVAILABLE',str(e))
        return StageResult('depth_anything_v2','OK','Checkpoint found; concrete Depth Anything V2 architecture must match configured checkpoint',metrics={'checkpoint':str(self.checkpoint),'config':self.config})
