from pathlib import Path
from ..core.types import StageResult
class SAM21:
    def __init__(self,checkpoint=None,config=None): self.checkpoint=Path(checkpoint) if checkpoint else None; self.config=config; self.model=None
    def load(self):
        if not self.checkpoint or not self.checkpoint.is_file(): return StageResult('sam2.1','MODEL_UNAVAILABLE','SAM 2.1 checkpoint missing',metrics={'required':str(self.checkpoint or 'models/sam2.1.pt')})
        try: import sam2
        except Exception as e:return StageResult('sam2.1','MODEL_UNAVAILABLE',f'SAM2 runtime unavailable: {e}')
        return StageResult('sam2.1','OK','SAM2 runtime imported; checkpoint/config wiring available',metrics={'checkpoint':str(self.checkpoint),'config':self.config})
