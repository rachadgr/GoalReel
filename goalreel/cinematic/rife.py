from pathlib import Path
from ..core.types import StageResult
class RIFE:
    def __init__(self,checkpoint=None):self.checkpoint=Path(checkpoint) if checkpoint else None
    def load(self):
        if not self.checkpoint or not self.checkpoint.is_file():return StageResult('rife','MODEL_UNAVAILABLE','RIFE checkpoint missing',metrics={'required':str(self.checkpoint or 'models/rife')})
        return StageResult('rife','MODEL_UNAVAILABLE','RIFE runtime wiring requires a compatible checkpoint/backend')
