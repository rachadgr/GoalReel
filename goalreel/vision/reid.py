from pathlib import Path
from ..core.types import StageResult
class PlayerReID:
    def __init__(self,weights=None): self.weights=Path(weights) if weights else None; self.backend=None
    def load(self):
        if not self.weights or not self.weights.is_file(): return StageResult('reid','MODEL_UNAVAILABLE','Re-ID checkpoint missing',metrics={'required':str(self.weights or 'models/reid.pth')})
        try: import torchreid
        except Exception as e: return StageResult('reid','MODEL_UNAVAILABLE',f'torchreid unavailable: {e}')
        try:
            self.backend=torchreid; return StageResult('reid','OK','Torchreid backend available',metrics={'weights':str(self.weights)})
        except Exception as e:return StageResult('reid','ERROR',str(e))
    def match(self,query,candidates):
        if self.backend is None: raise RuntimeError('REID_NOT_LOADED')
        raise NotImplementedError('Configure a concrete OSNet checkpoint/model transform before inference')
