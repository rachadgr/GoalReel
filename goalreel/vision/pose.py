from ..core.types import StageResult
class PoseBackend:
    def load(self): return StageResult('pose','MODEL_UNAVAILABLE','No pose checkpoint configured')
