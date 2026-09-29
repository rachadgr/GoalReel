from ..core.types import StageResult
class JerseyOCR:
    def __init__(self,backend=None): self.backend=backend; self.reader=None
    def load(self):
        if self.backend in (None,'','none'): return StageResult('jersey_ocr','MODEL_UNAVAILABLE','No OCR backend configured')
        if self.backend=='easyocr':
            try:
                import easyocr; self.reader=easyocr.Reader(['en'],gpu=False); return StageResult('jersey_ocr','OK','EasyOCR loaded')
            except Exception as e:return StageResult('jersey_ocr','MODEL_UNAVAILABLE',str(e))
        return StageResult('jersey_ocr','MODEL_UNAVAILABLE',f'Unsupported OCR backend: {self.backend}')
