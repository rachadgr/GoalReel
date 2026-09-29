from .base import NovelViewProvider
class UnavailableProvider(NovelViewProvider):
    name='unavailable'
    def status(self):return {'status':'MODEL_UNAVAILABLE','provider':self.name}
    def generate(self,request):return {'status':'MODEL_UNAVAILABLE','reason':'No novel-view backend configured'}
