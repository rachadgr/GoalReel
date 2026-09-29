from .base import NovelViewProvider
class MockProvider(NovelViewProvider):
    name='mock-test-only'
    def status(self):return {'status':'OK','provider':self.name,'test_only':True}
    def generate(self,request):return {'status':'OK','test_only':True}
