from ..providers.unavailable import UnavailableProvider
class NovelViewPlanner:
    def __init__(self,provider=None):self.provider=provider or UnavailableProvider()
    def generate(self,request):return self.provider.generate(request)
