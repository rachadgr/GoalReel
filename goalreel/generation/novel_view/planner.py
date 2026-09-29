from ..providers.unavailable import UnavailableProvider

def get_default_provider():
    """Choisit le provider novel-view selon la configuration réelle.

    ``GOALREEL_NOVEL_VIEW_BACKEND=seva`` -> SevaProvider ;
    ``mock`` -> MockProvider (tests uniquement) ; sinon UnavailableProvider.
    """
    import os
    backend = (os.getenv("GOALREEL_NOVEL_VIEW_BACKEND") or "unavailable").lower()
    if backend == "seva":
        try:
            from ..providers.seva import SevaProvider
            return SevaProvider()
        except Exception:
            return UnavailableProvider()
    if backend == "mock":
        from ..providers.mock import MockProvider
        return MockProvider()
    return UnavailableProvider()


class NovelViewPlanner:
    def __init__(self,provider=None):self.provider=provider or get_default_provider()
    def generate(self,request):return self.provider.generate(request)
    def status(self):return self.provider.status()
