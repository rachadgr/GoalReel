from abc import ABC,abstractmethod
class NovelViewProvider(ABC):
    name='base'
    @abstractmethod
    def status(self):...
    @abstractmethod
    def generate(self,request):...
