from dataclasses import dataclass,asdict
@dataclass
class SceneObject:
    object_id:str; kind:str; evidence_state:str; confidence:float; source_frames:list[int]
    def to_dict(self):return asdict(self)
class SceneRepresentation:
    def __init__(self):self.objects=[]
    def add(self,obj):self.objects.append(obj)
    def to_dict(self):return {'objects':[x.to_dict() for x in self.objects]}
