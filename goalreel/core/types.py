from dataclasses import dataclass, asdict, field
from typing import Any, Literal

Status = Literal["OK","MODEL_UNAVAILABLE","UNKNOWN","ERROR","QC_REJECTED"]
Evidence = Literal["VISIBLE_FROM_SOURCE","TEMPORALLY_INFERRED","DEPTH_INFERRED","MULTI_VIEW_SUPPORTED","GENERATED_ESTIMATE","UNKNOWN"]

@dataclass
class VideoInfo:
    path:str; duration_s:float; width:int; height:int; fps:float; frames:int
    video_codec:str; pixel_format:str; audio_codec:str|None=None
    audio_sample_rate:int|None=None; audio_channels:int|None=None; size_bytes:int=0
    def to_dict(self): return asdict(self)

@dataclass
class StageResult:
    name:str; status:Status; message:str; evidence:Evidence="UNKNOWN"
    metrics:dict[str,Any]=field(default_factory=dict); artifacts:list[str]=field(default_factory=list)
    def to_dict(self): return asdict(self)
