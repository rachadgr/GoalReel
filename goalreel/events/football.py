from dataclasses import dataclass,asdict
@dataclass
class Event:
    event_id:str; kind:str; start_frame:int; peak_frame:int; end_frame:int; confidence:float; evidence:list[str]; status:str
    def to_dict(self): return asdict(self)
class FootballEventEngine:
    def infer(self,tracks,ball=None,pose=None):
        # Evidence-constrained baseline: only emits generic motion/possession candidates when actual trajectories exist.
        if not tracks:return []
        events=[]
        # Do not claim pass/shot/goal without ball + pose/event evidence.
        for tid,frames in tracks.items():
            if len(frames)<3: continue
            xs=[(b[0]+b[2])/2 for b in frames]; ys=[(b[1]+b[3])/2 for b in frames]
            disp=((xs[-1]-xs[0])**2+(ys[-1]-ys[0])**2)**0.5
            if disp>40:
                events.append(Event(f'MOTION_{tid}', 'player_motion', frames[0]['frame'], frames[len(frames)//2]['frame'], frames[-1]['frame'], min(0.99,disp/300), [f'track:{tid}',f'displacement:{disp:.2f}'],'INFERRED').to_dict())
        return events
