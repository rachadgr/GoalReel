from dataclasses import dataclass,asdict
@dataclass
class Event:
    event_id:str; kind:str; start_frame:int; peak_frame:int; end_frame:int; confidence:float; evidence:list[str]; status:str
    def to_dict(self): return asdict(self)

def _bbox_of(item):
    """Récupère la bbox d'un item de trajectoire (dict {'bbox':..} ou liste)."""
    if isinstance(item, dict):
        return item.get('bbox')
    return item

class FootballEventEngine:
    def infer(self,tracks,ball=None,pose=None):
        # Evidence-constrained baseline: only emits generic motion/possession candidates when actual trajectories exist.
        if not tracks:return []
        events=[]
        # Do not claim pass/shot/goal without ball + pose/event evidence.
        for tid,frames in tracks.items():
            if len(frames)<3: continue
            boxes=[_bbox_of(f) for f in frames]
            boxes=[b for b in boxes if b is not None]
            if len(boxes)<3: continue
            xs=[(b[0]+b[2])/2 for b in boxes]; ys=[(b[1]+b[3])/2 for b in boxes]
            disp=((xs[-1]-xs[0])**2+(ys[-1]-ys[0])**2)**0.5
            if disp>40:
                def _frame(f,i):
                    return f['frame'] if isinstance(f,dict) and 'frame' in f else i
                events.append(Event(f'MOTION_{tid}', 'player_motion', _frame(frames[0],0), _frame(frames[len(frames)//2],len(frames)//2), _frame(frames[-1],len(frames)-1), min(0.99,disp/300), [f'track:{tid}',f'displacement:{disp:.2f}'],'INFERRED').to_dict())
        return events
