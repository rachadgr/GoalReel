import numpy as np
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)
@dataclass
class Track: track_id:int; bbox:np.ndarray; score:float; hits:int=1; lost:int=0
class ByteTrack:
    """Compact two-stage ByteTrack-style association; production tracker interface."""
    def __init__(self,high=0.5,low=0.1,match=0.3,max_lost=30): self.high=high; self.low=low; self.match=match; self.max_lost=max_lost; self.tracks=[]; self.next_id=1
    def _assoc(self,tracks,dets,thr):
        if not tracks or not dets:return [],list(range(len(tracks))),list(range(len(dets)))
        cost=np.array([[1-iou(t.bbox,d['bbox']) for d in dets] for t in tracks],dtype=np.float32)
        rr,cc=linear_sum_assignment(cost); mt=[]; ut=set(range(len(tracks))); ud=set(range(len(dets)))
        for r,c in zip(rr,cc):
            if 1-cost[r,c]>=thr: mt.append((r,c)); ut.discard(r); ud.discard(c)
        return mt,list(ut),list(ud)
    def update(self,dets):
        high=[d for d in dets if d.get('confidence',0)>=self.high]; low=[d for d in dets if self.low<=d.get('confidence',0)<self.high]
        m,ut,ud=self._assoc(self.tracks,high,self.match); updated=set(); used_high=set()
        for ti,di in m:
            t=self.tracks[ti]; t.bbox=np.asarray(high[di]['bbox'],float); t.score=high[di]['confidence']; t.lost=0; t.hits+=1; updated.add(ti); used_high.add(di)
        remaining=[self.tracks[i] for i in ut]; m2,_,_=self._assoc(remaining,low,self.match*0.8); used_low=set()
        for ri,di in m2:
            t=remaining[ri]; t.bbox=np.asarray(low[di]['bbox'],float); t.score=low[di]['confidence']; t.lost=0; t.hits+=1; updated.add(ut[ri]); used_low.add(di)
        for i,t in enumerate(self.tracks):
            if i not in updated:t.lost+=1
        for di,d in enumerate(high):
            if di not in used_high:self.tracks.append(Track(self.next_id,np.asarray(d['bbox'],float),float(d['confidence']))); self.next_id+=1
        self.tracks=[t for t in self.tracks if t.lost<=self.max_lost]
        return [{'track_id':t.track_id,'bbox':t.bbox.tolist(),'confidence':t.score,'hits':t.hits,'lost':t.lost} for t in self.tracks]
