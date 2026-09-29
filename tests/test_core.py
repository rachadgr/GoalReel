from pathlib import Path
from goalreel.core.ffprobe import probe
from goalreel.vision.bytetrack import ByteTrack
from goalreel.events.hero import score_hero

def test_probe():
    p=Path('/mnt/data/1000006866.mp4')
    if p.exists():
        i=probe(str(p)); assert i.width==1024 and i.height==576 and round(i.fps)==30

def test_tracker_creates_stable_id():
    t=ByteTrack(high=.2)
    a=t.update([{'bbox':[0,0,10,10],'confidence':.9}]); b=t.update([{'bbox':[1,0,11,10],'confidence':.9}])
    assert a[0]['track_id']==b[0]['track_id']

def test_hero_without_evidence_unknown():
    assert score_hero([])['status']=='UNKNOWN'
