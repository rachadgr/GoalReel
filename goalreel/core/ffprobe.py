import json, subprocess
from pathlib import Path
from .types import VideoInfo

def probe(path:str)->VideoInfo:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(path)
    cmd=['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]
    data=json.loads(subprocess.check_output(cmd,text=True))
    streams=data.get('streams',[]); v=next((s for s in streams if s.get('codec_type')=='video'),None)
    if not v: raise RuntimeError('NO_VIDEO_STREAM')
    a=next((s for s in streams if s.get('codec_type')=='audio'),None)
    num,den=(v.get('r_frame_rate','0/1').split('/')+[1])[:2]
    fps=float(num)/float(den or 1)
    frames=int(v.get('nb_frames') or round(float(data['format'].get('duration',0))*fps))
    return VideoInfo(str(p),float(data['format'].get('duration',0)),int(v['width']),int(v['height']),fps,frames,v.get('codec_name',''),v.get('pix_fmt',''),a.get('codec_name') if a else None,int(a['sample_rate']) if a and a.get('sample_rate') else None,int(a['channels']) if a and a.get('channels') else None,p.stat().st_size)
