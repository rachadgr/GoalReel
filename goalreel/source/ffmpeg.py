import subprocess
from pathlib import Path

def run(args):
    return subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',*map(str,args)],check=True,capture_output=True,text=True)

def render_vertical(source,out,width=1080,height=1920,fps=30):
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    # Real source-pixel reframe: no novel-view claim.
    vf=f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={width}:{height}:flags=lanczos,setsar=1,fps={fps},format=yuv420p"
    run(['-y','-i',source,'-vf',vf,'-c:v','libx264','-preset','veryfast','-crf','18','-c:a','aac','-b:a','192k','-movflags','+faststart',out])
    return str(out)

def validate_output(path):
    from ..core.ffprobe import probe
    return probe(path)
