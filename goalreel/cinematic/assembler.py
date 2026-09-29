from ..source.ffmpeg import run
from pathlib import Path

def concat_videos(inputs,out):
    Path(out).parent.mkdir(parents=True,exist_ok=True); lst=Path(out).with_suffix('.concat.txt'); lst.write_text(''.join(f"file '{Path(x).resolve()}'\n" for x in inputs)); run(['-y','-f','concat','-safe','0','-i',lst,'-c','copy',out]); return out
