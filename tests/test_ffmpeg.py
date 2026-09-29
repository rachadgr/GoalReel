from pathlib import Path
from goalreel.source.ffmpeg import render_vertical
from goalreel.qc.final import final_qc

def test_real_render(tmp_path):
    src=Path('/mnt/data/1000006866.mp4')
    if not src.exists():return
    out=tmp_path/'out.mp4'; render_vertical(str(src),str(out)); q=final_qc(str(out)); assert q['status']=='OK'
