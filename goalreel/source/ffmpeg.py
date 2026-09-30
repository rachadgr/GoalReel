import subprocess
from pathlib import Path

import numpy as np


def run(args):
    return subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',*map(str,args)],check=True,capture_output=True,text=True)


def render_vertical(source,out,width=1080,height=1920,fps=30,reframe=None):
    """Rendu vertical 9:16.

    Deux modes, choisis par la *preuve* disponible :

      * ``reframe`` fourni ET contenant de vraies cibles suivies
        (``reframe["targets"]`` non vide) => la caméra 9:16 **SUIT réellement**
        les coordonnées des joueurs/ballon détectés et suivis (pan horizontal
        par frame, lissé). C'est le comportement requis dès que la preuve de
        suivi existe.
      * ``reframe`` absent / sans cible => fallback déterministe : recadrage
        statique centré via FFmpeg (aucune prétention de suivi). Ce fallback
        n'est utilisé QUE lorsque la preuve de suivi est indisponible.

    Aucun « novel-view » n'est jamais revendiqué : on ne fait que recadrer de
    vrais pixels source.
    """
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    targets = (reframe or {}).get("targets") if reframe else None
    if targets:
        return _render_vertical_tracked(source,out,width,height,fps,targets)
    # --- Fallback (preuve de suivi indisponible) : recadrage statique centré ---
    # Real source-pixel reframe: no novel-view claim.
    vf=f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={width}:{height}:flags=lanczos,setsar=1,fps={fps},format=yuv420p"
    run(['-y','-i',source,'-vf',vf,'-c:v','libx264','-preset','veryfast','-crf','18','-c:a','aac','-b:a','192k','-movflags','+faststart',out])
    return str(out)


def _render_vertical_tracked(source,out,width,height,fps,targets):
    """Recadrage vertical piloté par les coordonnées suivies (pan horizontal).

    ``targets`` : mapping ``frame_idx -> cx`` (centre horizontal du sujet suivi,
    en pixels source). Les trous sont comblés par maintien de la dernière
    position connue, puis les positions sont lissées (EMA) pour éviter les
    saccades. Le crop garde toute la hauteur et suit le sujet en X.
    """
    import cv2

    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("VIDEO_OPEN_FAILED")
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps=cap.get(cv2.CAP_PROP_FPS) or fps
    out_fps=fps or src_fps

    # Crop 9:16 : toute la hauteur, largeur = H*9/16 (clampée à la source).
    crop_h=H
    crop_w=int(round(H*width/height))
    if crop_w>W:
        crop_w=W
    half=crop_w/2.0
    xmax=max(0,W-crop_w)

    # Ordre des clés triées pour un maintien cohérent.
    ckeys=sorted(int(k) for k in targets)
    alpha=0.25  # EMA de lissage
    smooth=None

    def cx_at(idx):
        # Maintien de la dernière position connue <= idx ; sinon première connue.
        cand=None
        for k in ckeys:
            if k<=idx:
                cand=targets[k]
            else:
                break
        if cand is None and ckeys:
            cand=targets[ckeys[0]]
        return cand

    proc=subprocess.Popen(
        ['ffmpeg','-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(out_fps),'-i','pipe:0',
         '-i',str(source),
         '-map','0:v:0','-map','1:a?',
         '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-r',str(out_fps),
         '-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart',str(out)],
        stdin=subprocess.PIPE)
    assert proc.stdin is not None
    idx=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            cx=cx_at(idx) if ckeys else None
            if cx is None:
                cx=W/2.0
            if smooth is None:
                smooth=float(cx)
            else:
                smooth=alpha*float(cx)+(1-alpha)*smooth
            x0=int(round(smooth-half))
            x0=max(0,min(xmax,x0))
            crop=frame[0:crop_h,x0:x0+crop_w]
            if crop.shape[0]!=crop_h or crop.shape[1]!=crop_w:
                crop=cv2.resize(frame,(crop_w,crop_h),interpolation=cv2.INTER_LINEAR)
            out_frame=cv2.resize(crop,(width,height),interpolation=cv2.INTER_LANCZOS4)
            try:
                proc.stdin.write(np.ascontiguousarray(out_frame).tobytes())
            except (BrokenPipeError, OSError):
                # ffmpeg a pu s'arrêter tôt (ex. ``-shortest`` lorsque la piste
                # audio est plus courte que la vidéo). On arrête proprement la
                # lecture au lieu de propager une erreur de pipe, puis on laisse
                # le ``finally`` réconcilier le processus.
                break
            idx+=1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
    if proc.returncode!=0:
        raise RuntimeError(f"FFMPEG_TRACKED_RENDER_FAILED rc={proc.returncode}")
    return str(out)


def validate_output(path):
    from ..core.ffprobe import probe
    return probe(path)
