from pathlib import Path
import cv2

def build_reference_pack(video,out_dir,frames):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(video); saved=[]
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f)); ok,im=cap.read()
        if ok:
            p=out/f'frame_{int(f):06d}.jpg'; cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,96]); saved.append({'frame':int(f),'path':str(p)})
    cap.release(); return {'frames':saved,'identity_prompt_status':'EVIDENCE_ONLY'}
