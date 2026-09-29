from pathlib import Path
import cv2, numpy as np

def analyze_video(path,out_dir,every_n=15):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError('VIDEO_OPEN_FAILED')
    idx=0; sharp=[]; motion=[]; prev=None; frames=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if idx%every_n==0:
            lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            p=out/f'frame_{idx:06d}.jpg'; cv2.imwrite(str(p),frame,[cv2.IMWRITE_JPEG_QUALITY,95])
            sharp.append({'frame':idx,'laplacian_variance':lap}); frames.append(str(p))
        if prev is not None: motion.append(float(cv2.absdiff(gray,prev).mean()))
        prev=gray; idx+=1
    cap.release()
    return {'frames_read':idx,'sampled_frames':frames,'sharpness':sharp,'motion_mean':float(np.mean(motion)) if motion else 0.0,'motion_p95':float(np.percentile(motion,95)) if motion else 0.0}

def temporal_window(center_frame,fps,total_frames,pre_s=2.0,post_s=2.0):
    pre=max(0,int(center_frame-pre_s*fps)); post=min(total_frames-1,int(center_frame+post_s*fps))
    return {'start_frame':pre,'peak_frame':int(center_frame),'end_frame':post,'pre_seconds':(center_frame-pre)/fps,'post_seconds':(post-center_frame)/fps}
