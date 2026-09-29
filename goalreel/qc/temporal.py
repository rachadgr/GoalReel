import cv2, numpy as np

def sequence_metrics(video):
    cap=cv2.VideoCapture(video); prev=None; diffs=[]; frames=0
    while True:
        ok,im=cap.read()
        if not ok:break
        g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        if prev is not None:diffs.append(float(cv2.absdiff(g,prev).mean()))
        prev=g;frames+=1
    cap.release();
    return {'status':'OK','frames':frames,'mean_frame_diff':float(np.mean(diffs)) if diffs else 0.0,'p99_frame_diff':float(np.percentile(diffs,99)) if diffs else 0.0}
