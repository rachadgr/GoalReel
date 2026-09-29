import cv2, numpy as np

def grade_frame(frame):
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB).astype(np.float32); l,a,b=cv2.split(lab); l=np.clip((l-128)*1.06+128,0,255); out=cv2.merge([l,a,b]).astype(np.uint8); return cv2.cvtColor(out,cv2.COLOR_LAB2BGR)
