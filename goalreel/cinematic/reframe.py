import cv2, numpy as np

def crop_to_target(frame,bbox,target_w=1080,target_h=1920,pad=2.2):
    h,w=frame.shape[:2]; x1,y1,x2,y2=map(float,bbox); cx=(x1+x2)/2; cy=(y1+y2)/2; bw=max(2,x2-x1)*pad; bh=max(2,y2-y1)*pad
    ar=target_w/target_h
    if bw/bh>ar: bh=bw/ar
    else: bw=bh*ar
    x1=max(0,int(cx-bw/2)); x2=min(w,int(cx+bw/2)); y1=max(0,int(cy-bh/2)); y2=min(h,int(cy+bh/2))
    crop=frame[y1:y2,x1:x2]
    if crop.size==0:return None
    return cv2.resize(crop,(target_w,target_h),interpolation=cv2.INTER_LANCZOS4)
