import cv2, numpy as np
from ..core.types import StageResult
class CameraEstimator:
    def estimate(self,video):
        cap=cv2.VideoCapture(video); ok,prev=cap.read();
        if not ok:return StageResult('camera_estimation','ERROR','Could not read source')
        prevg=cv2.resize(cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY),(320,180)); transforms=[]; i=1
        while i<min(90,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok,frame=cap.read();
            if not ok:break
            g=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(320,180))
            pts0,pts1=cv2.goodFeaturesToTrack(prevg,maxCorners=200,qualityLevel=.01,minDistance=8),None
            if pts0 is not None:
                pts1,st,_=cv2.calcOpticalFlowPyrLK(prevg,g,pts0,None)
                if pts1 is not None:
                    p0=pts0[st==1]; p1=pts1[st==1]
                    if len(p0)>=8:
                        M,_=cv2.estimateAffinePartial2D(p0,p1,method=cv2.RANSAC)
                        if M is not None: transforms.append({'frame':i,'dx':float(M[0,2]),'dy':float(M[1,2]),'scale':float((M[0,0]**2+M[1,0]**2)**0.5),'rotation_deg':float(np.degrees(np.arctan2(M[1,0],M[0,0])))})
            prevg=g;i+=1
        cap.release()
        return StageResult('camera_estimation','OK','Estimated 2D camera motion from optical flow',evidence='VISIBLE_FROM_SOURCE',metrics={'samples':len(transforms),'transforms':transforms[:50]})
