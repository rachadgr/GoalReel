from collections import defaultdict
class BallTracker:
    def __init__(self): self.history=[]
    def update(self,frame,detections):
        balls=[d for d in detections if d.get('class_name')=='ball' or d.get('class_id') in (-1,32)]
        item={'frame':frame,'candidates':balls,'status':'VISIBLE_FROM_SOURCE' if balls else 'UNKNOWN'}
        self.history.append(item); return item
