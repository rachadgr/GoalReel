from dataclasses import dataclass,asdict
@dataclass
class CameraKeyframe:
    frame:int; x:float; y:float; z:float; yaw:float; pitch:float; roll:float; fov:float; evidence_state:str
class VirtualCameraPlanner:
    def plan(self,width,height,target_track,event=None):
        if not target_track:return {'status':'UNKNOWN','reason':'TARGET_TRACK_REQUIRED'}
        # Reframe-only plan is source-pixel safe; novel 3D camera requires a loaded backend.
        frames=sorted(target_track,key=lambda x:x['frame']); keys=[]
        for i,t in enumerate(frames[::max(1,len(frames)//8)]):
            b=t['bbox']; cx=(b[0]+b[2])/2/width; cy=(b[1]+b[3])/2/height
            keys.append(asdict(CameraKeyframe(t['frame'],cx,cy,0,0,0,0,50,'VISIBLE_FROM_SOURCE')))
        return {'status':'OK','mode':'reframe_only','keyframes':keys,'novel_view_status':'UNAVAILABLE_UNTIL_BACKEND_CONFIGURED'}
