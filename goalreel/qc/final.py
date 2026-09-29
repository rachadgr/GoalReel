from ..core.ffprobe import probe

def final_qc(path,min_duration=0.1):
    info=probe(path); errors=[]
    if info.width!=1080 or info.height!=1920:errors.append('resolution')
    if abs(info.fps-30)>0.1:errors.append('fps')
    if info.video_codec!='h264':errors.append('codec')
    if info.pixel_format!='yuv420p':errors.append('pix_fmt')
    if info.duration_s<min_duration:errors.append('duration')
    return {'status':'OK' if not errors else 'QC_REJECTED','errors':errors,'video':info.to_dict()}
