def event_speed_plan(event):
    if not event:return {'status':'UNKNOWN'}
    return {'status':'OK','segments':[{'label':'pre','factor':1.0},{'label':'peak','factor':0.5},{'label':'post','factor':1.0}],'evidence':'EVENT_LINKED'}
