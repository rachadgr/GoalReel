class ShotPlanner:
    def plan(self,event,scene):
        if not event or event.get('status')=='UNKNOWN':return {'status':'UNKNOWN','shots':[]}
        return {'status':'OK','shots':[{'shot_id':'SHOT_001','event_id':event.get('event_id'),'type':'tracking','evidence':'EVENT_SUPPORTED'}]}
