def score_hero(events,track_stats=None):
    if not events:return {'status':'UNKNOWN','reason':'NO_EVIDENCE'}
    ranked=[]
    for e in events:
        s=float(e.get('confidence',0))
        if e.get('status')=='VERIFIED':s+=0.2
        ranked.append((s,e))
    ranked.sort(key=lambda x:x[0],reverse=True)
    return {'status':'OK','event':ranked[0][1],'score':ranked[0][0],'method':'evidence_score'}
