def check_event(source_event,generated_event):
    if not source_event or not generated_event:return {'status':'UNKNOWN','reason':'MISSING_EVENT'}
    same=source_event.get('kind')==generated_event.get('kind') and source_event.get('event_id')==generated_event.get('event_id')
    return {'status':'OK' if same else 'QC_REJECTED','event_locked':same}
