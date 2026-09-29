ALLOWED={'VISIBLE_FROM_SOURCE','TEMPORALLY_INFERRED','DEPTH_INFERRED','MULTI_VIEW_SUPPORTED','GENERATED_ESTIMATE','UNKNOWN'}
def evidence(state):
    if state not in ALLOWED: raise ValueError(state)
    return state
