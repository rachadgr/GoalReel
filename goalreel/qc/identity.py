def check_identity(source_embeddings=None,generated_embeddings=None,threshold=.55):
    if source_embeddings is None or generated_embeddings is None:return {'status':'UNKNOWN','reason':'NO_EMBEDDINGS'}
    return {'status':'OK' if source_embeddings>=threshold else 'QC_REJECTED','similarity':float(source_embeddings)}
