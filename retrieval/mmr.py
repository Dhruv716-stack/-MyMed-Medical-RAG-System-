from vectorstore.store import get_vectorstore

def get_mmr_retriever():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":5,"fetch_k":20,"lambda_mult":0.5})
    
    return retriever