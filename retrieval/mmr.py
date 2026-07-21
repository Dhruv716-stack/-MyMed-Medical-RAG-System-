from typing import Any, Dict, Optional

from vectorstore.store import get_vectorstore
from retrieval.hybrid import _build_tenant_filter

def get_mmr_retriever(user_id: Optional[str] = None, session_id: Optional[str] = None, restrict_to_user_upload: bool = False):
    vectorstore = get_vectorstore()

    search_kwargs: Dict[str, Any] = {"k": 8, "fetch_k": 20}  # v2 change (was k=5)

    tenant_filter = _build_tenant_filter(user_id, session_id, restrict_to_user_upload=restrict_to_user_upload)
    if tenant_filter is not None:
        search_kwargs["filter"] = tenant_filter

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    return retriever