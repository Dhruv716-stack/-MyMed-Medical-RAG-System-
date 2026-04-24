from typing import List

from pre_retrieval.query_rewritter import rewrite_query
from pre_retrieval.multi_query import generate_multi_queries
from pre_retrieval.ambiguity_detector import is_ambiguous_llm


def pre_retrieval_transform(query: str) -> List[str]:

    # Step 1: rewrite query
    rewritten_query = rewrite_query(query)

    # Step 2: detect ambiguity
    ambiguous = is_ambiguous_llm(rewritten_query)

    # Step 3: generate queries
    if ambiguous:
        queries = [rewritten_query] + generate_multi_queries(rewritten_query)
    else:
        queries = [rewritten_query]

    # Step 4: deduplicate
    queries = list(dict.fromkeys(queries))

    return queries