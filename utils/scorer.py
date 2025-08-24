from typing import Dict, List, Tuple
import numpy as np

def scale01(x, lo=0.0, hi=1.0):
    x = max(lo, min(hi, x))
    return x

def compute_semantic_similarity(vectorstore, query: str, k=5) -> Tuple[float, List[Dict]]:
    """Return top-k docs and a similarity score in [0,1]."""
    # similarity_search_with_relevance_scores returns (doc, score in 0..1)
    # If not available, fallback to similarity_search and compute proxy.
    try:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # Average relevance (already 0..1)
        rels = [r[1] for r in results] or [0.0]
        score = float(np.mean(rels))
        matches = [{"text": r[0].page_content, "score": r[1], "meta": r[0].metadata} for r in results]
        return score, matches
    except Exception:
        docs = vectorstore.similarity_search(query, k=k)
        # crude proxy if .with_relevance not supported
        score = 0.6 if docs else 0.0
        matches = [{"text": d.page_content, "score": None, "meta": d.metadata} for d in docs]
        return score, matches

def blend_score(sem_sim: float, kw_cov: float, llm_score_0_100: float) -> float:
    sem = scale01(sem_sim)        # 0..1
    kw  = scale01(kw_cov)         # 0..1
    llm = scale01(llm_score_0_100 / 100.0)
    final = 0.45*sem + 0.25*kw + 0.30*llm
    return round(final * 100, 1)
