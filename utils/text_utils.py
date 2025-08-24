import re
from rapidfuzz import fuzz
from collections import Counter

SKILL_HINTS = [
    # generic software/ATS skills — customize per domain
    "python","java","c++","javascript","react","node","sql","nosql","aws",
    "gcp","azure","docker","kubernetes","linux","git","rest","graphql",
    "pytorch","tensorflow","sklearn","nlp","ml","dl","langchain","streamlit",
    "fastapi","flask","spark","hadoop","airflow","tableau","power bi"
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9+#.-]+", text.lower())

def keyword_candidates_from_jd(jd_text: str, top_k=25):
    # naive keyword extraction + whitelist boost
    toks = tokenize(jd_text)
    stop = set(["and","or","with","the","a","an","for","of","to","in","on","at","by","is","as","be","are","from"])
    words = [t for t in toks if t not in stop and len(t) > 2]
    counts = Counter(words)
    # promote hints if present
    for h in SKILL_HINTS:
        if h in counts:
            counts[h] += 3
    return [w for w,_ in counts.most_common(top_k)]

def keyword_coverage(resume_text: str, jd_keywords: list[str]) -> tuple[float, list[str]]:
    res_norm = normalize(resume_text)
    present = []
    missing = []
    for kw in jd_keywords:
        # fuzzy contains
        hit = fuzz.partial_ratio(kw, res_norm) >= 85
        (present if hit else missing).append(kw)
    coverage = len(present) / max(1, len(jd_keywords))
    return coverage, missing
