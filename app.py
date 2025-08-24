import json
import streamlit as st

from utils.file_parser import extract_text
from utils.embeddings import get_embedding_model
from utils.vector_store import chunk_text, build_vectorstore
from utils.text_utils import keyword_candidates_from_jd, keyword_coverage
from utils.scorer import compute_semantic_similarity, blend_score
from chains.resume_chain import get_resume_chain

st.set_page_config(page_title="AI Resume Analyzer", layout="wide", page_icon="📄")
st.title("📄 AI Resume Analyzer (LLM + Embeddings)")

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Embeddings", ["OpenAI (default)", "Local (HF)"], index=0)
    k_neighbors = st.slider("Top-K similar chunks", 3, 12, 5)
    st.caption("Tip: Add a Job Description for best results.")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx"])
with col2:
    job_description = st.text_area("Paste Job Description (recommended)", height=220)

if uploaded_file:
    with st.spinner("🔍 Extracting resume text..."):
        resume_text = extract_text(uploaded_file)

    st.success(f"Extracted {len(resume_text)} characters.")
    if st.button("Analyze Resume", type="primary", use_container_width=True):
        # 1) Build vector index
        with st.spinner("📚 Building vector index..."):
            emb = get_embedding_model()
            resume_docs = chunk_text(resume_text, meta={"source": "resume"})
            jd_docs = chunk_text(job_description or "", meta={"source": "jd"})
            vs = build_vectorstore(resume_docs + jd_docs, emb, use_faiss=True)

        # 2) Semantic similarity (JD → Resume)
        with st.spinner("🔗 Computing semantic similarity..."):
            query = job_description if job_description else resume_text[:1500]
            sem_sim, matches = compute_semantic_similarity(vs, query=query, k=k_neighbors)

        # 3) Keyword coverage
        jd_kws = keyword_candidates_from_jd(job_description or resume_text)
        coverage, missing = keyword_coverage(resume_text, jd_kws)

        # 4) LLM scoring
        with st.spinner("🧠 LLM ATS scoring..."):
            chain = get_resume_chain()
            raw = chain.run({"resume_text": resume_text, "job_description": job_description})
            try:
                llm_json = json.loads(raw)
            except Exception:
                llm_json = {
                    "ats_score": 60,
                    "strengths": ["Clear structure"],
                    "weaknesses": ["Limited quantification of impact"],
                    "improvements": ["Add metrics to achievements", "Tailor skills to JD"],
                    "missing_keywords": []
                }

        # 5) Final blended score
        final_score = blend_score(sem_sim, coverage, llm_json.get("ats_score", 60))

        st.subheader("📊 Overall ATS Match")
        st.metric("Final Score", f"{final_score}/100")
        c1, c2, c3 = st.columns(3)
        c1.metric("Semantic Similarity", f"{round(sem_sim*100,1)}")
        c2.metric("Keyword Coverage", f"{round(coverage*100,1)}")
        c3.metric("LLM ATS Score", f"{llm_json.get('ats_score', 60)}")

        st.divider()
        st.subheader("🧩 Most Relevant Resume Snippets")
        for i, m in enumerate(matches, 1):
            st.caption(f"Snippet #{i} • relevance≈{round((m['score'] or 0)*100,1)}")
            st.write(m["text"])

        st.divider()
        st.subheader("🧠 LLM Feedback")
        st.markdown("**Strengths**")
        st.write(llm_json.get("strengths", []))
        st.markdown("**Weaknesses**")
        st.write(llm_json.get("weaknesses", []))
        st.markdown("**Improvements**")
        st.write(llm_json.get("improvements", []))

        st.divider()
        st.subheader("🔑 Keywords")
        st.markdown("**JD Keywords (top)**")
        st.write(jd_kws)
        st.markdown("**Missing in Resume (fuzzy)**")
        st.write(sorted(set(missing + llm_json.get("missing_keywords", []))))
