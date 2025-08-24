# utils/embeddings.py
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)
