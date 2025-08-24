from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
# Optional: from langchain_community.vectorstores import Chroma

def chunk_text(text: str, meta: dict, chunk_size=1200, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return [Document(page_content=c, metadata=meta) for c in splitter.split_text(text)]

def build_vectorstore(docs, embeddings, use_faiss=True):
    if use_faiss:
        return FAISS.from_documents(docs, embeddings)
    # Alternative: Chroma persistent store
    # return Chroma.from_documents(docs, embeddings, persist_directory=".chroma")
