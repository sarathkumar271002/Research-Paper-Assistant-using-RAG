from typing import Iterable
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

def get_or_create_chroma(collection_name: str, persist_directory: str) -> Chroma:
    return Chroma(collection_name=collection_name, persist_directory=persist_directory)

def add_documents(vs: Chroma, docs: Iterable[Document], embeddings):
    vs._embedding_function = embeddings
    vs.add_documents(list(docs))

def as_retriever(vs: Chroma, k=4, use_mmr=True):
    if use_mmr:
        return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k*3, 12)})
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
