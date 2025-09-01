from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_pdfs(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    return docs
