from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from rag.llm import groq_chat

SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer using ONLY the retrieved context. "
    "Cite sources inline with [filename, page]. If unsure, say you don't know."
)

QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source","unknown")
        page = d.metadata.get("page","?")
        parts.append(f"[source: {src}, p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

class QAChain:
    def __init__(self, retriever, model_name: str, temperature: float = 0.2):
        self.retriever = retriever
        self.llm = groq_chat(model_name, temperature)

    def invoke(self, inputs: Dict[str, str]):
        question = inputs["question"]
        context_docs = self.retriever.get_relevant_documents(question)
        context = _format_docs(context_docs)
        messages = QA_TEMPLATE.format_messages(question=question, context=context)
        answer = self.llm.invoke(messages).content
        return {
            "answer": answer,
            "sources": context_docs,
            "context": context_docs
        }


def build_qa_chain(retriever, model_name: str, temperature: float = 0.2) -> QAChain:
    return QAChain(retriever, model_name, temperature)

