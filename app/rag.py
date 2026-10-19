import os
import threading

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Example KB (replace with real docs later)
KB_DOCS = {
    "Billing": [
        "Refunds require approval from Billing; never promise refunds in replies.",
        "Invoices are generated on the 1st of each month; duplicate charges can occur if a retry happens."
    ],
    "Technical": [
        "Known login issue in v3.2.1 Android app; fix in v3.2.2. Workaround: clear cache.",
        "Password reset requires email confirmation. Check spam if not received."
    ],
    "Security": [
        "Advise immediate password reset + enable 2FA for suspicious logins.",
        "Forward phishing reports with full headers to security@."
    ],
    "General": [
        "Support SLA: 1 business day for Pro, 4 hours for Enterprise.",
        "Contact via chat, email, or portal."
    ]
}

# One in-memory vector store per category. The knowledge base is tiny, so it is
# embedded once per process on first use (no disk needed, which also makes the
# agent work on read-only serverless filesystems).
_vectorstores: dict[str, InMemoryVectorStore] = {}
_lock = threading.Lock()


def _build() -> None:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    for cat, texts in KB_DOCS.items():
        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents([Document(page_content=t) for t in texts])
        _vectorstores[cat] = store


def retrieve(category: str, query: str, top_k: int = 3):
    if not _vectorstores:
        with _lock:
            if not _vectorstores:
                _build()
    store = _vectorstores.get(category)
    if not store:
        return []
    results = store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]
