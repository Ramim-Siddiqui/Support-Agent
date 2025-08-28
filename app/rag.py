import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=os.getenv("GOOGLE_API_KEY")
)

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

# Build vectorstores
vectorstores = {}
for cat, texts in KB_DOCS.items():
    docs = [Document(page_content=t) for t in texts]
    vectorstores[cat] = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=f".chroma_{cat}"
    )

def retrieve(category: str, query: str, top_k: int = 3):
    retriever = vectorstores.get(category)
    if not retriever:
        return []
    results = retriever.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]
