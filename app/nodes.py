from __future__ import annotations
import os, csv, pathlib
from langchain_google_genai import ChatGoogleGenerativeAI
from .types import State, Category
from .rag import retrieve

CATEGORY_KEYWORDS: dict[Category, list[str]] = {
    "Billing": ["invoice", "refund", "charge", "payment", "bill", "credit"],
    "Technical": ["error", "bug", "login", "password", "app", "crash"],
    "Security": ["breach", "hacked", "phishing", "suspicious", "2fa", "security"],
    "General": ["question", "how", "help", "support", "info", "policy", "sla"],
}

def classify(state: State) -> dict:
    text = f"{state.get('subject','')} {state.get('description','')}".lower()
    best_category, best_score = "General", 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for k in kws if k in text)
        if score > best_score:
            best_category, best_score = cat, score
    return {"category": best_category, "history": [f"classify -> {best_category}"]}

def retrieval(state: State) -> dict:
    query = f"{state.get('subject','')} {state.get('description','')}"
    category = state.get("category", "General")
    docs = retrieve(category, query, top_k=3)
    return {"queries": [query], "context": docs, "history": [f"retrieval -> {len(docs)} docs"]}

def draft(state: State) -> dict:
    context = state.get("context", [])
    ticket = {"subject": state.get("subject",""), "description": state.get("description","")}

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = (
        "You are a helpful support agent. "
        "Write a concise, friendly reply grounded ONLY in the context. "
        "If the user asks for a refund, do NOT promise refunds; "
        "say you will escalate to Billing for approval.\n\n"
        f"Ticket: {ticket}\n\nContext:\n- " + "\n- ".join(context)
    )
    msg = llm.invoke(prompt).content

    return {"draft": msg, "drafts": [msg], "history": ["draft -> generated (Gemini)"]}

def review(state: State) -> dict:
    draft = (state.get("draft") or "").lower()
    feedback, approved = [], True

    if "refund" in draft and "escalate" not in draft and "approval" not in draft:
        approved, feedback = False, ["Do not promise refunds; escalate to Billing."]

    if "hi" not in draft and "hello" not in draft:
        approved, feedback = False, feedback + ["Start with a friendly greeting."]

    if "- " not in draft:
        approved, feedback = False, feedback + ["Ground the answer in retrieved context bullets."]

    if len(draft.strip()) < 40:
        approved, feedback = False, feedback + ["Draft too short; add next steps."]

    attempts = int(state.get("attempts", 0))
    return {
        "review": {"approved": approved, "feedback": "\n".join(feedback)},
        "reviews": [("approved" if approved else "rejected") + ": " + ("; ".join(feedback) or "ok")],
        "history": [f"review -> {'approved' if approved else 'rejected'}"],
        "attempts": attempts + (0 if approved else 1)
    }

def refine(state: State) -> dict:
    fb = (state.get("review", {}) or {}).get("feedback", "")
    extra_terms = " ".join([w for w in fb.split() if len(w) > 5])
    new_query = f"{state.get('subject','')} {state.get('description','')} {extra_terms}"
    category = state.get("category", "General")
    docs = retrieve(category, new_query, top_k=3)
    return {"queries": [new_query], "context": docs, "history": ["refine -> new docs from feedback"]}

def escalate(state: State) -> dict:
    path = pathlib.Path(__file__).resolve().parents[1] / "data/escalation_log.csv"
    exists = path.exists()
    row = {
        "subject": state.get("subject",""),
        "description": state.get("description",""),
        "category": state.get("category",""),
        "attempts": state.get("attempts", 0),
        "feedback": (state.get("review", {}) or {}).get("feedback",""),
        "drafts": " || ".join(state.get("drafts", [])),
        "reviews": " || ".join(state.get("reviews", [])),
        "queries": " | ".join(state.get("queries", [])),
        "context": " | ".join(state.get("context", [])),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: writer.writeheader()
        writer.writerow(row)
    return {"history": [f"escalate -> logged to {path.name}"]}
