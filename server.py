"""Web demo for the support-ticket agent.

    GET  /            demo page
    POST /api/resolve {"subject": "...", "description": "..."} -> full graph trace
    GET  /healthz     liveness

Run locally:  uvicorn server:app --reload
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.graph import graph  # noqa: E402

# Serverless filesystems are read-only except /tmp.
if os.getenv("VERCEL"):
    os.environ.setdefault("ESCALATION_LOG", "/tmp/escalation_log.csv")

app = FastAPI(title="Support Ticket Resolution Agent", docs_url="/docs", redoc_url=None)

# Let the portfolio site call the API from the browser (inline demo console).
_default_origins = "https://ramim-siddiqui.vercel.app,http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", _default_origins).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=86400,
)

# Simple per-IP rate limit so the public demo cannot burn the free API quota.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
_hits: dict[str, deque] = defaultdict(deque)


def _check_rate(ip: str) -> None:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Demo limit reached for this hour. Please try again later.")
    q.append(now)


class Ticket(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=2000)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/api/resolve")
def resolve(ticket: Ticket, request: Request):
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    _check_rate(ip)
    started = time.time()
    try:
        state = graph.invoke({"subject": ticket.subject, "description": ticket.description, "attempts": 0})
    except Exception as exc:  # surface model/API errors to the demo instead of a bare 500
        raise HTTPException(502, f"The agent failed: {exc.__class__.__name__}") from exc
    approved = bool((state.get("review") or {}).get("approved"))
    return {
        "outcome": "resolved" if approved else "escalated",
        "category": state.get("category"),
        "reply": state.get("draft") if approved else None,
        "context": list(dict.fromkeys(state.get("context", []))),
        "drafts": state.get("drafts", []),
        "reviews": state.get("reviews", []),
        "history": state.get("history", []),
        "attempts": state.get("attempts", 0),
        "seconds": round(time.time() - started, 1),
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return PAGE


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Support Ticket Agent · Demo</title>
<style>
  :root { --bg:#f5f7fb; --panel:#fff; --text:#17202b; --muted:#5d6a7a; --line:#dfe5ee; --accent:#3559e0; --ok:#16794a; --warn:#a15c00; }
  @media (prefers-color-scheme: dark) { :root { --bg:#0f1319; --panel:#171d26; --text:#e7ecf2; --muted:#9aa6b4; --line:#28313d; --accent:#7d9bff; --ok:#4cc38a; --warn:#f0a94b; } }
  * { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif }
  .wrap { max-width:1040px; margin:0 auto; padding:24px 16px 48px }
  h1 { font-size:22px; margin:0 0 4px } .sub { color:var(--muted); margin:0 0 20px }
  .grid { display:grid; gap:16px; grid-template-columns:1fr } @media (min-width:880px){ .grid { grid-template-columns:5fr 6fr } }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px }
  label { display:block; font-weight:600; font-size:13px; margin:0 0 6px } input, textarea { width:100%; font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:8px; padding:10px; margin-bottom:12px }
  textarea { min-height:120px; resize:vertical }
  button { font:inherit; border:0; border-radius:8px; padding:10px 16px; background:var(--accent); color:#fff; cursor:pointer } button:disabled { opacity:.6; cursor:wait }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin:0 0 14px } .chip { background:transparent; color:var(--accent); border:1px solid var(--line); padding:5px 10px; font-size:13px }
  .flow { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 14px } .step { font-size:12px; padding:3px 9px; border-radius:999px; border:1px solid var(--line); color:var(--muted) }
  .step.on { color:var(--text); border-color:var(--accent) }
  .badge { display:inline-block; font-weight:600; font-size:12px; padding:3px 10px; border-radius:999px; border:1px solid currentColor }
  .ok { color:var(--ok) } .warn { color:var(--warn) } .muted { color:var(--muted) }
  pre { white-space:pre-wrap; background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:12px; margin:8px 0 0; font:13px/1.5 ui-monospace,Consolas,monospace }
  h2 { font-size:15px; margin:16px 0 4px } ul { margin:6px 0 0; padding-left:18px } a { color:var(--accent) }
</style>
</head>
<body><div class="wrap">
  <h1>Support Ticket Resolution Agent</h1>
  <p class="sub">A LangGraph workflow with Google Gemini: it classifies a ticket, retrieves knowledge-base snippets, drafts a reply, and reviews the draft against rules before answering, retrying or escalating to a human. <a href="https://github.com/Ramim-Siddiqui/Support-Agent">Source on GitHub</a></p>
  <div class="grid">
    <div class="card">
      <div class="chips">
        <button class="chip" data-s="Charged twice this month" data-d="I see two charges for my subscription on my card this month. Can I get a refund for the duplicate?">Billing</button>
        <button class="chip" data-s="Can't log in on Android" data-d="Since the last update the Android app keeps rejecting my password and logging me out.">Technical</button>
        <button class="chip" data-s="Suspicious login alert" data-d="I got an email about a login from another country that wasn't me. I think my account was hacked.">Security</button>
        <button class="chip" data-s="Support response times" data-d="How quickly does support usually reply for Pro customers, and how can I contact you?">General</button>
      </div>
      <form id="f">
        <label for="s">Subject</label><input id="s" maxlength="200" required>
        <label for="d">Description</label><textarea id="d" maxlength="2000" required></textarea>
        <button id="go" type="submit">Run agent</button> <span id="t" class="muted"></span>
      </form>
    </div>
    <div class="card" id="out"><p class="muted">Pick an example or write a ticket, then run the agent. A run takes a few seconds; the first one after a quiet period can take up to 20 seconds while the server wakes up.</p></div>
  </div>
</div>
<script>
const $ = (q) => document.querySelector(q);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => { $("#s").value = b.dataset.s; $("#d").value = b.dataset.d; }));
$("#f").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#go").disabled = true; $("#t").textContent = "Running…";
  $("#out").innerHTML = '<p class="muted">The agent is working: classify → retrieve → draft → review…</p>';
  try {
    const r = await fetch("/api/resolve", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ subject: $("#s").value, description: $("#d").value }) });
    const data = await r.json();
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Please fill in a subject (3+ characters) and a description (10+ characters).");
    render(data);
  } catch (err) { $("#out").innerHTML = '<p class="warn">' + esc(err.message) + "</p>"; }
  finally { $("#go").disabled = false; $("#t").textContent = ""; }
});
function render(d) {
  const nodes = ["classify", "retrieval", "draft", "review", "refine", "escalate"];
  const seen = new Set(d.history.map((h) => h.split(" ")[0]));
  const flow = nodes.map((n) => '<span class="step' + (seen.has(n) ? " on" : "") + '">' + n + "</span>").join("");
  const badge = d.outcome === "resolved" ? '<span class="badge ok">Resolved</span>' : '<span class="badge warn">Escalated to a human</span>';
  $("#out").innerHTML =
    badge + ' <span class="muted">category: <b>' + esc(d.category) + "</b> · " + d.seconds + " s</span>" +
    '<div class="flow">' + flow + "</div>" +
    (d.reply ? "<h2>Reply sent to the customer</h2><pre>" + esc(d.reply) + "</pre>" : "<h2>Not sent</h2><p class=\"muted\">The draft failed review " + d.attempts + " times, so the ticket was logged for a human.</p>") +
    "<h2>Retrieved knowledge</h2><ul>" + d.context.map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
    "<h2>Review results</h2><ul>" + d.reviews.map((c) => "<li>" + esc(c) + "</li>").join("") + "</ul>" +
    "<h2>Trace</h2><pre>" + esc(d.history.join("\n")) + "</pre>";
}
</script>
</body></html>"""
