"""ISO 27001 Assistant: RAG chatbot for ISO/IEC 27001:2022 using Claude.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
then open http://127.0.0.1:8000
"""

import logging
import os
from pathlib import Path
from typing import List

import anthropic
import chromadb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from backend.rag_setup import BASE_DIR, COLLECTION_NAME, DB_PATH, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("iso-assistant")

# ---------- Settings ----------

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Chunks further away than this (cosine distance) are treated as unrelated.
# Measured on sample queries: ISO questions land at 0.37-0.74, off-topic
# questions ("capital of France", "overtime pay") at 0.85+.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.80"))
FRONTEND_DIR = BASE_DIR / "frontend"

SYSTEM_PROMPT = """You are an assistant that explains ISO/IEC 27001:2022 to people implementing an information security management system.

Answer using only the numbered sources in the user's message. Cite every claim with the source number in square brackets, like [1] or [2][3], and mention Annex A control numbers (for example A.7.7) when the sources give them.

If the sources do not cover the question, say that this guide does not cover it and suggest a related ISO 27001 topic the user could ask about instead. Do not fill gaps from general knowledge, and do not invent control numbers.

Keep answers short: two to five sentences, or a brief list when the user asks for steps. Write plain text without Markdown headings or bold."""

NOT_COVERED = (
    "I couldn't find anything about that in the ISO 27001 guide. "
    "Try asking about a specific control, such as access control, "
    "incident management or clear desk."
)

# ---------- Load models & DB at startup ----------

log.info("Loading embedding model %s ...", EMBEDDING_MODEL)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

db_client = chromadb.PersistentClient(path=str(DB_PATH))
try:
    collection = db_client.get_collection(name=COLLECTION_NAME)
except Exception as exc:  # the vector store was never built
    raise RuntimeError(
        "Vector store not found. Build it first with: python -m backend.rag_setup"
    ) from exc
log.info("Collection has %d chunks", collection.count())

# The SDK reads ANTHROPIC_API_KEY from the environment. Without a key the
# app still works and answers with the best-matching section.
llm_enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
claude = anthropic.Anthropic() if llm_enabled else None
log.info("Answer mode: %s", "claude" if llm_enabled else "search-only")

# ---------- API models ----------

app = FastAPI(
    title="ISO 27001 Assistant API",
    version="1.0.0",
    description="RAG chatbot for ISO/IEC 27001:2022 with answers written by Claude.",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(4, ge=1, le=10)


class Source(BaseModel):
    id: int
    control: str  # e.g. "A.7.7", or "" for general sections
    title: str
    text: str
    distance: float


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str  # "claude", "search-only", "fallback" (Claude call failed) or "none"
    sources: List[Source]


# ---------- Retrieval ----------

def retrieve(question: str, top_k: int) -> List[Source]:
    """Embed the question and return the closest chunks within MAX_DISTANCE."""
    results = collection.query(
        query_embeddings=embedding_model.encode([question]).tolist(),
        n_results=top_k,
    )
    sources: List[Source] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        if dist <= MAX_DISTANCE:
            sources.append(
                Source(
                    id=len(sources) + 1,
                    control=meta.get("control", ""),
                    title=meta.get("title", ""),
                    text=doc,
                    distance=round(dist, 3),
                )
            )
    return sources


# ---------- Generation ----------

def generate_answer(question: str, sources: List[Source]) -> str:
    """Ask Claude to answer from the retrieved sources, citing them as [n]."""
    numbered = "\n\n".join(f"[{s.id}] {s.text}" for s in sources)
    response = claude.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Sources:\n\n{numbered}\n\nQuestion: {question}"}
        ],
        # Short factual Q&A does not need deep reasoning.
        output_config={"effort": "low"},
        # If a safety classifier declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return "I can't answer that question."
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------- Endpoints ----------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "claude" if llm_enabled else "search-only",
        "chunks": collection.count(),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    sources = retrieve(question, payload.top_k)

    if not sources:
        return AskResponse(question=question, answer=NOT_COVERED, mode="none", sources=[])

    if not llm_enabled:
        return AskResponse(
            question=question, answer=sources[0].text, mode="search-only", sources=sources
        )

    try:
        return AskResponse(
            question=question,
            answer=generate_answer(question, sources),
            mode="claude",
            sources=sources,
        )
    except anthropic.AuthenticationError:
        log.error("Claude API key was rejected; answering from search only")
    except anthropic.RateLimitError:
        log.warning("Claude rate limit hit; answering from search only")
    except anthropic.APIStatusError as exc:
        log.error("Claude API error %s: %s", exc.status_code, exc.message)
    except anthropic.APIConnectionError:
        log.error("Could not reach the Claude API; answering from search only")

    return AskResponse(question=question, answer=sources[0].text, mode="fallback", sources=sources)


# The chat page is served by the same server, so there is one command to run
# and no CORS setup. Mounted last so the /api routes take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
