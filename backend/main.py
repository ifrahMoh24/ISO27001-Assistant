"""ISO 27001 Assistant: RAG chatbot for ISO/IEC 27001:2022 using Claude.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
then open http://127.0.0.1:8000
"""

import json
import logging
import os
from typing import Dict, Iterator, List, Literal

import anthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.rag_setup import BASE_DIR
from backend.retrieval import Source, collection, search

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("iso-assistant")

# ---------- Settings ----------

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
TOP_K = 4
MAX_HISTORY_MESSAGES = 6  # earlier turns sent to Claude (3 questions + 3 answers)
FRONTEND_DIR = BASE_DIR / "frontend"

SYSTEM_PROMPT = """You are an assistant that explains ISO/IEC 27001:2022 to people implementing an information security management system.

Answer the latest question using only the numbered sources in the latest user message. Cite every claim with the source number in square brackets, like [1] or [2][3], and mention Annex A control numbers (for example A.7.7) when the sources give them. Earlier messages are there so you understand follow-up questions; citation numbers in earlier answers referred to earlier sources, so do not reuse them.

If the sources do not cover the question, say that this guide does not cover it and suggest a related ISO 27001 topic the user could ask about instead. Do not fill gaps from general knowledge, and do not invent control numbers.

Keep answers short: two to five sentences, or a brief list when the user asks for steps. Write plain text without Markdown headings or bold."""

NOT_COVERED = (
    "I couldn't find anything about that in the ISO 27001 guide. "
    "Try asking about a specific control, such as access control, "
    "incident management or clear desk."
)

# The SDK reads ANTHROPIC_API_KEY from the environment. Without a key the
# app still works and answers with the best-matching section.
llm_enabled = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
claude = anthropic.Anthropic() if llm_enabled else None
log.info("Answer mode: %s", "claude" if llm_enabled else "search-only")

app = FastAPI(
    title="ISO 27001 Assistant API",
    version="1.1.0",
    description="RAG chatbot for ISO/IEC 27001:2022 with answers written by Claude.",
)

# ---------- API models ----------


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[Message] = Field(default_factory=list, max_length=40)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(TOP_K, ge=1, le=10)


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: str  # "claude", "search-only", "fallback" (Claude call failed) or "none"
    sources: List[Source]


# ---------- Retrieval with conversation context ----------

def retrieve_for_chat(message: str, history: List[Message]) -> List[Source]:
    """
    Search with the new message, and also with the previous question plus the
    new message, so follow-ups like "what about remote workers?" still find
    the right controls. Results are merged, keeping each chunk's best score.
    """
    queries = [message]
    previous_questions = [m.content for m in history if m.role == "user"]
    if previous_questions:
        queries.append(previous_questions[-1] + " " + message)

    best: Dict[str, Source] = {}
    for query in queries:
        for s in search(query, TOP_K):
            if s.text not in best or s.distance < best[s.text].distance:
                best[s.text] = s

    merged = sorted(best.values(), key=lambda s: s.distance)[:TOP_K]
    for i, s in enumerate(merged, start=1):
        s.id = i
    return merged


def build_messages(message: str, history: List[Message], sources: List[Source]) -> list:
    """Recent history (starting with a user turn) plus the new question with its sources."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    while recent and recent[0].role != "user":
        recent = recent[1:]

    numbered = "\n\n".join(f"[{s.id}] {s.text}" for s in sources)
    return [{"role": m.role, "content": m.content} for m in recent] + [
        {"role": "user", "content": f"Sources:\n\n{numbered}\n\nQuestion: {message}"}
    ]


CLAUDE_PARAMS = dict(
    model=CLAUDE_MODEL,
    max_tokens=16000,
    system=SYSTEM_PROMPT,
    # Short factual Q&A does not need deep reasoning.
    output_config={"effort": "low"},
    # If a safety classifier declines, the API retries on a fallback model.
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
)

# ---------- Endpoints ----------


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "claude" if llm_enabled else "search-only",
        "chunks": collection.count(),
    }


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat")
def chat(payload: ChatRequest):
    """
    Streams the answer as Server-Sent Events:
      sources  {mode, sources}   once, before the answer
      delta    {text}            pieces of the answer as Claude writes them
      done     {mode}            the final mode ("claude" or "fallback")
    """
    message = payload.message.strip()
    sources = retrieve_for_chat(message, payload.history)

    def events() -> Iterator[str]:
        if not sources:
            yield sse("sources", {"mode": "none", "sources": []})
            yield sse("delta", {"text": NOT_COVERED})
            yield sse("done", {"mode": "none"})
            return

        source_data = [s.model_dump() for s in sources]
        if not llm_enabled:
            yield sse("sources", {"mode": "search-only", "sources": source_data})
            yield sse("delta", {"text": sources[0].text})
            yield sse("done", {"mode": "search-only"})
            return

        yield sse("sources", {"mode": "claude", "sources": source_data})
        sent_text = False
        try:
            with claude.beta.messages.stream(
                messages=build_messages(message, payload.history, sources), **CLAUDE_PARAMS
            ) as stream:
                for text in stream.text_stream:
                    sent_text = True
                    yield sse("delta", {"text": text})
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield sse("replace", {"text": "I can't answer that question."})
            yield sse("done", {"mode": "claude"})
        except anthropic.APIError as exc:
            log.error("Claude API error: %s", exc)
            if sent_text:
                yield sse("delta", {"text": "\n\n(The answer was cut off by a connection error.)"})
                yield sse("done", {"mode": "claude"})
            else:
                yield sse("delta", {"text": sources[0].text})
                yield sse("done", {"mode": "fallback"})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    """Single question, single JSON answer (no history, no streaming)."""
    question = payload.question.strip()
    sources = search(question, payload.top_k)

    if not sources:
        return AskResponse(question=question, answer=NOT_COVERED, mode="none", sources=[])
    if not llm_enabled:
        return AskResponse(
            question=question, answer=sources[0].text, mode="search-only", sources=sources
        )

    try:
        response = claude.beta.messages.create(
            messages=build_messages(question, [], sources), **CLAUDE_PARAMS
        )
        if response.stop_reason == "refusal":
            answer = "I can't answer that question."
        else:
            answer = "".join(b.text for b in response.content if b.type == "text").strip()
        return AskResponse(question=question, answer=answer, mode="claude", sources=sources)
    except anthropic.APIError as exc:
        log.error("Claude API error: %s", exc)
        return AskResponse(
            question=question, answer=sources[0].text, mode="fallback", sources=sources
        )


# The chat page is served by the same server, so there is one command to run
# and no CORS setup. Mounted last so the /api routes take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
