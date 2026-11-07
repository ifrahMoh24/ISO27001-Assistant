# ISO 27001 Assistant

A chatbot that answers questions about **ISO/IEC 27001:2022** using **Retrieval-Augmented Generation (RAG)**. It finds the relevant Annex A controls in a local vector database, then **Claude** writes a short answer from those controls only, with numbered citations you can expand to check.

Built with **FastAPI**, **ChromaDB**, **SentenceTransformers** (`all-MiniLM-L6-v2`), the **Claude API** and a plain HTML/CSS/JavaScript chat page.

## How it works

```
Build once:  data/iso27001.txt ─► one chunk per control ─► embed (MiniLM) ─► ChromaDB

Each question:
  question ─► embed ─► nearest controls ─► drop if cosine distance > 0.80
                                                  │
                    ┌─────────────────────────────┴──────────────────────┐
              nothing left                                         numbered sources
                    │                                                    │
      "not covered by this guide"                Claude answers from the sources only, cites [n]
                                                                         │
                                                   answer + sources ─► chat page
```

- **Dataset**: all 93 Annex A controls of ISO 27001:2022 in four themes (Organizational A.5, People A.6, Physical A.7, Technological A.8), plus ISMS clauses 4–10, the Statement of Applicability and what changed from the 2013 version. The descriptions are plain-language summaries written for this project, not the official text.
- **Chunking**: one chunk per control, stored with its control number (e.g. `A.7.7`) as metadata.
- **Grounding**: Claude is instructed to answer only from the retrieved sources, cite each claim and say when the guide doesn't cover a question, instead of guessing.
- **Relevance cutoff**: on sample queries, ISO questions scored cosine distance 0.37–0.74 and off-topic ones ("What is the capital of France?") 0.85+, so anything beyond 0.80 is dropped and never reaches Claude.
- **Works without a key**: with no API key, or if the Claude call fails, the app answers with the best-matching section.

## Project structure

```
ISO27001-Assistant/
├── backend/
│   ├── main.py         # FastAPI app: retrieval, Claude answers, serves the chat page
│   └── rag_setup.py    # Builds the ChromaDB vector store
├── frontend/
│   └── index.html      # Chat page
├── data/
│   └── iso27001.txt    # ISO 27001:2022 dataset
└── requirements.txt
```

## Setup

Requires Python 3.9+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m backend.rag_setup
```

Set your Claude API key (from console.anthropic.com). Without it the app runs in search-only mode.

```bash
export ANTHROPIC_API_KEY=your-key-here
```

## Run

```bash
uvicorn backend.main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. The API docs are at <http://127.0.0.1:8000/docs>.

## API

`POST /api/ask`

```json
{ "question": "Do we need MFA?" }
```

Response:

```json
{
  "question": "Do we need MFA?",
  "answer": "ISO 27001:2022 expects secure authentication based on your access control policy, and MFA is the usual measure for critical systems and remote access [1].",
  "mode": "claude",
  "sources": [
    { "id": 1, "control": "A.8.5", "title": "A.8.5 Secure authentication (Technological controls)", "text": "...", "distance": 0.492 }
  ]
}
```

`mode` is `claude`, `search-only` (no API key), `fallback` (Claude call failed) or `none` (nothing relevant found).

`GET /api/health` reports the mode and the number of indexed chunks.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude answers |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used to write answers |
| `MAX_DISTANCE` | `0.80` | Cosine distance cutoff for relevant chunks |

## Limitations and next steps

- The dataset summarises the 93 controls; it is not the full standard or the ISO 27002 guidance.
- Each question is answered on its own, without chat history.
- Next: a small evaluation set (question → expected control) to measure retrieval accuracy, and streaming answers.
