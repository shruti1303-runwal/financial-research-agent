# 📊 Agentic Financial Research Assistant (RAG + MCP)

An AI agent that answers questions about company financial reports **with source citations**,
pulls **live market data**, and does **exact calculations**, choosing and chaining tools on its own.

- **LangChain agent** (Gemini) that plans multi-step tool calls
- **RAG over PostgreSQL + pgvector**: reports are chunked, embedded and searched by meaning
- **MCP server** exposing market-data + calculator tools (also works in Claude Desktop / Cursor)
- **Evaluation suite**: retrieval Hit@k / MRR / Recall, LLM-judged faithfulness, key-fact accuracy, citation rate, refusal on unanswerable questions
- **FastAPI backend** with a built-in chat UI and Swagger docs

```
                       ┌──────────────── FastAPI (/ask, /upload, chat UI) ────────────────┐
 user question ──────► │  LangChain agent (Gemini)                                        │
                       │    │  decides which tools to call, loops until it can answer    │
                       │    ├─► search_financial_reports ──► pgvector (PostgreSQL)        │
                       │    └─► MCP client ──stdio──► MCP server "finance-tools"          │
                       │                               ├ get_stock_quote / price_history │
                       │                               ├ get_key_metrics  (yfinance)     │
                       │                               └ calculate / growth_rate / cagr  │
                       └──► answer + [source, page] citations + tool-call trace ─────────┘
```

## Screenshots
The chat UI shows the answer with clickable citation chips, the agent's tool calls step by step, and the
retrieved source passages with their match score. It supports light and dark mode and adapts to phone screens.

<p>
  <img src="docs/screenshots/desktop.png" alt="Answer with citations and agent steps (desktop)" width="68%">
  <img src="docs/screenshots/mobile.png" alt="Same answer on a phone" width="26%">
</p>
<img src="docs/screenshots/dark.png" alt="Dark mode" width="68%">

## Results (retrieval evaluation, 12 questions, top-k = 5)
| Metric | Score |
|---|---|
| Retrieval Hit@5 | **100%** |
| Retrieval MRR | **0.95** |
| Retrieval Recall@5 | **100%** |

**What improved it:** every chunk is prefixed with its document title and section
(e.g. `[NovaTech Solutions Ltd. — Annual Report FY2025 > Financial Highlights]`). Before this, the
numbers table never named the company and ranked 7th–8th for "NovaTech net profit"; after it, the table ranks 1st.

## Example (what a typical run looks like)
**Q:** *Compare the FY2025 net profit margins of NovaTech and Greenfield Energy.*

Agent steps:
1. `search_financial_reports("NovaTech FY2025 net profit revenue")`
2. `search_financial_reports("Greenfield Energy FY2025 net profit revenue")`
3. `financial_ratio(655, 5120, "NovaTech net margin")` → 12.79%
4. `financial_ratio(760, 8940, "Greenfield net margin")` → 8.50%

**A:** NovaTech's net margin was ~12.8% [novatech_annual_report_fy2025.md, p.3] vs Greenfield's ~8.5% [greenfield_energy_annual_report_fy2025.md, p.3]; Greenfield's is lower because of heavy interest and depreciation costs…

## Project structure
```
financial-research-agent/
├── app/
│   ├── config.py        # settings from .env
│   ├── ingest.py        # load PDF/MD → chunk → embed → pgvector
│   ├── vectorstore.py   # PGVector store + retrieval
│   ├── mcp_server.py    # MCP server: market data + calculators
│   ├── agent.py         # LangChain agent: RAG tool + MCP tools
│   ├── api.py           # FastAPI backend
│   └── static/index.html# chat UI
├── data/reports/        # financial reports to index (2 sample reports included)
├── eval/
│   ├── questions.json   # custom question set with expected sources & facts
│   └── evaluate.py      # evaluation script → eval/report.md
├── docker-compose.yml   # PostgreSQL + pgvector
└── requirements.txt
```

---

## Step-by-step: build and run it

### Step 0 — Concepts you need (read once)
- **Embedding**: a list of numbers (here 768) that represents the *meaning* of text. Similar meaning → nearby vectors.
- **RAG (Retrieval-Augmented Generation)**: instead of trusting the LLM's memory, we *retrieve* relevant report passages and make the LLM answer from them. That gives citations and fewer hallucinations.
- **pgvector**: a PostgreSQL extension that stores vectors and finds the nearest ones (`<=>` cosine distance). You get a vector DB inside a normal SQL database.
- **Agent / tool calling**: the LLM can output "call tool X with arguments Y". LangChain runs the tool, feeds the result back, and the LLM continues. This loop is what makes it *agentic*.
- **MCP (Model Context Protocol)**: an open standard for exposing tools to LLMs. You write tools once in an MCP server and any MCP client (this agent, Claude Desktop, Cursor, VS Code) can use them.

### Step 1 — Install the prerequisites
1. **Python 3.10–3.12**
2. **Docker Desktop** (to run PostgreSQL + pgvector with one command): https://www.docker.com/products/docker-desktop
3. **A free Gemini API key**: https://aistudio.google.com/apikey → "Create API key"

### Step 2 — Set up the project
```bash
cd financial-research-agent
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```
Open `.env` and paste your Gemini key into `GOOGLE_API_KEY`.

### Step 3 — Start the database
```bash
docker compose up -d
```
This starts PostgreSQL with pgvector on port 5432 (user/password `postgres`, database `finrag`).
Check it is running: `docker ps` should show `finrag-db`.

> **No Docker?** Install PostgreSQL 16 and pgvector manually, create a database `finrag`,
> run `CREATE EXTENSION vector;` and update `DATABASE_URL` in `.env`.
>
> **No install at all?** Use a free hosted Postgres such as [Neon](https://neon.tech) (pgvector built in):
> run `CREATE EXTENSION vector;` in its SQL editor and set
> `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require`.

### Step 4 — Ingest the reports
```bash
python -m app.ingest --reset
```
You'll see something like `Ingested 15 pages/sections -> 15 chunks`.

**Learn:** open `app/ingest.py`. Each document is split into ~1,000-character chunks with
150 characters of overlap (so a sentence cut at a boundary still appears whole in one chunk).
Each chunk keeps `source` + `page` metadata, which is how citations work.

**Peek inside Postgres** (optional, great for interviews):
```bash
docker exec -it finrag-db psql -U postgres -d finrag -c "SELECT document, cmetadata FROM langchain_pg_embedding LIMIT 2;"
```

### Step 5 — Test the MCP server on its own
```bash
mcp dev app/mcp_server.py
```
This opens the **MCP Inspector** in your browser. Click *Tools → List tools*, pick `cagr`, enter
`3050`, `5120`, `3` and run it → `18.85`. Try `get_stock_quote` with `INFY.NS`.

**Learn:** in `mcp_server.py`, each `@mcp.tool()` function's **docstring** becomes the tool
description the LLM reads to decide when to use it. Good docstrings = better tool selection.
Note `calculate()` uses a safe AST evaluator instead of `eval()` (which could execute any code).

### Step 6 — Ask the agent from the terminal
```bash
python -m app.agent "What is NovaTech's 3-year revenue CAGR from FY2022 to FY2025?"
```
You'll see the tool calls, the answer and the sources.

**Learn:** in `agent.py`, `build_agent()` starts the MCP server as a subprocess (stdio),
loads its tools with `MultiServerMCPClient`, adds the local RAG tool, and calls `create_agent()`.
The `SYSTEM_PROMPT` enforces rules: retrieve first, cite everything, never do mental math.

### Step 7 — Run the API + chat UI
```bash
uvicorn app.api:app --reload
```
- Chat UI: http://localhost:8000 (click the example questions)
- API docs: http://localhost:8000/docs → try `POST /ask`
- Upload your own report: `POST /upload` with a PDF (e.g., a real annual report of Infosys or TCS)

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"What are the main risks Greenfield Energy highlights?"}'
```

### Step 8 — Evaluate it
```bash
python -m eval.evaluate
```
Runs every question in `eval/questions.json` and writes `eval/report.md`:

| Metric | What it measures |
|---|---|
| Retrieval Hit@5 | Was a correct section in the top 5 chunks? |
| MRR | How high the first correct chunk ranked |
| Faithfulness | LLM judge: share of answer claims supported by retrieved context/tool output |
| Key-fact accuracy | Does the answer contain the required numbers? |
| Citation rate | Did the answer cite a source? |
| Refusal accuracy | On an unanswerable question, did it say so instead of inventing? |

Copy your `eval/report.md` table into this README: real numbers impress reviewers.
Then experiment: change `CHUNK_SIZE`/`TOP_K` in `.env`, re-ingest, re-evaluate, and note what changed.

> The free Gemini tier has per-minute limits. If you see `429` errors during evaluation, wait a minute and re-run.

### Step 9 — Bonus: use your MCP server in Claude Desktop
Copy `claude_desktop_config.example.json` into Claude Desktop's config (Settings → Developer →
Edit Config), fix the paths, restart. Your finance tools now work inside Claude too, which shows
you understand MCP is a *standard*, not just a library.

### Step 10 — Add real reports
Put PDFs of real annual reports in `data/reports/` (most listed companies publish them on their
investor-relations page), run `python -m app.ingest --reset`, and add 5–10 questions about them
to `eval/questions.json`.

---

## How to explain it in an interview (30 seconds)
"I built a LangChain agent on Gemini that answers questions over company financial reports.
Reports are chunked and embedded into PostgreSQL with pgvector, and the agent retrieves passages
and cites file and page. Market data and calculators are exposed through an MCP server, so the
agent can chain steps, like retrieving revenue from two reports and calling a ratio tool to
compare margins, and the same tools also work in Claude Desktop. I evaluated retrieval with
Hit@k and MRR and answers with an LLM-judged faithfulness score on a custom question set,
including an unanswerable question to test refusal, and served it with FastAPI."

**Likely questions:** Why chunk overlap? · How do you stop hallucinations? (retrieve-first prompt,
citations, calculator tools, faithfulness eval, refusal test) · Why MCP instead of plain
LangChain tools? (reusable across clients, process isolation, standard schema) · How would you
improve retrieval? (hybrid BM25 + vector search, re-ranking, metadata filters by company/year).
