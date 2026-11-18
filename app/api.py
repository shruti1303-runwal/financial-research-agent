"""
FastAPI backend serving the agent.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs      Chat UI: http://localhost:8000
"""
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app import config
from app.agent import ask, build_agent
from app.ingest import ingest

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the agent once at startup (launches the MCP server subprocess)
    STATE["agent"], STATE["tools"] = await build_agent()
    yield


app = FastAPI(title="Agentic Financial Research Assistant", version="1.0", lifespan=lifespan)

# Let a UI hosted elsewhere (e.g. Netlify) call this API from the browser.
if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

# Simple per-visitor limit so a public demo can't burn the whole Gemini quota.
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request):
    if config.RATE_LIMIT_PER_HOUR <= 0:
        return
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "?")
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_HOUR:
        raise HTTPException(429, f"Demo limit: {config.RATE_LIMIT_PER_HOUR} questions per hour. "
                                 "Please try again later, or run it locally from GitHub.")
    q.append(now)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What was NovaTech's FY2025 net margin?"])


class Source(BaseModel):
    citation: str
    source: str | None = None
    page: int | None = None
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    tool_calls: list[dict]
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "model": config.CHAT_MODEL, "tools": STATE.get("tools", []),
            "uploads": config.ALLOW_UPLOADS}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest, request: Request):
    _rate_limit(request)
    try:
        out = await ask(STATE["agent"], req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    out.pop("contexts", None)
    return out


@app.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    """Upload a PDF / .md / .txt report and index it."""
    if not config.ALLOW_UPLOADS:
        raise HTTPException(403, "Uploads are disabled on this public demo. Run it locally to add reports.")
    if not file.filename.lower().endswith((".pdf", ".md", ".txt")):
        raise HTTPException(400, "Only .pdf, .md or .txt files")
    dest = config.REPORTS_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    n = ingest(config.REPORTS_DIR)
    return {"saved": dest.name, "total_chunks_indexed": n}


@app.get("/config.js")
def ui_config():
    # Served here, the UI talks to this same server. The Netlify copy of config.js
    # (app/static/config.js) points at the hosted API instead.
    return Response('window.API_BASE = "";', media_type="application/javascript")


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
