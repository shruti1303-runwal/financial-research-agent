"""
FastAPI backend serving the agent.

Run:  uvicorn app.api:app --reload
Docs: http://localhost:8000/docs      Chat UI: http://localhost:8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What was NovaTech's FY2025 net margin?"])


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
    return {"status": "ok", "model": config.CHAT_MODEL, "tools": STATE.get("tools", [])}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    try:
        out = await ask(STATE["agent"], req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    out.pop("contexts", None)
    return out


@app.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    """Upload a PDF / .md / .txt report and index it."""
    if not file.filename.lower().endswith((".pdf", ".md", ".txt")):
        raise HTTPException(400, "Only .pdf, .md or .txt files")
    dest = config.REPORTS_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    n = ingest(config.REPORTS_DIR)
    return {"saved": dest.name, "total_chunks_indexed": n}


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
