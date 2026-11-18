"""Central configuration, read from the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# "gemini-flash-latest" always points at Google's current Flash model.
# You can pin a specific one (see https://ai.google.dev/gemini-api/docs/models).
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-flash-latest")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/finrag")
COLLECTION = os.getenv("COLLECTION", "financial_reports")

REPORTS_DIR = ROOT / "data" / "reports"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

MCP_SERVER_PATH = ROOT / "app" / "mcp_server.py"

# Public deployment settings
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
ALLOW_UPLOADS = os.getenv("ALLOW_UPLOADS", "true").lower() == "true"
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "0"))   # 0 = unlimited
