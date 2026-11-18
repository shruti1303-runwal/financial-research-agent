"""
Ingest financial reports (PDF, .md, .txt) into pgvector.

Run:  python -m app.ingest            # add everything in data/reports
      python -m app.ingest --reset    # wipe the collection first

Pipeline: load -> split into overlapping chunks -> embed with Gemini -> store in Postgres.
Each chunk keeps metadata (source file, page, section) so answers can cite it.
"""
import argparse
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.vectorstore import get_vectorstore


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader
    title = path.stem.replace("_", " ").replace("-", " ")
    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text,
                                 metadata={"source": path.name, "page": i, "title": title}))
    return docs


def load_markdown(path: Path) -> list[Document]:
    """Split markdown by '## ' headings; treat each section as a 'page' for citations."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# (.+)$", text, flags=re.M)
    title = m.group(1).strip() if m else path.stem.replace("_", " ")
    parts = re.split(r"\n(?=## )", text)
    docs = []
    for i, part in enumerate(parts, start=1):
        heading = part.strip().splitlines()[0].lstrip("# ").strip() if part.strip() else ""
        docs.append(Document(page_content=part.strip(),
                             metadata={"source": path.name, "page": i, "section": heading,
                                       "title": title}))
    return docs


def load_reports(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() == ".pdf":
            docs += load_pdf(path)
        elif path.suffix.lower() in {".md", ".txt"}:
            docs += load_markdown(path)
    return docs


def split(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        # Contextual chunk header: a section like "## Financial Highlights" + a table of
        # numbers never names the company, so its embedding matched no company query.
        # Prefixing the document title (and section) makes every chunk self-describing.
        header = c.metadata.get("title", c.metadata["source"])
        section = c.metadata.get("section")
        if section and section != header:
            header += f" > {section}"
        c.page_content = f"[{header}]\n{c.page_content}"
    return chunks


def ingest(folder: Path = config.REPORTS_DIR, reset: bool = False, vectorstore=None) -> int:
    vs = vectorstore or get_vectorstore()
    if reset:
        vs.delete_collection()
        vs.create_collection()
    docs = load_reports(folder)
    chunks = split(docs)
    ids = [f"{c.metadata['source']}::{c.metadata['chunk_id']}" for c in chunks]
    # Batch to stay within embedding API limits
    for i in range(0, len(chunks), 50):
        vs.add_documents(chunks[i:i + 50], ids=ids[i:i + 50])
    print(f"Ingested {len(docs)} pages/sections -> {len(chunks)} chunks from {folder}")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--folder", default=str(config.REPORTS_DIR))
    a = ap.parse_args()
    ingest(Path(a.folder), reset=a.reset)
