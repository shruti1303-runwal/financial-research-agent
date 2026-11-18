"""pgvector store: embeddings live inside PostgreSQL."""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine

from app import config


@lru_cache
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        output_dimensionality=config.EMBEDDING_DIM,
        api_key=config.GOOGLE_API_KEY or None,
    )


@lru_cache
def get_engine():
    # pool_pre_ping: test each pooled connection before use and reconnect if it died.
    # Serverless Postgres (e.g. Neon) closes idle connections, which otherwise causes
    # "SSL connection has been closed unexpectedly" on the first query after a pause.
    return create_engine(config.DATABASE_URL, pool_pre_ping=True, pool_recycle=300)


@lru_cache
def get_vectorstore(embeddings=None) -> PGVector:
    return PGVector(
        embeddings=embeddings or get_embeddings(),
        collection_name=config.COLLECTION,
        connection=get_engine(),
        use_jsonb=True,
    )


def format_citation(doc: Document) -> str:
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    return f"{src}, p.{page}" if page is not None else src


def retrieve(query: str, k: int = None) -> list[tuple[Document, float]]:
    """Return the top-k chunks with their cosine distance (lower = closer)."""
    return get_vectorstore().similarity_search_with_score(query, k=k or config.TOP_K)
