"""
main.py
-------
FastAPI application entry point for GreenGrid Exchange RAG AI service.

Includes:
  - CORS middleware
  - Document ingestion router
  - Health and root endpoints
  - Startup hook to ensure OpenSearch index exists
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.document_routes import router as document_router
from app.services import opensearch_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run startup tasks before the application starts accepting requests.
    """
    logger.info("Starting %s", settings.APP_NAME)
    try:
        opensearch_service.ensure_index_exists()
        logger.info("OpenSearch index ready: %s", settings.OPENSEARCH_INDEX_NAME)
    except Exception as exc:
        # Log but do not crash — app can still serve requests if OS is temporarily unavailable
        logger.error("OpenSearch startup check failed: %s", exc)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "RAG-powered AI assistant for GreenGrid Exchange. "
        "Provides document ingestion, semantic retrieval, and AI-guided "
        "answers about electricity credits, source classification, and marketplace rules."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(document_router)

# AI/RAG router will be added here in the next phase:
# from app.routes.ai_routes import router as ai_router
# app.include_router(ai_router)


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "greengrid-rag-ai",
    }
