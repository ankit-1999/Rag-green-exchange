import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.query_schema import QueryRequest, QueryResponse
from app.services import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a question using indexed document chunks",
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    try:
        return rag_service.answer_question(request)
    except ValueError as exc:
        logger.warning("Invalid query request: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("RAG upstream failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream service error: {exc}",
        ) from exc
