import logging
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.schemas.credit_schema import (
    CreateCreditRequest,
    CreditAuditRecord,
    CreditResponse,
    CreditTransferRequest,
)
from app.services import credit_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Credits"])


@router.post(
    "/createcredit",
    response_model=CreditResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new energy credit",
)
async def create_credit(request: CreateCreditRequest) -> CreditResponse:
    try:
        return credit_service.create_credit(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/listings",
    response_model=List[CreditResponse],
    status_code=status.HTTP_200_OK,
    summary="List all credits",
)
async def list_credits() -> List[CreditResponse]:
    return credit_service.list_credits()


@router.get(
    "/audit",
    response_model=List[CreditAuditRecord],
    status_code=status.HTTP_200_OK,
    summary="List all credit operations",
)
async def list_credit_audit() -> List[CreditAuditRecord]:
    return credit_service.list_credit_audit()


@router.post(
    "/credit/transfer",
    response_model=CreditResponse,
    status_code=status.HTTP_200_OK,
    summary="Transfer credit ownership",
)
async def transfer_credit(request: CreditTransferRequest) -> CreditResponse:
    try:
        return credit_service.transfer_credit(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
