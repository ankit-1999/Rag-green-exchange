import logging
from typing import List, Union

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
    response_model=Union[CreditResponse, List[CreditResponse]],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new energy credit",
)
async def create_credit(
    request: Union[CreateCreditRequest, List[CreateCreditRequest]],
) -> Union[CreditResponse, List[CreditResponse]]:
    try:
        requests = request if isinstance(request, list) else [request]
        credits = [credit_service.create_credit(item) for item in requests]
        return credits if isinstance(request, list) else credits[0]
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


@router.get(
    "/credits/created-by/{user_id}",
    response_model=List[CreditResponse],
    status_code=status.HTTP_200_OK,
    summary="List all credits created by a specific user",
)
async def list_credits_created_by_user(user_id: str) -> List[CreditResponse]:
    return credit_service.list_credits_created_by_user(user_id)


@router.get(
    "/audit/{credit_id}",
    response_model=List[CreditAuditRecord],
    status_code=status.HTTP_200_OK,
    summary="List audit trail for one credit",
)
async def list_credit_audit_by_credit_id(credit_id: str) -> List[CreditAuditRecord]:
    return credit_service.list_credit_audit_by_credit_id(credit_id)


@router.post(
    "/credit/transfer",
    response_model=Union[CreditResponse, List[CreditResponse]],
    status_code=status.HTTP_200_OK,
    summary="Transfer credit ownership",
)
async def transfer_credit(
    request: Union[CreditTransferRequest, List[CreditTransferRequest]],
) -> Union[CreditResponse, List[CreditResponse]]:
    try:
        requests = request if isinstance(request, list) else [request]
        transferred = [credit_service.transfer_credit(item) for item in requests]
        return transferred if isinstance(request, list) else transferred[0]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
