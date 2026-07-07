from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4

from app.schemas.credit_schema import (
    CreateCreditRequest,
    CreditAuditRecord,
    CreditResponse,
    CreditTransferRequest,
)
from app.services import user_service

_credit_store: Dict[str, CreditResponse] = {}
_credit_audit_store: List[CreditAuditRecord] = []
_credit_code_counter = 100


def _next_credit_code() -> str:
    global _credit_code_counter
    _credit_code_counter += 1
    return f"EC-{_credit_code_counter}"


def create_credit(request: CreateCreditRequest) -> CreditResponse:
    if user_service.get_user(request.user_id) is None:
        raise ValueError(f"User not found: {request.user_id}")

    credit = CreditResponse(
        credit_id=f"credit_{uuid4().hex[:8]}",
        credit_code=_next_credit_code(),
        user_id=request.user_id,
        credit_type=request.credit_type,
        price=float(request.price),
        created_at=datetime.now(timezone.utc),
    )
    _credit_store[credit.credit_id] = credit

    _credit_audit_store.append(
        CreditAuditRecord(
            event_id=f"evt_{uuid4().hex[:8]}",
            operation="create",
            credit_id=credit.credit_id,
            source_user_id=request.user_id,
            destination_user_id=request.user_id,
            created_at=datetime.now(timezone.utc),
            details=f"Created {credit.credit_type.value} credit at price {credit.price}",
        )
    )

    return credit


def list_credits() -> List[CreditResponse]:
    return list(_credit_store.values())


def list_credit_audit() -> List[CreditAuditRecord]:
    return list(_credit_audit_store)


def get_credit_by_reference(reference: str) -> CreditResponse:
    """Lookup credit by internal `credit_id` or business `credit_code` (e.g. EC-101)."""
    ref = reference.strip()
    if not ref:
        raise ValueError("Credit reference is required")

    direct = _credit_store.get(ref)
    if direct is not None:
        return direct

    ref_upper = ref.upper()
    for credit in _credit_store.values():
        if credit.credit_code.upper() == ref_upper:
            return credit

    raise ValueError(f"Credit not found: {reference}")


def transfer_credit(request: CreditTransferRequest) -> CreditResponse:
    credit = _credit_store.get(request.credit_id)
    if credit is None:
        raise ValueError(f"Credit not found: {request.credit_id}")

    if credit.user_id != request.source_user_id:
        raise ValueError(
            f"Credit {request.credit_id} is not owned by source user {request.source_user_id}"
        )

    if user_service.get_user(request.destination_user_id) is None:
        raise ValueError(f"Destination user not found: {request.destination_user_id}")

    if user_service.get_user(request.source_user_id) is None:
        raise ValueError(f"Source user not found: {request.source_user_id}")

    updated_credit = credit.model_copy(update={"user_id": request.destination_user_id})
    _credit_store[request.credit_id] = updated_credit

    _credit_audit_store.append(
        CreditAuditRecord(
            event_id=f"evt_{uuid4().hex[:8]}",
            operation="transfer",
            credit_id=request.credit_id,
            source_user_id=request.source_user_id,
            destination_user_id=request.destination_user_id,
            created_at=datetime.now(timezone.utc),
            details=(
                f"Transferred ownership from {request.source_user_id} "
                f"to {request.destination_user_id}"
            ),
        )
    )

    return updated_credit
