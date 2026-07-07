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


def _get_user_display_name(user_id: str) -> str:
    user = user_service.get_user(user_id)
    if user is None:
        return user_id
    return f"{user.first_name} {user.last_name}".strip()


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
            source_name=_get_user_display_name(request.user_id),
            destination_user_id=request.user_id,
            destination_name=_get_user_display_name(request.user_id),
            created_at=datetime.now(timezone.utc),
            details=f"Created {credit.credit_type.value} credit at price {credit.price}",
        )
    )

    return credit


def list_credits() -> List[CreditResponse]:
    return list(_credit_store.values())


def list_credit_audit() -> List[CreditAuditRecord]:
    return list(_credit_audit_store)


def list_credits_created_by_user(user_id: str) -> List[CreditResponse]:
    """Return credits originally created by a given user (from create audit trail)."""
    created_credit_ids = {
        rec.credit_id
        for rec in _credit_audit_store
        if rec.operation == "create" and rec.source_user_id == user_id
    }
    credits: List[CreditResponse] = []
    for credit_id in created_credit_ids:
        credit = _credit_store.get(credit_id)
        if credit is not None:
            credits.append(credit)
    return credits


def list_credit_audit_by_credit_id(credit_id: str) -> List[CreditAuditRecord]:
    """Return all audit records related to a single credit id."""
    return [rec for rec in _credit_audit_store if rec.credit_id == credit_id]


def get_credit_history_timeline(reference: str) -> Dict:
    """
    Build a human-readable timeline for a credit by code/id.

    Returns machine + text timeline so LLM can directly answer history questions.
    """
    credit = get_credit_by_reference(reference)
    records = list_credit_audit_by_credit_id(credit.credit_id)
    records_sorted = sorted(records, key=lambda rec: rec.created_at)

    timeline_items: List[Dict] = []
    for rec in records_sorted:
        action = f"{rec.source_name} -> {rec.destination_name} on {rec.created_at.isoformat()}"
        timeline_items.append(
            {
                "timestamp": rec.created_at.isoformat(),
                "operation": rec.operation,
                "action": action,
                "details": rec.details,
                "source_name": rec.source_name,
                "destination_name": rec.destination_name,
            }
        )

    if timeline_items:
        timeline_text = "\n".join(
            f"- {item['action']}"
            for item in timeline_items
        )
    else:
        timeline_text = "No history records found for this credit."

    return {
        "credit_id": credit.credit_id,
        "credit_code": credit.credit_code,
        "current_owner_user_id": credit.user_id,
        "timeline": timeline_items,
        "timeline_text": timeline_text,
    }


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
            source_name=_get_user_display_name(request.source_user_id),
            destination_user_id=request.destination_user_id,
            destination_name=_get_user_display_name(request.destination_user_id),
            created_at=datetime.now(timezone.utc),
            details=(
                f"Transferred ownership from {request.source_user_id} "
                f"to {request.destination_user_id}"
            ),
        )
    )

    return updated_credit
