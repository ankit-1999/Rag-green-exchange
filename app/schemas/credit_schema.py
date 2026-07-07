from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class CreditType(str, Enum):
    SOLAR = "solar"
    WIND = "wind"
    COAL = "coal"


class CreateCreditRequest(BaseModel):
    user_id: str = Field(..., min_length=3)
    credit_type: CreditType
    price: float = Field(..., gt=0)


class CreditResponse(BaseModel):
    credit_id: str
    credit_code: str
    user_id: str
    credit_type: CreditType
    price: float
    created_at: datetime


class CreditTransferRequest(BaseModel):
    credit_id: str = Field(..., min_length=3)
    source_user_id: str = Field(..., min_length=3)
    destination_user_id: str = Field(..., min_length=3)


class CreditAuditRecord(BaseModel):
    event_id: str
    operation: Literal["create", "transfer"]
    credit_id: str
    source_user_id: str
    destination_user_id: str
    created_at: datetime
    details: str
