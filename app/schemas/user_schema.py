from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    SELLER = "seller"
    BUYER = "buyer"
    ADMIN = "admin"


class CreateUserRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=1, le=120)
    city: str = Field(..., min_length=1, max_length=100)
    role: UserRole


class UserResponse(BaseModel):
    user_id: str
    first_name: str
    last_name: str
    age: int
    city: str
    role: UserRole
    created_at: datetime


class GetUserRequest(BaseModel):
    user_id: str = Field(..., min_length=3)


class UserUpdateRequest(BaseModel):
    first_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    age: Optional[int] = Field(default=None, ge=1, le=120)
    city: Optional[str] = Field(default=None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
