from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from app.schemas.user_schema import CreateUserRequest, UserResponse

_user_store: Dict[str, UserResponse] = {}


def create_user(request: CreateUserRequest) -> UserResponse:
    user = UserResponse(
        user_id=f"user_{uuid4().hex[:8]}",
        first_name=request.first_name,
        last_name=request.last_name,
        age=request.age,
        city=request.city,
        role=request.role,
        created_at=datetime.now(timezone.utc),
    )
    _user_store[user.user_id] = user
    return user


def get_user(user_id: str) -> Optional[UserResponse]:
    return _user_store.get(user_id)


def list_users() -> List[UserResponse]:
    return list(_user_store.values())
