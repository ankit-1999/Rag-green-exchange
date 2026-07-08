import logging
from typing import List, Union

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.user_schema import CreateUserRequest, UserResponse
from app.services import user_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Users"])


@router.post(
    "/createuser",
    response_model=Union[UserResponse, List[UserResponse]],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
)
async def create_user(
    request: Union[CreateUserRequest, List[CreateUserRequest]],
) -> Union[UserResponse, List[UserResponse]]:
    requests = request if isinstance(request, list) else [request]
    users = [user_service.create_user(item) for item in requests]
    return users if isinstance(request, list) else users[0]


@router.get(
    "/getuser",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get user details by user_id",
)
async def get_user(user_id: str = Query(..., min_length=3)) -> UserResponse:
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )
    return user


@router.get(
    "/getusers",
    response_model=List[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="List all users",
)
async def get_users() -> List[UserResponse]:
    return user_service.list_users()
