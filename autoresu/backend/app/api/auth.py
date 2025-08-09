"""
Authentication API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any

from app.models.user import User, UserCreate, UserLogin, UserResponse
from app.services.auth_service import (
	register_new_user,
	login_user,
	refresh_token as refresh_access_token,
	get_current_active_user,
	auth_service,
)
from app.utils.exceptions import AppException

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
	try:
		user_obj = await register_new_user(user)
		return UserResponse(
			id=str(user_obj.id),
			email=user_obj.email,
			username=user_obj.username,
			profile=user_obj.profile,
			preferences=user_obj.preferences,
			role=user_obj.role,
			is_active=user_obj.is_active,
			is_verified=user_obj.is_verified,
			created_at=user_obj.created_at,
			updated_at=user_obj.updated_at,
			resumes_created=user_obj.resumes_created,
			jobs_applied=user_obj.jobs_applied,
		)
	except AppException as e:
		raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post("/login")
async def login(payload: UserLogin) -> Dict[str, Any]:
	try:
		return await login_user(payload)
	except AppException as e:
		raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post("/refresh")
async def refresh(body: Dict[str, str]) -> Dict[str, Any]:
	try:
		token = body.get("refresh_token")
		if not token:
			raise HTTPException(status_code=400, detail="refresh_token is required")
		return await refresh_access_token(token)
	except AppException as e:
		raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_active_user)):
	return UserResponse(
		id=str(current_user.id),
		email=current_user.email,
		username=current_user.username,
		profile=current_user.profile,
		preferences=current_user.preferences,
		role=current_user.role,
		is_active=current_user.is_active,
		is_verified=current_user.is_verified,
		created_at=current_user.created_at,
		updated_at=current_user.updated_at,
		resumes_created=current_user.resumes_created,
		jobs_applied=current_user.jobs_applied,
	)


@router.post("/change-password")
async def change_password(body: Dict[str, str], current_user: User = Depends(get_current_active_user)):
	current = body.get("current_password")
	new = body.get("new_password")
	if not current or not new:
		raise HTTPException(status_code=400, detail="current_password and new_password required")
	try:
		await auth_service.change_password(current_user, current, new)
		return {"message": "Password updated"}
	except AppException as e:
		raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post("/api-key")
async def generate_api_key(current_user: User = Depends(get_current_active_user)):
	key = auth_service.generate_api_key(current_user)
	return {"api_key": key}
