"""
GitHub integration endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List

from app.models.user import User
from app.services.auth_service import get_current_active_user
from app.services.github_importer import import_github_profile

router = APIRouter()


@router.get("/import")
async def import_profile(username: str = Query(None), current_user: User = Depends(get_current_active_user)):
	"""Import GitHub data for the provided username or the current user's profile"""
	handle = username or current_user.profile.github_username
	if not handle:
		raise HTTPException(status_code=400, detail="GitHub username not provided")

	data = await import_github_profile(handle)
	return {"username": handle, "data": data}
