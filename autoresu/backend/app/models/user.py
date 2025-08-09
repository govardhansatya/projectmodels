"""
User model for authentication and profile management
"""
from beanie import Document
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    PREMIUM = "premium"

class UserProfile(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    github_username: Optional[str] = None
    linkedin_url: Optional[str] = None
    bio: Optional[str] = None

class UserPreferences(BaseModel):
    preferred_roles: List[str] = []
    target_companies: List[str] = []
    salary_range: Optional[Dict[str, int]] = None
    remote_preference: Optional[bool] = None
    notification_settings: Dict[str, bool] = {
        "email_notifications": True,
        "job_alerts": True,
        "skill_recommendations": True
    }

class User(Document):
    email: EmailStr = Field(..., unique=True, index=True)
    username: Optional[str] = Field(None, unique=True, index=True)
    hashed_password: Optional[str] = None

    # OAuth fields
    google_id: Optional[str] = Field(None, unique=True, sparse=True)
    github_id: Optional[str] = Field(None, unique=True, sparse=True)

    # Profile
    profile: UserProfile = Field(default_factory=UserProfile)
    preferences: UserPreferences = Field(default_factory=UserPreferences)

    # System fields
    role: UserRole = UserRole.USER
    is_active: bool = True
    is_verified: bool = False

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    # Usage tracking
    api_calls_count: int = 0
    resumes_created: int = 0
    jobs_applied: int = 0

    class Settings:
        name = "users"
        indexes = [
            "email",
            "username",
            "google_id",
            "github_id",
            "created_at",
        ]

class UserCreate(BaseModel):
    email: EmailStr
    username: Optional[str] = None
    password: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    profile: Optional[UserProfile] = None
    preferences: Optional[UserPreferences] = None

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    username: Optional[str]
    profile: UserProfile
    preferences: UserPreferences
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    resumes_created: int
    jobs_applied: int
