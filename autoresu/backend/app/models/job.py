"""
Job model for storing job postings and matches
"""
from beanie import Document, Link
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.models.user import User
from app.models.resume import Resume

class JobType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"

class WorkLocation(str, Enum):
    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"

class Salary(BaseModel):
    min_amount: Optional[int] = None
    max_amount: Optional[int] = None
    currency: str = "USD"
    period: str = "yearly"  # yearly, monthly, hourly

class Company(BaseModel):
    name: str
    logo_url: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None

class Job(Document):
    # Basic job information
    title: str = Field(..., index=True)
    description: str
    company: Company

    # Job details
    job_type: JobType
    experience_level: ExperienceLevel
    work_location: WorkLocation
    location: Optional[str] = None

    # Requirements and qualifications
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    required_experience: Optional[int] = None  # years
    education_requirements: List[str] = []

    # Compensation
    salary: Optional[Salary] = None
    benefits: List[str] = []

    # Application details
    application_url: Optional[str] = None
    application_email: Optional[str] = None
    application_deadline: Optional[datetime] = None

    # Source information
    source: str = "manual"  # manual, linkedin, indeed, etc.
    source_id: Optional[str] = None
    source_url: Optional[str] = None

    # AI-generated content
    summary: Optional[str] = None
    key_responsibilities: List[str] = []
    ideal_candidate: Optional[str] = None

    # Embeddings for similarity search
    embedding: Optional[List[float]] = None

    # Analytics
    views: int = 0
    applications: int = 0

    # Status
    is_active: bool = True
    featured: bool = False

    # Timestamps
    posted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "jobs"
        indexes = [
            "title", 
            "company.name", 
            "job_type", 
            "experience_level",
            "work_location",
            "required_skills",
            "is_active",
            "posted_at",
            "created_at"
        ]

class MatchScore(BaseModel):
    overall_score: float
    skill_match: float
    experience_match: float
    title_match: float
    location_match: float
    details: Dict[str, Any] = {}

class JobMatch(Document):
    user: Link[User]
    resume: Link[Resume]
    job: Link[Job]

    # Match analysis
    match_score: MatchScore

    # Detailed analysis
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    skill_gaps: List[str] = []

    # Recommendations
    recommendations: List[str] = []
    improvement_suggestions: List[str] = []

    # User interaction
    is_favorite: bool = False
    is_applied: bool = False
    applied_at: Optional[datetime] = None
    user_rating: Optional[int] = None  # 1-5 stars
    user_notes: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "job_matches"
        indexes = [
            "user",
            "resume", 
            "job",
            "match_score.overall_score",
            "is_favorite",
            "is_applied",
            "created_at"
        ]

class JobCreate(BaseModel):
    title: str
    description: str
    company: Company
    job_type: JobType
    experience_level: ExperienceLevel
    work_location: WorkLocation
    location: Optional[str] = None
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    salary: Optional[Salary] = None

class JobUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    featured: Optional[bool] = None

class JobResponse(BaseModel):
    id: str
    title: str
    company: Company
    job_type: JobType
    experience_level: ExperienceLevel
    work_location: WorkLocation
    location: Optional[str]
    required_skills: List[str]
    salary: Optional[Salary]
    created_at: datetime
    is_active: bool

class JobMatchResponse(BaseModel):
    id: str
    job: JobResponse
    match_score: MatchScore
    matching_skills: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    is_favorite: bool
    is_applied: bool
    created_at: datetime
