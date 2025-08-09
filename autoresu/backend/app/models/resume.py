"""
Resume model for storing and managing resume data
"""
from beanie import Document, Link
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.models.user import User

class ResumeStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None

class Experience(BaseModel):
    company: str
    position: str
    location: Optional[str] = None
    start_date: str
    end_date: Optional[str] = None
    current: bool = False
    description: List[str] = []
    achievements: List[str] = []
    technologies: List[str] = []

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    relevant_coursework: List[str] = []

class Project(BaseModel):
    name: str
    description: str
    technologies: List[str] = []
    github_url: Optional[str] = None
    demo_url: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    highlights: List[str] = []

class Certification(BaseModel):
    name: str
    issuer: str
    date_obtained: Optional[str] = None
    expiry_date: Optional[str] = None
    credential_id: Optional[str] = None
    url: Optional[str] = None

class QualityScores(BaseModel):
    overall_score: float = 0.0
    sections: Dict[str, float] = {}
    improvements: List[str] = []
    strengths: List[str] = []

class ResumeVersion(Document):
    resume_id: str = Field(..., index=True)
    version_number: int
    name: str

    # Core resume sections
    summary: Optional[str] = None
    contact_info: ContactInfo = Field(default_factory=ContactInfo)
    experience: List[Experience] = []
    education: List[Education] = []
    projects: List[Project] = []
    skills: List[str] = []
    certifications: List[Certification] = []

    # Custom sections
    custom_sections: Dict[str, Any] = {}

    # AI-generated content
    generated_content: Dict[str, Any] = {}

    # Quality analysis
    quality_scores: QualityScores = Field(default_factory=QualityScores)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: int = 0

    class Settings:
        name = "resume_versions"
        indexes = ["resume_id", "version_number", "created_at"]

class Resume(Document):
    user: Link[User]

    # Basic info
    title: str = Field(..., index=True)
    description: Optional[str] = None
    status: ResumeStatus = ResumeStatus.DRAFT

    # Current version
    current_version: int = 1
    versions: List[Link[ResumeVersion]] = []

    # Target job information
    target_roles: List[str] = []
    target_companies: List[str] = []
    target_keywords: List[str] = []

    # AI Enhancement data
    github_data: Optional[Dict[str, Any]] = None
    enhancement_suggestions: List[Dict[str, Any]] = []

    # File information
    original_file_path: Optional[str] = None
    original_file_name: Optional[str] = None
    file_type: Optional[str] = None

    # Embeddings for similarity search
    embedding: Optional[List[float]] = None

    # Analytics
    views: int = 0
    downloads: int = 0
    job_matches_count: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_enhanced_at: Optional[datetime] = None

    class Settings:
        name = "resumes"
        indexes = ["title", "status", "target_roles", "created_at", "updated_at"]

class ResumeCreate(BaseModel):
    title: str
    description: Optional[str] = None
    target_roles: List[str] = []

class ResumeUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ResumeStatus] = None
    target_roles: Optional[List[str]] = None
    target_companies: Optional[List[str]] = None

class ResumeResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    status: ResumeStatus
    current_version: int
    target_roles: List[str]
    created_at: datetime
    updated_at: datetime
    quality_scores: QualityScores
    views: int
    downloads: int
