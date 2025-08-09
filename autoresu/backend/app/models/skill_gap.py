"""
Skill gap analysis model
"""
from beanie import Document, Link
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.models.user import User
from app.models.resume import Resume

class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    LANGUAGE = "language"
    CERTIFICATION = "certification"
    TOOL = "tool"
    FRAMEWORK = "framework"

class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

class LearningResource(BaseModel):
    title: str
    url: str
    provider: str
    type: str  # course, tutorial, book, certification
    duration: Optional[str] = None
    cost: Optional[str] = None
    rating: Optional[float] = None
    difficulty: Optional[str] = None

class SkillRecommendation(BaseModel):
    skill_name: str
    category: SkillCategory
    priority_score: float  # 0-1, higher is more important
    market_demand: float  # 0-1, how in-demand this skill is
    learning_difficulty: float  # 0-1, how hard to learn
    time_to_learn: Optional[str] = None  # estimated time

    # Why this skill is recommended
    reasoning: List[str] = []

    # Related jobs that require this skill
    related_jobs: List[str] = []

    # Learning resources
    resources: List[LearningResource] = []

class CurrentSkill(BaseModel):
    name: str
    category: SkillCategory
    level: SkillLevel
    years_experience: Optional[int] = None
    last_used: Optional[datetime] = None
    verified: bool = False  # verified through projects/experience

class SkillGap(Document):
    user: Link[User]
    resume: Link[Resume]

    # Analysis metadata
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    target_roles: List[str] = []

    # Current skills assessment
    current_skills: List[CurrentSkill] = []

    # Skill gaps identified
    missing_skills: List[SkillRecommendation] = []
    skills_to_improve: List[SkillRecommendation] = []

    # Market analysis
    trending_skills: List[str] = []
    declining_skills: List[str] = []

    # Recommendations summary
    high_priority_skills: List[str] = []  # Most important to learn
    quick_wins: List[str] = []  # Easy to learn, high impact

    # Progress tracking
    learning_goals: List[Dict[str, Any]] = []
    completed_goals: List[Dict[str, Any]] = []

    # AI-generated insights
    summary: Optional[str] = None
    career_advice: List[str] = []

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "skill_gaps"
        indexes = [
            "user",
            "resume", 
            "analysis_date",
            "target_roles",
            "created_at"
        ]

class SkillGapCreate(BaseModel):
    target_roles: List[str] = []

class SkillGapResponse(BaseModel):
    id: str
    analysis_date: datetime
    target_roles: List[str]
    current_skills: List[CurrentSkill]
    missing_skills: List[SkillRecommendation]
    high_priority_skills: List[str]
    quick_wins: List[str]
    summary: Optional[str]
    career_advice: List[str]

class LearningProgress(BaseModel):
    skill_name: str
    progress_percentage: int  # 0-100
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    resources_used: List[LearningResource] = []
    notes: Optional[str] = None
