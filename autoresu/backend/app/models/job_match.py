"""
Job match analysis model for detailed matching results
"""
from beanie import Document, Link
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from app.models.user import User
from app.models.resume import Resume
from app.models.job import Job

class MatchLevel(str, Enum):
    EXCELLENT = "excellent"  # 80-100%
    GOOD = "good"           # 60-79%
    FAIR = "fair"           # 40-59%
    POOR = "poor"           # 0-39%

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"

class SectionMatch(BaseModel):
    section_name: str
    score: float  # 0-1
    weight: float  # importance weight
    details: Dict[str, Any] = {}
    suggestions: List[str] = []

class SkillMatch(BaseModel):
    skill_name: str
    match_type: str  # "exact", "similar", "missing"
    confidence: float  # 0-1
    importance: float  # 0-1, how important this skill is for the job
    alternative_skills: List[str] = []

class ExperienceMatch(BaseModel):
    required_years: Optional[int] = None
    candidate_years: Optional[int] = None
    match_ratio: float  # candidate_years / required_years
    relevant_experience: List[str] = []
    missing_experience: List[str] = []

class EducationMatch(BaseModel):
    meets_requirements: bool
    candidate_education: List[str] = []
    required_education: List[str] = []
    alternative_qualifications: List[str] = []

class SalaryMatch(BaseModel):
    job_salary_min: Optional[int] = None
    job_salary_max: Optional[int] = None
    candidate_expectation: Optional[int] = None
    match_status: str  # "within_range", "below_range", "above_range", "unknown"
    negotiation_potential: float  # 0-1

class LocationMatch(BaseModel):
    job_location: Optional[str] = None
    candidate_location: Optional[str] = None
    remote_option: bool = False
    relocation_required: bool = False
    distance_km: Optional[float] = None
    match_score: float  # 0-1

class JobMatchAnalysis(Document):
    user: Link[User]
    resume: Link[Resume]
    job: Link[Job]

    # Analysis metadata
    analysis_id: str = Field(..., unique=True, index=True)
    status: AnalysisStatus = AnalysisStatus.PENDING
    ai_model_used: str = "gemini-pro"

    # Overall match results
    overall_score: float  # 0-1
    match_level: MatchLevel
    confidence_score: float  # 0-1, how confident we are in the analysis

    # Detailed section matches
    section_matches: List[SectionMatch] = []

    # Skill analysis
    skill_matches: List[SkillMatch] = []
    matching_skills_count: int = 0
    missing_skills_count: int = 0
    total_skills_analyzed: int = 0

    # Experience analysis
    experience_match: ExperienceMatch = Field(default_factory=ExperienceMatch)

    # Education analysis
    education_match: EducationMatch = Field(default_factory=EducationMatch)

    # Compensation analysis
    salary_match: SalaryMatch = Field(default_factory=SalaryMatch)

    # Location analysis
    location_match: LocationMatch = Field(default_factory=LocationMatch)

    # AI-generated insights
    strengths: List[str] = []  # What makes this a good match
    weaknesses: List[str] = []  # What are the gaps
    recommendations: List[str] = []  # How to improve the match
    next_steps: List[str] = []  # Actionable advice

    # Improvement suggestions
    resume_improvements: List[str] = []
    skill_development: List[str] = []
    experience_gaps: List[str] = []

    # Match explanation
    why_good_match: List[str] = []
    why_not_perfect: List[str] = []
    deal_breakers: List[str] = []

    # Ranking and scoring details
    ranking_factors: Dict[str, float] = {}  # What contributed to the score
    comparison_metrics: Dict[str, Any] = {}

    # User feedback
    user_rating: Optional[int] = None  # 1-5 stars for match quality
    user_feedback: Optional[str] = None
    is_useful: Optional[bool] = None

    # Application tracking
    application_intent: Optional[str] = None  # "will_apply", "maybe", "no"
    application_date: Optional[datetime] = None
    application_status: Optional[str] = None

    # Processing metadata
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "job_match_analyses"
        indexes = [
            "user",
            "resume",
            "job",
            "analysis_id",
            "status",
            "overall_score",
            "match_level",
            "created_at",
            "completed_at"
        ]

class JobMatchCreate(BaseModel):
    resume_id: str
    job_id: str

class JobMatchUpdate(BaseModel):
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None
    is_useful: Optional[bool] = None
    application_intent: Optional[str] = None

class QuickMatchResult(BaseModel):
    """Lightweight match result for quick comparisons"""
    job_id: str
    overall_score: float
    match_level: MatchLevel
    top_strengths: List[str]
    top_gaps: List[str]

class JobMatchResponse(BaseModel):
    id: str
    analysis_id: str
    overall_score: float
    match_level: MatchLevel
    confidence_score: float
    
    # Key insights
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Detailed scores
    skills_score: float
    experience_score: float
    education_score: float
    location_score: float
    
    # Counts
    matching_skills_count: int
    missing_skills_count: int
    
    # Status
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime]

class BulkMatchRequest(BaseModel):
    resume_id: str
    job_ids: List[str]
    max_results: Optional[int] = 50

class BulkMatchResponse(BaseModel):
    resume_id: str
    total_jobs_analyzed: int
    completed_analyses: int
    failed_analyses: int
    matches: List[QuickMatchResult]

class MatchFilters(BaseModel):
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    match_levels: Optional[List[MatchLevel]] = None
    has_skills: Optional[List[str]] = None
    missing_skills: Optional[List[str]] = None
    min_salary: Optional[int] = None
    max_salary: Optional[int] = None
    locations: Optional[List[str]] = None
    remote_only: Optional[bool] = None
    
class MatchStatistics(BaseModel):
    total_matches: int
    average_score: float
    score_distribution: Dict[str, int]  # score ranges -> count
    match_level_distribution: Dict[MatchLevel, int]
    top_matching_skills: List[str]
    most_missing_skills: List[str]
    best_matches: List[QuickMatchResult]
    improvement_opportunities: List[str]
