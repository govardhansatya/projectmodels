"""
Job matching service for finding relevant jobs for resumes
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.models.job import Job, JobMatch, MatchScore
from app.models.resume import Resume
from app.models.job_match import JobMatchAnalysis, MatchLevel, AnalysisStatus, QuickMatchResult
from app.models.user import User
from app.ml.classifiers import job_match_classifier, JobMatchResult
from app.ml.embeddings import embedding_manager
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

class JobMatcherService:
    """Service for matching jobs with resumes"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the job matcher service"""
        try:
            logger.info("Initializing Job Matcher Service...")
            # Ensure classifiers are loaded
            await job_match_classifier.load_model()
            self.initialized = True
            logger.info("Job Matcher Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Job Matcher Service: {e}")
            raise
    
    async def find_matching_jobs(self, resume_id: str, limit: int = 20, 
                                min_score: float = 0.3) -> List[QuickMatchResult]:
        """Find jobs that match a given resume"""
        try:
            # Get resume
            resume = await Resume.get(resume_id)
            if not resume:
                raise ValueError(f"Resume {resume_id} not found")
            
            # Get all active jobs
            jobs = await Job.find(Job.is_active == True).limit(100).to_list()
            
            if not jobs:
                return []
            
            # Perform quick matching
            matches = []
            for job in jobs:
                try:
                    # Convert models to dict for ML processing
                    job_data = await self._job_to_dict(job)
                    resume_data = await self._resume_to_dict(resume)
                    
                    # Get match result
                    match_result = await job_match_classifier.predict_match(job_data, resume_data)
                    
                    if match_result.match_score >= min_score:
                        quick_result = QuickMatchResult(
                            job_id=str(job.id),
                            overall_score=match_result.match_score,
                            match_level=match_result.match_level,
                            top_strengths=match_result.reasons[:3],
                            top_gaps=match_result.missing_skills[:3]
                        )
                        matches.append(quick_result)
                        
                except Exception as e:
                    logger.error(f"Error matching job {job.id}: {e}")
                    continue
            
            # Sort by score and limit results
            matches.sort(key=lambda x: x.overall_score, reverse=True)
            return matches[:limit]
            
        except Exception as e:
            logger.error(f"Error finding matching jobs: {e}")
            return []
    
    async def analyze_job_match(self, resume_id: str, job_id: str) -> JobMatchAnalysis:
        """Perform detailed analysis of a job-resume match"""
        try:
            # Get resume and job
            resume = await Resume.get(resume_id)
            job = await Job.get(job_id)
            
            if not resume or not job:
                raise ValueError("Resume or job not found")
            
            # Check if analysis already exists
            existing = await JobMatchAnalysis.find_one(
                JobMatchAnalysis.resume.id == resume_id,
                JobMatchAnalysis.job.id == job_id
            )
            
            if existing and existing.status == AnalysisStatus.COMPLETED:
                return existing
            
            # Create new analysis
            analysis = JobMatchAnalysis(
                user=resume.user,
                resume=resume,
                job=job,
                analysis_id=f"{resume_id}_{job_id}_{int(datetime.utcnow().timestamp())}",
                status=AnalysisStatus.ANALYZING
            )
            
            await analysis.insert()
            
            try:
                # Perform detailed analysis
                await self._perform_detailed_analysis(analysis, resume, job)
                
                # Update status
                analysis.status = AnalysisStatus.COMPLETED
                analysis.completed_at = datetime.utcnow()
                
                await analysis.save()
                
                return analysis
                
            except Exception as e:
                analysis.status = AnalysisStatus.FAILED
                analysis.error_message = str(e)
                await analysis.save()
                raise
                
        except Exception as e:
            logger.error(f"Error analyzing job match: {e}")
            raise
    
    async def _perform_detailed_analysis(self, analysis: JobMatchAnalysis, 
                                       resume: Resume, job: Job):
        """Perform detailed AI-powered analysis"""
        start_time = datetime.utcnow()
        
        # Convert to dict format
        job_data = await self._job_to_dict(job)
        resume_data = await self._resume_to_dict(resume)
        
        # Get ML classifier results
        match_result = await job_match_classifier.predict_match(job_data, resume_data)
        
        # Update basic match info
        analysis.overall_score = match_result.match_score
        analysis.match_level = match_result.match_level
        analysis.confidence_score = match_result.confidence
        
        # Set skill information
        analysis.matching_skills_count = len(match_result.matching_skills)
        analysis.missing_skills_count = len(match_result.missing_skills)
        analysis.total_skills_analyzed = len(job_data.get('required_skills', [])) + len(job_data.get('preferred_skills', []))
        
        # Create skill matches
        analysis.skill_matches = await self._analyze_skills(job_data, resume_data, match_result)
        
        # Analyze experience
        analysis.experience_match = await self._analyze_experience(job_data, resume_data)
        
        # Analyze education
        analysis.education_match = await self._analyze_education(job_data, resume_data)
        
        # Analyze salary compatibility
        analysis.salary_match = await self._analyze_salary(job_data, resume_data)
        
        # Analyze location compatibility
        analysis.location_match = await self._analyze_location(job_data, resume_data)
        
        # Generate AI insights
        await self._generate_ai_insights(analysis, job, resume)
        
        # Calculate processing time
        end_time = datetime.utcnow()
        analysis.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
    
    async def _analyze_skills(self, job_data: Dict, resume_data: Dict, 
                            match_result: JobMatchResult) -> List[Dict[str, Any]]:
        """Analyze skill matches in detail"""
        from app.models.job_match import SkillMatch
        
        skill_matches = []
        job_skills = set(job_data.get('required_skills', []) + job_data.get('preferred_skills', []))
        resume_skills = set(resume_data.get('skills', []))
        
        # Analyze each job skill
        for skill in job_skills:
            skill_lower = skill.lower()
            
            # Check for exact match
            if skill_lower in [rs.lower() for rs in resume_skills]:
                match_type = "exact"
                confidence = 1.0
            else:
                # Check for similar skills using embeddings
                similarity_scores = await embedding_manager.batch_similarity(
                    skill, list(resume_skills)
                )
                max_similarity = max(similarity_scores) if similarity_scores else 0.0
                
                if max_similarity > 0.7:
                    match_type = "similar"
                    confidence = max_similarity
                else:
                    match_type = "missing"
                    confidence = 0.0
            
            # Determine importance (simplified)
            importance = 0.9 if skill in job_data.get('required_skills', []) else 0.6
            
            skill_match = {
                'skill_name': skill,
                'match_type': match_type,
                'confidence': confidence,
                'importance': importance,
                'alternative_skills': []
            }
            
            skill_matches.append(skill_match)
        
        return skill_matches
    
    async def _analyze_experience(self, job_data: Dict, resume_data: Dict) -> Dict[str, Any]:
        """Analyze experience match"""
        from app.models.job_match import ExperienceMatch
        
        required_years = job_data.get('required_experience', 0)
        candidate_years = resume_data.get('total_experience_years', 0)
        
        match_ratio = candidate_years / required_years if required_years > 0 else 1.0
        
        return {
            'required_years': required_years,
            'candidate_years': candidate_years,
            'match_ratio': min(match_ratio, 2.0),  # Cap at 2x requirements
            'relevant_experience': [],
            'missing_experience': []
        }
    
    async def _analyze_education(self, job_data: Dict, resume_data: Dict) -> Dict[str, Any]:
        """Analyze education match"""
        from app.models.job_match import EducationMatch
        
        candidate_education = [edu.get('degree', '') for edu in resume_data.get('education', [])]
        required_education = job_data.get('education_requirements', [])
        
        # Simple check - if no requirements, assume met
        meets_requirements = len(required_education) == 0 or len(candidate_education) > 0
        
        return {
            'meets_requirements': meets_requirements,
            'candidate_education': candidate_education,
            'required_education': required_education,
            'alternative_qualifications': []
        }
    
    async def _analyze_salary(self, job_data: Dict, resume_data: Dict) -> Dict[str, Any]:
        """Analyze salary compatibility"""
        from app.models.job_match import SalaryMatch
        
        salary_data = job_data.get('salary', {})
        candidate_expectation = resume_data.get('expected_salary', 0)
        
        job_min = salary_data.get('min_amount', 0) if salary_data else 0
        job_max = salary_data.get('max_amount', 0) if salary_data else 0
        
        if job_min and job_max and candidate_expectation:
            if job_min <= candidate_expectation <= job_max:
                match_status = "within_range"
            elif candidate_expectation < job_min:
                match_status = "below_range"
            else:
                match_status = "above_range"
        else:
            match_status = "unknown"
        
        return {
            'job_salary_min': job_min,
            'job_salary_max': job_max,
            'candidate_expectation': candidate_expectation,
            'match_status': match_status,
            'negotiation_potential': 0.5  # Default
        }
    
    async def _analyze_location(self, job_data: Dict, resume_data: Dict) -> Dict[str, Any]:
        """Analyze location compatibility"""
        from app.models.job_match import LocationMatch
        
        job_location = job_data.get('location', '')
        candidate_location = resume_data.get('location', '')
        remote_option = job_data.get('work_location') == 'remote'
        
        if remote_option:
            match_score = 1.0
            relocation_required = False
        elif job_location and candidate_location:
            # Simple location matching (can be enhanced with geographic data)
            match_score = 1.0 if job_location.lower() in candidate_location.lower() else 0.3
            relocation_required = match_score < 0.5
        else:
            match_score = 0.5
            relocation_required = False
        
        return {
            'job_location': job_location,
            'candidate_location': candidate_location,
            'remote_option': remote_option,
            'relocation_required': relocation_required,
            'distance_km': None,
            'match_score': match_score
        }
    
    async def _generate_ai_insights(self, analysis: JobMatchAnalysis, job: Job, resume: Resume):
        """Generate AI-powered insights and recommendations"""
        try:
            # Prepare context for AI
            resume_text = await self._create_resume_text(resume)
            job_text = f"{job.title}\n{job.description}\nRequired skills: {', '.join(job.required_skills)}"
            
            # Get detailed analysis from AI
            ai_analysis = await ai_service.analyze_job_match(resume_text, job_text)
            
            if ai_analysis:
                analysis.strengths = ai_analysis.get('matching_skills', [])[:5]
                analysis.weaknesses = ai_analysis.get('missing_skills', [])[:5]
                analysis.recommendations = ai_analysis.get('recommendations', [])[:5]
                analysis.resume_improvements = ai_analysis.get('improvement_suggestions', [])[:5]
                
                # Generate explanations
                if analysis.overall_score >= 0.8:
                    analysis.why_good_match = [
                        "Strong skill alignment with job requirements",
                        "Experience level matches expectations",
                        "Profile aligns with job description"
                    ]
                elif analysis.overall_score >= 0.6:
                    analysis.why_good_match = ["Good foundational match with room for growth"]
                    analysis.why_not_perfect = ["Some skill gaps need to be addressed"]
                else:
                    analysis.why_not_perfect = [
                        "Significant skill gaps present",
                        "Experience level may not fully meet requirements"
                    ]
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            # Fallback to rule-based insights
            analysis.strengths = ["Analysis completed successfully"]
            analysis.recommendations = ["Review job requirements carefully"]
    
    async def _job_to_dict(self, job: Job) -> Dict[str, Any]:
        """Convert Job model to dictionary for ML processing"""
        return {
            'id': str(job.id),
            'title': job.title,
            'description': job.description,
            'required_skills': job.required_skills,
            'preferred_skills': job.preferred_skills,
            'required_experience': job.required_experience,
            'location': job.location,
            'work_location': job.work_location.value if job.work_location else None,
            'remote_option': job.work_location == 'remote' if job.work_location else False,
            'salary_range': {
                'min': job.salary.min_amount if job.salary else None,
                'max': job.salary.max_amount if job.salary else None
            } if job.salary else None,
            'experience_level': job.experience_level.value if job.experience_level else None,
            'experience_level_numeric': self._experience_to_numeric(job.experience_level.value if job.experience_level else 'mid'),
            'education_requirements': job.education_requirements,
            'company_name': job.company.name if job.company else None,
            'industry': job.company.industry if job.company else None,
            'job_type': job.job_type.value if job.job_type else None
        }
    
    async def _resume_to_dict(self, resume: Resume) -> Dict[str, Any]:
        """Convert Resume model to dictionary for ML processing"""
        # Get current version
        current_version = None
        if resume.versions:
            # Get the latest version
            versions = await resume.fetch_all_links()
            if versions and 'versions' in versions:
                current_version = versions['versions'][-1] if versions['versions'] else None
        
        if not current_version:
            # Use basic resume data
            return {
                'id': str(resume.id),
                'skills': [],
                'total_experience_years': 0,
                'education': [],
                'experience': [],
                'contact_info': {},
                'summary': '',
                'location': '',
                'expected_salary': None
            }
        
        # Extract experience years
        total_years = 0
        experience_descriptions = []
        
        if hasattr(current_version, 'experience') and current_version.experience:
            for exp in current_version.experience:
                # Simple year calculation (could be enhanced)
                total_years += 2  # Assume 2 years per position on average
                if hasattr(exp, 'description') and exp.description:
                    experience_descriptions.extend(exp.description)
        
        return {
            'id': str(resume.id),
            'skills': current_version.skills if hasattr(current_version, 'skills') else [],
            'total_experience_years': total_years,
            'education': [
                {
                    'degree': edu.degree,
                    'institution': edu.institution,
                    'field_of_study': getattr(edu, 'field_of_study', None)
                } for edu in (current_version.education if hasattr(current_version, 'education') and current_version.education else [])
            ],
            'experience': experience_descriptions,
            'experience_descriptions': experience_descriptions,
            'contact_info': current_version.contact_info.__dict__ if hasattr(current_version, 'contact_info') and current_version.contact_info else {},
            'summary': getattr(current_version, 'summary', ''),
            'location': getattr(current_version.contact_info, 'location', '') if hasattr(current_version, 'contact_info') and current_version.contact_info else '',
            'expected_salary': None  # Could be added to resume model
        }
    
    def _experience_to_numeric(self, experience_level: str) -> int:
        """Convert experience level to numeric years"""
        mapping = {
            'entry': 0,
            'junior': 1,
            'mid': 3,
            'senior': 5,
            'lead': 7,
            'executive': 10
        }
        return mapping.get(experience_level.lower(), 3)
    
    async def _create_resume_text(self, resume: Resume) -> str:
        """Create a text representation of the resume"""
        # Get current version
        current_version = None
        if resume.versions:
            versions = await resume.fetch_all_links()
            if versions and 'versions' in versions:
                current_version = versions['versions'][-1] if versions['versions'] else None
        
        if not current_version:
            return f"Resume: {resume.title}"
        
        parts = []
        
        # Add summary
        if hasattr(current_version, 'summary') and current_version.summary:
            parts.append(f"Summary: {current_version.summary}")
        
        # Add experience
        if hasattr(current_version, 'experience') and current_version.experience:
            parts.append("Experience:")
            for exp in current_version.experience:
                exp_text = f"{exp.position} at {exp.company}"
                if hasattr(exp, 'description') and exp.description:
                    exp_text += f" - {' '.join(exp.description)}"
                parts.append(exp_text)
        
        # Add skills
        if hasattr(current_version, 'skills') and current_version.skills:
            parts.append(f"Skills: {', '.join(current_version.skills)}")
        
        return '\n'.join(parts)

# Global job matcher service instance
job_matcher_service = JobMatcherService()

# Convenience functions
async def find_jobs_for_resume(resume_id: str, limit: int = 20) -> List[QuickMatchResult]:
    """Find matching jobs for a resume"""
    return await job_matcher_service.find_matching_jobs(resume_id, limit)

async def analyze_job_resume_match(resume_id: str, job_id: str) -> JobMatchAnalysis:
    """Analyze match between a job and resume"""
    return await job_matcher_service.analyze_job_match(resume_id, job_id)

# Initialize job matcher on startup
async def initialize_job_matcher():
    """Initialize job matcher on application startup"""
    logger.info("Initializing job matcher...")
    await job_matcher_service.initialize()
    logger.info("Job matcher initialization complete")
