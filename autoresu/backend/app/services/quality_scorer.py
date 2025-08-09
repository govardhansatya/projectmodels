"""
Quality scoring service for resume analysis
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.ml.classifiers import quality_scorer
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

class QualityScorerService:
    """Service for scoring resume quality"""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the quality scorer service"""
        try:
            logger.info("Initializing Quality Scorer Service...")
            self.initialized = True
            logger.info("Quality Scorer Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Quality Scorer Service: {e}")
            raise
    
    async def score_resume(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score resume quality using ML and AI"""
        try:
            # Use ML classifier for initial scoring
            quality_result = await quality_scorer.score_resume(resume_data)
            
            # Enhance with AI insights if available
            try:
                ai_score = await self._get_ai_quality_score(resume_data)
                if ai_score:
                    # Merge AI insights with ML results
                    quality_result.suggestions.extend(ai_score.get('suggestions', []))
                    quality_result.strengths.extend(ai_score.get('strengths', []))
                    
                    # Adjust overall score based on AI analysis
                    ai_overall = ai_score.get('overall_score', 0) / 100.0  # Convert to 0-1 scale
                    if ai_overall > 0:
                        # Weighted average of ML and AI scores
                        quality_result.overall_score = (quality_result.overall_score * 0.7 + ai_overall * 0.3)
            
            except Exception as e:
                logger.warning(f"AI scoring failed, using ML results only: {e}")
            
            return {
                "overall_score": quality_result.overall_score,
                "section_scores": quality_result.section_scores,
                "suggestions": quality_result.suggestions[:10],  # Limit suggestions
                "strengths": quality_result.strengths[:5],       # Limit strengths
                "weaknesses": quality_result.weaknesses[:5],     # Limit weaknesses
                "ats_friendly": self._check_ats_compatibility(resume_data),
                "word_count": self._count_words(resume_data),
                "completeness": self._calculate_completeness(resume_data)
            }
            
        except Exception as e:
            logger.error(f"Error scoring resume: {e}")
            # Return basic score
            return {
                "overall_score": 0.5,
                "section_scores": {},
                "suggestions": ["Complete resume analysis failed. Please try again."],
                "strengths": [],
                "weaknesses": [],
                "ats_friendly": False,
                "word_count": 0,
                "completeness": 0.0
            }
    
    async def _get_ai_quality_score(self, resume_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get AI-powered quality score"""
        try:
            # Create resume text for AI analysis
            resume_text = self._create_resume_text(resume_data)
            
            if not resume_text.strip():
                return None
            
            # Get AI score
            ai_result = await ai_service.score_resume_quality(resume_text)
            return ai_result
            
        except Exception as e:
            logger.error(f"AI quality scoring failed: {e}")
            return None
    
    def _create_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Create text representation of resume for AI analysis"""
        parts = []
        
        # Add summary
        if resume_data.get('summary'):
            parts.append(f"Summary: {resume_data['summary']}")
        
        # Add experience
        experience = resume_data.get('experience', [])
        if experience:
            parts.append("Experience:")
            for exp in experience:
                exp_text = f"{exp.get('position', '')} at {exp.get('company', '')}"
                if exp.get('description'):
                    exp_text += f" - {exp['description']}"
                parts.append(exp_text)
        
        # Add education
        education = resume_data.get('education', [])
        if education:
            parts.append("Education:")
            for edu in education:
                edu_text = f"{edu.get('degree', '')} from {edu.get('institution', '')}"
                parts.append(edu_text)
        
        # Add skills
        skills = resume_data.get('skills', [])
        if skills:
            parts.append(f"Skills: {', '.join(skills)}")
        
        return '\n'.join(parts)
    
    def _check_ats_compatibility(self, resume_data: Dict[str, Any]) -> bool:
        """Check if resume is ATS-friendly"""
        score = 0
        max_score = 10
        
        # Check for standard sections
        required_sections = ['experience', 'skills', 'contact_info']
        for section in required_sections:
            if resume_data.get(section):
                score += 2
        
        # Check for clear job titles and company names
        experience = resume_data.get('experience', [])
        if experience:
            has_clear_titles = all(exp.get('position') and exp.get('company') for exp in experience)
            if has_clear_titles:
                score += 2
        
        # Check for skills section with relevant keywords
        skills = resume_data.get('skills', [])
        if len(skills) >= 5:
            score += 2
        
        return score >= 6  # 60% threshold
    
    def _count_words(self, resume_data: Dict[str, Any]) -> int:
        """Count total words in resume"""
        text = self._create_resume_text(resume_data)
        return len(text.split())
    
    def _calculate_completeness(self, resume_data: Dict[str, Any]) -> float:
        """Calculate resume completeness percentage"""
        sections = [
            'contact_info',
            'summary', 
            'experience',
            'education',
            'skills'
        ]
        
        completed = sum(1 for section in sections if resume_data.get(section))
        return completed / len(sections)
    
    async def get_improvement_suggestions(self, resume_data: Dict[str, Any], 
                                        target_role: Optional[str] = None) -> List[str]:
        """Get specific improvement suggestions"""
        suggestions = []
        
        # Check completeness
        if not resume_data.get('summary'):
            suggestions.append("Add a professional summary to highlight your key achievements")
        
        if not resume_data.get('skills') or len(resume_data.get('skills', [])) < 5:
            suggestions.append("Include more relevant technical and soft skills")
        
        # Check experience descriptions
        experience = resume_data.get('experience', [])
        if experience:
            for exp in experience:
                if not exp.get('description') or len(str(exp.get('description', ''))) < 50:
                    suggestions.append(f"Expand the description for {exp.get('position', 'position')} at {exp.get('company', 'company')}")
                    break
        
        # Check for quantified achievements
        resume_text = self._create_resume_text(resume_data)
        if not any(char.isdigit() for char in resume_text):
            suggestions.append("Include quantified achievements and metrics in your experience descriptions")
        
        # Role-specific suggestions
        if target_role:
            role_lower = target_role.lower()
            
            if 'software' in role_lower or 'developer' in role_lower:
                if not any('github' in skill.lower() for skill in resume_data.get('skills', [])):
                    suggestions.append("Consider adding GitHub profile and relevant programming languages")
            
            elif 'manager' in role_lower or 'lead' in role_lower:
                if not any('leadership' in skill.lower() for skill in resume_data.get('skills', [])):
                    suggestions.append("Highlight leadership and management experience")
        
        return suggestions[:5]  # Limit to top 5 suggestions

# Global quality scorer service instance
quality_scorer_service = QualityScorerService()

# Convenience function
async def score_resume_quality_service(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """Score resume quality"""
    return await quality_scorer_service.score_resume(resume_data)

# Initialize quality scorer on startup
async def initialize_quality_scorer():
    """Initialize quality scorer on application startup"""
    logger.info("Initializing quality scorer...")
    await quality_scorer_service.initialize()
    logger.info("Quality scorer initialization complete")
