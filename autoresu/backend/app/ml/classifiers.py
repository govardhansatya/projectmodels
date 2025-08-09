"""
Classification models for AI Resume Builder
Handles job matching, skill categorization, and quality scoring
"""
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import joblib
import os
from datetime import datetime

from app.config.settings import settings
from app.ml.embeddings import embedding_manager
from app.ml.training import model_trainer

logger = logging.getLogger(__name__)

class MatchLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT_SKILLS = "soft_skills"
    DOMAIN_SPECIFIC = "domain_specific"
    TOOLS = "tools"
    LANGUAGES = "languages"
    CERTIFICATIONS = "certifications"

@dataclass
class JobMatchResult:
    job_id: str
    resume_id: str
    match_score: float
    match_level: MatchLevel
    confidence: float
    reasons: List[str]
    missing_skills: List[str]
    matching_skills: List[str]
    experience_match: float
    salary_match: bool
    location_match: bool

@dataclass
class SkillClassificationResult:
    skill_name: str
    category: SkillCategory
    confidence: float
    related_skills: List[str]
    importance_score: float

@dataclass
class QualityScoreResult:
    overall_score: float
    section_scores: Dict[str, float]
    suggestions: List[str]
    strengths: List[str]
    weaknesses: List[str]

class JobMatchClassifier:
    """Classifier for job-resume matching"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.is_loaded = False
    
    async def load_model(self) -> bool:
        """Load the trained job matching model"""
        try:
            if 'job_matching' in model_trainer.models:
                self.model = model_trainer.models['job_matching']
                self.scaler = model_trainer.scalers.get('job_matching')
                self.is_loaded = True
                logger.info("Job matching model loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading job matching model: {e}")
            return False
    
    async def predict_match(self, job_data: Dict[str, Any], 
                           resume_data: Dict[str, Any]) -> JobMatchResult:
        """Predict job-resume match"""
        try:
            if not self.is_loaded:
                await self.load_model()
            
            # Extract features
            features = await self._extract_features(job_data, resume_data)
            
            # Predict if model is available
            if self.model and self.scaler:
                feature_array = np.array([list(features.values())])
                feature_array_scaled = self.scaler.transform(feature_array)
                
                match_probability = self.model.predict_proba(feature_array_scaled)[0][1]
                confidence = max(match_probability, 1 - match_probability)
            else:
                # Fallback to rule-based scoring
                match_probability = await self._rule_based_match(features)
                confidence = 0.7
            
            # Determine match level
            match_level = self._get_match_level(match_probability)
            
            # Generate explanations
            reasons, missing_skills, matching_skills = await self._generate_explanations(
                job_data, resume_data, features
            )
            
            return JobMatchResult(
                job_id=job_data.get('id', ''),
                resume_id=resume_data.get('id', ''),
                match_score=float(match_probability),
                match_level=match_level,
                confidence=float(confidence),
                reasons=reasons,
                missing_skills=missing_skills,
                matching_skills=matching_skills,
                experience_match=features.get('experience_match', 0.0),
                salary_match=features.get('salary_in_range', 0) == 1,
                location_match=features.get('location_match', 0) == 1
            )
            
        except Exception as e:
            logger.error(f"Error predicting job match: {e}")
            # Return default result
            return JobMatchResult(
                job_id=job_data.get('id', ''),
                resume_id=resume_data.get('id', ''),
                match_score=0.0,
                match_level=MatchLevel.POOR,
                confidence=0.0,
                reasons=["Error in analysis"],
                missing_skills=[],
                matching_skills=[],
                experience_match=0.0,
                salary_match=False,
                location_match=False
            )
    
    async def _extract_features(self, job_data: Dict[str, Any], 
                               resume_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for job-resume pair"""
        features = {}
        
        # Skills matching
        job_skills = set([skill.lower() for skill in job_data.get('required_skills', []) + job_data.get('preferred_skills', [])])
        resume_skills = set([skill.lower() for skill in resume_data.get('skills', [])])
        
        features['skills_overlap'] = len(job_skills.intersection(resume_skills))
        features['skills_coverage'] = len(job_skills.intersection(resume_skills)) / len(job_skills) if job_skills else 0
        features['total_resume_skills'] = len(resume_skills)
        features['total_job_skills'] = len(job_skills)
        
        # Experience matching
        resume_experience = resume_data.get('total_experience_years', 0)
        required_experience = job_data.get('experience_level_numeric', 2)
        
        features['years_experience'] = resume_experience
        features['required_experience'] = required_experience
        features['experience_match'] = min(resume_experience / required_experience, 1.0) if required_experience > 0 else 1.0
        
        # Education
        features['has_degree'] = 1 if resume_data.get('education') else 0
        features['education_level'] = len(resume_data.get('education', []))
        
        # Location
        job_location = job_data.get('location', '').lower()
        resume_location = resume_data.get('location', '').lower()
        features['location_match'] = 1 if job_location and resume_location and job_location in resume_location else 0
        
        # Salary
        salary_range = job_data.get('salary_range', {})
        expected_salary = resume_data.get('expected_salary', 0)
        
        if salary_range and expected_salary:
            salary_min = salary_range.get('min', 0)
            salary_max = salary_range.get('max', 0)
            features['salary_in_range'] = 1 if salary_min <= expected_salary <= salary_max else 0
            features['salary_expectation_ratio'] = expected_salary / ((salary_min + salary_max) / 2) if (salary_min + salary_max) > 0 else 1
        else:
            features['salary_in_range'] = 0
            features['salary_expectation_ratio'] = 1
        
        # Remote work
        features['remote_preference_match'] = 1 if job_data.get('remote_option', False) else 0
        
        # Semantic similarity
        job_text = f"{job_data.get('title', '')} {job_data.get('description', '')}"
        resume_text = f"{resume_data.get('summary', '')} {' '.join(resume_data.get('experience_descriptions', []))}"
        
        if job_text.strip() and resume_text.strip():
            similarity = await embedding_manager.calculate_similarity(job_text, resume_text)
            features['semantic_similarity'] = similarity
        else:
            features['semantic_similarity'] = 0.0
        
        return features
    
    async def _rule_based_match(self, features: Dict[str, float]) -> float:
        """Rule-based matching when ML model is not available"""
        score = 0.0
        
        # Skills matching (40% weight)
        skills_score = features.get('skills_coverage', 0) * 0.4
        score += skills_score
        
        # Experience matching (25% weight)
        exp_score = features.get('experience_match', 0) * 0.25
        score += exp_score
        
        # Semantic similarity (20% weight)
        semantic_score = features.get('semantic_similarity', 0) * 0.2
        score += semantic_score
        
        # Location match (10% weight)
        location_score = features.get('location_match', 0) * 0.1
        score += location_score
        
        # Salary match (5% weight)
        salary_score = features.get('salary_in_range', 0) * 0.05
        score += salary_score
        
        return min(score, 1.0)
    
    def _get_match_level(self, score: float) -> MatchLevel:
        """Convert score to match level"""
        if score >= 0.8:
            return MatchLevel.EXCELLENT
        elif score >= 0.6:
            return MatchLevel.GOOD
        elif score >= 0.4:
            return MatchLevel.FAIR
        else:
            return MatchLevel.POOR
    
    async def _generate_explanations(self, job_data: Dict[str, Any], 
                                   resume_data: Dict[str, Any], 
                                   features: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
        """Generate explanations for the match"""
        reasons = []
        missing_skills = []
        matching_skills = []
        
        # Skills analysis
        job_skills = set([skill.lower() for skill in job_data.get('required_skills', []) + job_data.get('preferred_skills', [])])
        resume_skills = set([skill.lower() for skill in resume_data.get('skills', [])])
        
        matching_skills = list(job_skills.intersection(resume_skills))
        missing_skills = list(job_skills - resume_skills)
        
        if matching_skills:
            reasons.append(f"Strong skill match with {len(matching_skills)} relevant skills")
        
        if missing_skills:
            reasons.append(f"Missing {len(missing_skills)} required skills")
        
        # Experience analysis
        exp_match = features.get('experience_match', 0)
        if exp_match >= 1.0:
            reasons.append("Meets or exceeds experience requirements")
        elif exp_match >= 0.7:
            reasons.append("Good experience level for the role")
        else:
            reasons.append("Limited experience for this role")
        
        # Semantic similarity
        semantic_sim = features.get('semantic_similarity', 0)
        if semantic_sim >= 0.7:
            reasons.append("Excellent content alignment with job requirements")
        elif semantic_sim >= 0.5:
            reasons.append("Good content alignment with job requirements")
        
        # Location and salary
        if features.get('location_match', 0) == 1:
            reasons.append("Location preference matches")
        
        if features.get('salary_in_range', 0) == 1:
            reasons.append("Salary expectations align with job offer")
        
        return reasons, missing_skills, matching_skills

class SkillClassifier:
    """Classifier for skill categorization"""
    
    def __init__(self):
        self.skill_categories = {
            'programming': SkillCategory.TECHNICAL,
            'software': SkillCategory.TECHNICAL,
            'database': SkillCategory.TECHNICAL,
            'framework': SkillCategory.TECHNICAL,
            'communication': SkillCategory.SOFT_SKILLS,
            'leadership': SkillCategory.SOFT_SKILLS,
            'management': SkillCategory.SOFT_SKILLS,
            'language': SkillCategory.LANGUAGES,
            'certification': SkillCategory.CERTIFICATIONS
        }
        
        self.technical_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'aws', 'azure',
            'docker', 'kubernetes', 'git', 'jenkins', 'terraform'
        ]
        
        self.soft_skill_keywords = [
            'communication', 'leadership', 'teamwork', 'problem-solving',
            'analytical', 'creative', 'organized', 'detail-oriented'
        ]
    
    async def classify_skill(self, skill_name: str) -> SkillClassificationResult:
        """Classify a single skill"""
        try:
            skill_lower = skill_name.lower()
            
            # Direct keyword matching
            category = SkillCategory.TECHNICAL  # Default
            confidence = 0.5
            
            if any(keyword in skill_lower for keyword in self.technical_keywords):
                category = SkillCategory.TECHNICAL
                confidence = 0.9
            elif any(keyword in skill_lower for keyword in self.soft_skill_keywords):
                category = SkillCategory.SOFT_SKILLS
                confidence = 0.9
            elif 'language' in skill_lower or any(lang in skill_lower for lang in ['english', 'spanish', 'french', 'german']):
                category = SkillCategory.LANGUAGES
                confidence = 0.8
            elif any(cert in skill_lower for cert in ['certified', 'certification', 'aws', 'azure', 'cisco']):
                category = SkillCategory.CERTIFICATIONS
                confidence = 0.8
            
            # Find related skills using embeddings
            related_skills = await self._find_related_skills(skill_name)
            
            # Calculate importance score (simplified)
            importance_score = self._calculate_importance_score(skill_name, category)
            
            return SkillClassificationResult(
                skill_name=skill_name,
                category=category,
                confidence=confidence,
                related_skills=related_skills,
                importance_score=importance_score
            )
            
        except Exception as e:
            logger.error(f"Error classifying skill '{skill_name}': {e}")
            return SkillClassificationResult(
                skill_name=skill_name,
                category=SkillCategory.TECHNICAL,
                confidence=0.0,
                related_skills=[],
                importance_score=0.5
            )
    
    async def _find_related_skills(self, skill_name: str) -> List[str]:
        """Find skills related to the given skill"""
        try:
            # This would ideally use a pre-built skill similarity index
            # For now, return some basic related skills
            skill_lower = skill_name.lower()
            
            related_map = {
                'python': ['django', 'flask', 'pandas', 'numpy'],
                'javascript': ['react', 'node.js', 'typescript', 'vue'],
                'java': ['spring', 'hibernate', 'maven', 'gradle'],
                'sql': ['mysql', 'postgresql', 'oracle', 'mongodb'],
                'aws': ['ec2', 's3', 'lambda', 'cloudformation'],
                'react': ['javascript', 'redux', 'jsx', 'typescript']
            }
            
            for key, related in related_map.items():
                if key in skill_lower:
                    return related
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding related skills: {e}")
            return []
    
    def _calculate_importance_score(self, skill_name: str, category: SkillCategory) -> float:
        """Calculate importance score for a skill"""
        # Simplified importance scoring
        skill_lower = skill_name.lower()
        
        # High-demand technical skills
        high_demand = ['python', 'javascript', 'react', 'aws', 'docker', 'kubernetes']
        if any(skill in skill_lower for skill in high_demand):
            return 0.9
        
        # Important soft skills
        important_soft = ['communication', 'leadership', 'problem-solving']
        if any(skill in skill_lower for skill in important_soft):
            return 0.8
        
        # Category-based scoring
        if category == SkillCategory.TECHNICAL:
            return 0.7
        elif category == SkillCategory.SOFT_SKILLS:
            return 0.6
        elif category == SkillCategory.CERTIFICATIONS:
            return 0.8
        else:
            return 0.5

class QualityScorer:
    """Classifier for resume quality scoring"""
    
    def __init__(self):
        self.section_weights = {
            'summary': 0.2,
            'experience': 0.3,
            'skills': 0.2,
            'education': 0.15,
            'formatting': 0.1,
            'completeness': 0.05
        }
    
    async def score_resume(self, resume_data: Dict[str, Any]) -> QualityScoreResult:
        """Score resume quality"""
        try:
            section_scores = {}
            suggestions = []
            strengths = []
            weaknesses = []
            
            # Score each section
            section_scores['summary'] = self._score_summary(resume_data.get('summary', ''))
            section_scores['experience'] = self._score_experience(resume_data.get('experience', []))
            section_scores['skills'] = self._score_skills(resume_data.get('skills', []))
            section_scores['education'] = self._score_education(resume_data.get('education', []))
            section_scores['formatting'] = self._score_formatting(resume_data)
            section_scores['completeness'] = self._score_completeness(resume_data)
            
            # Calculate overall score
            overall_score = sum(
                score * self.section_weights[section] 
                for section, score in section_scores.items()
            )
            
            # Generate suggestions and feedback
            suggestions, strengths, weaknesses = self._generate_feedback(section_scores, resume_data)
            
            return QualityScoreResult(
                overall_score=overall_score,
                section_scores=section_scores,
                suggestions=suggestions,
                strengths=strengths,
                weaknesses=weaknesses
            )
            
        except Exception as e:
            logger.error(f"Error scoring resume: {e}")
            return QualityScoreResult(
                overall_score=0.0,
                section_scores={},
                suggestions=["Error analyzing resume"],
                strengths=[],
                weaknesses=[]
            )
    
    def _score_summary(self, summary: str) -> float:
        """Score the resume summary section"""
        if not summary:
            return 0.0
        
        score = 0.0
        
        # Length check
        word_count = len(summary.split())
        if 50 <= word_count <= 150:
            score += 0.4
        elif 30 <= word_count <= 200:
            score += 0.3
        elif word_count > 0:
            score += 0.1
        
        # Keywords and phrases
        positive_phrases = ['experienced', 'skilled', 'proven', 'successful', 'expert']
        if any(phrase in summary.lower() for phrase in positive_phrases):
            score += 0.2
        
        # Action words
        action_words = ['led', 'managed', 'developed', 'implemented', 'created']
        if any(word in summary.lower() for word in action_words):
            score += 0.2
        
        # Quantifiable achievements
        if any(char.isdigit() for char in summary):
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_experience(self, experience: List[Dict[str, Any]]) -> float:
        """Score the experience section"""
        if not experience:
            return 0.0
        
        score = 0.0
        
        # Number of positions
        if len(experience) >= 3:
            score += 0.3
        elif len(experience) >= 1:
            score += 0.2
        
        # Description quality
        for exp in experience:
            description = exp.get('description', '')
            if len(description) > 100:
                score += 0.1
            
            # Action words and achievements
            action_words = ['led', 'managed', 'developed', 'improved', 'increased']
            if any(word in description.lower() for word in action_words):
                score += 0.1
            
            # Quantifiable results
            if any(char.isdigit() for char in description):
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_skills(self, skills: List[str]) -> float:
        """Score the skills section"""
        if not skills:
            return 0.0
        
        score = 0.0
        
        # Number of skills
        if len(skills) >= 10:
            score += 0.4
        elif len(skills) >= 5:
            score += 0.3
        elif len(skills) >= 1:
            score += 0.2
        
        # Skill diversity
        technical_count = sum(1 for skill in skills if any(
            tech in skill.lower() for tech in ['python', 'java', 'sql', 'aws']
        ))
        
        if technical_count >= 5:
            score += 0.3
        elif technical_count >= 3:
            score += 0.2
        
        # Soft skills presence
        soft_skills = ['communication', 'leadership', 'teamwork']
        if any(soft in ' '.join(skills).lower() for soft in soft_skills):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_education(self, education: List[Dict[str, Any]]) -> float:
        """Score the education section"""
        if not education:
            return 0.5  # Not everyone needs formal education
        
        score = 0.5  # Base score for having education
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            
            # Degree level
            if 'phd' in degree or 'doctorate' in degree:
                score += 0.3
            elif 'master' in degree or 'mba' in degree:
                score += 0.2
            elif 'bachelor' in degree:
                score += 0.15
            elif 'associate' in degree:
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_formatting(self, resume_data: Dict[str, Any]) -> float:
        """Score formatting and structure"""
        score = 0.0
        
        # Required sections present
        required_sections = ['contact_info', 'experience', 'skills']
        present_sections = sum(1 for section in required_sections if resume_data.get(section))
        score += (present_sections / len(required_sections)) * 0.5
        
        # Contact information completeness
        contact = resume_data.get('contact_info', {})
        if contact.get('email'):
            score += 0.2
        if contact.get('phone'):
            score += 0.1
        if contact.get('location'):
            score += 0.1
        if contact.get('linkedin'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_completeness(self, resume_data: Dict[str, Any]) -> float:
        """Score overall completeness"""
        sections = ['summary', 'experience', 'skills', 'education', 'contact_info']
        completed = sum(1 for section in sections if resume_data.get(section))
        return completed / len(sections)
    
    def _generate_feedback(self, section_scores: Dict[str, float], 
                          resume_data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Generate detailed feedback"""
        suggestions = []
        strengths = []
        weaknesses = []
        
        # Analyze each section
        for section, score in section_scores.items():
            if score >= 0.8:
                strengths.append(f"Excellent {section} section")
            elif score <= 0.4:
                weaknesses.append(f"Weak {section} section needs improvement")
                
                # Specific suggestions
                if section == 'summary' and not resume_data.get('summary'):
                    suggestions.append("Add a professional summary highlighting your key achievements")
                elif section == 'experience' and len(resume_data.get('experience', [])) < 2:
                    suggestions.append("Add more detailed work experience with quantifiable achievements")
                elif section == 'skills' and len(resume_data.get('skills', [])) < 5:
                    suggestions.append("Include more relevant technical and soft skills")
        
        # General suggestions
        if section_scores.get('completeness', 0) < 0.8:
            suggestions.append("Complete all resume sections for better impact")
        
        if not strengths:
            strengths.append("Resume shows professional experience")
        
        return suggestions, strengths, weaknesses

# Global classifier instances
job_match_classifier = JobMatchClassifier()
skill_classifier = SkillClassifier()
quality_scorer = QualityScorer()

# Convenience functions
async def match_job_resume(job_data: Dict[str, Any], resume_data: Dict[str, Any]) -> JobMatchResult:
    """Match a job with a resume"""
    return await job_match_classifier.predict_match(job_data, resume_data)

async def classify_skills(skills: List[str]) -> List[SkillClassificationResult]:
    """Classify a list of skills"""
    results = []
    for skill in skills:
        result = await skill_classifier.classify_skill(skill)
        results.append(result)
    return results

async def score_resume_quality(resume_data: Dict[str, Any]) -> QualityScoreResult:
    """Score resume quality"""
    return await quality_scorer.score_resume(resume_data)

# Initialize classifiers
async def initialize_classifiers():
    """Initialize classifiers on application startup"""
    logger.info("Initializing classifiers...")
    await job_match_classifier.load_model()
    logger.info("Classifiers initialization complete")
