"""
Skill Gap Detection Service for analyzing differences between user skills and job requirements.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timezone

from ..models.resume import Resume
from ..models.job import Job
from ..models.skill_gap import SkillGap, SkillGapAnalysis
from ..ml.classifiers import SkillClassifier, QualityScorer
from ..ml.embeddings import EmbeddingManager
from ..services.ai_service import AIService
from ..utils.helpers import normalize_skill_name, calculate_match_score, extract_keywords

logger = logging.getLogger(__name__)

class SkillGapDetector:
    """Service for detecting and analyzing skill gaps between resumes and job requirements."""
    
    def __init__(self, ai_service: AIService, embedding_manager: EmbeddingManager):
        self.ai_service = ai_service
        self.embedding_manager = embedding_manager
        self.skill_classifier = SkillClassifier()
        self.quality_scorer = QualityScorer()
    
    async def analyze_skill_gap(self, resume: Resume, job: Job) -> SkillGapAnalysis:
        """
        Perform comprehensive skill gap analysis between a resume and job posting.
        
        Args:
            resume: The user's resume
            job: The target job posting
            
        Returns:
            SkillGapAnalysis containing detailed gap analysis
        """
        try:
            logger.info(f"Starting skill gap analysis for resume {resume.id} and job {job.id}")
            
            # Extract skills from resume and job
            resume_skills = self._extract_resume_skills(resume)
            job_requirements = self._extract_job_requirements(job)
            
            # Perform skill matching
            matching_skills, missing_skills, additional_skills = self._match_skills(
                resume_skills, job_requirements
            )
            
            # Calculate skill similarity scores
            skill_similarities = await self._calculate_skill_similarities(
                resume_skills, job_requirements
            )
            
            # Analyze skill categories
            skill_categories = self._categorize_skills(missing_skills)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                missing_skills, skill_categories, job
            )
            
            # Calculate overall gap score
            gap_score = self._calculate_gap_score(
                len(matching_skills), len(missing_skills), len(job_requirements)
            )
            
            # Create skill gap analysis
            analysis = SkillGapAnalysis(
                resume_id=str(resume.id),
                job_id=str(job.id),
                matching_skills=matching_skills,
                missing_skills=missing_skills,
                additional_skills=additional_skills,
                skill_similarities=skill_similarities,
                skill_categories=skill_categories,
                gap_score=gap_score,
                recommendations=recommendations,
                analysis_date=datetime.now(timezone.utc)
            )
            
            logger.info(f"Skill gap analysis completed with gap score: {gap_score}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in skill gap analysis: {str(e)}")
            raise
    
    def _extract_resume_skills(self, resume: Resume) -> List[str]:
        """Extract all skills from resume."""
        skills = []
        
        # Direct skills
        if resume.skills:
            skills.extend(resume.skills)
        
        # Skills from experience descriptions
        if resume.experience:
            for exp in resume.experience:
                if exp.description:
                    skills.extend(self._extract_skills_from_text(exp.description))
        
        # Skills from projects
        if resume.projects:
            for project in resume.projects:
                if project.description:
                    skills.extend(self._extract_skills_from_text(project.description))
                if project.technologies:
                    skills.extend(project.technologies)
        
        # Normalize and deduplicate
        normalized_skills = [normalize_skill_name(skill) for skill in skills]
        return list(set(filter(None, normalized_skills)))
    
    def _extract_job_requirements(self, job: Job) -> List[str]:
        """Extract skill requirements from job posting."""
        requirements = []
        
        # Direct requirements
        if job.required_skills:
            requirements.extend(job.required_skills)
        
        if job.preferred_skills:
            requirements.extend(job.preferred_skills)
        
        # Extract from job description
        if job.description:
            requirements.extend(self._extract_skills_from_text(job.description))
        
        # Extract from qualifications
        if hasattr(job, 'qualifications') and job.qualifications:
            requirements.extend(self._extract_skills_from_text(job.qualifications))
        
        # Normalize and deduplicate
        normalized_requirements = [normalize_skill_name(skill) for skill in requirements]
        return list(set(filter(None, normalized_requirements)))
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract potential skills from text using keyword matching and AI classification."""
        # Common technology keywords
        tech_keywords = [
            'python', 'java', 'javascript', 'typescript', 'react', 'vue', 'angular',
            'node.js', 'express', 'django', 'flask', 'spring', 'hibernate',
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'git', 'github', 'gitlab', 'jira', 'confluence',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'pandas', 'numpy', 'scikit-learn', 'spark', 'hadoop',
            'html', 'css', 'sass', 'bootstrap', 'tailwind',
            'rest api', 'graphql', 'microservices', 'agile', 'scrum'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        # Match known keywords
        for keyword in tech_keywords:
            if keyword in text_lower:
                found_skills.append(keyword)
        
        # Extract other potential skills using regex patterns
        keywords = extract_keywords(text, min_length=2)
        
        # Filter keywords that might be skills (length > 2, not common words)
        potential_skills = [
            kw for kw in keywords 
            if len(kw) > 2 and kw not in ['experience', 'development', 'management', 'team']
        ]
        
        found_skills.extend(potential_skills)
        
        return list(set(found_skills))
    
    def _match_skills(self, resume_skills: List[str], job_requirements: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Match skills between resume and job requirements."""
        resume_skills_set = set(resume_skills)
        job_requirements_set = set(job_requirements)
        
        # Exact matches
        matching_skills = list(resume_skills_set.intersection(job_requirements_set))
        
        # Missing skills (in job but not in resume)
        missing_skills = list(job_requirements_set - resume_skills_set)
        
        # Additional skills (in resume but not required by job)
        additional_skills = list(resume_skills_set - job_requirements_set)
        
        # Try fuzzy matching for missing skills
        fuzzy_matches = self._fuzzy_match_skills(missing_skills, resume_skills)
        
        # Move fuzzy matches from missing to matching
        for missing_skill, matched_skill in fuzzy_matches.items():
            if missing_skill in missing_skills:
                missing_skills.remove(missing_skill)
                matching_skills.append(f"{missing_skill} (~{matched_skill})")
        
        return matching_skills, missing_skills, additional_skills
    
    def _fuzzy_match_skills(self, missing_skills: List[str], resume_skills: List[str]) -> Dict[str, str]:
        """Perform fuzzy matching to find similar skills."""
        matches = {}
        
        # Skill synonyms mapping
        skill_synonyms = {
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'react.js': 'react',
            'vue.js': 'vue',
            'node.js': 'nodejs',
            'postgresql': 'postgres',
            'mysql': 'sql',
            'ai': 'artificial intelligence',
            'ml': 'machine learning'
        }
        
        for missing_skill in missing_skills:
            # Check direct synonyms
            for resume_skill in resume_skills:
                if (skill_synonyms.get(missing_skill) == resume_skill or 
                    skill_synonyms.get(resume_skill) == missing_skill):
                    matches[missing_skill] = resume_skill
                    break
            
            # Check partial matches
            if missing_skill not in matches:
                for resume_skill in resume_skills:
                    if (missing_skill in resume_skill or resume_skill in missing_skill):
                        if abs(len(missing_skill) - len(resume_skill)) <= 3:
                            matches[missing_skill] = resume_skill
                            break
        
        return matches
    
    async def _calculate_skill_similarities(self, resume_skills: List[str], job_requirements: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity between skills using embeddings."""
        similarities = {}
        
        try:
            # Create embeddings for all skills
            all_skills = resume_skills + job_requirements
            embeddings = await self.embedding_manager.create_embeddings(all_skills)
            
            # Calculate similarities between resume skills and job requirements
            for i, resume_skill in enumerate(resume_skills):
                max_similarity = 0.0
                for j, job_skill in enumerate(job_requirements):
                    # Calculate cosine similarity
                    similarity = self.embedding_manager.calculate_similarity(
                        embeddings[i], embeddings[len(resume_skills) + j]
                    )
                    max_similarity = max(max_similarity, similarity)
                
                similarities[resume_skill] = max_similarity
        
        except Exception as e:
            logger.warning(f"Error calculating skill similarities: {str(e)}")
            # Fallback to basic text similarity
            for resume_skill in resume_skills:
                max_similarity = 0.0
                for job_skill in job_requirements:
                    similarity = self._basic_text_similarity(resume_skill, job_skill)
                    max_similarity = max(max_similarity, similarity)
                similarities[resume_skill] = max_similarity
        
        return similarities
    
    def _basic_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity using character overlap."""
        if not text1 or not text2:
            return 0.0
        
        text1, text2 = text1.lower(), text2.lower()
        
        # Calculate Jaccard similarity
        set1, set2 = set(text1), set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize skills into different types."""
        categories = {
            'programming_languages': [],
            'frameworks': [],
            'databases': [],
            'cloud_platforms': [],
            'tools': [],
            'soft_skills': [],
            'other': []
        }
        
        # Skill category mappings
        category_mappings = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'frameworks': [
                'react', 'vue', 'angular', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'tensorflow', 'pytorch', 'scikit-learn'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'cassandra', 'dynamodb', 'sql', 'nosql'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'heroku', 'vercel'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'jenkins', 'jira', 'confluence',
                'slack', 'figma', 'photoshop', 'illustrator'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum'
            ]
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            categorized = False
            
            for category, keywords in category_mappings.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categories[category].append(skill)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    async def _generate_recommendations(self, missing_skills: List[str], skill_categories: Dict[str, List[str]], job: Job) -> List[str]:
        """Generate learning recommendations for missing skills."""
        recommendations = []
        
        if not missing_skills:
            return ["Great! You have all the required skills for this position."]
        
        try:
            # Generate AI-powered recommendations
            prompt = f"""
            Based on the missing skills for a {job.title} position at {job.company}, 
            provide specific learning recommendations.
            
            Missing skills: {', '.join(missing_skills[:10])}
            
            Please provide 3-5 actionable recommendations for acquiring these skills,
            including specific resources, courses, or learning paths.
            Format as a bulleted list.
            """
            
            ai_recommendations = await self.ai_service.generate_content(prompt)
            
            if ai_recommendations:
                # Parse AI recommendations
                recommendations.extend(self._parse_ai_recommendations(ai_recommendations))
        
        except Exception as e:
            logger.warning(f"Error generating AI recommendations: {str(e)}")
        
        # Fallback to rule-based recommendations
        if not recommendations:
            recommendations = self._generate_fallback_recommendations(missing_skills, skill_categories)
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def _parse_ai_recommendations(self, ai_text: str) -> List[str]:
        """Parse AI-generated recommendations into a list."""
        recommendations = []
        
        # Split by bullet points or numbered lists
        lines = ai_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or 
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                # Clean up the line
                clean_line = line.lstrip('•-*0123456789. ').strip()
                if clean_line:
                    recommendations.append(clean_line)
        
        return recommendations
    
    def _generate_fallback_recommendations(self, missing_skills: List[str], skill_categories: Dict[str, List[str]]) -> List[str]:
        """Generate rule-based recommendations as fallback."""
        recommendations = []
        
        # Programming languages
        if 'programming_languages' in skill_categories:
            langs = skill_categories['programming_languages']
            recommendations.append(
                f"Consider learning {', '.join(langs[:3])} through online platforms like "
                "Codecademy, freeCodeCamp, or LeetCode"
            )
        
        # Frameworks
        if 'frameworks' in skill_categories:
            frameworks = skill_categories['frameworks']
            recommendations.append(
                f"Build projects using {', '.join(frameworks[:3])} to gain hands-on experience"
            )
        
        # Cloud platforms
        if 'cloud_platforms' in skill_categories:
            platforms = skill_categories['cloud_platforms']
            recommendations.append(
                f"Get certified in {', '.join(platforms[:2])} through official training programs"
            )
        
        # Databases
        if 'databases' in skill_categories:
            dbs = skill_categories['databases']
            recommendations.append(
                f"Practice {', '.join(dbs[:3])} through database design projects and tutorials"
            )
        
        # General recommendation
        if len(missing_skills) > 5:
            recommendations.append(
                "Focus on the most critical skills first, then gradually expand your skill set"
            )
        
        # If no specific recommendations generated
        if not recommendations:
            recommendations.append(
                "Consider taking online courses or tutorials to learn the missing skills"
            )
            recommendations.append(
                "Build personal projects that incorporate these technologies"
            )
            recommendations.append(
                "Join developer communities and forums to learn from others"
            )
        
        return recommendations
    
    def _calculate_gap_score(self, matching_count: int, missing_count: int, total_requirements: int) -> float:
        """Calculate overall skill gap score (0-100, higher is better)."""
        if total_requirements == 0:
            return 100.0
        
        match_percentage = (matching_count / total_requirements) * 100
        
        # Apply penalty for missing critical skills
        penalty = min(missing_count * 10, 50)  # Max 50% penalty
        
        final_score = max(0, match_percentage - penalty)
        
        return round(final_score, 2)
    
    async def compare_multiple_jobs(self, resume: Resume, jobs: List[Job]) -> Dict[str, SkillGapAnalysis]:
        """Compare a resume against multiple job postings."""
        analyses = {}
        
        for job in jobs:
            try:
                analysis = await self.analyze_skill_gap(resume, job)
                analyses[str(job.id)] = analysis
            except Exception as e:
                logger.error(f"Error analyzing job {job.id}: {str(e)}")
                continue
        
        return analyses
    
    def get_skill_learning_path(self, missing_skills: List[str]) -> Dict[str, Any]:
        """Generate a structured learning path for missing skills."""
        learning_path = {
            'beginner': [],
            'intermediate': [],
            'advanced': [],
            'estimated_time': '3-6 months'
        }
        
        # Categorize skills by difficulty
        beginner_skills = ['html', 'css', 'git', 'basic programming concepts']
        intermediate_skills = ['javascript', 'python', 'sql', 'rest api']
        advanced_skills = ['machine learning', 'microservices', 'system design']
        
        for skill in missing_skills:
            skill_lower = skill.lower()
            
            if any(beginner in skill_lower for beginner in beginner_skills):
                learning_path['beginner'].append(skill)
            elif any(intermediate in skill_lower for intermediate in intermediate_skills):
                learning_path['intermediate'].append(skill)
            elif any(advanced in skill_lower for advanced in advanced_skills):
                learning_path['advanced'].append(skill)
            else:
                # Default to intermediate
                learning_path['intermediate'].append(skill)
        
        # Adjust time estimate based on number of skills
        total_skills = len(missing_skills)
        if total_skills <= 3:
            learning_path['estimated_time'] = '1-2 months'
        elif total_skills <= 6:
            learning_path['estimated_time'] = '3-4 months'
        else:
            learning_path['estimated_time'] = '6-12 months'
        
        return learning_path