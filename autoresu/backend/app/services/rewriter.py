"""
Resume Rewriter Service for improving and optimizing resume content using AI.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from datetime import datetime, timezone

from ..models.resume import Resume
from ..models.job import Job
from ..services.ai_service import AIService
from ..ml.classifiers import QualityScorer
from ..utils.helpers import clean_text, extract_keywords

logger = logging.getLogger(__name__)

class ResumeRewriter:
    """Service for rewriting and improving resume content using AI."""
    
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service
        self.quality_scorer = QualityScorer()
    
    async def rewrite_resume(self, resume: Resume, target_job: Optional[Job] = None, 
                           style: str = "professional") -> Dict[str, Any]:
        """
        Rewrite entire resume for better impact and ATS optimization.
        
        Args:
            resume: The original resume to rewrite
            target_job: Optional target job for tailored optimization
            style: Writing style ('professional', 'creative', 'technical', 'executive')
            
        Returns:
            Dictionary containing rewritten resume sections and improvement metrics
        """
        try:
            logger.info(f"Starting resume rewrite for resume {resume.id}")
            
            rewritten_sections = {}
            improvement_metrics = {}
            
            # Rewrite summary/objective
            if resume.summary:
                rewritten_sections['summary'] = await self._rewrite_summary(
                    resume.summary, target_job, style
                )
                improvement_metrics['summary'] = await self._calculate_improvement(
                    resume.summary, rewritten_sections['summary']
                )
            
            # Rewrite experience descriptions
            if resume.experience:
                rewritten_sections['experience'] = []
                for exp in resume.experience:
                    rewritten_exp = await self._rewrite_experience(exp, target_job, style)
                    rewritten_sections['experience'].append(rewritten_exp)
                
                improvement_metrics['experience'] = await self._calculate_experience_improvement(
                    resume.experience, rewritten_sections['experience']
                )
            
            # Rewrite project descriptions
            if resume.projects:
                rewritten_sections['projects'] = []
                for project in resume.projects:
                    rewritten_project = await self._rewrite_project(project, target_job, style)
                    rewritten_sections['projects'].append(rewritten_project)
                
                improvement_metrics['projects'] = await self._calculate_projects_improvement(
                    resume.projects, rewritten_sections['projects']
                )
            
            # Optimize skills section
            if resume.skills:
                rewritten_sections['skills'] = await self._optimize_skills(
                    resume.skills, target_job
                )
                improvement_metrics['skills'] = self._calculate_skills_improvement(
                    resume.skills, rewritten_sections['skills']
                )
            
            # Generate overall improvement score
            overall_score = self._calculate_overall_improvement(improvement_metrics)
            
            result = {
                'rewritten_sections': rewritten_sections,
                'improvement_metrics': improvement_metrics,
                'overall_improvement_score': overall_score,
                'rewrite_date': datetime.now(timezone.utc),
                'style_used': style,
                'target_job_id': str(target_job.id) if target_job else None
            }
            
            logger.info(f"Resume rewrite completed with improvement score: {overall_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error in resume rewrite: {str(e)}")
            raise
    
    async def _rewrite_summary(self, original_summary: str, target_job: Optional[Job], 
                             style: str) -> Dict[str, Any]:
        """Rewrite resume summary/objective."""
        try:
            # Create context for AI rewriting
            context = f"Style: {style}\n"
            
            if target_job:
                context += f"Target Position: {target_job.title} at {target_job.company}\n"
                context += f"Job Requirements: {', '.join(target_job.required_skills[:10]) if target_job.required_skills else 'Not specified'}\n"
            
            prompt = f"""
            Rewrite the following resume summary to be more impactful and ATS-friendly.
            
            {context}
            
            Original Summary:
            {original_summary}
            
            Requirements:
            1. Make it more engaging and action-oriented
            2. Include relevant keywords for ATS optimization
            3. Quantify achievements where possible
            4. Keep it concise (2-3 sentences)
            5. Match the {style} tone
            
            Provide the rewritten summary only, without explanations.
            """
            
            rewritten_text = await self.ai_service.generate_content(prompt)
            
            return {
                'original': original_summary,
                'rewritten': rewritten_text.strip(),
                'word_count_change': len(rewritten_text.split()) - len(original_summary.split()),
                'keyword_density': self._calculate_keyword_density(rewritten_text, target_job)
            }
            
        except Exception as e:
            logger.warning(f"Error rewriting summary: {str(e)}")
            return {
                'original': original_summary,
                'rewritten': original_summary,
                'word_count_change': 0,
                'keyword_density': 0
            }
    
    async def _rewrite_experience(self, experience: Dict[str, Any], target_job: Optional[Job], 
                                style: str) -> Dict[str, Any]:
        """Rewrite individual experience entry."""
        try:
            context = f"Style: {style}\n"
            context += f"Position: {experience.get('position', 'Not specified')}\n"
            context += f"Company: {experience.get('company', 'Not specified')}\n"
            
            if target_job:
                context += f"Target Position: {target_job.title}\n"
                context += f"Relevant Skills: {', '.join(target_job.required_skills[:5]) if target_job.required_skills else 'Not specified'}\n"
            
            original_description = experience.get('description', '')
            
            prompt = f"""
            Rewrite the following job experience description to be more impactful and ATS-optimized.
            
            {context}
            
            Original Description:
            {original_description}
            
            Requirements:
            1. Use strong action verbs
            2. Quantify achievements with specific numbers/percentages
            3. Focus on results and impact
            4. Include relevant keywords
            5. Use bullet points format
            6. Keep {style} tone
            
            Provide only the rewritten bullet points, without explanations.
            """
            
            rewritten_description = await self.ai_service.generate_content(prompt)
            
            # Parse bullet points
            bullet_points = self._parse_bullet_points(rewritten_description)
            
            rewritten_experience = experience.copy()
            rewritten_experience['description'] = rewritten_description.strip()
            rewritten_experience['bullet_points'] = bullet_points
            rewritten_experience['improvement_metrics'] = {
                'action_verbs_added': self._count_action_verbs(rewritten_description) - self._count_action_verbs(original_description),
                'quantified_achievements': self._count_quantified_achievements(rewritten_description),
                'keyword_relevance': self._calculate_keyword_density(rewritten_description, target_job)
            }
            
            return rewritten_experience
            
        except Exception as e:
            logger.warning(f"Error rewriting experience: {str(e)}")
            return experience
    
    async def _rewrite_project(self, project: Dict[str, Any], target_job: Optional[Job], 
                             style: str) -> Dict[str, Any]:
        """Rewrite project description."""
        try:
            context = f"Style: {style}\n"
            context += f"Project: {project.get('title', 'Not specified')}\n"
            context += f"Technologies: {', '.join(project.get('technologies', [])) if project.get('technologies') else 'Not specified'}\n"
            
            if target_job:
                context += f"Target Position: {target_job.title}\n"
            
            original_description = project.get('description', '')
            
            prompt = f"""
            Rewrite the following project description to showcase technical skills and impact.
            
            {context}
            
            Original Description:
            {original_description}
            
            Requirements:
            1. Emphasize technical achievements
            2. Include specific technologies used
            3. Quantify impact (users, performance, etc.)
            4. Show problem-solving skills
            5. Keep {style} tone
            
            Provide only the rewritten description, without explanations.
            """
            
            rewritten_description = await self.ai_service.generate_content(prompt)
            
            rewritten_project = project.copy()
            rewritten_project['description'] = rewritten_description.strip()
            rewritten_project['improvement_metrics'] = {
                'technical_terms_added': self._count_technical_terms(rewritten_description) - self._count_technical_terms(original_description),
                'quantified_impact': self._count_quantified_achievements(rewritten_description),
                'keyword_relevance': self._calculate_keyword_density(rewritten_description, target_job)
            }
            
            return rewritten_project
            
        except Exception as e:
            logger.warning(f"Error rewriting project: {str(e)}")
            return project
    
    async def _optimize_skills(self, skills: List[str], target_job: Optional[Job]) -> Dict[str, Any]:
        """Optimize skills section for ATS and relevance."""
        try:
            optimized_skills = skills.copy()
            
            if target_job and target_job.required_skills:
                # Prioritize skills that match job requirements
                job_skills = [skill.lower() for skill in target_job.required_skills]
                
                # Separate matching and non-matching skills
                matching_skills = []
                other_skills = []
                
                for skill in skills:
                    if skill.lower() in job_skills:
                        matching_skills.append(skill)
                    else:
                        other_skills.append(skill)
                
                # Reorder: matching skills first
                optimized_skills = matching_skills + other_skills
                
                # Add missing high-priority skills if relevant
                missing_skills = []
                for job_skill in target_job.required_skills[:5]:  # Top 5 required skills
                    if job_skill.lower() not in [s.lower() for s in skills]:
                        missing_skills.append(job_skill)
            
            return {
                'original_skills': skills,
                'optimized_skills': optimized_skills,
                'missing_critical_skills': missing_skills if target_job else [],
                'relevance_score': self._calculate_skills_relevance(optimized_skills, target_job),
                'suggestions': self._generate_skill_suggestions(skills, target_job)
            }
            
        except Exception as e:
            logger.warning(f"Error optimizing skills: {str(e)}")
            return {
                'original_skills': skills,
                'optimized_skills': skills,
                'missing_critical_skills': [],
                'relevance_score': 0,
                'suggestions': []
            }
    
    def _parse_bullet_points(self, text: str) -> List[str]:
        """Parse bullet points from text."""
        bullet_points = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                clean_point = line.lstrip('•-* ').strip()
                if clean_point:
                    bullet_points.append(clean_point)
        
        return bullet_points
    
    def _count_action_verbs(self, text: str) -> int:
        """Count action verbs in text."""
        action_verbs = [
            'achieved', 'administered', 'analyzed', 'architected', 'built', 'collaborated',
            'created', 'delivered', 'designed', 'developed', 'enhanced', 'established',
            'executed', 'implemented', 'improved', 'increased', 'led', 'managed',
            'optimized', 'organized', 'reduced', 'resolved', 'spearheaded', 'streamlined'
        ]
        
        count = 0
        text_lower = text.lower()
        for verb in action_verbs:
            count += text_lower.count(verb)
        
        return count
    
    def _count_quantified_achievements(self, text: str) -> int:
        """Count quantified achievements (numbers, percentages)."""
        # Look for numbers and percentages
        number_patterns = [
            r'\d+%',  # Percentages
            r'\d+\+',  # Numbers with plus
            r'\$\d+',  # Dollar amounts
            r'\d+k',   # Thousands (e.g., 10k users)
            r'\d+m',   # Millions
            r'\d+x',   # Multipliers
            r'\d+ (users|customers|projects|years|months|weeks)',  # Specific quantities
        ]
        
        count = 0
        for pattern in number_patterns:
            matches = re.findall(pattern, text.lower())
            count += len(matches)
        
        return count
    
    def _count_technical_terms(self, text: str) -> int:
        """Count technical terms in text."""
        technical_terms = [
            'api', 'database', 'framework', 'algorithm', 'architecture', 'deployment',
            'optimization', 'integration', 'automation', 'scalability', 'performance',
            'security', 'testing', 'debugging', 'monitoring', 'analytics'
        ]
        
        count = 0
        text_lower = text.lower()
        for term in technical_terms:
            count += text_lower.count(term)
        
        return count
    
    def _calculate_keyword_density(self, text: str, target_job: Optional[Job]) -> float:
        """Calculate keyword density for ATS optimization."""
        if not target_job or not target_job.required_skills:
            return 0.0
        
        text_lower = text.lower()
        total_keywords = len(target_job.required_skills)
        found_keywords = 0
        
        for skill in target_job.required_skills:
            if skill.lower() in text_lower:
                found_keywords += 1
        
        return (found_keywords / total_keywords) * 100 if total_keywords > 0 else 0.0
    
    def _calculate_skills_relevance(self, skills: List[str], target_job: Optional[Job]) -> float:
        """Calculate how relevant skills are to target job."""
        if not target_job or not target_job.required_skills:
            return 100.0  # Default high score if no target
        
        skill_set = set(skill.lower() for skill in skills)
        job_skill_set = set(skill.lower() for skill in target_job.required_skills)
        
        matching_skills = skill_set.intersection(job_skill_set)
        
        return (len(matching_skills) / len(job_skill_set)) * 100 if job_skill_set else 0.0
    
    def _generate_skill_suggestions(self, current_skills: List[str], target_job: Optional[Job]) -> List[str]:
        """Generate skill improvement suggestions."""
        suggestions = []
        
        if not target_job:
            return ["Consider adding more specific technical skills relevant to your target roles"]
        
        # Find missing critical skills
        current_skill_set = set(skill.lower() for skill in current_skills)
        job_skill_set = set(skill.lower() for skill in target_job.required_skills or [])
        
        missing_skills = job_skill_set - current_skill_set
        
        if missing_skills:
            suggestions.append(f"Add these missing skills: {', '.join(list(missing_skills)[:5])}")
        
        # Check for skill groupings
        if 'python' in current_skill_set and 'machine learning' not in current_skill_set:
            suggestions.append("Consider adding machine learning frameworks like TensorFlow or scikit-learn")
        
        if 'javascript' in current_skill_set and 'react' not in current_skill_set:
            suggestions.append("Consider adding popular JavaScript frameworks like React or Vue.js")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    async def _calculate_improvement(self, original: str, rewritten: str) -> Dict[str, Any]:
        """Calculate improvement metrics between original and rewritten text."""
        try:
            # Basic metrics
            original_words = len(original.split())
            rewritten_words = len(rewritten.split())
            
            # Action verb counts
            original_action_verbs = self._count_action_verbs(original)
            rewritten_action_verbs = self._count_action_verbs(rewritten)
            
            # Quantified achievements
            original_achievements = self._count_quantified_achievements(original)
            rewritten_achievements = self._count_quantified_achievements(rewritten)
            
            # Calculate improvement score
            improvement_score = 0
            
            if rewritten_action_verbs > original_action_verbs:
                improvement_score += 20
            
            if rewritten_achievements > original_achievements:
                improvement_score += 30
            
            if rewritten_words > original_words:
                improvement_score += 10
            
            # Readability improvement (simplified)
            if len(rewritten.split('.')) > len(original.split('.')):
                improvement_score += 10
            
            return {
                'word_count_change': rewritten_words - original_words,
                'action_verbs_added': rewritten_action_verbs - original_action_verbs,
                'achievements_added': rewritten_achievements - original_achievements,
                'improvement_score': min(improvement_score, 100)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating improvement: {str(e)}")
            return {
                'word_count_change': 0,
                'action_verbs_added': 0,
                'achievements_added': 0,
                'improvement_score': 0
            }
    
    async def _calculate_experience_improvement(self, original_experience: List[Dict], 
                                             rewritten_experience: List[Dict]) -> Dict[str, Any]:
        """Calculate improvement metrics for experience section."""
        if not original_experience or not rewritten_experience:
            return {'overall_improvement': 0}
        
        total_improvement = 0
        improvements = []
        
        for i, (orig, rewritten) in enumerate(zip(original_experience, rewritten_experience)):
            if 'improvement_metrics' in rewritten:
                improvements.append(rewritten['improvement_metrics'])
                total_improvement += rewritten['improvement_metrics'].get('action_verbs_added', 0)
        
        return {
            'total_action_verbs_added': total_improvement,
            'entries_improved': len(improvements),
            'average_improvement': total_improvement / len(improvements) if improvements else 0,
            'individual_improvements': improvements
        }
    
    async def _calculate_projects_improvement(self, original_projects: List[Dict], 
                                            rewritten_projects: List[Dict]) -> Dict[str, Any]:
        """Calculate improvement metrics for projects section."""
        if not original_projects or not rewritten_projects:
            return {'overall_improvement': 0}
        
        total_technical_terms = 0
        improvements = []
        
        for i, (orig, rewritten) in enumerate(zip(original_projects, rewritten_projects)):
            if 'improvement_metrics' in rewritten:
                improvements.append(rewritten['improvement_metrics'])
                total_technical_terms += rewritten['improvement_metrics'].get('technical_terms_added', 0)
        
        return {
            'total_technical_terms_added': total_technical_terms,
            'projects_improved': len(improvements),
            'average_improvement': total_technical_terms / len(improvements) if improvements else 0,
            'individual_improvements': improvements
        }
    
    def _calculate_skills_improvement(self, original_skills: List[str], 
                                    optimized_skills: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvement metrics for skills section."""
        return {
            'skills_reordered': original_skills != optimized_skills.get('optimized_skills', []),
            'relevance_score': optimized_skills.get('relevance_score', 0),
            'missing_skills_identified': len(optimized_skills.get('missing_critical_skills', [])),
            'suggestions_provided': len(optimized_skills.get('suggestions', []))
        }
    
    def _calculate_overall_improvement(self, improvement_metrics: Dict[str, Any]) -> float:
        """Calculate overall improvement score."""
        total_score = 0
        section_count = 0
        
        for section, metrics in improvement_metrics.items():
            if isinstance(metrics, dict):
                if 'improvement_score' in metrics:
                    total_score += metrics['improvement_score']
                    section_count += 1
                elif section == 'experience' and 'average_improvement' in metrics:
                    total_score += metrics['average_improvement'] * 10  # Scale up
                    section_count += 1
                elif section == 'skills' and 'relevance_score' in metrics:
                    total_score += metrics['relevance_score']
                    section_count += 1
        
        return total_score / section_count if section_count > 0 else 0.0
    
    async def suggest_improvements(self, resume: Resume, target_job: Optional[Job] = None) -> Dict[str, Any]:
        """Generate improvement suggestions without full rewrite."""
        suggestions = {
            'summary': [],
            'experience': [],
            'skills': [],
            'general': [],
            'ats_optimization': []
        }
        
        try:
            # Summary suggestions
            if resume.summary:
                if len(resume.summary.split()) < 20:
                    suggestions['summary'].append("Consider expanding your summary to 20-30 words for better impact")
                
                if self._count_action_verbs(resume.summary) == 0:
                    suggestions['summary'].append("Add strong action verbs to make your summary more dynamic")
            
            # Experience suggestions
            if resume.experience:
                for i, exp in enumerate(resume.experience):
                    if exp.description:
                        if self._count_quantified_achievements(exp.description) == 0:
                            suggestions['experience'].append(f"Add quantified achievements to {exp.position} role")
                        
                        if self._count_action_verbs(exp.description) < 2:
                            suggestions['experience'].append(f"Use more action verbs in {exp.position} description")
            
            # Skills suggestions
            if target_job and target_job.required_skills:
                current_skills = set(skill.lower() for skill in resume.skills or [])
                required_skills = set(skill.lower() for skill in target_job.required_skills)
                missing_skills = required_skills - current_skills
                
                if missing_skills:
                    suggestions['skills'].append(f"Consider adding these relevant skills: {', '.join(list(missing_skills)[:3])}")
            
            # ATS optimization
            if target_job:
                keyword_density = self._calculate_keyword_density(
                    resume.summary + ' ' + ' '.join([exp.description or '' for exp in resume.experience or []]),
                    target_job
                )
                
                if keyword_density < 30:
                    suggestions['ats_optimization'].append("Include more keywords from the job posting to improve ATS compatibility")
            
            # General suggestions
            if not resume.projects:
                suggestions['general'].append("Consider adding a projects section to showcase your work")
            
            if not resume.certifications:
                suggestions['general'].append("Add relevant certifications to strengthen your profile")
        
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
        
        return suggestions
    
    async def quick_optimize(self, text: str, context: str = "resume") -> str:
        """Quick text optimization for individual sections."""
        try:
            prompt = f"""
            Optimize the following {context} text to be more professional and impactful:
            
            {text}
            
            Requirements:
            1. Use strong action verbs
            2. Make it more concise and clear
            3. Improve professional tone
            4. Keep the same meaning
            
            Provide only the optimized text, without explanations.
            """
            
            optimized_text = await self.ai_service.generate_content(prompt)
            return optimized_text.strip()
            
        except Exception as e:
            logger.warning(f"Error in quick optimization: {str(e)}")
            return text