"""
Resume builder service for generating formatted resume documents
"""
import logging
import os
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import json

from app.models.resume import ResumeVersion
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ResumeBuilderService:
    """Service for building formatted resume documents"""
    
    def __init__(self):
        self.templates = {
            'modern': self._build_modern_template,
            'classic': self._build_classic_template,
            'minimal': self._build_minimal_template
        }
        
        self.output_dir = os.path.join(settings.upload_dir, "generated")
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def build_resume(self, resume_version: ResumeVersion, 
                          format: str = "pdf", template: str = "modern") -> str:
        """Build a formatted resume document"""
        try:
            if format.lower() != "pdf":
                raise ValueError(f"Unsupported format: {format}")
            
            if template not in self.templates:
                raise ValueError(f"Unsupported template: {template}")
            
            # Generate PDF
            output_path = await self._generate_pdf(resume_version, template)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error building resume: {e}")
            raise
    
    async def _generate_pdf(self, resume_version: ResumeVersion, template: str) -> str:
        """Generate PDF resume"""
        try:
            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resume_{resume_version.resume_id}_{timestamp}.pdf"
            output_path = os.path.join(self.output_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get template builder
            template_builder = self.templates[template]
            
            # Build content
            story = template_builder(resume_version)
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Resume PDF generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            raise
    
    def _build_modern_template(self, resume_version: ResumeVersion) -> list:
        """Build modern template resume"""
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            leftIndent=20
        )
        
        # Header with contact info
        story.extend(self._build_header(resume_version, title_style, styles['Normal']))
        
        # Summary
        if resume_version.summary:
            story.append(Paragraph("PROFESSIONAL SUMMARY", heading_style))
            story.append(Paragraph(resume_version.summary, body_style))
            story.append(Spacer(1, 12))
        
        # Experience
        if resume_version.experience:
            story.append(Paragraph("WORK EXPERIENCE", heading_style))
            for exp in resume_version.experience:
                story.extend(self._build_experience_section(exp, body_style))
            story.append(Spacer(1, 12))
        
        # Education
        if resume_version.education:
            story.append(Paragraph("EDUCATION", heading_style))
            for edu in resume_version.education:
                story.extend(self._build_education_section(edu, body_style))
            story.append(Spacer(1, 12))
        
        # Skills
        if resume_version.skills:
            story.append(Paragraph("SKILLS", heading_style))
            skills_text = " • ".join(resume_version.skills)
            story.append(Paragraph(skills_text, body_style))
            story.append(Spacer(1, 12))
        
        # Projects
        if resume_version.projects:
            story.append(Paragraph("PROJECTS", heading_style))
            for project in resume_version.projects:
                story.extend(self._build_project_section(project, body_style))
        
        # Certifications
        if resume_version.certifications:
            story.append(Paragraph("CERTIFICATIONS", heading_style))
            for cert in resume_version.certifications:
                story.extend(self._build_certification_section(cert, body_style))
        
        return story
    
    def _build_classic_template(self, resume_version: ResumeVersion) -> list:
        """Build classic template resume"""
        story = []
        styles = getSampleStyleSheet()
        
        # Classic styling
        title_style = ParagraphStyle(
            'ClassicTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'ClassicHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            borderWidth=1,
            borderColor=colors.black
        )
        
        # Build with classic styling
        story.extend(self._build_header(resume_version, title_style, styles['Normal']))
        
        # Add sections similar to modern but with classic styling
        if resume_version.summary:
            story.append(Paragraph("Summary", heading_style))
            story.append(Paragraph(resume_version.summary, styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Continue with other sections...
        
        return story
    
    def _build_minimal_template(self, resume_version: ResumeVersion) -> list:
        """Build minimal template resume"""
        story = []
        styles = getSampleStyleSheet()
        
        # Minimal styling
        title_style = ParagraphStyle(
            'MinimalTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=15,
            alignment=TA_LEFT
        )
        
        heading_style = ParagraphStyle(
            'MinimalHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.HexColor('#666666')
        )
        
        # Build with minimal styling
        story.extend(self._build_header(resume_version, title_style, styles['Normal']))
        
        # Add sections with minimal styling
        # Similar structure but cleaner appearance
        
        return story
    
    def _build_header(self, resume_version: ResumeVersion, title_style, normal_style) -> list:
        """Build resume header with contact information"""
        header = []
        
        # Name (use version name or extract from contact info)
        name = "Professional Resume"
        if resume_version.contact_info and hasattr(resume_version.contact_info, 'email'):
            # Extract name from email if available (simple approach)
            if resume_version.contact_info.email:
                name_part = resume_version.contact_info.email.split('@')[0]
                name = name_part.replace('.', ' ').replace('_', ' ').title()
        
        header.append(Paragraph(name, title_style))
        
        # Contact information
        contact_parts = []
        if resume_version.contact_info:
            if hasattr(resume_version.contact_info, 'email') and resume_version.contact_info.email:
                contact_parts.append(resume_version.contact_info.email)
            if hasattr(resume_version.contact_info, 'phone') and resume_version.contact_info.phone:
                contact_parts.append(resume_version.contact_info.phone)
            if hasattr(resume_version.contact_info, 'location') and resume_version.contact_info.location:
                contact_parts.append(resume_version.contact_info.location)
            if hasattr(resume_version.contact_info, 'linkedin') and resume_version.contact_info.linkedin:
                contact_parts.append(resume_version.contact_info.linkedin)
        
        if contact_parts:
            contact_style = ParagraphStyle(
                'Contact',
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=10,
                spaceAfter=20
            )
            contact_text = " | ".join(contact_parts)
            header.append(Paragraph(contact_text, contact_style))
        
        return header
    
    def _build_experience_section(self, experience, style) -> list:
        """Build experience section"""
        section = []
        
        # Job title and company
        job_style = ParagraphStyle(
            'JobTitle',
            parent=style,
            fontSize=12,
            spaceAfter=3,
            leftIndent=0
        )
        
        title_text = f"<b>{experience.position}</b> - {experience.company}"
        if experience.location:
            title_text += f" ({experience.location})"
        
        section.append(Paragraph(title_text, job_style))
        
        # Dates
        date_style = ParagraphStyle(
            'JobDates',
            parent=style,
            fontSize=10,
            spaceAfter=6,
            leftIndent=0,
            textColor=colors.HexColor('#666666')
        )
        
        date_text = experience.start_date
        if experience.end_date:
            date_text += f" - {experience.end_date}"
        elif experience.current:
            date_text += " - Present"
        
        section.append(Paragraph(date_text, date_style))
        
        # Description
        if experience.description:
            for desc in experience.description:
                bullet_style = ParagraphStyle(
                    'Bullet',
                    parent=style,
                    leftIndent=20,
                    bulletIndent=10,
                    bulletText='•'
                )
                section.append(Paragraph(desc, bullet_style))
        
        section.append(Spacer(1, 10))
        return section
    
    def _build_education_section(self, education, style) -> list:
        """Build education section"""
        section = []
        
        edu_style = ParagraphStyle(
            'Education',
            parent=style,
            fontSize=11,
            spaceAfter=3,
            leftIndent=0
        )
        
        edu_text = f"<b>{education.degree}</b>"
        if hasattr(education, 'field_of_study') and education.field_of_study:
            edu_text += f" in {education.field_of_study}"
        edu_text += f" - {education.institution}"
        
        section.append(Paragraph(edu_text, edu_style))
        
        if hasattr(education, 'graduation_date') and education.graduation_date:
            date_style = ParagraphStyle(
                'EduDate',
                parent=style,
                fontSize=10,
                spaceAfter=6,
                textColor=colors.HexColor('#666666')
            )
            section.append(Paragraph(education.graduation_date, date_style))
        
        section.append(Spacer(1, 8))
        return section
    
    def _build_project_section(self, project, style) -> list:
        """Build project section"""
        section = []
        
        proj_style = ParagraphStyle(
            'ProjectTitle',
            parent=style,
            fontSize=11,
            spaceAfter=3,
            leftIndent=0
        )
        
        section.append(Paragraph(f"<b>{project.name}</b>", proj_style))
        section.append(Paragraph(project.description, style))
        
        if project.technologies:
            tech_text = f"Technologies: {', '.join(project.technologies)}"
            tech_style = ParagraphStyle(
                'ProjectTech',
                parent=style,
                fontSize=10,
                spaceAfter=6,
                textColor=colors.HexColor('#666666')
            )
            section.append(Paragraph(tech_text, tech_style))
        
        section.append(Spacer(1, 8))
        return section
    
    def _build_certification_section(self, certification, style) -> list:
        """Build certification section"""
        section = []
        
        cert_text = f"<b>{certification.name}</b> - {certification.issuer}"
        if hasattr(certification, 'date_obtained') and certification.date_obtained:
            cert_text += f" ({certification.date_obtained})"
        
        section.append(Paragraph(cert_text, style))
        section.append(Spacer(1, 6))
        
        return section

# Global resume builder service instance
resume_builder_service = ResumeBuilderService()

# Convenience function
async def build_resume_pdf(resume_version: ResumeVersion, template: str = "modern") -> str:
    """Build a resume PDF"""
    return await resume_builder_service.build_resume(resume_version, "pdf", template)
