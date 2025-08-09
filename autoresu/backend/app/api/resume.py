"""
Resume API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import os

from app.models.user import User
from app.models.resume import (
    Resume, ResumeVersion, ResumeCreate, ResumeUpdate, ResumeResponse,
    ContactInfo, Experience, Education, Project, Certification
)
from app.services.auth_service import get_current_user
from app.services.resume_parser import resume_parser_service
from app.services.resume_builder import resume_builder_service
from app.services.ai_service import ai_service
from app.services.quality_scorer import quality_scorer_service
from app.ml.classifiers import score_resume_quality
from app.utils.exceptions import AppException
from app.config.settings import settings
from app.utils.validators import validate_file_type, validate_file_size

router = APIRouter()

@router.post("/", response_model=ResumeResponse)
async def create_resume(
    resume_data: ResumeCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new resume"""
    try:
        # Create resume
        resume = Resume(
            user=current_user,
            title=resume_data.title,
            description=resume_data.description,
            target_roles=resume_data.target_roles
        )
        
        await resume.insert()
        
        # Create initial version
        initial_version = ResumeVersion(
            resume_id=str(resume.id),
            version_number=1,
            name="Initial Version"
        )
        
        await initial_version.insert()
        resume.versions = [initial_version]
        await resume.save()
        
        return await _resume_to_response(resume)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[ResumeResponse])
async def list_resumes(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """List user's resumes"""
    try:
        resumes = await Resume.find(
            Resume.user.id == current_user.id
        ).skip(skip).limit(limit).to_list()
        
        return [await _resume_to_response(resume) for resume in resumes]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{resume_id}", response_model=ResumeResponse)
async def get_resume(
    resume_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific resume"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        return await _resume_to_response(resume)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{resume_id}", response_model=ResumeResponse)
async def update_resume(
    resume_id: str,
    resume_data: ResumeUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a resume"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Update fields
        if resume_data.title is not None:
            resume.title = resume_data.title
        if resume_data.description is not None:
            resume.description = resume_data.description
        if resume_data.status is not None:
            resume.status = resume_data.status
        if resume_data.target_roles is not None:
            resume.target_roles = resume_data.target_roles
        if resume_data.target_companies is not None:
            resume.target_companies = resume_data.target_companies
        
        resume.updated_at = datetime.utcnow()
        await resume.save()
        
        return await _resume_to_response(resume)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a resume"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Delete all versions first
        await ResumeVersion.find(ResumeVersion.resume_id == resume_id).delete()
        
        # Delete resume
        await resume.delete()
        
        return {"message": "Resume deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{resume_id}/upload")
async def upload_resume_file(
    resume_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a resume file for parsing"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Validate file type and size
        if not validate_file_type(file.filename, allowed_types=settings.allowed_extensions):
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}")
        
        # Save file
        upload_dir = os.path.join(settings.upload_dir, str(current_user.id))
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{resume_id}_{file.filename}")
        
        content = await file.read()
        if not validate_file_size(len(content), max_size_mb=settings.max_file_size // (1024 * 1024)):
            raise HTTPException(status_code=400, detail=f"File too large. Max {settings.max_file_size // (1024*1024)} MB")
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
    # Parse resume
        parsed_data = await resume_parser_service.parse_resume(file_path)
        
        # Create new version with parsed data
        version_number = resume.current_version + 1
        new_version = ResumeVersion(
            resume_id=resume_id,
            version_number=version_number,
            name=f"Uploaded from {file.filename}",
            summary=parsed_data.get('summary'),
            contact_info=ContactInfo(**parsed_data.get('contact_info', {})),
            experience=[Experience(**exp) for exp in parsed_data.get('experience', [])],
            education=[Education(**edu) for edu in parsed_data.get('education', [])],
            skills=parsed_data.get('skills', []),
            certifications=[Certification(**cert) for cert in parsed_data.get('certifications', [])]
        )
        
        await new_version.insert()
        
        # Update resume
        resume.current_version = version_number
        resume.versions.append(new_version)
        resume.original_file_path = file_path
        resume.original_file_name = file.filename
        resume.updated_at = datetime.utcnow()
        
        await resume.save()

        # Upsert embedding to Pinecone for semantic search
        try:
            resume_text = await _compose_resume_text(new_version)
            await ai_service.store_resume_embedding(
                resume_id=str(resume.id),
                resume_text=resume_text,
                metadata={
                    "title": resume.title,
                    "user_id": str(current_user.id),
                    "type": "resume",
                },
            )
        except Exception:
            pass
        
        return {"message": "Resume uploaded and parsed successfully", "version": version_number}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{resume_id}/versions/{version_number}")
async def get_resume_version(
    resume_id: str,
    version_number: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific resume version"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_number == version_number
        )
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return version
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{resume_id}/versions/{version_number}")
async def update_resume_version(
    resume_id: str,
    version_number: int,
    version_data: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Update a resume version"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_number == version_number
        )
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        # Update version fields
        for field, value in version_data.items():
            if hasattr(version, field):
                setattr(version, field, value)
        
        await version.save()
        
        # Update resume timestamp
        resume.updated_at = datetime.utcnow()
        await resume.save()
        
        return version
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{resume_id}/enhance")
async def enhance_resume(
    resume_id: str,
    target_role: str = Form(...),
    sections: List[str] = Form(default=["summary", "experience"]),
    current_user: User = Depends(get_current_user)
):
    """AI-enhance resume content"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get current version
        current_version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_number == resume.current_version
        )
        
        if not current_version:
            raise HTTPException(status_code=404, detail="Current version not found")
        
        # Enhance specified sections
        enhanced_data = {}
        
        for section in sections:
            if section == "summary" and current_version.summary:
                enhanced_data["summary"] = await ai_service.enhance_resume_section(
                    "summary", current_version.summary, target_role
                )
            
            elif section == "experience" and current_version.experience:
                enhanced_exp = []
                for exp in current_version.experience:
                    if exp.description:
                        enhanced_desc = await ai_service.enhance_resume_section(
                            "experience", " ".join(exp.description), target_role
                        )
                        exp.description = enhanced_desc.split(". ")
                    enhanced_exp.append(exp)
                enhanced_data["experience"] = enhanced_exp
        
        # Create new enhanced version
        version_number = resume.current_version + 1
        enhanced_version = ResumeVersion(
            resume_id=resume_id,
            version_number=version_number,
            name=f"AI Enhanced for {target_role}",
            summary=enhanced_data.get("summary", current_version.summary),
            contact_info=current_version.contact_info,
            experience=enhanced_data.get("experience", current_version.experience),
            education=current_version.education,
            skills=current_version.skills,
            certifications=current_version.certifications,
            projects=current_version.projects,
            generated_content={"enhanced_sections": sections, "target_role": target_role}
        )
        
        await enhanced_version.insert()
        
        # Update resume
        resume.current_version = version_number
        resume.versions.append(enhanced_version)
        resume.last_enhanced_at = datetime.utcnow()
        resume.updated_at = datetime.utcnow()
        
        await resume.save()
        
        # Update embedding after enhancement
        try:
            resume_text = await _compose_resume_text(enhanced_version)
            await ai_service.store_resume_embedding(
                resume_id=str(resume.id),
                resume_text=resume_text,
                metadata={
                    "title": resume.title,
                    "user_id": str(current_user.id),
                    "type": "resume",
                    "target_role": target_role,
                },
            )
        except Exception:
            pass

        return {"message": "Resume enhanced successfully", "version": version_number}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{resume_id}/score")
async def score_resume(
    resume_id: str,
    current_user: User = Depends(get_current_user)
):
    """Score resume quality"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get current version
        current_version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_number == resume.current_version
        )
        
        if not current_version:
            raise HTTPException(status_code=404, detail="Current version not found")
        
        # Convert to dict for scoring
        resume_data = await _version_to_dict(current_version)
        
        # Score using ML classifier
        quality_result = await score_resume_quality(resume_data)
        
        # Update version with scores
        current_version.quality_scores.overall_score = quality_result.overall_score
        current_version.quality_scores.sections = quality_result.section_scores
        current_version.quality_scores.improvements = quality_result.suggestions
        current_version.quality_scores.strengths = quality_result.strengths
        
        await current_version.save()
        
        return {
            "overall_score": quality_result.overall_score,
            "section_scores": quality_result.section_scores,
            "suggestions": quality_result.suggestions,
            "strengths": quality_result.strengths,
            "weaknesses": quality_result.weaknesses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{resume_id}/export")
async def export_resume(
    resume_id: str,
    format: str = Form("pdf"),
    template: str = Form("modern"),
    current_user: User = Depends(get_current_user)
):
    """Export resume to PDF or other formats"""
    try:
        resume = await Resume.get(resume_id)
        
        if not resume or resume.user.id != current_user.id:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get current version
        current_version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == resume_id,
            ResumeVersion.version_number == resume.current_version
        )
        
        if not current_version:
            raise HTTPException(status_code=404, detail="Current version not found")
        
        # Generate resume file
        file_path = await resume_builder_service.build_resume(
            current_version, format, template
        )
        
        # Update download count
        resume.downloads += 1
        await resume.save()
        
        return FileResponse(
            path=file_path,
            filename=f"{resume.title}.{format}",
            media_type=f"application/{format}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

async def _resume_to_response(resume: Resume) -> ResumeResponse:
    """Convert Resume model to response format"""
    # Get current version for quality scores
    quality_scores = {"overall_score": 0.0, "sections": {}, "improvements": [], "strengths": []}
    
    if resume.current_version > 0:
        current_version = await ResumeVersion.find_one(
            ResumeVersion.resume_id == str(resume.id),
            ResumeVersion.version_number == resume.current_version
        )
        
        if current_version and current_version.quality_scores:
            quality_scores = {
                "overall_score": current_version.quality_scores.overall_score,
                "sections": current_version.quality_scores.sections,
                "improvements": current_version.quality_scores.improvements,
                "strengths": current_version.quality_scores.strengths
            }
    
    return ResumeResponse(
        id=str(resume.id),
        title=resume.title,
        description=resume.description,
        status=resume.status,
        current_version=resume.current_version,
        target_roles=resume.target_roles,
        created_at=resume.created_at,
        updated_at=resume.updated_at,
        quality_scores=quality_scores,
        views=resume.views,
        downloads=resume.downloads
    )

async def _version_to_dict(version: ResumeVersion) -> Dict[str, Any]:
    """Convert ResumeVersion to dictionary for ML processing"""
    return {
        "summary": version.summary or "",
        "experience": [
            {
                "company": exp.company,
                "position": exp.position,
                "description": " ".join(exp.description) if exp.description else "",
                "achievements": exp.achievements if hasattr(exp, 'achievements') else []
            }
            for exp in (version.experience or [])
        ],
        "skills": version.skills or [],
        "education": [
            {
                "degree": edu.degree,
                "institution": edu.institution,
                "field_of_study": getattr(edu, 'field_of_study', None)
            }
            for edu in (version.education or [])
        ],
        "contact_info": {
            "email": getattr(version.contact_info, 'email', None),
            "phone": getattr(version.contact_info, 'phone', None),
            "location": getattr(version.contact_info, 'location', None),
            "linkedin": getattr(version.contact_info, 'linkedin', None)
        } if version.contact_info else {}
    }

async def _compose_resume_text(version: ResumeVersion) -> str:
    parts = [version.summary or ""]
    if version.experience:
        for exp in version.experience:
            if exp.description:
                parts.append(" ".join(exp.description))
    if version.skills:
        parts.append(", ".join(version.skills))
    return "\n".join([p for p in parts if p])
