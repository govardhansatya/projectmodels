"""
Analytics API endpoints
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime, timedelta

from app.models.user import User
from app.models.resume import Resume
from app.models.job import Job, JobMatch
from app.services.auth_service import get_current_active_user
from app.ml.training import train_models, get_training_status
from app.services.ai_service import ai_service
from app.models.resume import ResumeVersion

router = APIRouter()


@router.get("/overview")
async def overview(current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
	since = datetime.utcnow() - timedelta(days=30)
	resume_count = await Resume.find(Resume.user.id == current_user.id).count()
	jobs_count = await Job.find(Job.is_active == True).count()
	matches_count = await JobMatch.find(JobMatch.user.id == current_user.id).count()

	recent_resumes = await Resume.find(Resume.user.id == current_user.id).sort("-updated_at").limit(5).to_list()

	return {
		"resumes": resume_count,
		"active_jobs": jobs_count,
		"matches": matches_count,
		"recent_resumes": [
			{
				"id": str(r.id),
				"title": r.title,
				"updated_at": r.updated_at,
				"downloads": r.downloads,
			}
			for r in recent_resumes
		],
		"since": since,
	}


@router.post("/ml/train")
async def trigger_training(current_user: User = Depends(get_current_active_user)):
	"""Trigger training of ML models (admin/premium could be enforced later)"""
	result = await train_models()
	return result


@router.get("/ml/status")
async def training_status(current_user: User = Depends(get_current_active_user)):
	return await get_training_status()


@router.post("/vector/backfill")
async def backfill_vectors(current_user: User = Depends(get_current_active_user)):
	"""Backfill Pinecone index with existing jobs and resumes"""
	# Jobs
	jobs = await Job.find(Job.is_active == True).to_list()
	job_count = 0
	for j in jobs:
		try:
			text = f"{j.title}\n{j.description}\nRequired: {', '.join(j.required_skills or [])}"
			await ai_service.store_job_embedding(
				job_id=str(j.id),
				job_text=text,
				metadata={
					"title": j.title,
					"company": j.company.name if j.company else None,
					"type": "job",
				},
			)
			job_count += 1
		except Exception:
			continue

	# Resumes (latest version per resume)
	resumes = await Resume.find_all().to_list()
	resume_count = 0
	for r in resumes:
		try:
			version = await ResumeVersion.find_one(
				ResumeVersion.resume_id == str(r.id),
				ResumeVersion.version_number == r.current_version,
			)
			if not version:
				continue
			parts = [version.summary or ""]
			if version.experience:
				for exp in version.experience:
					if exp.description:
						parts.append(" ".join(exp.description))
			if version.skills:
				parts.append(", ".join(version.skills))
			text = "\n".join([p for p in parts if p])
			await ai_service.store_resume_embedding(
				resume_id=str(r.id),
				resume_text=text,
				metadata={
					"title": r.title,
					"user_id": str(r.user.id) if r.user else None,
					"type": "resume",
				},
			)
			resume_count += 1
		except Exception:
			continue

	return {"jobs_indexed": job_count, "resumes_indexed": resume_count}
