"""
Job API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.user import User
from app.models.job import Job, JobCreate, JobUpdate, JobResponse
from app.services.auth_service import get_current_active_user
from app.services.ai_service import ai_service
from app.services.job_matcher import job_matcher_service
from app.models.job_match import JobMatchAnalysis, QuickMatchResult

router = APIRouter()


def _job_to_text(job: Job) -> str:
	parts = [job.title or "", job.description or ""]
	if job.required_skills:
		parts.append("Required: " + ", ".join(job.required_skills))
	if job.preferred_skills:
		parts.append("Preferred: " + ", ".join(job.preferred_skills))
	if job.company and job.company.name:
		parts.append(f"Company: {job.company.name}")
	return "\n".join([p for p in parts if p])


@router.post("/", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(payload: JobCreate, current_user: User = Depends(get_current_active_user)):
	try:
		job = Job(**payload.dict())
		job.created_at = datetime.utcnow()
		job.updated_at = datetime.utcnow()
		await job.insert()

		# Store job embedding to Pinecone
		try:
			text = _job_to_text(job)
			await ai_service.store_job_embedding(
				job_id=str(job.id),
				job_text=text,
				metadata={
					"title": job.title,
					"company": job.company.name if job.company else None,
					"experience_level": job.experience_level.value if job.experience_level else None,
					"type": "job",
				},
			)
		except Exception:
			# Don't block job creation if embedding fails
			pass

		return JobResponse(
			id=str(job.id),
			title=job.title,
			company=job.company,
			job_type=job.job_type,
			experience_level=job.experience_level,
			work_location=job.work_location,
			location=job.location,
			required_skills=job.required_skills,
			salary=job.salary,
			created_at=job.created_at,
			is_active=job.is_active,
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[JobResponse])
async def list_jobs(
	skip: int = Query(0, ge=0),
	limit: int = Query(50, ge=1, le=100),
	only_active: bool = Query(True),
	current_user: User = Depends(get_current_active_user),
):
	try:
		query = Job.find(Job.is_active == True) if only_active else Job.find_all()
		jobs = await query.skip(skip).limit(limit).to_list()
		return [
			JobResponse(
				id=str(j.id),
				title=j.title,
				company=j.company,
				job_type=j.job_type,
				experience_level=j.experience_level,
				work_location=j.work_location,
				location=j.location,
				required_skills=j.required_skills,
				salary=j.salary,
				created_at=j.created_at,
				is_active=j.is_active,
			)
			for j in jobs
		]
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=Job)
async def get_job(job_id: str, current_user: User = Depends(get_current_active_user)):
	job = await Job.get(job_id)
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")
	return job


@router.put("/{job_id}", response_model=Job)
async def update_job(job_id: str, payload: JobUpdate, current_user: User = Depends(get_current_active_user)):
	job = await Job.get(job_id)
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")

	data = payload.dict(exclude_unset=True)
	for k, v in data.items():
		setattr(job, k, v)
	job.updated_at = datetime.utcnow()
	await job.save()

	# Update embedding
	try:
		text = _job_to_text(job)
		await ai_service.store_job_embedding(
			job_id=str(job.id),
			job_text=text,
			metadata={
				"title": job.title,
				"company": job.company.name if job.company else None,
				"experience_level": job.experience_level.value if job.experience_level else None,
				"type": "job",
			},
		)
	except Exception:
		pass

	return job


@router.delete("/{job_id}")
async def delete_job(job_id: str, current_user: User = Depends(get_current_active_user)):
	job = await Job.get(job_id)
	if not job:
		raise HTTPException(status_code=404, detail="Job not found")
	await job.delete()
	return {"message": "Job deleted"}


@router.get("/match/{resume_id}", response_model=List[QuickMatchResult])
async def match_jobs(resume_id: str, limit: int = Query(20, ge=1, le=100), current_user: User = Depends(get_current_active_user)):
	return await job_matcher_service.find_matching_jobs(resume_id=resume_id, limit=limit)


@router.post("/{job_id}/analyze/{resume_id}", response_model=JobMatchAnalysis)
async def analyze(job_id: str, resume_id: str, current_user: User = Depends(get_current_active_user)):
	return await job_matcher_service.analyze_job_match(resume_id=resume_id, job_id=job_id)


@router.post("/vector-search")
async def vector_search(body: Dict[str, Any], current_user: User = Depends(get_current_active_user)):
	"""Search jobs by semantic similarity. Body may include resume_text or resume_id."""
	resume_text = body.get("resume_text")
	resume_id = body.get("resume_id")

	if not resume_text and not resume_id:
		raise HTTPException(status_code=400, detail="resume_text or resume_id required")

	if not resume_text and resume_id:
		# Build text from resume latest version
		from app.models.resume import ResumeVersion, Resume
		resume = await Resume.get(resume_id)
		if not resume:
			raise HTTPException(status_code=404, detail="Resume not found")
		version = await ResumeVersion.find_one(
			ResumeVersion.resume_id == str(resume.id),
			ResumeVersion.version_number == resume.current_version,
		)
		if not version:
			raise HTTPException(status_code=404, detail="Resume version not found")
		resume_text = " ".join([
			version.summary or "",
			" ".join([" ".join(exp.description) for exp in (version.experience or []) if exp.description]),
			", ".join(version.skills or []),
		])

	results = await ai_service.search_jobs(resume_text=resume_text, top_k=body.get("top_k", 10))
	return {"results": results}
