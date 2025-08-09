"""
AI Service integrating Gemini, Groq, and Pinecone
"""
import google.generativeai as genai
from groq import Groq
import pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
import aiohttp
import json
import numpy as np

from app.config.settings import settings

logger = logging.getLogger(__name__)


class AIService:
    def __init__(self):
        self.gemini_client = None
        self.groq_client = None
        self.pinecone_index = None
        self.embedding_model = None
        self.initialized = False

    async def initialize(self):
        """Initialize AI services"""
        try:
            # Initialize Gemini
            genai.configure(api_key=settings.gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini client initialized")

            # Initialize Groq
            self.groq_client = Groq(api_key=settings.groq_api_key)
            logger.info("Groq client initialized")

            # Initialize Pinecone
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Create or connect to index
            if settings.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine"
                )
                logger.info(f"Created Pinecone index: {settings.pinecone_index_name}")
            
            self.pinecone_index = pinecone.Index(settings.pinecone_index_name)
            logger.info(f"Connected to Pinecone index: {settings.pinecone_index_name}")

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded embedding model: {settings.embedding_model}")

            self.initialized = True
            logger.info("AI Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def generate_resume_content(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate resume content using Gemini"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            full_prompt = f"""
            You are an expert resume writer. Generate professional resume content based on the following:
            
            Context: {json.dumps(context, indent=2)}
            
            Request: {prompt}
            
            Please provide clear, professional, and ATS-friendly content.
            """
            
            response = await self._call_gemini_async(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating resume content with Gemini: {e}")
            raise

    async def analyze_job_match(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Analyze job match using Groq"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            prompt = f"""
            Analyze the match between this resume and job description. Provide a detailed analysis in JSON format.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Return JSON with:
            - overall_score (0-100)
            - skill_match_score (0-100)
            - experience_match_score (0-100)
            - education_match_score (0-100)
            - matching_skills (array)
            - missing_skills (array)
            - recommendations (array)
            - improvement_suggestions (array)
            
            Only return valid JSON, no additional text.
            """
            
            response = await self._call_groq_async(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing job match with Groq: {e}")
            raise

    async def enhance_resume_section(self, section_type: str, content: str, target_role: str) -> str:
        """Enhance specific resume section using Gemini"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            prompt = f"""
            Enhance the following {section_type} section for a {target_role} position.
            Make it more impactful, quantified, and ATS-friendly.
            
            Current content:
            {content}
            
            Please provide enhanced content that:
            1. Uses strong action verbs
            2. Includes specific metrics where possible
            3. Highlights relevant skills and achievements
            4. Is optimized for ATS scanning
            5. Maintains professional tone
            
            Return only the enhanced content, no additional formatting.
            """
            
            response = await self._call_gemini_async(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error enhancing resume section with Gemini: {e}")
            raise

    async def generate_content(self, prompt: str) -> str:
        """General content generation helper using Gemini"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        try:
            resp = await self._call_gemini_async(prompt)
            return getattr(resp, "text", "") or ""
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            raise

    async def generate_skill_recommendations(self, current_skills: List[str], target_role: str) -> List[str]:
        """Generate skill recommendations using Groq"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            prompt = f"""
            Given the current skills and target role, recommend additional skills to learn.
            
            Current Skills: {', '.join(current_skills)}
            Target Role: {target_role}
            
            Return a JSON array of 5-10 recommended skills that would be valuable for this role.
            Include both technical and soft skills. Only return the JSON array, no additional text.
            """
            
            response = await self._call_groq_async(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error generating skill recommendations with Groq: {e}")
            raise

    async def score_resume_quality(self, resume_text: str) -> Dict[str, Any]:
        """Score resume quality using Gemini"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            prompt = f"""
            Analyze the quality of this resume and provide scores and recommendations.
            
            Resume:
            {resume_text}
            
            Return JSON with:
            - overall_score (0-100)
            - section_scores (object with scores for each section)
            - strengths (array of strings)
            - improvements (array of strings)
            - ats_score (0-100, how ATS-friendly it is)
            - readability_score (0-100)
            
            Only return valid JSON, no additional text.
            """
            
            response = await self._call_gemini_async(prompt)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error scoring resume quality with Gemini: {e}")
            raise

    async def search_similar_resumes(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar resumes using Pinecone"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            # Generate embedding for the resume
            embedding = self.get_embeddings([resume_text])[0]
            
            # Search in Pinecone
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Error searching similar resumes: {e}")
            raise

    async def store_resume_embedding(self, resume_id: str, resume_text: str, metadata: Dict[str, Any]):
        """Store resume embedding in Pinecone"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            # Generate embedding
            embedding = self.get_embeddings([resume_text])[0]
            
            # Store in Pinecone
            self.pinecone_index.upsert([
                {
                    "id": resume_id,
                    "values": embedding,
                    "metadata": metadata
                }
            ])
            
            logger.info(f"Stored embedding for resume {resume_id}")
        except Exception as e:
            logger.error(f"Error storing resume embedding: {e}")
            raise

    async def store_job_embedding(self, job_id: str, job_text: str, metadata: Dict[str, Any]):
        """Store job embedding in Pinecone with a job_ prefix and type filter"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")

        try:
            embedding = self.get_embeddings([job_text])[0]

            # Ensure metadata contains type for filtering
            metadata = {**(metadata or {}), "type": "job"}

            self.pinecone_index.upsert([
                {
                    "id": f"job_{job_id}",
                    "values": embedding,
                    "metadata": metadata
                }
            ])

            logger.info(f"Stored embedding for job {job_id}")
        except Exception as e:
            logger.error(f"Error storing job embedding: {e}")
            raise

    async def search_jobs(self, resume_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for matching jobs using vector similarity"""
        if not self.initialized:
            raise RuntimeError("AI Service not initialized")
        
        try:
            # Generate embedding for the resume
            embedding = self.get_embeddings([resume_text])[0]
            
            # Search for jobs (assuming job embeddings are stored with "job_" prefix)
            results = self.pinecone_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"type": "job"}
            )
            
            return [
                {
                    "job_id": match.id.replace("job_", ""),
                    "similarity_score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            raise

    async def _call_gemini_async(self, prompt: str) -> Any:
        """Make async call to Gemini"""
        def _call_gemini():
            return self.gemini_client.generate_content(prompt)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call_gemini)

    async def _call_groq_async(self, prompt: str) -> str:
        """Make async call to Groq"""
        def _call_groq():
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="mixtral-8x7b-32768",
                max_tokens=settings.max_tokens,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call_groq)


# Global AI service instance
ai_service = AIService()
