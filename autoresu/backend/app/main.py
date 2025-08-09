"""
Main FastAPI application
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from app.config.settings import settings
from app.config.database import connect_to_mongo, close_mongo_connection
from app.api import auth, resume, job, analytics, github
from app.utils.exceptions import AppException
from app.ml.embeddings import initialize_embeddings
from app.ml.classifiers import initialize_classifiers
from app.services.job_matcher import initialize_job_matcher
from app.services.quality_scorer import initialize_quality_scorer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Resume Builder API...")
    
    # Create upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)
    
    # Connect to database
    await connect_to_mongo()
    
    # Initialize AI services
    from app.services.ai_service import ai_service
    await ai_service.initialize()
    # Initialize local embeddings and classifiers
    await initialize_embeddings()
    await initialize_classifiers()
    # Initialize services dependent on models
    await initialize_job_matcher()
    await initialize_quality_scorer()
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Resume Builder API...")
    await close_mongo_connection()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="AI-powered resume builder with intelligent job matching",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# Static files
if os.path.exists(settings.upload_dir):
    app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")

# Exception handlers
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from app.config.database import ping_database
    
    db_status = await ping_database()
    
    return {
        "status": "healthy" if db_status else "unhealthy",
        "version": settings.version,
        "database": "connected" if db_status else "disconnected"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "docs": "/docs"
    }

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(resume.router, prefix="/api/v1/resumes", tags=["resumes"])
app.include_router(job.router, prefix="/api/v1/jobs", tags=["jobs"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(github.router, prefix="/api/v1/github", tags=["github"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )