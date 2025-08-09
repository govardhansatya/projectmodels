#!/usr/bin/env python3
"""
Setup script for AI Resume Builder Enhanced Features
This script initializes the enhanced ML and AI features
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def setup_enhanced_features():
    """Setup enhanced AI and ML features"""
    logger.info("🚀 Starting AI Resume Builder Enhanced Features Setup...")
    
    try:
        # Import all necessary modules
        logger.info("📦 Importing modules...")
        from app.config.database import connect_to_mongo, close_mongo_connection
        from app.ml.training import initialize_models
        from app.ml.embeddings import initialize_embeddings
        from app.ml.classifiers import initialize_classifiers
        from app.services.ai_service import ai_service
        from app.services.job_matcher import initialize_job_matcher
        from app.services.quality_scorer import initialize_quality_scorer
        
        # Connect to database
        logger.info("🗄️ Connecting to database...")
        await connect_to_mongo()
        
        # Initialize AI service
        logger.info("🤖 Initializing AI services...")
        await ai_service.initialize()
        
        # Initialize embeddings
        logger.info("🔢 Initializing embeddings...")
        await initialize_embeddings()
        
        # Initialize ML models
        logger.info("🧠 Initializing ML models...")
        await initialize_models()
        
        # Initialize classifiers
        logger.info("📊 Initializing classifiers...")
        await initialize_classifiers()
        
        # Initialize job matcher
        logger.info("🎯 Initializing job matcher...")
        await initialize_job_matcher()
        
        # Initialize quality scorer
        logger.info("⭐ Initializing quality scorer...")
        await initialize_quality_scorer()
        
        # Create sample data if needed
        logger.info("📄 Setting up sample data...")
        await setup_sample_data()
        
        # Close database connection
        await close_mongo_connection()
        
        logger.info("✅ Enhanced features setup completed successfully!")
        logger.info("")
        logger.info("🎉 Your AI Resume Builder is now ready with enhanced features:")
        logger.info("   • AI-powered resume enhancement")
        logger.info("   • Intelligent job matching")
        logger.info("   • Skill gap analysis")
        logger.info("   • Quality scoring")
        logger.info("   • Vector-based semantic search")
        logger.info("")
        logger.info("💡 To start the application:")
        logger.info("   cd backend && uvicorn app.main:app --reload")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        raise

async def setup_sample_data():
    """Setup sample data for testing"""
    try:
        from app.models.job import Job, Company, JobType, ExperienceLevel, WorkLocation, Salary
        from app.models.user import User, UserRole
        
        # Check if we already have sample data
        existing_jobs = await Job.find().limit(1).to_list()
        if existing_jobs:
            logger.info("Sample data already exists, skipping...")
            return
        
        logger.info("Creating sample job postings...")
        
        # Sample companies
        companies = [
            Company(
                name="TechCorp Inc",
                industry="Technology",
                size="500-1000",
                location="San Francisco, CA",
                description="Leading technology company"
            ),
            Company(
                name="StartupXYZ",
                industry="FinTech",
                size="50-100",
                location="New York, NY",
                description="Innovative fintech startup"
            ),
            Company(
                name="Enterprise Solutions",
                industry="Consulting",
                size="1000+",
                location="Chicago, IL",
                description="Enterprise consulting firm"
            )
        ]
        
        # Sample jobs
        sample_jobs = [
            {
                "title": "Senior Software Engineer",
                "description": "We are looking for a Senior Software Engineer to join our dynamic team. You will be responsible for designing, developing, and maintaining scalable web applications using modern technologies.",
                "company": companies[0],
                "job_type": JobType.FULL_TIME,
                "experience_level": ExperienceLevel.SENIOR,
                "work_location": WorkLocation.HYBRID,
                "location": "San Francisco, CA",
                "required_skills": ["Python", "JavaScript", "React", "Node.js", "PostgreSQL", "AWS"],
                "preferred_skills": ["Docker", "Kubernetes", "GraphQL", "TypeScript"],
                "required_experience": 5,
                "salary": Salary(min_amount=120000, max_amount=180000, currency="USD", period="yearly"),
                "benefits": ["Health Insurance", "401k", "Remote Work", "Professional Development"]
            },
            {
                "title": "Frontend Developer",
                "description": "Join our frontend team to build beautiful and responsive user interfaces. Experience with React and modern CSS frameworks required.",
                "company": companies[1],
                "job_type": JobType.FULL_TIME,
                "experience_level": ExperienceLevel.MID,
                "work_location": WorkLocation.REMOTE,
                "location": "Remote",
                "required_skills": ["JavaScript", "React", "CSS", "HTML", "Git"],
                "preferred_skills": ["TypeScript", "Redux", "Tailwind CSS", "Jest"],
                "required_experience": 3,
                "salary": Salary(min_amount=80000, max_amount=120000, currency="USD", period="yearly"),
                "benefits": ["Health Insurance", "Flexible Hours", "Remote Work"]
            },
            {
                "title": "Data Scientist",
                "description": "Analyze complex datasets and build machine learning models to drive business insights. Strong statistical background required.",
                "company": companies[2],
                "job_type": JobType.FULL_TIME,
                "experience_level": ExperienceLevel.MID,
                "work_location": WorkLocation.ONSITE,
                "location": "Chicago, IL",
                "required_skills": ["Python", "SQL", "Machine Learning", "Statistics", "Pandas", "Scikit-learn"],
                "preferred_skills": ["TensorFlow", "PyTorch", "R", "Tableau", "AWS"],
                "required_experience": 3,
                "salary": Salary(min_amount=90000, max_amount=140000, currency="USD", period="yearly"),
                "benefits": ["Health Insurance", "401k", "Professional Development", "Conference Budget"]
            },
            {
                "title": "DevOps Engineer",
                "description": "Build and maintain CI/CD pipelines, manage cloud infrastructure, and ensure system reliability and scalability.",
                "company": companies[0],
                "job_type": JobType.FULL_TIME,
                "experience_level": ExperienceLevel.SENIOR,
                "work_location": WorkLocation.HYBRID,
                "location": "San Francisco, CA",
                "required_skills": ["AWS", "Docker", "Kubernetes", "Jenkins", "Linux", "Python"],
                "preferred_skills": ["Terraform", "Ansible", "Monitoring", "Security"],
                "required_experience": 4,
                "salary": Salary(min_amount=110000, max_amount=160000, currency="USD", period="yearly"),
                "benefits": ["Health Insurance", "401k", "Stock Options", "Remote Work"]
            },
            {
                "title": "Product Manager",
                "description": "Lead product strategy and development, work with cross-functional teams to deliver innovative solutions.",
                "company": companies[1],
                "job_type": JobType.FULL_TIME,
                "experience_level": ExperienceLevel.MID,
                "work_location": WorkLocation.HYBRID,
                "location": "New York, NY",
                "required_skills": ["Product Management", "Agile", "Analytics", "Communication", "Strategy"],
                "preferred_skills": ["SQL", "A/B Testing", "User Research", "Roadmapping"],
                "required_experience": 4,
                "salary": Salary(min_amount=100000, max_amount=150000, currency="USD", period="yearly"),
                "benefits": ["Health Insurance", "401k", "Equity", "Professional Development"]
            }
        ]
        
        # Create jobs
        for job_data in sample_jobs:
            job = Job(**job_data)
            await job.insert()
        
        logger.info(f"Created {len(sample_jobs)} sample job postings")
        
        # Generate embeddings for sample jobs if embedding service is available
        try:
            from app.ml.embeddings import embedding_manager
            
            jobs = await Job.find().to_list()
            job_documents = []
            
            for job in jobs:
                job_text = f"{job.title} {job.description} {' '.join(job.required_skills)}"
                job_documents.append({
                    'id': str(job.id),
                    'text': job_text,
                    'title': job.title,
                    'company': job.company.name,
                    'type': 'job'
                })
            
            if job_documents:
                await embedding_manager.create_index('jobs', job_documents, 'text')
                logger.info("Created job embeddings index")
        
        except Exception as e:
            logger.warning(f"Could not create job embeddings: {e}")
        
    except Exception as e:
        logger.error(f"Error setting up sample data: {e}")

def check_environment():
    """Check if environment is properly configured"""
    logger.info("🔍 Checking environment configuration...")
    
    required_env_vars = [
        'MONGODB_URL',
        'DATABASE_NAME',
        'SECRET_KEY',
        'GEMINI_API_KEY',
        'GROQ_API_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please create a .env file with the required variables")
        logger.error("See .env.example for reference")
        return False
    
    logger.info("✅ Environment configuration looks good")
    return True

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating necessary directories...")
    
    directories = [
        "backend/uploads",
        "backend/data",
        "backend/data/models",
        "backend/data/embeddings",
        "backend/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("✅ Directories created")

def main():
    """Main setup function"""
    print("🤖 AI Resume Builder - Enhanced Features Setup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run async setup
    try:
        asyncio.run(setup_enhanced_features())
    except KeyboardInterrupt:
        logger.info("Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
