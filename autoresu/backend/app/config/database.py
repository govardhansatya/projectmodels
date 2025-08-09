"""
Database connection and initialization
"""
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.config.settings import settings
from app.models.user import User
from app.models.resume import Resume, ResumeVersion
from app.models.job import Job, JobMatch
from app.models.skill_gap import SkillGap
from app.models.job_match import JobMatchAnalysis
import logging

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    database = None

db = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        db.client = AsyncIOMotorClient(settings.mongodb_url)
        db.database = db.client[settings.database_name]

        # Initialize Beanie with document models
        await init_beanie(
            database=db.database,
            document_models=[
                User,
                Resume,
                ResumeVersion,
                Job,
                JobMatch,
                SkillGap,
                JobMatchAnalysis,
            ]
        )
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("Disconnected from MongoDB")

async def get_database():
    return db.database

async def ping_database():
    """Ping database to check connection"""
    try:
        if db.client:
            await db.client.admin.command('ping')
            return True
        return False
    except Exception as e:
        logger.error(f"Database ping failed: {e}")
        return False
