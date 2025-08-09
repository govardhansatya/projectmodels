"""
Application settings and configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os


class Settings(BaseSettings):
    # Basic app info
    app_name: str = "AI Resume Builder"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database
    mongodb_url: str = Field(..., env="MONGODB_URL")
    database_name: str = Field(..., env="DATABASE_NAME")
    
    # JWT Authentication
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=30, env="REFRESH_TOKEN_EXPIRE_DAYS")
    algorithm: str = "HS256"
    
    # AI Services
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    
    # Pinecone
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="resume-embeddings", env="PINECONE_INDEX_NAME")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # GitHub OAuth
    github_client_id: Optional[str] = Field(default=None, env="GITHUB_CLIENT_ID")
    github_client_secret: Optional[str] = Field(default=None, env="GITHUB_CLIENT_SECRET")
    
    # File Upload
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: List[str] = Field(default=["pdf", "docx", "doc", "txt"], env="ALLOWED_EXTENSIONS")
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    data_dir: str = Field(default="data", env="DATA_DIR")
    
    # Email (optional)
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: Optional[int] = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    
    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # CORS
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # ML Model Settings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    max_tokens: int = Field(default=4000, env="MAX_TOKENS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
