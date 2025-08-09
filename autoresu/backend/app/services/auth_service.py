"""
Authentication service for user management and JWT handling
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets

from app.models.user import User, UserCreate, UserLogin
from app.config.settings import settings
from app.utils.exceptions import AppException

logger = logging.getLogger(__name__)

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Service for user authentication and authorization"""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
        self.refresh_token_expire_days = settings.refresh_token_expire_days
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a new access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a new refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != token_type:
                raise JWTError("Invalid token type")
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token expired")
            
            return payload
            
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                error_code="INVALID_TOKEN"
            )
    
    async def register_user(self, user_data: UserCreate) -> User:
        """Register a new user"""
        try:
            # Check if user already exists
            existing_user = await User.find_one(User.email == user_data.email)
            if existing_user:
                raise AppException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User with this email already exists",
                    error_code="USER_EXISTS"
                )
            
            # Check username if provided
            if user_data.username:
                existing_username = await User.find_one(User.username == user_data.username)
                if existing_username:
                    raise AppException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Username already taken",
                        error_code="USERNAME_EXISTS"
                    )
            
            # Hash password
            hashed_password = self.get_password_hash(user_data.password)
            
            # Create user
            user = User(
                email=user_data.email,
                username=user_data.username,
                hashed_password=hashed_password
            )
            
            # Set profile data if provided
            if user_data.first_name or user_data.last_name:
                user.profile.first_name = user_data.first_name
                user.profile.last_name = user_data.last_name
            
            await user.insert()
            
            logger.info(f"New user registered: {user.email}")
            return user
            
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register user",
                error_code="REGISTRATION_FAILED"
            )
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password"""
        try:
            user = await User.find_one(User.email == email)
            
            if not user:
                return None
            
            if not user.is_active:
                raise AppException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is deactivated",
                    error_code="ACCOUNT_DEACTIVATED"
                )
            
            if not user.hashed_password:
                # User registered via OAuth, no password set
                raise AppException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Please use OAuth login for this account",
                    error_code="OAUTH_ONLY_ACCOUNT"
                )
            
            if not self.verify_password(password, user.hashed_password):
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await user.save()
            
            return user
            
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def login(self, login_data: UserLogin) -> Dict[str, Any]:
        """Login user and return tokens"""
        user = await self.authenticate_user(login_data.email, login_data.password)
        
        if not user:
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                error_code="INVALID_CREDENTIALS"
            )
        
        # Create tokens
        access_token = self.create_access_token({"sub": str(user.id), "email": user.email})
        refresh_token = self.create_refresh_token({"sub": str(user.id), "email": user.email})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "role": user.role,
                "profile": user.profile
            }
        }
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            payload = self.verify_token(refresh_token, "refresh")
            user_id = payload.get("sub")
            
            if not user_id:
                raise AppException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token",
                    error_code="INVALID_REFRESH_TOKEN"
                )
            
            # Get user
            user = await User.get(user_id)
            if not user or not user.is_active:
                raise AppException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive",
                    error_code="USER_INACTIVE"
                )
            
            # Create new access token
            access_token = self.create_access_token({"sub": str(user.id), "email": user.email})
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to refresh token",
                error_code="REFRESH_FAILED"
            )
    
    async def get_user_from_token(self, token: str) -> User:
        """Get user from access token"""
        payload = self.verify_token(token, "access")
        user_id = payload.get("sub")
        
        if not user_id:
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                error_code="INVALID_TOKEN_PAYLOAD"
            )
        
        user = await User.get(user_id)
        if not user:
            raise AppException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                error_code="USER_NOT_FOUND"
            )
        
        if not user.is_active:
            raise AppException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated",
                error_code="ACCOUNT_DEACTIVATED"
            )
        
        return user
    
    async def change_password(self, user: User, current_password: str, new_password: str) -> bool:
        """Change user password"""
        try:
            # Verify current password
            if not user.hashed_password or not self.verify_password(current_password, user.hashed_password):
                raise AppException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is incorrect",
                    error_code="INVALID_CURRENT_PASSWORD"
                )
            
            # Hash new password
            user.hashed_password = self.get_password_hash(new_password)
            user.updated_at = datetime.utcnow()
            
            await user.save()
            
            logger.info(f"Password changed for user: {user.email}")
            return True
            
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to change password",
                error_code="PASSWORD_CHANGE_FAILED"
            )
    
    async def reset_password_request(self, email: str) -> bool:
        """Request password reset (would typically send email)"""
        try:
            user = await User.find_one(User.email == email)
            
            if not user:
                # Don't reveal if user exists for security
                return True
            
            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            reset_expires = datetime.utcnow() + timedelta(hours=1)
            
            # Store reset token (would typically save to database)
            # For now, just log it (in production, send via email)
            logger.info(f"Password reset token for {email}: {reset_token}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error requesting password reset: {e}")
            return False
    
    def generate_api_key(self, user: User) -> str:
        """Generate API key for user"""
        # Create long-lived token for API access
        api_token = self.create_access_token(
            {"sub": str(user.id), "email": user.email, "api_key": True},
            expires_delta=timedelta(days=365)  # 1 year expiration
        )
        return api_token

# Global auth service instance
auth_service = AuthService()

# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """FastAPI dependency to get current authenticated user"""
    token = credentials.credentials
    return await auth_service.get_user_from_token(token)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to get current active user"""
    if not current_user.is_active:
        raise AppException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
            error_code="ACCOUNT_DEACTIVATED"
        )
    return current_user

async def get_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    """FastAPI dependency to get admin user"""
    if current_user.role != "admin":
        raise AppException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
            error_code="ADMIN_REQUIRED"
        )
    return current_user

# Convenience functions
async def register_new_user(user_data: UserCreate) -> User:
    """Register a new user"""
    return await auth_service.register_user(user_data)

async def login_user(login_data: UserLogin) -> Dict[str, Any]:
    """Login user"""
    return await auth_service.login(login_data)

async def refresh_token(refresh_token: str) -> Dict[str, Any]:
    """Refresh access token"""
    return await auth_service.refresh_access_token(refresh_token)
