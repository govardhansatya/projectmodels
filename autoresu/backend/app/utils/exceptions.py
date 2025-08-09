"""
Custom exceptions for the AI Resume Builder application
"""
from typing import Any, Dict, Optional


class AppException(Exception):
    """Base application exception"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.headers = headers or {}
        super().__init__(detail)


class ValidationError(AppException):
    """Validation error exception"""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        super().__init__(
            status_code=422,
            detail=detail,
            error_code="VALIDATION_ERROR"
        )
        self.field = field


class AuthenticationError(AppException):
    """Authentication error exception"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=401,
            detail=detail,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(AppException):
    """Authorization error exception"""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=403,
            detail=detail,
            error_code="AUTHORIZATION_ERROR"
        )


class NotFoundError(AppException):
    """Resource not found exception"""
    
    def __init__(self, detail: str = "Resource not found", resource_type: Optional[str] = None):
        super().__init__(
            status_code=404,
            detail=detail,
            error_code="NOT_FOUND"
        )
        self.resource_type = resource_type


class ConflictError(AppException):
    """Resource conflict exception"""
    
    def __init__(self, detail: str = "Resource conflict"):
        super().__init__(
            status_code=409,
            detail=detail,
            error_code="CONFLICT_ERROR"
        )


class RateLimitError(AppException):
    """Rate limit exceeded exception"""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            detail=detail,
            error_code="RATE_LIMIT_ERROR"
        )


class ServiceUnavailableError(AppException):
    """Service unavailable exception"""
    
    def __init__(self, detail: str = "Service temporarily unavailable"):
        super().__init__(
            status_code=503,
            detail=detail,
            error_code="SERVICE_UNAVAILABLE"
        )


class FileUploadError(AppException):
    """File upload error exception"""
    
    def __init__(self, detail: str = "File upload failed"):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="FILE_UPLOAD_ERROR"
        )


class AIServiceError(AppException):
    """AI service error exception"""
    
    def __init__(self, detail: str = "AI service error", service: Optional[str] = None):
        super().__init__(
            status_code=502,
            detail=detail,
            error_code="AI_SERVICE_ERROR"
        )
        self.service = service


class DatabaseError(AppException):
    """Database error exception"""
    
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="DATABASE_ERROR"
        )


class BusinessLogicError(AppException):
    """Business logic error exception"""
    
    def __init__(self, detail: str, error_code: Optional[str] = None):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code=error_code or "BUSINESS_LOGIC_ERROR"
        )


# Convenience functions for common exceptions

def not_found(resource_type: str, resource_id: str) -> NotFoundError:
    """Create a not found exception for a specific resource"""
    return NotFoundError(
        detail=f"{resource_type} with id '{resource_id}' not found",
        resource_type=resource_type
    )


def unauthorized(detail: str = "Authentication required") -> AuthenticationError:
    """Create an authentication error"""
    return AuthenticationError(detail)


def forbidden(detail: str = "Access denied") -> AuthorizationError:
    """Create an authorization error"""
    return AuthorizationError(detail)


def validation_error(field: str, message: str) -> ValidationError:
    """Create a validation error for a specific field"""
    return ValidationError(
        detail=f"Validation error for field '{field}': {message}",
        field=field
    )


def conflict(resource_type: str, detail: str = None) -> ConflictError:
    """Create a conflict error"""
    if detail is None:
        detail = f"{resource_type} already exists"
    return ConflictError(detail)


def ai_service_error(service: str, detail: str = None) -> AIServiceError:
    """Create an AI service error"""
    if detail is None:
        detail = f"{service} service is currently unavailable"
    return AIServiceError(detail, service)


def file_too_large(max_size: str) -> FileUploadError:
    """Create a file too large error"""
    return FileUploadError(f"File size exceeds maximum allowed size of {max_size}")


def unsupported_file_type(allowed_types: list) -> FileUploadError:
    """Create an unsupported file type error"""
    return FileUploadError(
        f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
    )


def rate_limit_exceeded(limit: str, window: str) -> RateLimitError:
    """Create a rate limit exceeded error"""
    return RateLimitError(
        f"Rate limit of {limit} requests per {window} exceeded"
    )


def database_connection_error() -> DatabaseError:
    """Create a database connection error"""
    return DatabaseError("Unable to connect to database")


def invalid_token() -> AuthenticationError:
    """Create an invalid token error"""
    return AuthenticationError("Invalid or expired token")


def token_expired() -> AuthenticationError:
    """Create a token expired error"""
    return AuthenticationError("Token has expired")


def user_not_found() -> NotFoundError:
    """Create a user not found error"""
    return NotFoundError("User not found", "User")


def resume_not_found(resume_id: str) -> NotFoundError:
    """Create a resume not found error"""
    return NotFoundError(f"Resume with id '{resume_id}' not found", "Resume")


def job_not_found(job_id: str) -> NotFoundError:
    """Create a job not found error"""
    return NotFoundError(f"Job with id '{job_id}' not found", "Job")


def insufficient_credits() -> BusinessLogicError:
    """Create an insufficient credits error"""
    return BusinessLogicError(
        "Insufficient credits to perform this action",
        "INSUFFICIENT_CREDITS"
    )


def feature_not_available() -> BusinessLogicError:
    """Create a feature not available error"""
    return BusinessLogicError(
        "This feature is not available in your current plan",
        "FEATURE_NOT_AVAILABLE"
    )


def account_not_verified() -> AuthorizationError:
    """Create an account not verified error"""
    return AuthorizationError("Account email not verified")


def account_suspended() -> AuthorizationError:
    """Create an account suspended error"""
    return AuthorizationError("Account has been suspended")


def maintenance_mode() -> ServiceUnavailableError:
    """Create a maintenance mode error"""
    return ServiceUnavailableError("System is currently under maintenance")


def external_service_timeout(service: str) -> ServiceUnavailableError:
    """Create an external service timeout error"""
    return ServiceUnavailableError(f"Timeout connecting to {service} service")
