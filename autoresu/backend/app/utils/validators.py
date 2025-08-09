"""
Validation utilities for the AI Resume Builder application.
"""

import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from email_validator import validate_email, EmailNotValidError
from pydantic import BaseModel, validator
import mimetypes

def validate_email_address(email: str) -> bool:
    """Validate email address format."""
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format."""
    # Remove all non-digit characters
    digits = re.sub(r'[^\d]', '', phone)
    
    # Check if it's a valid length (10-15 digits)
    return 10 <= len(digits) <= 15

def validate_url(url: str) -> bool:
    """Validate URL format."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """Validate file type based on extension."""
    if allowed_types is None:
        allowed_types = ['pdf', 'doc', 'docx', 'txt']
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in allowed_types

def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate file size."""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength and return detailed feedback."""
    result = {
        'is_valid': True,
        'score': 0,
        'errors': [],
        'suggestions': []
    }
    
    # Check minimum length
    if len(password) < 8:
        result['errors'].append('Password must be at least 8 characters long')
        result['is_valid'] = False
    else:
        result['score'] += 20
    
    # Check for uppercase letters
    if not re.search(r'[A-Z]', password):
        result['errors'].append('Password must contain at least one uppercase letter')
        result['is_valid'] = False
    else:
        result['score'] += 20
    
    # Check for lowercase letters
    if not re.search(r'[a-z]', password):
        result['errors'].append('Password must contain at least one lowercase letter')
        result['is_valid'] = False
    else:
        result['score'] += 20
    
    # Check for numbers
    if not re.search(r'\d', password):
        result['errors'].append('Password must contain at least one number')
        result['is_valid'] = False
    else:
        result['score'] += 20
    
    # Check for special characters
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result['errors'].append('Password must contain at least one special character')
        result['is_valid'] = False
    else:
        result['score'] += 20
    
    # Additional strength checks
    if len(password) >= 12:
        result['score'] += 10
    
    if not re.search(r'(.)\1{2,}', password):  # No repeated characters
        result['score'] += 10
    else:
        result['suggestions'].append('Avoid repeating characters')
    
    # Common password patterns
    common_patterns = [
        r'123', r'abc', r'qwerty', r'password', r'admin', r'login'
    ]
    
    for pattern in common_patterns:
        if re.search(pattern, password.lower()):
            result['suggestions'].append('Avoid common patterns or dictionary words')
            result['score'] = max(0, result['score'] - 20)
            break
    
    return result

def validate_resume_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate resume data structure."""
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    required_fields = ['personal_info', 'summary']
    
    for field in required_fields:
        if field not in data or not data[field]:
            result['errors'].append(f'Missing required field: {field}')
            result['is_valid'] = False
    
    # Validate personal info
    if 'personal_info' in data:
        personal_info = data['personal_info']
        
        if 'email' in personal_info:
            if not validate_email_address(personal_info['email']):
                result['errors'].append('Invalid email address format')
                result['is_valid'] = False
        
        if 'phone' in personal_info:
            if not validate_phone_number(personal_info['phone']):
                result['warnings'].append('Phone number format may be invalid')
        
        if 'linkedin' in personal_info:
            if personal_info['linkedin'] and not validate_url(personal_info['linkedin']):
                result['warnings'].append('LinkedIn URL format may be invalid')
        
        if 'github' in personal_info:
            if personal_info['github'] and not validate_url(personal_info['github']):
                result['warnings'].append('GitHub URL format may be invalid')
    
    # Validate experience entries
    if 'experience' in data:
        for i, exp in enumerate(data['experience']):
            if not exp.get('position'):
                result['warnings'].append(f'Experience entry {i+1} missing position title')
            
            if not exp.get('company'):
                result['warnings'].append(f'Experience entry {i+1} missing company name')
            
            # Validate dates
            start_date = exp.get('start_date')
            end_date = exp.get('end_date')
            
            if start_date and end_date and end_date != 'Present':
                try:
                    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    
                    if start > end:
                        result['warnings'].append(f'Experience entry {i+1} has start date after end date')
                except ValueError:
                    result['warnings'].append(f'Experience entry {i+1} has invalid date format')
    
    # Validate education entries
    if 'education' in data:
        for i, edu in enumerate(data['education']):
            if not edu.get('degree'):
                result['warnings'].append(f'Education entry {i+1} missing degree')
            
            if not edu.get('institution'):
                result['warnings'].append(f'Education entry {i+1} missing institution')
    
    # Validate skills
    if 'skills' in data:
        if not isinstance(data['skills'], list) or len(data['skills']) == 0:
            result['warnings'].append('No skills listed')
    
    return result

def validate_job_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate job posting data structure."""
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    required_fields = ['title', 'company', 'description']
    
    for field in required_fields:
        if field not in data or not data[field]:
            result['errors'].append(f'Missing required field: {field}')
            result['is_valid'] = False
    
    # Validate salary range
    if 'salary_min' in data and 'salary_max' in data:
        try:
            salary_min = float(data['salary_min'])
            salary_max = float(data['salary_max'])
            
            if salary_min > salary_max:
                result['warnings'].append('Minimum salary is greater than maximum salary')
            
            if salary_min < 0 or salary_max < 0:
                result['warnings'].append('Salary values should be positive')
        except (ValueError, TypeError):
            result['warnings'].append('Invalid salary format')
    
    # Validate location
    if 'location' in data:
        location = data['location']
        if not isinstance(location, str) or len(location.strip()) == 0:
            result['warnings'].append('Location should be a non-empty string')
    
    # Validate required skills
    if 'required_skills' in data:
        if not isinstance(data['required_skills'], list):
            result['warnings'].append('Required skills should be a list')
        elif len(data['required_skills']) == 0:
            result['warnings'].append('No required skills specified')
    
    # Validate job type
    valid_job_types = ['full-time', 'part-time', 'contract', 'internship', 'temporary']
    if 'job_type' in data:
        if data['job_type'] not in valid_job_types:
            result['warnings'].append(f'Invalid job type. Must be one of: {", ".join(valid_job_types)}')
    
    # Validate experience level
    valid_experience_levels = ['entry', 'junior', 'mid', 'senior', 'lead', 'executive']
    if 'experience_level' in data:
        if data['experience_level'] not in valid_experience_levels:
            result['warnings'].append(f'Invalid experience level. Must be one of: {", ".join(valid_experience_levels)}')
    
    return result

class ResumeValidator(BaseModel):
    """Pydantic model for resume validation."""
    
    personal_info: Dict[str, Any]
    summary: str
    experience: Optional[List[Dict[str, Any]]] = []
    education: Optional[List[Dict[str, Any]]] = []
    skills: Optional[List[str]] = []
    projects: Optional[List[Dict[str, Any]]] = []
    certifications: Optional[List[Dict[str, Any]]] = []
    
    @validator('personal_info')
    def validate_personal_info(cls, v):
        if 'email' in v:
            if not validate_email_address(v['email']):
                raise ValueError('Invalid email address')
        return v
    
    @validator('summary')
    def validate_summary(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Summary must be at least 10 characters long')
        return v

class JobValidator(BaseModel):
    """Pydantic model for job posting validation."""
    
    title: str
    company: str
    description: str
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    job_type: Optional[str] = 'full-time'
    experience_level: Optional[str] = 'mid'
    required_skills: Optional[List[str]] = []
    preferred_skills: Optional[List[str]] = []
    
    @validator('title')
    def validate_title(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Job title must be at least 2 characters long')
        return v
    
    @validator('company')
    def validate_company(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Company name must be at least 2 characters long')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if len(v.strip()) < 20:
            raise ValueError('Job description must be at least 20 characters long')
        return v
    
    @validator('salary_min', 'salary_max')
    def validate_salary(cls, v):
        if v is not None and v < 0:
            raise ValueError('Salary must be positive')
        return v
    
    @validator('job_type')
    def validate_job_type(cls, v):
        valid_types = ['full-time', 'part-time', 'contract', 'internship', 'temporary']
        if v not in valid_types:
            raise ValueError(f'Job type must be one of: {", ".join(valid_types)}')
        return v
    
    @validator('experience_level')
    def validate_experience_level(cls, v):
        valid_levels = ['entry', 'junior', 'mid', 'senior', 'lead', 'executive']
        if v not in valid_levels:
            raise ValueError(f'Experience level must be one of: {", ".join(valid_levels)}')
        return v

def validate_search_query(query: str) -> bool:
    """Validate search query."""
    if not query or not isinstance(query, str):
        return False
    
    # Remove extra whitespace
    query = query.strip()
    
    # Check minimum length
    if len(query) < 2:
        return False
    
    # Check maximum length
    if len(query) > 500:
        return False
    
    # Check for malicious patterns
    malicious_patterns = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'data:text\/html',  # Data URLs
        r'vbscript:',  # VBScript protocol
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    
    return True

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent XSS and other attacks."""
    if not text:
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove script-related content
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    return text.strip()

def validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Remove whitespace
    api_key = api_key.strip()
    
    # Check minimum length
    if len(api_key) < 10:
        return False
    
    # Check for alphanumeric characters and common API key characters
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', api_key):
        return False
    
    return True

def validate_pagination_params(page: int, page_size: int) -> Dict[str, Any]:
    """Validate pagination parameters."""
    result = {
        'is_valid': True,
        'errors': [],
        'page': page,
        'page_size': page_size
    }
    
    # Validate page number
    if page < 1:
        result['errors'].append('Page number must be greater than 0')
        result['is_valid'] = False
        result['page'] = 1
    
    # Validate page size
    if page_size < 1:
        result['errors'].append('Page size must be greater than 0')
        result['is_valid'] = False
        result['page_size'] = 10
    elif page_size > 100:
        result['errors'].append('Page size cannot exceed 100')
        result['is_valid'] = False
        result['page_size'] = 100
    
    return result