"""
Utility helper functions for the AI Resume Builder application.
"""

import re
import uuid
import hashlib
import secrets
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import json
from pathlib import Path
import mimetypes
import os

def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-.,;:()!?@#$%&*+=<>{}[\]|\\/"\'`~]', '', text)
    
    return text

def extract_email(text: str) -> Optional[str]:
    """Extract email address from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from text."""
    # Multiple phone number patterns
    patterns = [
        r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
        r'\+?([0-9]{1,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})[-.\s]?([0-9]{3,4})',
        r'\(([0-9]{3})\)\s*([0-9]{3})[-.\s]?([0-9]{4})'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Clean and format the phone number
            digits = ''.join(''.join(match) for match in matches[0] if isinstance(match, (list, tuple)))
            if isinstance(matches[0], str):
                digits = re.sub(r'[^\d]', '', matches[0])
            
            if 10 <= len(digits) <= 15:
                return digits
    
    return None

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def calculate_match_score(resume_skills: List[str], job_requirements: List[str]) -> float:
    """Calculate basic match score between resume skills and job requirements."""
    if not resume_skills or not job_requirements:
        return 0.0
    
    # Convert to lowercase for comparison
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_requirements_lower = [req.lower() for req in job_requirements]
    
    # Count matches
    matches = sum(1 for req in job_requirements_lower if req in resume_skills_lower)
    
    # Calculate percentage
    return (matches / len(job_requirements)) * 100

def normalize_skill_name(skill: str) -> str:
    """Normalize skill names for better matching."""
    if not skill:
        return ""
    
    # Convert to lowercase and remove extra spaces
    skill = skill.lower().strip()
    
    # Common skill variations mapping
    skill_mappings = {
        'js': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'c++': 'cpp',
        'c#': 'csharp',
        'react.js': 'react',
        'vue.js': 'vue',
        'node.js': 'nodejs',
        'express.js': 'express',
        'postgresql': 'postgres',
        'mysql': 'sql',
        'mongodb': 'mongo',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'nlp': 'natural language processing'
    }
    
    return skill_mappings.get(skill, skill)

def get_file_type(filename: str) -> str:
    """Get file type from filename."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type.split('/')[1]
    
    # Fallback to file extension
    return Path(filename).suffix.lower().lstrip('.')

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount."""
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$"
    }
    
    symbol = currency_symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def convert_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert object to dictionary, handling nested objects."""
    if hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                if hasattr(value, '__dict__'):
                    result[key] = convert_to_dict(value)
                elif isinstance(value, list):
                    result[key] = [convert_to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: convert_to_dict(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                else:
                    result[key] = value
        return result
    elif isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    else:
        return obj

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be safe for file system."""
    # Remove or replace unsafe characters
    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    safe_filename = safe_filename.strip(' .')
    
    # Ensure it's not empty
    if not safe_filename:
        safe_filename = f"file_{generate_id()[:8]}"
    
    return safe_filename

def parse_json_safely(json_string: str) -> Optional[Dict[str, Any]]:
    """Safely parse JSON string."""
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return None

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0

def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate if file size is within limits."""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def extract_keywords(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text."""
    # Remove common words (stop words)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
        'just', 'don', 'should', 'now'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter words
    keywords = []
    for word in words:
        if len(word) >= min_length and word not in stop_words:
            keywords.append(word)
    
    return list(set(keywords))  # Remove duplicates

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity using common words."""
    words1 = set(extract_keywords(text1))
    words2 = set(extract_keywords(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def format_date(date: datetime, format_string: str = "%Y-%m-%d") -> str:
    """Format datetime object to string."""
    return date.strftime(format_string)

def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)

class TextProcessor:
    """Advanced text processing utilities."""
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """Extract common resume sections from text."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'summary': r'(?i)(summary|profile|objective|about)\s*:?\s*\n',
            'experience': r'(?i)(experience|work\s+history|employment)\s*:?\s*\n',
            'education': r'(?i)(education|academic\s+background)\s*:?\s*\n',
            'skills': r'(?i)(skills|technical\s+skills|competencies)\s*:?\s*\n',
            'projects': r'(?i)(projects|portfolio)\s*:?\s*\n',
            'certifications': r'(?i)(certifications?|certificates?)\s*:?\s*\n',
            'awards': r'(?i)(awards?|honors?|achievements?)\s*:?\s*\n'
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.search(pattern, text)
            if matches:
                start = matches.end()
                # Find next section or end of text
                next_sections = []
                for other_pattern in section_patterns.values():
                    if other_pattern != pattern:
                        next_match = re.search(other_pattern, text[start:])
                        if next_match:
                            next_sections.append(start + next_match.start())
                
                end = min(next_sections) if next_sections else len(text)
                sections[section_name] = text[start:end].strip()
        
        return sections