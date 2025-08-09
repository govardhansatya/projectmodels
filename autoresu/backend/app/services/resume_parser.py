"""
Resume parsing service for extracting data from uploaded files
"""
import logging
import os
import re
from typing import Dict, List, Any, Optional
import pdfplumber
from docx import Document
import asyncio

from app.services.ai_service import ai_service
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ResumeParserService:
    """Service for parsing resume files and extracting structured data"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.doc', '.txt']
        
    async def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and extract structured data"""
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Extract text based on file type
            text = await self._extract_text(file_path, ext)
            
            if not text.strip():
                raise ValueError("No text could be extracted from the file")
            
            # Parse text using AI and rule-based methods
            parsed_data = await self._parse_text(text)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {e}")
            raise
    
    async def _extract_text(self, file_path: str, file_ext: str) -> str:
        """Extract text from file based on type"""
        if file_ext == '.pdf':
            return await self._extract_pdf_text(file_path)
        elif file_ext in ['.docx', '.doc']:
            return await self._extract_docx_text(file_path)
        elif file_ext == '.txt':
            return await self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise ValueError("Failed to extract text from PDF file")
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text_parts = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            raise ValueError("Failed to extract text from DOCX file")
    
    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading TXT file: {e}")
                raise ValueError("Failed to read text file")
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            raise ValueError("Failed to extract text from TXT file")
    
    async def _parse_text(self, text: str) -> Dict[str, Any]:
        """Parse extracted text using AI and rule-based methods"""
        try:
            # Use AI for primary parsing
            ai_parsed = await self._ai_parse_text(text)
            
            # Use rule-based parsing as fallback/supplement
            rule_parsed = await self._rule_based_parse(text)
            
            # Merge results, preferring AI parsing but using rule-based as fallback
            merged_data = self._merge_parsed_data(ai_parsed, rule_parsed)
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error parsing text: {e}")
            # Fallback to rule-based parsing only
            return await self._rule_based_parse(text)
    
    async def _ai_parse_text(self, text: str) -> Dict[str, Any]:
        """Use AI to parse resume text"""
        try:
            prompt = f"""
            Parse the following resume text and extract structured information in JSON format.
            
            Resume text:
            {text[:4000]}  # Limit text to avoid token limits
            
            Return a JSON object with the following structure:
            {{
                "contact_info": {{
                    "email": "string or null",
                    "phone": "string or null",
                    "location": "string or null",
                    "linkedin": "string or null",
                    "github": "string or null"
                }},
                "summary": "string or null",
                "experience": [
                    {{
                        "company": "string",
                        "position": "string",
                        "location": "string or null",
                        "start_date": "string",
                        "end_date": "string or null",
                        "current": "boolean",
                        "description": ["array of strings"],
                        "achievements": ["array of strings"]
                    }}
                ],
                "education": [
                    {{
                        "institution": "string",
                        "degree": "string",
                        "field_of_study": "string or null",
                        "graduation_date": "string or null",
                        "gpa": "number or null"
                    }}
                ],
                "skills": ["array of strings"],
                "projects": [
                    {{
                        "name": "string",
                        "description": "string",
                        "technologies": ["array of strings"],
                        "github_url": "string or null"
                    }}
                ],
                "certifications": [
                    {{
                        "name": "string",
                        "issuer": "string",
                        "date_obtained": "string or null"
                    }}
                ]
            }}
            
            Only return valid JSON, no additional text.
            """
            
            response = await ai_service.generate_resume_content(prompt, {})
            
            # Parse JSON response
            import json
            try:
                parsed_data = json.loads(response)
                return self._validate_ai_parsed_data(parsed_data)
            except json.JSONDecodeError:
                logger.warning("AI returned invalid JSON, falling back to rule-based parsing")
                return {}
                
        except Exception as e:
            logger.error(f"AI parsing failed: {e}")
            return {}
    
    def _validate_ai_parsed_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean AI-parsed data"""
        validated = {}
        
        # Validate contact info
        contact_info = data.get('contact_info', {})
        validated['contact_info'] = {
            'email': self._validate_email(contact_info.get('email')),
            'phone': self._clean_phone(contact_info.get('phone')),
            'location': contact_info.get('location'),
            'linkedin': self._validate_url(contact_info.get('linkedin')),
            'github': self._validate_url(contact_info.get('github'))
        }
        
        # Validate other fields
        validated['summary'] = data.get('summary')
        validated['experience'] = data.get('experience', [])
        validated['education'] = data.get('education', [])
        validated['skills'] = data.get('skills', [])
        validated['projects'] = data.get('projects', [])
        validated['certifications'] = data.get('certifications', [])
        
        return validated
    
    async def _rule_based_parse(self, text: str) -> Dict[str, Any]:
        """Parse text using rule-based methods"""
        parsed_data = {
            'contact_info': {},
            'summary': None,
            'experience': [],
            'education': [],
            'skills': [],
            'projects': [],
            'certifications': []
        }
        
        # Extract contact information
        parsed_data['contact_info'] = self._extract_contact_info(text)
        
        # Extract skills
        parsed_data['skills'] = self._extract_skills(text)
        
        # Extract sections
        sections = self._identify_sections(text)
        
        # Extract summary
        parsed_data['summary'] = self._extract_summary(text, sections)
        
        # Extract experience
        parsed_data['experience'] = self._extract_experience(text, sections)
        
        # Extract education
        parsed_data['education'] = self._extract_education(text, sections)
        
        return parsed_data
    
    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information using regex"""
        contact_info = {}
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        contact_info['email'] = emails[0] if emails else None
        
        # Phone
        phone_patterns = [
            r'\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                if isinstance(phones[0], tuple):
                    contact_info['phone'] = f"({phones[0][0]}) {phones[0][1]}-{phones[0][2]}"
                else:
                    contact_info['phone'] = phones[0]
                break
        
        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        contact_info['linkedin'] = f"https://{linkedin_matches[0]}" if linkedin_matches else None
        
        # GitHub
        github_pattern = r'github\.com/[\w-]+'
        github_matches = re.findall(github_pattern, text, re.IGNORECASE)
        contact_info['github'] = f"https://{github_matches[0]}" if github_matches else None
        
        return contact_info
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills using keyword matching"""
        # Common technical skills
        technical_skills = [
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'linux',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn'
        ]
        
        # Soft skills
        soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
            'project management', 'agile', 'scrum', 'collaboration'
        ]
        
        all_skills = technical_skills + soft_skills
        found_skills = []
        
        text_lower = text.lower()
        for skill in all_skills:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def _identify_sections(self, text: str) -> Dict[str, int]:
        """Identify section headers and their positions"""
        section_patterns = {
            'summary': r'\b(?:summary|profile|objective|about)\b',
            'experience': r'\b(?:experience|work|employment|professional)\b',
            'education': r'\b(?:education|academic|qualifications)\b',
            'skills': r'\b(?:skills|technical|competencies)\b',
            'projects': r'\b(?:projects|portfolio)\b',
            'certifications': r'\b(?:certifications|certificates|licenses)\b'
        }
        
        sections = {}
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line_lower) and len(line.strip()) < 50:
                    sections[section_name] = i
                    break
        
        return sections
    
    def _extract_summary(self, text: str, sections: Dict[str, int]) -> Optional[str]:
        """Extract summary/objective section"""
        if 'summary' not in sections:
            return None
        
        lines = text.split('\n')
        start_idx = sections['summary']
        
        # Find next section or end of text
        next_section_idx = len(lines)
        for section_name, idx in sections.items():
            if idx > start_idx and idx < next_section_idx:
                next_section_idx = idx
        
        # Extract text between sections
        summary_lines = []
        for i in range(start_idx + 1, min(next_section_idx, start_idx + 10)):
            if i < len(lines):
                line = lines[i].strip()
                if line and not re.match(r'^[A-Z\s]+$', line):  # Skip all-caps headers
                    summary_lines.append(line)
        
        return ' '.join(summary_lines) if summary_lines else None
    
    def _extract_experience(self, text: str, sections: Dict[str, int]) -> List[Dict[str, Any]]:
        """Extract work experience"""
        if 'experience' not in sections:
            return []
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated parsing
        return []
    
    def _extract_education(self, text: str, sections: Dict[str, int]) -> List[Dict[str, Any]]:
        """Extract education information"""
        if 'education' not in sections:
            return []
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated parsing
        return []
    
    def _merge_parsed_data(self, ai_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge AI and rule-based parsing results"""
        merged = {}
        
        # Use AI data as primary, rule-based as fallback
        for key in ['contact_info', 'summary', 'experience', 'education', 'skills', 'projects', 'certifications']:
            ai_value = ai_data.get(key)
            rule_value = rule_data.get(key)
            
            if ai_value:
                merged[key] = ai_value
            elif rule_value:
                merged[key] = rule_value
            else:
                merged[key] = [] if key != 'summary' and key != 'contact_info' else ({} if key == 'contact_info' else None)
        
        # For skills, merge both lists
        ai_skills = set(ai_data.get('skills', []))
        rule_skills = set(rule_data.get('skills', []))
        merged['skills'] = list(ai_skills.union(rule_skills))
        
        return merged
    
    def _validate_email(self, email: Optional[str]) -> Optional[str]:
        """Validate email format"""
        if not email:
            return None
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return email if re.match(email_pattern, email) else None
    
    def _clean_phone(self, phone: Optional[str]) -> Optional[str]:
        """Clean and validate phone number"""
        if not phone:
            return None
        
        # Remove non-digit characters
        digits = re.sub(r'\D', '', phone)
        
        # Format US phone number
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        
        return phone
    
    def _validate_url(self, url: Optional[str]) -> Optional[str]:
        """Validate URL format"""
        if not url:
            return None
        
        # Add https if missing
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
        
        return url

# Global resume parser service instance
resume_parser_service = ResumeParserService()

# Convenience function
async def parse_resume_file(file_path: str) -> Dict[str, Any]:
    """Parse a resume file"""
    return await resume_parser_service.parse_resume(file_path)
