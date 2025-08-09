# AI Resume Builder

A comprehensive AI-powered resume builder with intelligent job matching, skill gap analysis, and enhancement suggestions.

## Features

### Core Features
- 🤖 **AI-Powered Resume Enhancement**: Leverages Gemini and Groq AI models for intelligent content optimization
- 📄 **Multi-Format Support**: Import from PDF, DOCX, and manual input
- 🎯 **Smart Job Matching**: Advanced algorithms to match resumes with job postings
- 📊 **Skill Gap Analysis**: Identifies missing skills and provides improvement suggestions
- 🔄 **Version Control**: Track and manage multiple resume versions
- 📈 **Quality Scoring**: Comprehensive resume quality assessment
- 🐙 **GitHub Integration**: Automatically import projects and experience from GitHub

### AI & ML Features
- **Vector Embeddings**: Pinecone for semantic search and similarity matching
- **Natural Language Processing**: Advanced text analysis and content generation
- **Machine Learning Models**: XGBoost and LightGBM for job matching algorithms
- **Semantic Search**: Find relevant jobs and experiences using vector similarity

### Authentication & Security
- 🔐 **JWT Authentication**: Secure user authentication and authorization
- 👤 **User Management**: Complete user profile and preference management
- 🔒 **Role-based Access**: Admin and user role management

### Analytics & Insights
- 📊 **Dashboard Analytics**: Comprehensive insights into resume performance
- 📈 **Job Market Trends**: Analysis of job market requirements
- 🎯 **Success Metrics**: Track application success rates and improvements

## Tech Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **MongoDB**: Document database with Beanie ODM
- **Pinecone**: Vector database for semantic search
- **Redis**: Caching and session management
- **Celery**: Background task processing

### AI & ML
- **Google Gemini**: Advanced language model for content generation
- **Groq**: High-performance AI inference
- **Sentence Transformers**: Text embeddings
- **spaCy & NLTK**: Natural language processing
- **scikit-learn**: Machine learning utilities

### Frontend
- **React**: Modern UI framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client for API communication

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- MongoDB Atlas account
- Pinecone account
- Google AI API key
- Groq API key

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-resume
```

2. Create environment file:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. Backend setup:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

4. Frontend setup:
```bash
cd frontend
npm install
npm start
```

5. Docker setup (alternative):
```bash
docker-compose up --build
```

## Environment Variables

```env
# Database
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/airesume
DATABASE_NAME=airesume

# AI Services
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=resume-embeddings

# Authentication
SECRET_KEY=your_secret_key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_URL=redis://localhost:6379

# GitHub
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Project Structure

```
ai-resume/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── config/         # Configuration
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   ├── ml/             # ML models and training
│   │   └── utils/          # Utilities
│   └── requirements.txt
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── utils/          # Utilities
│   └── package.json
├── docker-compose.yml
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
