"""
Machine Learning training module for AI Resume Builder
Handles training of job matching and skill analysis models
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from datetime import datetime

from app.config.settings import settings
from app.models.job import Job
from app.models.resume import Resume
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main class for training ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_dir = os.path.join(settings.data_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
    async def prepare_job_matching_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for job matching model"""
        logger.info("Preparing job matching training data...")
        
        # Fetch jobs and resumes from database
        jobs = await Job.find_all().to_list()
        resumes = await Resume.find_all().to_list()
        
        training_data = []
        
        for resume in resumes:
            for job in jobs:
                # Calculate features for job-resume pair
                features = await self._extract_job_resume_features(job, resume)
                
                # Simulate match score (in real scenario, this would come from user feedback)
                match_score = await self._calculate_similarity_score(job, resume)
                
                features['match_score'] = match_score
                features['is_match'] = 1 if match_score > 0.7 else 0
                
                training_data.append(features)
        
        df = pd.DataFrame(training_data)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['match_score', 'is_match']]
        X = df[feature_columns]
        y = df['is_match']
        
        logger.info(f"Prepared {len(df)} training samples with {len(feature_columns)} features")
        return X, y
    
    async def _extract_job_resume_features(self, job: Job, resume: Resume) -> Dict[str, Any]:
        """Extract features from job-resume pair"""
        features = {}
        
        # Skills matching features
        job_skills = set([skill.lower() for skill in job.required_skills + job.preferred_skills])
        resume_skills = set([skill.skill_name.lower() for skill in resume.skills])
        
        features['skills_overlap'] = len(job_skills.intersection(resume_skills))
        features['skills_coverage'] = len(job_skills.intersection(resume_skills)) / len(job_skills) if job_skills else 0
        features['total_resume_skills'] = len(resume_skills)
        features['total_job_skills'] = len(job_skills)
        
        # Experience features
        features['years_experience'] = resume.total_experience_years
        features['required_experience'] = job.experience_level_numeric if hasattr(job, 'experience_level_numeric') else 2
        features['experience_match'] = min(features['years_experience'] / features['required_experience'], 1.0) if features['required_experience'] > 0 else 0
        
        # Education features
        features['has_degree'] = 1 if resume.education else 0
        features['education_level'] = len(resume.education) if resume.education else 0
        
        # Location features
        features['location_match'] = 1 if job.location and resume.contact_info and job.location.lower() in resume.contact_info.location.lower() else 0
        
        # Salary features
        if job.salary_range and resume.expected_salary:
            salary_min = job.salary_range.get('min', 0)
            salary_max = job.salary_range.get('max', 0)
            expected = resume.expected_salary
            
            features['salary_in_range'] = 1 if salary_min <= expected <= salary_max else 0
            features['salary_expectation_ratio'] = expected / ((salary_min + salary_max) / 2) if (salary_min + salary_max) > 0 else 1
        else:
            features['salary_in_range'] = 0
            features['salary_expectation_ratio'] = 1
        
        # Company size preference
        features['company_size_match'] = 1  # Default, can be enhanced with actual preferences
        
        # Job type features
        features['remote_preference_match'] = 1 if job.remote_option else 0
        
        return features
    
    async def _calculate_similarity_score(self, job: Job, resume: Resume) -> float:
        """Calculate semantic similarity between job and resume"""
        try:
            # Create job description text
            job_text = f"{job.title} {job.description} {' '.join(job.required_skills)}"
            
            # Create resume text
            resume_text = f"{resume.summary or ''} {' '.join([exp.description for exp in resume.experience])} {' '.join([skill.skill_name for skill in resume.skills])}"
            
            # Get embeddings
            job_embedding = await ai_service.get_embeddings([job_text])
            resume_embedding = await ai_service.get_embeddings([resume_text])
            
            # Calculate cosine similarity
            similarity = np.dot(job_embedding[0], resume_embedding[0]) / (
                np.linalg.norm(job_embedding[0]) * np.linalg.norm(resume_embedding[0])
            )
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def train_job_matching_model(self) -> Dict[str, Any]:
        """Train the job matching model"""
        logger.info("Training job matching model...")
        
        # Prepare data
        X, y = await self.prepare_job_matching_data()
        
        if len(X) == 0:
            logger.warning("No training data available for job matching model")
            return {"error": "No training data available"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_score = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_score': cv_score
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_score:.4f}")
            
            if cv_score > best_score:
                best_score = cv_score
                best_model = (name, model)
        
        if best_model:
            # Save best model
            model_name, model = best_model
            model_path = os.path.join(self.model_dir, "job_matching_model.joblib")
            scaler_path = os.path.join(self.model_dir, "job_matching_scaler.joblib")
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Store in memory
            self.models['job_matching'] = model
            self.scalers['job_matching'] = scaler
            
            logger.info(f"Best model ({model_name}) saved with CV score: {best_score:.4f}")
            
            return {
                "success": True,
                "best_model": model_name,
                "best_score": best_score,
                "results": results,
                "feature_names": list(X.columns),
                "training_samples": len(X)
            }
        
        return {"error": "Failed to train any model"}
    
    async def train_skill_gap_model(self) -> Dict[str, Any]:
        """Train model for skill gap analysis"""
        logger.info("Training skill gap analysis model...")
        
        # Prepare skill gap training data
        training_data = await self._prepare_skill_gap_data()
        
        if not training_data:
            return {"error": "No training data available for skill gap model"}
        
        # This is a more complex model that could use NLP techniques
        # For now, we'll use a simple similarity-based approach
        
        skill_embeddings = {}
        skill_categories = {}
        
        # Get embeddings for all unique skills
        all_skills = set()
        for data in training_data:
            all_skills.update(data['skills'])
        
        all_skills = list(all_skills)
        embeddings = await ai_service.get_embeddings(all_skills)
        
        for skill, embedding in zip(all_skills, embeddings):
            skill_embeddings[skill] = embedding
        
        # Save skill embeddings
        embeddings_path = os.path.join(self.model_dir, "skill_embeddings.joblib")
        joblib.dump(skill_embeddings, embeddings_path)
        
        self.models['skill_embeddings'] = skill_embeddings
        
        logger.info(f"Skill embeddings saved for {len(all_skills)} skills")
        
        return {
            "success": True,
            "skills_processed": len(all_skills),
            "model_type": "embedding_based"
        }
    
    async def _prepare_skill_gap_data(self) -> List[Dict[str, Any]]:
        """Prepare training data for skill gap analysis"""
        jobs = await Job.find_all().to_list()
        resumes = await Resume.find_all().to_list()
        
        training_data = []
        
        for job in jobs:
            job_skills = job.required_skills + job.preferred_skills
            
            training_data.append({
                'job_title': job.title,
                'skills': job_skills,
                'experience_level': job.experience_level,
                'industry': getattr(job, 'industry', 'unknown')
            })
        
        for resume in resumes:
            resume_skills = [skill.skill_name for skill in resume.skills]
            
            training_data.append({
                'job_title': 'resume',
                'skills': resume_skills,
                'experience_level': f"{resume.total_experience_years}_years",
                'industry': 'various'
            })
        
        return training_data
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load job matching model
            job_model_path = os.path.join(self.model_dir, "job_matching_model.joblib")
            job_scaler_path = os.path.join(self.model_dir, "job_matching_scaler.joblib")
            
            if os.path.exists(job_model_path) and os.path.exists(job_scaler_path):
                self.models['job_matching'] = joblib.load(job_model_path)
                self.scalers['job_matching'] = joblib.load(job_scaler_path)
                logger.info("Job matching model loaded successfully")
            
            # Load skill embeddings
            embeddings_path = os.path.join(self.model_dir, "skill_embeddings.joblib")
            if os.path.exists(embeddings_path):
                self.models['skill_embeddings'] = joblib.load(embeddings_path)
                logger.info("Skill embeddings loaded successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    async def retrain_all_models(self) -> Dict[str, Any]:
        """Retrain all models"""
        results = {}
        
        # Train job matching model
        job_matching_result = await self.train_job_matching_model()
        results['job_matching'] = job_matching_result
        
        # Train skill gap model
        skill_gap_result = await self.train_skill_gap_model()
        results['skill_gap'] = skill_gap_result
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'model_directory': self.model_dir,
            'last_updated': datetime.now().isoformat()
        }
        
        # Add model-specific info
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                stat = os.stat(model_path)
                info[f'{model_name}_last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        return info

# Global model trainer instance
model_trainer = ModelTrainer()

# Training functions for API endpoints
async def train_models():
    """Train all models"""
    return await model_trainer.retrain_all_models()

async def get_training_status():
    """Get status of trained models"""
    return model_trainer.get_model_info()

# Initialize models on startup
async def initialize_models():
    """Initialize models on application startup"""
    logger.info("Initializing ML models...")
    
    # Try to load existing models
    loaded = model_trainer.load_models()
    
    if not loaded:
        logger.info("No existing models found, training new models...")
        await model_trainer.retrain_all_models()
    
    logger.info("ML models initialization complete")
