# MongoDB init script
db = db.getSiblingDB('airesume');

db.createUser({
  user: 'airesume_user',
  pwd: 'airesume_password',
  roles: [
    {
      role: 'readWrite',
      db: 'airesume'
    }
  ]
});

// Create collections with indexes
db.createCollection('users');
db.createCollection('resumes');
db.createCollection('resume_versions');
db.createCollection('jobs');
db.createCollection('job_matches');
db.createCollection('skill_gaps');

// Create indexes
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "created_at": 1 });

db.resumes.createIndex({ "title": 1 });
db.resumes.createIndex({ "status": 1 });
db.resumes.createIndex({ "target_roles": 1 });
db.resumes.createIndex({ "created_at": 1 });
db.resumes.createIndex({ "updated_at": 1 });

db.resume_versions.createIndex({ "resume_id": 1 });
db.resume_versions.createIndex({ "version_number": 1 });
db.resume_versions.createIndex({ "created_at": 1 });

db.jobs.createIndex({ "title": 1 });
db.jobs.createIndex({ "company.name": 1 });
db.jobs.createIndex({ "job_type": 1 });
db.jobs.createIndex({ "experience_level": 1 });
db.jobs.createIndex({ "required_skills": 1 });
db.jobs.createIndex({ "created_at": 1 });
db.jobs.createIndex({ "is_active": 1 });

db.job_matches.createIndex({ "user": 1 });
db.job_matches.createIndex({ "resume": 1 });
db.job_matches.createIndex({ "job": 1 });
db.job_matches.createIndex({ "match_score.overall_score": 1 });
db.job_matches.createIndex({ "is_favorite": 1 });
db.job_matches.createIndex({ "is_applied": 1 });
db.job_matches.createIndex({ "created_at": 1 });

console.log('Database initialized successfully!');
