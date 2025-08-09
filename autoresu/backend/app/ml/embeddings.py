"""
Embeddings module for AI Resume Builder
Handles text embeddings for semantic search and similarity matching
"""
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime

from app.config.settings import settings
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings and vector search operations"""
    
    def __init__(self):
        self.model = None
        self.indexes = {}
        self.documents = {}
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.embeddings_dir = os.path.join(settings.data_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            logger.info("Initializing sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model initialized with dimension: {self.embedding_dim}")
            
            # Load existing indexes
            await self.load_indexes()
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts"""
        if not self.model:
            await self.initialize()
        
        try:
            # Clean and preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Truncate if too long (model limit is usually 512 tokens)
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    async def create_index(self, index_name: str, documents: List[Dict[str, Any]], 
                          text_field: str = 'text') -> bool:
        """Create a new FAISS index for documents"""
        try:
            logger.info(f"Creating index '{index_name}' with {len(documents)} documents")
            
            # Extract texts for embedding
            texts = [doc.get(text_field, '') for doc in documents]
            
            # Generate embeddings
            embeddings = await self.get_embeddings(texts)
            
            # Create FAISS index
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            # Store index and documents
            self.indexes[index_name] = index
            self.documents[index_name] = documents
            
            # Save to disk
            await self.save_index(index_name)
            
            logger.info(f"Index '{index_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            return False
    
    async def search(self, index_name: str, query: str, k: int = 10, 
                    threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents in an index"""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index '{index_name}' not found")
                return []
            
            # Get query embedding
            query_embedding = await self.get_embeddings([query])
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            scores, indices = self.indexes[index_name].search(
                query_embedding.astype('float32'), k
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1 or score < threshold:
                    continue
                
                result = {
                    'document': self.documents[index_name][idx],
                    'score': float(score),
                    'rank': i + 1
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index '{index_name}': {e}")
            return []
    
    async def add_documents(self, index_name: str, documents: List[Dict[str, Any]], 
                           text_field: str = 'text') -> bool:
        """Add new documents to an existing index"""
        try:
            if index_name not in self.indexes:
                logger.warning(f"Index '{index_name}' not found, creating new one")
                return await self.create_index(index_name, documents, text_field)
            
            # Extract texts for embedding
            texts = [doc.get(text_field, '') for doc in documents]
            
            # Generate embeddings
            embeddings = await self.get_embeddings(texts)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.indexes[index_name].add(embeddings.astype('float32'))
            
            # Add to documents
            self.documents[index_name].extend(documents)
            
            # Save updated index
            await self.save_index(index_name)
            
            logger.info(f"Added {len(documents)} documents to index '{index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to index '{index_name}': {e}")
            return False
    
    async def update_document(self, index_name: str, doc_id: str, 
                             updated_doc: Dict[str, Any], text_field: str = 'text') -> bool:
        """Update a document in the index (requires rebuilding)"""
        try:
            if index_name not in self.documents:
                return False
            
            # Find and update document
            documents = self.documents[index_name]
            updated = False
            
            for i, doc in enumerate(documents):
                if doc.get('id') == doc_id:
                    documents[i] = updated_doc
                    updated = True
                    break
            
            if not updated:
                return False
            
            # Rebuild index with updated documents
            return await self.create_index(index_name, documents, text_field)
            
        except Exception as e:
            logger.error(f"Error updating document in index '{index_name}': {e}")
            return False
    
    async def delete_document(self, index_name: str, doc_id: str, 
                             text_field: str = 'text') -> bool:
        """Delete a document from the index (requires rebuilding)"""
        try:
            if index_name not in self.documents:
                return False
            
            # Remove document
            documents = self.documents[index_name]
            original_count = len(documents)
            documents = [doc for doc in documents if doc.get('id') != doc_id]
            
            if len(documents) == original_count:
                return False  # Document not found
            
            # Rebuild index without deleted document
            return await self.create_index(index_name, documents, text_field)
            
        except Exception as e:
            logger.error(f"Error deleting document from index '{index_name}': {e}")
            return False
    
    async def save_index(self, index_name: str) -> bool:
        """Save index and documents to disk"""
        try:
            if index_name not in self.indexes:
                return False
            
            # Save FAISS index
            index_path = os.path.join(self.embeddings_dir, f"{index_name}.index")
            faiss.write_index(self.indexes[index_name], index_path)
            
            # Save documents
            docs_path = os.path.join(self.embeddings_dir, f"{index_name}_docs.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents[index_name], f)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'document_count': len(self.documents[index_name]),
                'embedding_dim': self.embedding_dim
            }
            metadata_path = os.path.join(self.embeddings_dir, f"{index_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index '{index_name}' saved to disk")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index '{index_name}': {e}")
            return False
    
    async def load_index(self, index_name: str) -> bool:
        """Load index and documents from disk"""
        try:
            # Load FAISS index
            index_path = os.path.join(self.embeddings_dir, f"{index_name}.index")
            if not os.path.exists(index_path):
                return False
            
            self.indexes[index_name] = faiss.read_index(index_path)
            
            # Load documents
            docs_path = os.path.join(self.embeddings_dir, f"{index_name}_docs.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents[index_name] = pickle.load(f)
            
            logger.info(f"Index '{index_name}' loaded from disk")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index '{index_name}': {e}")
            return False
    
    async def load_indexes(self):
        """Load all available indexes from disk"""
        try:
            if not os.path.exists(self.embeddings_dir):
                return
            
            # Find all index files
            for filename in os.listdir(self.embeddings_dir):
                if filename.endswith('.index'):
                    index_name = filename[:-6]  # Remove .index extension
                    await self.load_index(index_name)
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
    
    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an index"""
        if index_name not in self.indexes:
            return None
        
        index = self.indexes[index_name]
        documents = self.documents.get(index_name, [])
        
        return {
            'name': index_name,
            'total_vectors': index.ntotal,
            'dimension': index.d,
            'document_count': len(documents),
            'is_trained': index.is_trained
        }
    
    def list_indexes(self) -> List[str]:
        """List all available indexes"""
        return list(self.indexes.keys())
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            embeddings = await self.get_embeddings([text1, text2])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def get_document_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single document"""
        try:
            embeddings = await self.get_embeddings([text])
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error getting document embedding: {e}")
            return None
    
    async def batch_similarity(self, query: str, texts: List[str]) -> List[float]:
        """Calculate similarity between a query and multiple texts"""
        try:
            all_texts = [query] + texts
            embeddings = await self.get_embeddings(all_texts)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            query_embedding = embeddings[0]
            text_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = []
            for text_embedding in text_embeddings:
                similarity = np.dot(query_embedding, text_embedding)
                similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating batch similarities: {e}")
            return [0.0] * len(texts)

# Global embedding manager instance
embedding_manager = EmbeddingManager()

# Convenience functions
async def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for texts"""
    return await embedding_manager.get_embeddings(texts)

async def create_job_index(jobs: List[Dict[str, Any]]) -> bool:
    """Create index for job postings"""
    return await embedding_manager.create_index('jobs', jobs, 'description')

async def create_resume_index(resumes: List[Dict[str, Any]]) -> bool:
    """Create index for resumes"""
    return await embedding_manager.create_index('resumes', resumes, 'content')

async def search_jobs(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar jobs"""
    return await embedding_manager.search('jobs', query, k)

async def search_resumes(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """Search for similar resumes"""
    return await embedding_manager.search('resumes', query, k)

async def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    return await embedding_manager.calculate_similarity(text1, text2)

# Initialize embeddings on startup
async def initialize_embeddings():
    """Initialize embeddings on application startup"""
    logger.info("Initializing embeddings...")
    await embedding_manager.initialize()
    logger.info("Embeddings initialization complete")
