"""
Embedding model and utility functions for creating and handling embeddings.
"""
import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)
from huggingface_hub import login

# Replace with your actual token

class EmbeddingModel:
    """Handles document and query embedding using a pretrained language model."""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier for the embedding model
        """
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for a text passage.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use CLS token embedding as the document/query representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings.squeeze()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy.ndarray: The embedding vectors as a 2D array
        """
        if not texts:
            return np.array([])
            
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
            
def save_embeddings(embeddings: Dict[str, np.ndarray], file_path: str) -> None:
    """
    Save embeddings to disk.
    
    Args:
        embeddings: Dictionary mapping document IDs to embedding vectors
        file_path: Path to save the embeddings
    """
    try:
        torch.save(embeddings, file_path)
        logger.info(f"Saved {len(embeddings)} embeddings to {file_path}")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
        raise

def load_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """
    Load embeddings from disk.
    
    Args:
        file_path: Path to the embeddings file
        
    Returns:
        Dict: Dictionary mapping document IDs to embedding vectors
    """
    try:
        embeddings = torch.load(file_path)
        logger.info(f"Loaded {len(embeddings)} embeddings from {file_path}")
        return embeddings
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise