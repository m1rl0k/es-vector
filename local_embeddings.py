"""
Pure local embedding service using sentence-transformers
No external API dependencies required
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    """Local embedding service using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the local embedding service

        Args:
            model_name: Name of the sentence-transformer model to use
                       Default: "all-MiniLM-L6-v2" (384 dimensions, fast, good quality)
                       Alternatives: "all-mpnet-base-v2" (768 dims, slower, better quality)
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing

        Returns:
            numpy array of embeddings (1D for single text, 2D for multiple texts)
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )

        # Return 1D array for single text, 2D for multiple texts
        if single_text and len(embeddings.shape) > 1:
            return embeddings[0]  # Return first (and only) embedding as 1D array

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.embedding_dim

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)

# Global instance for easy access
embedding_service = LocalEmbeddingService()

def get_embedding_service(model_name: str = None) -> LocalEmbeddingService:
    """Get the global embedding service instance"""
    global embedding_service
    if model_name and model_name != embedding_service.model_name:
        embedding_service = LocalEmbeddingService(model_name)
    return embedding_service
