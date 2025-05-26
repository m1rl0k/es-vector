"""
Document chunking module for processing large texts into manageable chunks
Supports multiple chunking strategies for optimal RAG performance
"""
import re
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SEMANTIC_SLIDING = "semantic_sliding"
    RECURSIVE = "recursive"

@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    source_document_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    word_count: int
    char_count: int
    strategy: str
    overlap_with_previous: bool = False
    overlap_with_next: bool = False

@dataclass
class DocumentChunk:
    """Represents a chunk of a document"""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for indexing"""
        return {
            "text": self.text,
            "chunk_id": self.metadata.chunk_id,
            "source_document_id": self.metadata.source_document_id,
            "chunk_index": self.metadata.chunk_index,
            "total_chunks": self.metadata.total_chunks,
            "start_char": self.metadata.start_char,
            "end_char": self.metadata.end_char,
            "word_count": self.metadata.word_count,
            "char_count": self.metadata.char_count,
            "strategy": self.metadata.strategy,
            "overlap_with_previous": self.metadata.overlap_with_previous,
            "overlap_with_next": self.metadata.overlap_with_next
        }

class DocumentChunker:
    """Main document chunking class with multiple strategies"""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        """
        Initialize document chunker

        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to avoid tiny chunks
            max_chunk_size: Maximum chunk size to enforce limits
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_separator = re.compile(r'\n\s*\n')

    def chunk_document(self,
                      text: str,
                      document_id: str,
                      strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
                      **kwargs) -> List[DocumentChunk]:
        """
        Chunk a document using the specified strategy

        Args:
            text: The text to chunk
            document_id: Unique identifier for the source document
            strategy: Chunking strategy to use
            **kwargs: Additional parameters for specific strategies

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        logger.info(f"Chunking document {document_id} ({len(text)} chars) using {strategy.value}")

        if strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text, document_id, **kwargs)
        elif strategy == ChunkingStrategy.SENTENCE_BASED:
            chunks = self._chunk_sentence_based(text, document_id, **kwargs)
        elif strategy == ChunkingStrategy.PARAGRAPH_BASED:
            chunks = self._chunk_paragraph_based(text, document_id, **kwargs)
        elif strategy == ChunkingStrategy.SEMANTIC_SLIDING:
            chunks = self._chunk_semantic_sliding(text, document_id, **kwargs)
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._chunk_recursive(text, document_id, **kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks

    def _chunk_fixed_size(self, text: str, document_id: str, **kwargs) -> List[DocumentChunk]:
        """Simple fixed-size chunking with overlap"""
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.chunk_overlap)

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= self.min_chunk_size:
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    source_document_id=document_id,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    start_char=start,
                    end_char=end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    strategy=ChunkingStrategy.FIXED_SIZE.value,
                    overlap_with_previous=start > 0,
                    overlap_with_next=end < len(text)
                )

                chunks.append(DocumentChunk(chunk_text, metadata))
                chunk_index += 1

            start = max(start + chunk_size - overlap, start + 1)

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _chunk_sentence_based(self, text: str, document_id: str, **kwargs) -> List[DocumentChunk]:
        """Chunk based on sentence boundaries"""
        target_size = kwargs.get('chunk_size', self.chunk_size)

        # Split into sentences more carefully
        import re
        # Find sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)

        if len(sentences) <= 1:
            return self._chunk_fixed_size(text, document_id, **kwargs)

        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding this sentence would exceed target size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) > target_size and current_chunk:
                # Save current chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    end_char = start_char + len(current_chunk)
                    metadata = ChunkMetadata(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        source_document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        start_char=start_char,
                        end_char=end_char,
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        strategy=ChunkingStrategy.SENTENCE_BASED.value
                    )
                    chunks.append(DocumentChunk(current_chunk, metadata))
                    chunk_index += 1
                    start_char = end_char

                current_chunk = sentence
            else:
                current_chunk = potential_chunk

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                source_document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=0,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                strategy=ChunkingStrategy.SENTENCE_BASED.value
            )
            chunks.append(DocumentChunk(current_chunk, metadata))

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _chunk_paragraph_based(self, text: str, document_id: str, **kwargs) -> List[DocumentChunk]:
        """Chunk based on paragraph boundaries"""
        target_size = kwargs.get('chunk_size', self.chunk_size)

        # Split into paragraphs
        paragraphs = self.paragraph_separator.split(text)
        if not paragraphs:
            return self._chunk_sentence_based(text, document_id, **kwargs)

        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed target size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(potential_chunk) > target_size and current_chunk:
                # Save current chunk
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    end_char = start_char + len(current_chunk)
                    metadata = ChunkMetadata(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        source_document_id=document_id,
                        chunk_index=chunk_index,
                        total_chunks=0,
                        start_char=start_char,
                        end_char=end_char,
                        word_count=len(current_chunk.split()),
                        char_count=len(current_chunk),
                        strategy=ChunkingStrategy.PARAGRAPH_BASED.value
                    )
                    chunks.append(DocumentChunk(current_chunk, metadata))
                    chunk_index += 1
                    start_char = end_char

                current_chunk = paragraph
            else:
                current_chunk = potential_chunk

        # Add final chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                source_document_id=document_id,
                chunk_index=chunk_index,
                total_chunks=0,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                word_count=len(current_chunk.split()),
                char_count=len(current_chunk),
                strategy=ChunkingStrategy.PARAGRAPH_BASED.value
            )
            chunks.append(DocumentChunk(current_chunk, metadata))

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _chunk_semantic_sliding(self, text: str, document_id: str, **kwargs) -> List[DocumentChunk]:
        """Semantic sliding window chunking with overlap"""
        chunk_size = kwargs.get('chunk_size', self.chunk_size)
        overlap = kwargs.get('overlap', self.chunk_overlap)

        # First try sentence-based chunking
        sentence_chunks = self._chunk_sentence_based(text, document_id, chunk_size=chunk_size)

        if len(sentence_chunks) <= 1:
            return sentence_chunks

        # Create sliding windows with semantic overlap
        chunks = []
        chunk_index = 0

        for i in range(len(sentence_chunks)):
            # Determine overlap range
            start_idx = max(0, i - 1) if i > 0 and overlap > 0 else i
            end_idx = min(len(sentence_chunks), i + 2) if i < len(sentence_chunks) - 1 and overlap > 0 else i + 1

            # Combine chunks in the window
            combined_text = ""
            start_char = sentence_chunks[start_idx].metadata.start_char
            end_char = sentence_chunks[end_idx - 1].metadata.end_char

            for j in range(start_idx, end_idx):
                if combined_text:
                    combined_text += " "
                combined_text += sentence_chunks[j].text

            if len(combined_text.strip()) >= self.min_chunk_size:
                metadata = ChunkMetadata(
                    chunk_id=f"{document_id}_semantic_chunk_{chunk_index}",
                    source_document_id=document_id,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=end_char,
                    word_count=len(combined_text.split()),
                    char_count=len(combined_text),
                    strategy=ChunkingStrategy.SEMANTIC_SLIDING.value,
                    overlap_with_previous=i > 0,
                    overlap_with_next=i < len(sentence_chunks) - 1
                )
                chunks.append(DocumentChunk(combined_text, metadata))
                chunk_index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        return chunks

    def _chunk_recursive(self, text: str, document_id: str, **kwargs) -> List[DocumentChunk]:
        """Recursive chunking that tries multiple strategies"""
        target_size = kwargs.get('chunk_size', self.chunk_size)

        # If text is small enough, return as single chunk
        if len(text) <= target_size:
            metadata = ChunkMetadata(
                chunk_id=f"{document_id}_chunk_0",
                source_document_id=document_id,
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=len(text),
                word_count=len(text.split()),
                char_count=len(text),
                strategy=ChunkingStrategy.RECURSIVE.value
            )
            return [DocumentChunk(text, metadata)]

        # Try paragraph-based first
        if '\n\n' in text:
            chunks = self._chunk_paragraph_based(text, document_id, **kwargs)
            if self._chunks_are_good_size(chunks, target_size):
                # Update strategy to recursive
                for chunk in chunks:
                    chunk.metadata.strategy = ChunkingStrategy.RECURSIVE.value
                return chunks

        # Try sentence-based
        if '.' in text or '!' in text or '?' in text:
            chunks = self._chunk_sentence_based(text, document_id, **kwargs)
            if self._chunks_are_good_size(chunks, target_size):
                # Update strategy to recursive
                for chunk in chunks:
                    chunk.metadata.strategy = ChunkingStrategy.RECURSIVE.value
                return chunks

        # Fall back to fixed-size
        chunks = self._chunk_fixed_size(text, document_id, **kwargs)
        # Update strategy to recursive
        for chunk in chunks:
            chunk.metadata.strategy = ChunkingStrategy.RECURSIVE.value
        return chunks

    def _chunks_are_good_size(self, chunks: List[DocumentChunk], target_size: int) -> bool:
        """Check if chunks are within acceptable size range"""
        if not chunks:
            return False

        for chunk in chunks:
            if chunk.metadata.char_count > self.max_chunk_size:
                return False
            if chunk.metadata.char_count < self.min_chunk_size:
                return False

        # Check if chunks are reasonably close to target size
        avg_size = sum(chunk.metadata.char_count for chunk in chunks) / len(chunks)
        return abs(avg_size - target_size) / target_size < 0.5  # Within 50% of target

    def merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks

        merged_chunks = []
        current_chunk = None

        for chunk in chunks:
            if chunk.metadata.char_count < self.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with current chunk
                    merged_text = current_chunk.text + " " + chunk.text
                    current_chunk.text = merged_text
                    current_chunk.metadata.end_char = chunk.metadata.end_char
                    current_chunk.metadata.char_count = len(merged_text)
                    current_chunk.metadata.word_count = len(merged_text.split())
            else:
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(chunk)

        if current_chunk is not None:
            merged_chunks.append(current_chunk)

        # Update chunk indices and total count
        for i, chunk in enumerate(merged_chunks):
            chunk.metadata.chunk_index = i
            chunk.metadata.total_chunks = len(merged_chunks)

        return merged_chunks

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}

        char_counts = [chunk.metadata.char_count for chunk in chunks]
        word_counts = [chunk.metadata.word_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(char_counts) / len(chunks),
            "min_chunk_size": min(char_counts),
            "max_chunk_size": max(char_counts),
            "avg_word_count": sum(word_counts) / len(chunks),
            "strategy": chunks[0].metadata.strategy if chunks else None
        }