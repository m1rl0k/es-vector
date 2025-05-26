"""
Test suite for document chunking functionality
"""
import pytest
import tempfile
import os
from document_chunker import DocumentChunker, ChunkingStrategy, DocumentChunk
from enhanced_rag import EnhancedRAGSystem
from rag_test import es_client

class TestDocumentChunker:
    """Test suite for DocumentChunker class"""

    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.chunker = DocumentChunker(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
            max_chunk_size=1000
        )

        # Sample texts for testing
        cls.short_text = "This is a short text that should not be chunked."

        cls.medium_text = """
        This is a medium-length text for testing document chunking.
        It contains multiple sentences and should be chunked appropriately.
        The chunking algorithm should respect sentence boundaries when possible.
        This text is designed to test the sentence-based chunking strategy.
        """

        cls.long_text = """
        This is a comprehensive document about artificial intelligence and machine learning.

        Artificial Intelligence (AI) is a broad field of computer science focused on creating
        systems that can perform tasks that typically require human intelligence. These tasks
        include learning, reasoning, problem-solving, perception, and language understanding.

        Machine Learning is a subset of AI that focuses on the development of algorithms and
        statistical models that enable computer systems to improve their performance on a
        specific task through experience. Instead of being explicitly programmed to perform
        a task, machine learning systems learn from data.

        Deep Learning is a subset of machine learning that uses artificial neural networks
        with multiple layers (hence "deep") to model and understand complex patterns in data.
        Deep learning has been particularly successful in areas such as image recognition,
        natural language processing, and speech recognition.

        Natural Language Processing (NLP) is a field that combines computational linguistics
        with machine learning and deep learning models to give computers the ability to
        understand, interpret, and generate human language in a valuable way.

        The applications of AI are vast and growing rapidly. In healthcare, AI is being used
        for medical diagnosis, drug discovery, and personalized treatment plans. In finance,
        AI powers algorithmic trading, fraud detection, and risk assessment. In transportation,
        AI enables autonomous vehicles and traffic optimization systems.
        """

    def test_chunker_initialization(self):
        """Test chunker initialization with different parameters"""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.min_chunk_size == 100  # default
        assert chunker.max_chunk_size == 2000  # default

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "test_doc_1",
            ChunkingStrategy.FIXED_SIZE
        )

        assert len(chunks) > 1, "Long text should be split into multiple chunks"

        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.metadata.strategy == ChunkingStrategy.FIXED_SIZE.value
            assert chunk.metadata.source_document_id == "test_doc_1"
            assert chunk.metadata.char_count <= self.chunker.max_chunk_size
            assert chunk.metadata.char_count >= self.chunker.min_chunk_size

    def test_sentence_based_chunking(self):
        """Test sentence-based chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "test_doc_2",
            ChunkingStrategy.SENTENCE_BASED
        )

        assert len(chunks) > 1, "Long text should be split into multiple chunks"

        for chunk in chunks:
            assert chunk.metadata.strategy == ChunkingStrategy.SENTENCE_BASED.value
            # Check that chunks are reasonable size
            assert chunk.metadata.char_count >= self.chunker.min_chunk_size
            assert chunk.metadata.char_count <= self.chunker.max_chunk_size

    def test_paragraph_based_chunking(self):
        """Test paragraph-based chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "test_doc_3",
            ChunkingStrategy.PARAGRAPH_BASED
        )

        assert len(chunks) > 1, "Long text should be split into multiple chunks"

        for chunk in chunks:
            assert chunk.metadata.strategy == ChunkingStrategy.PARAGRAPH_BASED.value
            # Check that chunks are reasonable size
            assert chunk.metadata.char_count >= self.chunker.min_chunk_size
            assert chunk.metadata.char_count <= self.chunker.max_chunk_size

    def test_recursive_chunking(self):
        """Test recursive chunking strategy"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "test_doc_4",
            ChunkingStrategy.RECURSIVE
        )

        assert len(chunks) > 1, "Long text should be split into multiple chunks"

        for chunk in chunks:
            assert chunk.metadata.strategy == ChunkingStrategy.RECURSIVE.value
            assert chunk.metadata.char_count <= self.chunker.max_chunk_size

    def test_semantic_sliding_chunking(self):
        """Test semantic sliding window chunking"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "test_doc_5",
            ChunkingStrategy.SEMANTIC_SLIDING
        )

        assert len(chunks) >= 1, "Should create at least one chunk"

        for chunk in chunks:
            assert chunk.metadata.strategy == ChunkingStrategy.SEMANTIC_SLIDING.value

    def test_short_text_handling(self):
        """Test handling of short texts that don't need chunking"""
        chunks = self.chunker.chunk_document(
            self.short_text,
            "short_doc",
            ChunkingStrategy.RECURSIVE
        )

        assert len(chunks) == 1, "Short text should result in single chunk"
        assert chunks[0].text.strip() == self.short_text.strip()
        assert chunks[0].metadata.total_chunks == 1

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text"""
        empty_chunks = self.chunker.chunk_document("", "empty_doc", ChunkingStrategy.FIXED_SIZE)
        assert len(empty_chunks) == 0, "Empty text should result in no chunks"

        whitespace_chunks = self.chunker.chunk_document("   \n\n   ", "whitespace_doc", ChunkingStrategy.FIXED_SIZE)
        assert len(whitespace_chunks) == 0, "Whitespace-only text should result in no chunks"

    def test_chunk_metadata(self):
        """Test chunk metadata completeness and accuracy"""
        chunks = self.chunker.chunk_document(
            self.medium_text,
            "metadata_test",
            ChunkingStrategy.SENTENCE_BASED
        )

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata

            # Check required fields
            assert metadata.chunk_id == f"metadata_test_chunk_{i}"
            assert metadata.source_document_id == "metadata_test"
            assert metadata.chunk_index == i
            assert metadata.total_chunks == len(chunks)
            assert metadata.char_count == len(chunk.text)
            assert metadata.word_count == len(chunk.text.split())
            assert metadata.start_char >= 0
            assert metadata.end_char > metadata.start_char

    def test_chunk_statistics(self):
        """Test chunk statistics calculation"""
        chunks = self.chunker.chunk_document(
            self.long_text,
            "stats_test",
            ChunkingStrategy.RECURSIVE
        )

        stats = self.chunker.get_chunk_statistics(chunks)

        assert stats['total_chunks'] == len(chunks)
        assert stats['total_characters'] > 0
        assert stats['total_words'] > 0
        assert stats['avg_chunk_size'] > 0
        assert stats['min_chunk_size'] <= stats['avg_chunk_size'] <= stats['max_chunk_size']
        assert stats['strategy'] == ChunkingStrategy.RECURSIVE.value

    def test_merge_small_chunks(self):
        """Test merging of small chunks"""
        # Create some artificially small chunks
        small_chunks = self.chunker.chunk_document(
            "A. B. C. D. E.",  # Very short sentences
            "small_test",
            ChunkingStrategy.SENTENCE_BASED,
            chunk_size=10  # Force very small chunks
        )

        merged_chunks = self.chunker.merge_small_chunks(small_chunks)

        # Should have fewer chunks after merging
        assert len(merged_chunks) <= len(small_chunks)

        # All merged chunks should meet minimum size requirement
        for chunk in merged_chunks:
            assert chunk.metadata.char_count >= self.chunker.min_chunk_size or len(merged_chunks) == 1

class TestEnhancedRAGSystem:
    """Test suite for EnhancedRAGSystem class"""

    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.test_index = "test-enhanced-rag"
        cls.rag_system = EnhancedRAGSystem(
            index_name=cls.test_index,
            chunk_size=300,
            chunk_overlap=50
        )

        # Ensure Elasticsearch is running
        if not es_client.ping():
            pytest.skip("Elasticsearch is not running")

    def setup_method(self):
        """Setup for each test method"""
        # Clean up any existing data
        try:
            if es_client.indices.exists(index=self.test_index):
                es_client.indices.delete(index=self.test_index)
        except:
            pass

        # Recreate the RAG system for each test
        self.rag_system = EnhancedRAGSystem(
            index_name=self.test_index,
            chunk_size=300,
            chunk_overlap=50
        )

    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        try:
            if es_client.indices.exists(index=cls.test_index):
                es_client.indices.delete(index=cls.test_index)
        except:
            pass

    def test_add_document_with_chunking(self):
        """Test adding a document with automatic chunking"""
        long_document = """
        This is a test document for the enhanced RAG system. It contains multiple
        paragraphs and should be automatically chunked when added to the system.

        The first paragraph discusses the importance of document chunking in RAG systems.
        Proper chunking ensures that relevant information can be retrieved efficiently
        while maintaining context and semantic coherence.

        The second paragraph explains how different chunking strategies can be applied
        depending on the type of content and the specific use case requirements.
        """

        chunk_ids = self.rag_system.add_document(
            document_id="test_doc_1",
            text=long_document,
            title="Test Document",
            metadata={"category": "test", "author": "test_suite"}
        )

        assert len(chunk_ids) > 1, "Long document should be split into multiple chunks"

        # Verify chunks were indexed
        for chunk_id in chunk_ids:
            doc = es_client.get(index=self.test_index, id=chunk_id)
            assert doc['found'], f"Chunk {chunk_id} should be indexed"
            assert 'text_embedding' in doc['_source'], "Chunk should have embedding"

    def test_search_with_chunking(self):
        """Test searching with chunked documents"""
        # Add a document first
        test_text = """
        Machine learning is a powerful technique for data analysis. It enables computers
        to learn patterns from data without being explicitly programmed. Deep learning,
        a subset of machine learning, uses neural networks with multiple layers.
        """

        self.rag_system.add_document(
            document_id="ml_doc",
            text=test_text,
            title="Machine Learning Basics"
        )

        # Search for relevant content
        results = self.rag_system.search("machine learning patterns", k=3)

        assert len(results) > 0, "Should find relevant chunks"

        for result in results:
            assert 'chunk_id' in result
            assert 'score' in result
            assert 'text' in result
            assert 'document_id' in result
            assert result['document_id'] == "ml_doc"

    def test_get_document_chunks(self):
        """Test retrieving all chunks for a document"""
        # First add a document
        test_text = """
        This is a test document for chunk retrieval. It should be split into multiple
        chunks to test the retrieval functionality. Each chunk should be properly
        indexed and retrievable by document ID.
        """

        self.rag_system.add_document(
            document_id="test_doc_1",
            text=test_text,
            title="Test Document"
        )

        chunks = self.rag_system.get_document_chunks("test_doc_1")

        assert len(chunks) > 0, "Should retrieve document chunks"

        # Verify chunks are ordered by index
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_index'] == i, "Chunks should be ordered by index"

    def test_delete_document(self):
        """Test deleting a document and all its chunks"""
        # Add a document to delete
        self.rag_system.add_document(
            document_id="delete_test",
            text="This document will be deleted.",
            title="Delete Test"
        )

        # Verify it exists
        chunks_before = self.rag_system.get_document_chunks("delete_test")
        assert len(chunks_before) > 0, "Document should exist before deletion"

        # Delete the document
        deleted_count = self.rag_system.delete_document("delete_test")
        assert deleted_count > 0, "Should delete at least one chunk"

        # Verify it's gone
        chunks_after = self.rag_system.get_document_chunks("delete_test")
        assert len(chunks_after) == 0, "Document should be gone after deletion"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
