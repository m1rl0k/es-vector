"""
Comprehensive test suite for the local RAG system
"""
import pytest
import time
import numpy as np
from elasticsearch import Elasticsearch
from local_embeddings import get_embedding_service
from rag_test import (
    create_index, add_document, add_sample_documents, 
    search_documents, es_client
)

class TestLocalRAGSystem:
    """Test suite for local RAG system"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.test_index = "test-rag-local"
        cls.embedding_service = get_embedding_service()
        
        # Ensure Elasticsearch is running
        if not es_client.ping():
            pytest.skip("Elasticsearch is not running")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        try:
            if es_client.indices.exists(index=cls.test_index):
                es_client.indices.delete(index=cls.test_index)
        except:
            pass
    
    def test_elasticsearch_connection(self):
        """Test Elasticsearch connection"""
        assert es_client.ping(), "Should connect to Elasticsearch"
    
    def test_embedding_service(self):
        """Test local embedding service"""
        text = "This is a test sentence"
        embedding = self.embedding_service.encode(text)
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert len(embedding.shape) == 1, "Should be 1D array"
        assert embedding.shape[0] > 0, "Should have positive dimensions"
    
    def test_embedding_consistency(self):
        """Test that same text produces same embeddings"""
        text = "Consistent embedding test"
        embedding1 = self.embedding_service.encode(text)
        embedding2 = self.embedding_service.encode(text)
        
        np.testing.assert_array_almost_equal(
            embedding1, embedding2, decimal=6,
            err_msg="Same text should produce identical embeddings"
        )
    
    def test_embedding_similarity(self):
        """Test embedding similarity calculation"""
        text1 = "Machine learning is fascinating"
        text2 = "AI and ML are interesting topics"
        text3 = "I like pizza and pasta"
        
        emb1 = self.embedding_service.encode(text1)
        emb2 = self.embedding_service.encode(text2)
        emb3 = self.embedding_service.encode(text3)
        
        sim_related = self.embedding_service.similarity(emb1, emb2)
        sim_unrelated = self.embedding_service.similarity(emb1, emb3)
        
        assert sim_related > sim_unrelated, "Related texts should be more similar"
    
    def test_create_index(self):
        """Test index creation"""
        create_index(self.test_index)
        
        assert es_client.indices.exists(index=self.test_index), "Index should exist"
        
        # Check mapping
        mapping = es_client.indices.get_mapping(index=self.test_index)
        properties = mapping[self.test_index]['mappings']['properties']
        
        assert 'text' in properties, "Should have text field"
        assert 'text_embedding' in properties, "Should have embedding field"
        assert properties['text_embedding']['type'] == 'dense_vector', "Should be dense vector"
    
    def test_add_document(self):
        """Test adding documents"""
        if not es_client.indices.exists(index=self.test_index):
            create_index(self.test_index)
        
        doc_id = "test_doc_1"
        text = "This is a test document for RAG system"
        title = "Test Document"
        metadata = {"category": "test", "importance": "high"}
        
        response = add_document(self.test_index, doc_id, text, title, metadata)
        
        assert response['result'] in ['created', 'updated'], "Document should be added"
        
        # Refresh index and verify document
        es_client.indices.refresh(index=self.test_index)
        doc = es_client.get(index=self.test_index, id=doc_id)
        
        assert doc['_source']['text'] == text, "Text should match"
        assert doc['_source']['title'] == title, "Title should match"
        assert doc['_source']['metadata'] == metadata, "Metadata should match"
        assert 'text_embedding' in doc['_source'], "Should have embedding"
    
    def test_add_sample_documents(self):
        """Test adding sample documents"""
        if not es_client.indices.exists(index=self.test_index):
            create_index(self.test_index)
        
        add_sample_documents(self.test_index)
        
        # Check document count
        es_client.indices.refresh(index=self.test_index)
        count_response = es_client.count(index=self.test_index)
        
        assert count_response['count'] >= 5, "Should have at least 5 sample documents"
    
    def test_search_documents(self):
        """Test document search functionality"""
        if not es_client.indices.exists(index=self.test_index):
            create_index(self.test_index)
            add_sample_documents(self.test_index)
        
        query = "RAG and AI responses"
        response = search_documents(self.test_index, query, k=3)
        
        assert 'hits' in response, "Response should have hits"
        assert len(response['hits']['hits']) > 0, "Should return results"
        
        # Check that results have required fields
        for hit in response['hits']['hits']:
            source = hit['_source']
            assert 'text' in source, "Result should have text"
            assert 'text_embedding' in source, "Result should have embedding"
            assert '_score' in hit, "Result should have score"
    
    def test_search_relevance(self):
        """Test search relevance"""
        if not es_client.indices.exists(index=self.test_index):
            create_index(self.test_index)
            add_sample_documents(self.test_index)
        
        # Search for RAG-related content
        response = search_documents(self.test_index, "RAG retrieval generation", k=5)
        hits = response['hits']['hits']
        
        # The first result should be most relevant
        assert len(hits) > 0, "Should return results"
        
        # Check that scores are in descending order
        scores = [hit['_score'] for hit in hits]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    
    def test_performance_benchmark(self):
        """Basic performance test"""
        if not es_client.indices.exists(index=self.test_index):
            create_index(self.test_index)
            add_sample_documents(self.test_index)
        
        query = "machine learning embeddings"
        
        # Measure search time
        start_time = time.time()
        response = search_documents(self.test_index, query)
        search_time = time.time() - start_time
        
        assert search_time < 5.0, f"Search should complete in under 5 seconds, took {search_time:.2f}s"
        assert len(response['hits']['hits']) > 0, "Should return results"

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
