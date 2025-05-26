"""
Performance benchmark for the local RAG system
"""
import time
import statistics
import numpy as np
from typing import List, Dict
from local_embeddings import get_embedding_service
from rag_test import create_index, add_document, search_documents, es_client

class RAGBenchmark:
    """Benchmark suite for RAG system performance"""
    
    def __init__(self, index_name: str = "benchmark-rag"):
        self.index_name = index_name
        self.embedding_service = get_embedding_service()
        self.results = {}
    
    def setup_benchmark_data(self, num_docs: int = 100):
        """Create benchmark dataset"""
        print(f"Setting up benchmark with {num_docs} documents...")
        
        # Create index
        if es_client.indices.exists(index=self.index_name):
            es_client.indices.delete(index=self.index_name)
        create_index(self.index_name)
        
        # Generate sample documents
        sample_texts = [
            "Machine learning algorithms for data analysis and pattern recognition",
            "Natural language processing techniques for text understanding",
            "Computer vision applications in image and video analysis",
            "Deep learning neural networks for complex problem solving",
            "Artificial intelligence systems for automated decision making",
            "Data science methodologies for extracting insights from data",
            "Statistical modeling approaches for predictive analytics",
            "Information retrieval systems for document search and ranking",
            "Knowledge representation and reasoning in AI systems",
            "Distributed computing frameworks for large-scale data processing"
        ]
        
        # Add documents
        start_time = time.time()
        for i in range(num_docs):
            text = sample_texts[i % len(sample_texts)]
            add_document(
                self.index_name,
                f"doc_{i}",
                f"{text} Document {i} with additional content for testing purposes.",
                f"Document {i}",
                {"category": f"cat_{i % 5}", "doc_id": i}
            )
        
        es_client.indices.refresh(index=self.index_name)
        setup_time = time.time() - start_time
        print(f"✓ Setup completed in {setup_time:.2f}s")
        return setup_time
    
    def benchmark_embedding_generation(self, num_texts: int = 100) -> Dict:
        """Benchmark embedding generation performance"""
        print(f"Benchmarking embedding generation for {num_texts} texts...")
        
        test_texts = [f"Test sentence number {i} for embedding benchmark" for i in range(num_texts)]
        
        # Single text embedding
        single_times = []
        for text in test_texts[:10]:  # Test first 10 individually
            start = time.time()
            self.embedding_service.encode(text)
            single_times.append(time.time() - start)
        
        # Batch embedding
        start = time.time()
        self.embedding_service.encode(test_texts)
        batch_time = time.time() - start
        
        results = {
            "single_avg": statistics.mean(single_times),
            "single_std": statistics.stdev(single_times) if len(single_times) > 1 else 0,
            "batch_total": batch_time,
            "batch_per_text": batch_time / num_texts,
            "speedup": statistics.mean(single_times) / (batch_time / num_texts)
        }
        
        print(f"✓ Single text: {results['single_avg']:.4f}s ± {results['single_std']:.4f}s")
        print(f"✓ Batch processing: {results['batch_per_text']:.4f}s per text")
        print(f"✓ Batch speedup: {results['speedup']:.2f}x")
        
        return results
    
    def benchmark_search_performance(self, num_queries: int = 50) -> Dict:
        """Benchmark search performance"""
        print(f"Benchmarking search performance for {num_queries} queries...")
        
        test_queries = [
            "machine learning algorithms",
            "natural language processing",
            "computer vision applications",
            "deep learning networks",
            "artificial intelligence systems",
            "data science methods",
            "statistical modeling",
            "information retrieval",
            "knowledge representation",
            "distributed computing"
        ]
        
        search_times = []
        result_counts = []
        
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            
            start = time.time()
            response = search_documents(self.index_name, query, k=5)
            search_time = time.time() - start
            
            search_times.append(search_time)
            result_counts.append(len(response['hits']['hits']))
        
        results = {
            "avg_time": statistics.mean(search_times),
            "std_time": statistics.stdev(search_times) if len(search_times) > 1 else 0,
            "min_time": min(search_times),
            "max_time": max(search_times),
            "avg_results": statistics.mean(result_counts),
            "queries_per_second": 1 / statistics.mean(search_times)
        }
        
        print(f"✓ Average search time: {results['avg_time']:.4f}s ± {results['std_time']:.4f}s")
        print(f"✓ Search range: {results['min_time']:.4f}s - {results['max_time']:.4f}s")
        print(f"✓ Queries per second: {results['queries_per_second']:.1f}")
        print(f"✓ Average results per query: {results['avg_results']:.1f}")
        
        return results
    
    def benchmark_scalability(self, doc_counts: List[int] = [10, 50, 100, 500]) -> Dict:
        """Benchmark system scalability with different document counts"""
        print("Benchmarking scalability...")
        
        scalability_results = {}
        
        for doc_count in doc_counts:
            print(f"\nTesting with {doc_count} documents...")
            
            # Setup data
            setup_time = self.setup_benchmark_data(doc_count)
            
            # Test search performance
            test_query = "machine learning algorithms"
            search_times = []
            
            for _ in range(10):  # 10 search queries
                start = time.time()
                search_documents(self.index_name, test_query, k=5)
                search_times.append(time.time() - start)
            
            scalability_results[doc_count] = {
                "setup_time": setup_time,
                "avg_search_time": statistics.mean(search_times),
                "std_search_time": statistics.stdev(search_times) if len(search_times) > 1 else 0
            }
            
            print(f"✓ {doc_count} docs: {scalability_results[doc_count]['avg_search_time']:.4f}s search")
        
        return scalability_results
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("=" * 60)
        print("LOCAL RAG SYSTEM PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        # Check system
        if not es_client.ping():
            print("✗ Elasticsearch not available")
            return
        
        print(f"✓ Elasticsearch connected")
        print(f"✓ Embedding model: {self.embedding_service.model_name}")
        print(f"✓ Embedding dimension: {self.embedding_service.get_embedding_dimension()}")
        print()
        
        # Setup benchmark data
        self.setup_benchmark_data(100)
        print()
        
        # Run benchmarks
        self.results['embedding'] = self.benchmark_embedding_generation(100)
        print()
        
        self.results['search'] = self.benchmark_search_performance(50)
        print()
        
        self.results['scalability'] = self.benchmark_scalability([10, 50, 100])
        print()
        
        # Summary
        print("=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Embedding generation: {self.results['embedding']['batch_per_text']:.4f}s per text")
        print(f"Search performance: {self.results['search']['avg_time']:.4f}s per query")
        print(f"Queries per second: {self.results['search']['queries_per_second']:.1f}")
        print(f"Batch speedup: {self.results['embedding']['speedup']:.2f}x")
        
        # Cleanup
        if es_client.indices.exists(index=self.index_name):
            es_client.indices.delete(index=self.index_name)
        
        return self.results

if __name__ == "__main__":
    benchmark = RAGBenchmark()
    results = benchmark.run_full_benchmark()
