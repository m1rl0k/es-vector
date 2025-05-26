"""
Unified workflow that runs both basic RAG and enhanced RAG with document chunking
Provides comprehensive comparison and results output
"""
import time
import json
import logging
from typing import Dict, List, Any
from document_chunker import DocumentChunker, ChunkingStrategy
from enhanced_rag import EnhancedRAGSystem
from rag_test import es_client, embedding_service, create_index, add_sample_documents, search_documents
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedRAGWorkflow:
    """Unified workflow for comparing basic and enhanced RAG systems"""

    def __init__(self):
        self.basic_index = "workflow-basic-rag"
        self.enhanced_index = "workflow-enhanced-rag"
        self.test_queries = [
            "What is machine learning and how does it work?",
            "Explain artificial intelligence applications",
            "How does RAG improve AI responses?",
            "What are vector databases used for?",
            "Describe natural language processing techniques"
        ]
        self.results = {
            "basic_rag": {},
            "enhanced_rag": {},
            "comparison": {},
            "performance": {}
        }

    def setup_test_documents(self) -> List[Dict[str, str]]:
        """Create comprehensive test documents"""
        return [
            {
                "id": "ai_comprehensive",
                "title": "Comprehensive Guide to Artificial Intelligence",
                "text": """
                Artificial Intelligence (AI) is a transformative technology that encompasses machine learning,
                deep learning, natural language processing, and computer vision. Modern AI systems can perform
                complex tasks like image recognition, language translation, and decision-making.

                Machine Learning is a subset of AI that enables systems to learn from data without explicit
                programming. It includes supervised learning (with labeled data), unsupervised learning
                (finding patterns in unlabeled data), and reinforcement learning (learning through interaction).

                Deep Learning uses neural networks with multiple layers to process complex patterns. These
                networks can handle unstructured data like images, text, and audio, making them powerful
                for applications like computer vision and natural language processing.

                Natural Language Processing (NLP) allows machines to understand and generate human language.
                Modern NLP uses transformer architectures and large language models to achieve human-like
                text understanding and generation capabilities.

                AI applications span healthcare (medical diagnosis, drug discovery), finance (fraud detection,
                algorithmic trading), transportation (autonomous vehicles), and entertainment (recommendation
                systems, content generation).
                """
            },
            {
                "id": "rag_systems",
                "title": "Retrieval-Augmented Generation Systems",
                "text": """
                Retrieval-Augmented Generation (RAG) combines information retrieval with text generation to
                create more accurate and contextual AI responses. RAG systems first retrieve relevant
                information from a knowledge base, then use this information to generate informed responses.

                Vector databases are essential for RAG systems, storing document embeddings that enable
                semantic search. These databases can quickly find relevant information based on meaning
                rather than just keyword matching.

                Document chunking is crucial for RAG systems when dealing with large documents. Proper
                chunking strategies ensure that relevant information is retrievable while maintaining
                context and semantic coherence.

                Hybrid search combines vector similarity search with traditional text search to improve
                retrieval accuracy. This approach leverages both semantic understanding and exact keyword
                matching for better results.

                RAG systems excel in applications requiring up-to-date information, domain-specific
                knowledge, and factual accuracy. They're particularly valuable for question-answering
                systems, chatbots, and knowledge management platforms.
                """
            },
            {
                "id": "ml_algorithms",
                "title": "Machine Learning Algorithms and Techniques",
                "text": """
                Machine learning algorithms can be categorized into several types based on their learning
                approach and application domain. Understanding these categories helps in selecting the
                right algorithm for specific problems.

                Supervised learning algorithms learn from labeled training data to make predictions on
                new data. Popular algorithms include linear regression, decision trees, random forests,
                support vector machines, and neural networks.

                Unsupervised learning algorithms find patterns in data without labeled examples. Common
                techniques include clustering (k-means, hierarchical clustering), dimensionality reduction
                (PCA, t-SNE), and association rule learning.

                Ensemble methods combine multiple algorithms to improve performance. Random forests combine
                multiple decision trees, while gradient boosting builds models sequentially to correct
                previous errors.

                Feature engineering is crucial for machine learning success. It involves selecting,
                transforming, and creating relevant features from raw data to improve model performance
                and interpretability.
                """
            }
        ]

    def run_basic_rag_workflow(self) -> Dict[str, Any]:
        """Run the basic RAG system workflow"""
        logger.info("🚀 Starting Basic RAG Workflow")

        start_time = time.time()

        # Setup basic RAG
        if es_client.indices.exists(index=self.basic_index):
            es_client.indices.delete(index=self.basic_index)

        create_index(self.basic_index)
        add_sample_documents(self.basic_index)

        setup_time = time.time() - start_time

        # Test search performance
        search_results = []
        search_times = []

        for query in self.test_queries:
            start_search = time.time()
            response = search_documents(self.basic_index, query, k=3)
            search_time = time.time() - start_search
            search_times.append(search_time)

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    "score": hit['_score'],
                    "text_preview": hit['_source']['text'][:200] + "...",
                    "title": hit['_source'].get('title', 'No title')
                })

            search_results.append({
                "query": query,
                "results": results,
                "search_time": search_time,
                "total_hits": response['hits']['total']['value']
            })

        # Get system stats
        stats_response = es_client.indices.stats(index=self.basic_index)
        index_stats = stats_response['indices'][self.basic_index]['total']

        basic_results = {
            "setup_time": setup_time,
            "avg_search_time": statistics.mean(search_times),
            "total_documents": index_stats['docs']['count'],
            "index_size_bytes": index_stats['store']['size_in_bytes'],
            "search_results": search_results,
            "performance_metrics": {
                "min_search_time": min(search_times),
                "max_search_time": max(search_times),
                "queries_per_second": 1 / statistics.mean(search_times)
            }
        }

        logger.info(f"✅ Basic RAG completed in {setup_time:.2f}s")
        return basic_results

    def run_enhanced_rag_workflow(self) -> Dict[str, Any]:
        """Run the enhanced RAG system workflow with document chunking"""
        logger.info("🚀 Starting Enhanced RAG Workflow")

        start_time = time.time()

        # Setup enhanced RAG
        rag_system = EnhancedRAGSystem(
            index_name=self.enhanced_index,
            chunk_size=600,
            chunk_overlap=100,
            chunking_strategy=ChunkingStrategy.RECURSIVE
        )

        # Add test documents with chunking
        test_docs = self.setup_test_documents()
        total_chunks = 0

        for doc in test_docs:
            chunk_ids = rag_system.add_document(
                document_id=doc["id"],
                text=doc["text"],
                title=doc["title"],
                metadata={"category": "test", "workflow": "enhanced"}
            )
            total_chunks += len(chunk_ids)

        setup_time = time.time() - start_time

        # Test search performance with chunking
        search_results = []
        search_times = []

        for query in self.test_queries:
            start_search = time.time()
            results = rag_system.search(query, k=3, include_chunk_context=True)
            search_time = time.time() - start_search
            search_times.append(search_time)

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result['score'],
                    "text_preview": result['text'][:200] + "...",
                    "title": result['title'],
                    "document_id": result['document_id'],
                    "chunk_info": f"Chunk {result['chunk_index'] + 1}/{result['total_chunks']}",
                    "chunk_size": result['chunk_size'],
                    "strategy": result['strategy'],
                    "has_context": bool(result.get('context', {}).get('previous_chunks') or
                                      result.get('context', {}).get('next_chunks'))
                })

            search_results.append({
                "query": query,
                "results": formatted_results,
                "search_time": search_time,
                "total_results": len(results)
            })

        # Get enhanced system stats
        system_stats = rag_system.get_system_statistics()

        enhanced_results = {
            "setup_time": setup_time,
            "avg_search_time": statistics.mean(search_times),
            "total_chunks": system_stats['total_chunks'],
            "total_documents": len(test_docs),
            "index_size_bytes": system_stats['index_size_bytes'],
            "strategy_distribution": system_stats['strategy_distribution'],
            "search_results": search_results,
            "performance_metrics": {
                "min_search_time": min(search_times),
                "max_search_time": max(search_times),
                "queries_per_second": 1 / statistics.mean(search_times),
                "avg_chunks_per_doc": total_chunks / len(test_docs)
            }
        }

        logger.info(f"✅ Enhanced RAG completed in {setup_time:.2f}s")
        return enhanced_results

    def compare_systems(self, basic_results: Dict[str, Any], enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare basic and enhanced RAG systems"""
        logger.info("📊 Comparing RAG Systems")

        comparison = {
            "performance_comparison": {
                "setup_time": {
                    "basic": basic_results['setup_time'],
                    "enhanced": enhanced_results['setup_time'],
                    "difference": enhanced_results['setup_time'] - basic_results['setup_time'],
                    "winner": "basic" if basic_results['setup_time'] < enhanced_results['setup_time'] else "enhanced"
                },
                "search_speed": {
                    "basic_avg": basic_results['avg_search_time'],
                    "enhanced_avg": enhanced_results['avg_search_time'],
                    "speedup": basic_results['avg_search_time'] / enhanced_results['avg_search_time'],
                    "winner": "basic" if basic_results['avg_search_time'] < enhanced_results['avg_search_time'] else "enhanced"
                },
                "throughput": {
                    "basic_qps": basic_results['performance_metrics']['queries_per_second'],
                    "enhanced_qps": enhanced_results['performance_metrics']['queries_per_second'],
                    "improvement": enhanced_results['performance_metrics']['queries_per_second'] /
                                 basic_results['performance_metrics']['queries_per_second']
                }
            },
            "data_comparison": {
                "basic_documents": basic_results['total_documents'],
                "enhanced_chunks": enhanced_results['total_chunks'],
                "enhanced_documents": enhanced_results['total_documents'],
                "chunking_ratio": enhanced_results['total_chunks'] / enhanced_results['total_documents'],
                "index_size_basic": basic_results['index_size_bytes'],
                "index_size_enhanced": enhanced_results['index_size_bytes'],
                "size_difference": enhanced_results['index_size_bytes'] - basic_results['index_size_bytes']
            },
            "feature_comparison": {
                "basic_features": [
                    "Simple document indexing",
                    "Hybrid search (vector + text)",
                    "Basic metadata",
                    "Single document per entry"
                ],
                "enhanced_features": [
                    "Automatic document chunking",
                    "Multiple chunking strategies",
                    "Context preservation",
                    "Chunk-level search",
                    "Rich metadata tracking",
                    "Neighboring chunk context",
                    "Document management"
                ]
            }
        }

        return comparison

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive report of both systems"""
        report = []
        report.append("=" * 100)
        report.append("UNIFIED RAG WORKFLOW RESULTS")
        report.append("=" * 100)

        # Basic RAG Results
        basic = self.results['basic_rag']
        report.append("\n📋 BASIC RAG SYSTEM RESULTS")
        report.append("-" * 50)
        report.append(f"Setup Time: {basic['setup_time']:.3f}s")
        report.append(f"Total Documents: {basic['total_documents']}")
        report.append(f"Index Size: {basic['index_size_bytes'] / 1024:.1f} KB")
        report.append(f"Average Search Time: {basic['avg_search_time']:.4f}s")
        report.append(f"Queries per Second: {basic['performance_metrics']['queries_per_second']:.1f}")

        report.append("\nSample Search Results:")
        for i, search in enumerate(basic['search_results'][:2], 1):
            report.append(f"  Query {i}: '{search['query']}'")
            report.append(f"    Results: {search['total_hits']} hits in {search['search_time']:.4f}s")
            if search['results']:
                top_result = search['results'][0]
                report.append(f"    Top Result: {top_result['title']} (Score: {top_result['score']:.4f})")

        # Enhanced RAG Results
        enhanced = self.results['enhanced_rag']
        report.append("\n🚀 ENHANCED RAG SYSTEM RESULTS")
        report.append("-" * 50)
        report.append(f"Setup Time: {enhanced['setup_time']:.3f}s")
        report.append(f"Total Documents: {enhanced['total_documents']}")
        report.append(f"Total Chunks: {enhanced['total_chunks']}")
        report.append(f"Average Chunks per Document: {enhanced['performance_metrics']['avg_chunks_per_doc']:.1f}")
        report.append(f"Index Size: {enhanced['index_size_bytes'] / 1024:.1f} KB")
        report.append(f"Average Search Time: {enhanced['avg_search_time']:.4f}s")
        report.append(f"Queries per Second: {enhanced['performance_metrics']['queries_per_second']:.1f}")

        report.append("\nChunking Strategy Distribution:")
        for strategy, count in enhanced['strategy_distribution'].items():
            report.append(f"  {strategy}: {count} chunks")

        report.append("\nSample Search Results:")
        for i, search in enumerate(enhanced['search_results'][:2], 1):
            report.append(f"  Query {i}: '{search['query']}'")
            report.append(f"    Results: {search['total_results']} chunks in {search['search_time']:.4f}s")
            if search['results']:
                top_result = search['results'][0]
                report.append(f"    Top Result: {top_result['title']} - {top_result['chunk_info']}")
                report.append(f"      Score: {top_result['score']:.4f}, Size: {top_result['chunk_size']} chars")
                report.append(f"      Strategy: {top_result['strategy']}, Has Context: {top_result['has_context']}")

        # Comparison Results
        comparison = self.results['comparison']
        report.append("\n⚖️  SYSTEM COMPARISON")
        report.append("-" * 50)

        perf = comparison['performance_comparison']
        report.append(f"Setup Time Winner: {perf['setup_time']['winner'].upper()}")
        report.append(f"  Basic: {perf['setup_time']['basic']:.3f}s")
        report.append(f"  Enhanced: {perf['setup_time']['enhanced']:.3f}s")
        report.append(f"  Difference: {perf['setup_time']['difference']:.3f}s")

        report.append(f"\nSearch Speed Winner: {perf['search_speed']['winner'].upper()}")
        report.append(f"  Basic: {perf['search_speed']['basic_avg']:.4f}s")
        report.append(f"  Enhanced: {perf['search_speed']['enhanced_avg']:.4f}s")
        report.append(f"  Speedup: {perf['search_speed']['speedup']:.2f}x")

        data = comparison['data_comparison']
        report.append(f"\nData Organization:")
        report.append(f"  Basic: {data['basic_documents']} documents")
        report.append(f"  Enhanced: {data['enhanced_documents']} documents → {data['enhanced_chunks']} chunks")
        report.append(f"  Chunking Ratio: {data['chunking_ratio']:.1f} chunks per document")
        report.append(f"  Index Size Difference: {data['size_difference'] / 1024:.1f} KB")

        # Feature Comparison
        features = comparison['feature_comparison']
        report.append(f"\nFeature Comparison:")
        report.append(f"  Basic RAG Features:")
        for feature in features['basic_features']:
            report.append(f"    • {feature}")
        report.append(f"  Enhanced RAG Features:")
        for feature in features['enhanced_features']:
            report.append(f"    • {feature}")

        # Performance Summary
        report.append("\n🏆 PERFORMANCE SUMMARY")
        report.append("-" * 50)

        if perf['search_speed']['winner'] == 'enhanced':
            report.append(f"✅ Enhanced RAG is {perf['search_speed']['speedup']:.2f}x faster for search")
        else:
            report.append(f"⚠️  Basic RAG is faster for search by {1/perf['search_speed']['speedup']:.2f}x")

        report.append(f"📊 Enhanced RAG provides {data['chunking_ratio']:.1f}x more granular search")
        report.append(f"🎯 Enhanced RAG offers superior context preservation and metadata")
        report.append(f"🔧 Enhanced RAG supports multiple chunking strategies")

        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("Use Basic RAG when:")
        report.append("  • Working with small, well-structured documents")
        report.append("  • Simple search requirements")
        report.append("  • Minimal setup time is critical")

        report.append("\nUse Enhanced RAG when:")
        report.append("  • Processing large documents")
        report.append("  • Need context preservation")
        report.append("  • Require granular search results")
        report.append("  • Want rich metadata and chunk relationships")
        report.append("  • Building production RAG systems")

        report.append("\n" + "=" * 100)

        return "\n".join(report)

    def save_results_to_json(self, filename: str = "unified_rag_results.json"):
        """Save all results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"💾 Results saved to {filename}")

    def run_unified_workflow(self) -> Dict[str, Any]:
        """Run the complete unified workflow"""
        logger.info("🎯 Starting Unified RAG Workflow")

        # Check Elasticsearch connection
        if not es_client.ping():
            raise ConnectionError("Elasticsearch is not running. Please start it with: make start")

        workflow_start = time.time()

        try:
            # Run basic RAG workflow
            self.results['basic_rag'] = self.run_basic_rag_workflow()

            # Run enhanced RAG workflow
            self.results['enhanced_rag'] = self.run_enhanced_rag_workflow()

            # Compare systems
            self.results['comparison'] = self.compare_systems(
                self.results['basic_rag'],
                self.results['enhanced_rag']
            )

            # Add overall performance metrics
            total_time = time.time() - workflow_start
            self.results['performance'] = {
                "total_workflow_time": total_time,
                "basic_rag_time": self.results['basic_rag']['setup_time'],
                "enhanced_rag_time": self.results['enhanced_rag']['setup_time'],
                "comparison_overhead": total_time - (
                    self.results['basic_rag']['setup_time'] +
                    self.results['enhanced_rag']['setup_time']
                )
            }

            # Generate and display report
            report = self.generate_comprehensive_report()
            print(report)

            # Save results to JSON
            self.save_results_to_json()

            logger.info(f"🎉 Unified workflow completed in {total_time:.2f}s")

            return self.results

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise
        finally:
            # Cleanup indices
            self.cleanup()

    def cleanup(self):
        """Clean up test indices"""
        try:
            for index in [self.basic_index, self.enhanced_index]:
                if es_client.indices.exists(index=index):
                    es_client.indices.delete(index=index)
                    logger.info(f"🧹 Cleaned up index: {index}")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")

def main():
    """Main function to run the unified workflow"""
    try:
        workflow = UnifiedRAGWorkflow()
        results = workflow.run_unified_workflow()

        print("\n" + "=" * 100)
        print("WORKFLOW SUMMARY")
        print("=" * 100)
        print(f"✅ Basic RAG: {results['basic_rag']['total_documents']} documents indexed")
        print(f"✅ Enhanced RAG: {results['enhanced_rag']['total_chunks']} chunks from {results['enhanced_rag']['total_documents']} documents")
        print(f"⏱️  Total Time: {results['performance']['total_workflow_time']:.2f}s")
        print(f"📊 Results saved to: unified_rag_results.json")
        print(f"🎯 Workflow completed successfully!")

        return results

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return None

if __name__ == "__main__":
    main()
