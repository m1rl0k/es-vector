"""
Demonstration of document chunking capabilities
Shows different chunking strategies and their effects
"""
import logging
from document_chunker import DocumentChunker, ChunkingStrategy
from enhanced_rag import EnhancedRAGSystem
from rag_test import es_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_documents():
    """Create sample documents for demonstration"""
    return {
        "ai_overview": """
        Artificial Intelligence: A Comprehensive Overview
        
        Artificial Intelligence (AI) represents one of the most transformative technologies of our time. 
        It encompasses a broad range of techniques and approaches designed to enable machines to perform 
        tasks that typically require human intelligence.
        
        Historical Development
        The concept of artificial intelligence dates back to ancient times, but modern AI began in the 
        1950s with pioneers like Alan Turing, John McCarthy, and Marvin Minsky. The field has experienced 
        several waves of optimism and setbacks, known as "AI winters," but has seen remarkable progress 
        in recent decades.
        
        Core Technologies
        Machine Learning forms the backbone of modern AI systems. It enables computers to learn from 
        data without being explicitly programmed for every task. Deep Learning, a subset of machine 
        learning, uses artificial neural networks with multiple layers to model complex patterns.
        
        Natural Language Processing (NLP) allows machines to understand, interpret, and generate human 
        language. Recent advances in transformer architectures have revolutionized this field, leading 
        to powerful language models.
        
        Computer Vision enables machines to interpret and understand visual information from the world. 
        Applications range from medical image analysis to autonomous vehicle navigation.
        
        Current Applications
        AI is transforming numerous industries. In healthcare, AI assists with medical diagnosis, drug 
        discovery, and personalized treatment plans. Financial services use AI for fraud detection, 
        algorithmic trading, and risk assessment.
        
        Transportation is being revolutionized by autonomous vehicles and intelligent traffic management 
        systems. Entertainment platforms use AI for content recommendation and creation.
        
        Future Prospects
        The future of AI holds immense promise and challenges. Advances in quantum computing may 
        accelerate AI capabilities. However, ethical considerations around bias, privacy, and job 
        displacement require careful attention.
        
        Artificial General Intelligence (AGI) remains a long-term goal, representing AI systems that 
        can match or exceed human cognitive abilities across all domains.
        """,
        
        "ml_fundamentals": """
        Machine Learning Fundamentals
        
        Machine Learning (ML) is a subset of artificial intelligence that focuses on developing 
        algorithms and statistical models that enable computer systems to improve their performance 
        on a specific task through experience.
        
        Types of Machine Learning
        
        Supervised Learning involves training algorithms on labeled datasets. The algorithm learns 
        to map inputs to outputs based on example input-output pairs. Common applications include 
        classification and regression tasks.
        
        Unsupervised Learning works with unlabeled data to discover hidden patterns or structures. 
        Clustering, dimensionality reduction, and association rule learning are typical unsupervised 
        learning tasks.
        
        Reinforcement Learning involves an agent learning to make decisions by interacting with an 
        environment. The agent receives rewards or penalties based on its actions and learns to 
        maximize cumulative reward over time.
        
        Key Algorithms
        
        Linear Regression is one of the simplest ML algorithms, used for predicting continuous values. 
        It assumes a linear relationship between input features and the target variable.
        
        Decision Trees create a model that predicts target values by learning simple decision rules 
        inferred from data features. They are easy to interpret but can overfit complex datasets.
        
        Random Forest combines multiple decision trees to create a more robust and accurate model. 
        It reduces overfitting and provides better generalization.
        
        Support Vector Machines (SVM) find the optimal boundary between different classes by 
        maximizing the margin between them. They work well for both linear and non-linear problems.
        
        Neural Networks are inspired by biological neural networks and consist of interconnected 
        nodes (neurons) that process information. Deep neural networks with many layers enable 
        complex pattern recognition.
        """,
        
        "data_science_workflow": """
        Data Science Workflow and Best Practices
        
        Data science is an interdisciplinary field that combines statistical analysis, machine 
        learning, and domain expertise to extract insights from data.
        
        The Data Science Process
        
        Problem Definition is the first and most crucial step. It involves understanding the 
        business problem, defining success metrics, and determining what questions need to be answered.
        
        Data Collection involves gathering relevant data from various sources. This may include 
        databases, APIs, web scraping, surveys, or sensor data. Data quality and completeness 
        are critical considerations.
        
        Data Exploration and Analysis helps understand the data's structure, quality, and patterns. 
        Exploratory Data Analysis (EDA) uses statistical summaries and visualizations to gain insights.
        
        Data Preprocessing includes cleaning, transforming, and preparing data for analysis. This 
        involves handling missing values, outliers, and inconsistencies, as well as feature 
        engineering and selection.
        
        Model Development involves selecting appropriate algorithms, training models, and tuning 
        hyperparameters. Cross-validation helps ensure model generalizability.
        
        Model Evaluation assesses model performance using appropriate metrics. For classification 
        tasks, metrics include accuracy, precision, recall, and F1-score. For regression, common 
        metrics are mean squared error and R-squared.
        
        Deployment and Monitoring involve putting models into production and continuously monitoring 
        their performance. Models may degrade over time due to data drift or changing conditions.
        
        Best Practices
        
        Version Control is essential for tracking changes in code, data, and models. Git and 
        specialized ML platforms help manage the complexity of data science projects.
        
        Documentation ensures reproducibility and knowledge transfer. Well-documented code and 
        analysis make it easier for others to understand and build upon your work.
        
        Ethical Considerations include ensuring fairness, avoiding bias, protecting privacy, and 
        considering the societal impact of data science applications.
        """
    }

def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies"""
    print("=" * 80)
    print("DOCUMENT CHUNKING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Initialize chunker
    chunker = DocumentChunker(
        chunk_size=800,
        chunk_overlap=150,
        min_chunk_size=100,
        max_chunk_size=1200
    )
    
    # Test different strategies
    strategies = [
        ChunkingStrategy.FIXED_SIZE,
        ChunkingStrategy.SENTENCE_BASED,
        ChunkingStrategy.PARAGRAPH_BASED,
        ChunkingStrategy.RECURSIVE
    ]
    
    for doc_id, text in documents.items():
        print(f"\nDocument: {doc_id}")
        print(f"Original length: {len(text)} characters")
        print("-" * 60)
        
        for strategy in strategies:
            chunks = chunker.chunk_document(text, doc_id, strategy)
            stats = chunker.get_chunk_statistics(chunks)
            
            print(f"\n{strategy.value.upper()} Strategy:")
            print(f"  Chunks created: {stats['total_chunks']}")
            print(f"  Average chunk size: {stats['avg_chunk_size']:.0f} chars")
            print(f"  Size range: {stats['min_chunk_size']}-{stats['max_chunk_size']} chars")
            
            # Show first chunk as example
            if chunks:
                first_chunk = chunks[0]
                preview = first_chunk.text[:200] + "..." if len(first_chunk.text) > 200 else first_chunk.text
                print(f"  First chunk preview: {preview}")

def demonstrate_enhanced_rag():
    """Demonstrate enhanced RAG system with chunking"""
    print("\n" + "=" * 80)
    print("ENHANCED RAG SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Check Elasticsearch connection
    if not es_client.ping():
        print("❌ Elasticsearch is not running. Please start it with: make start")
        return
    
    print("✅ Connected to Elasticsearch")
    
    # Initialize enhanced RAG system
    rag_system = EnhancedRAGSystem(
        index_name="demo-chunking-rag",
        chunk_size=600,
        chunk_overlap=100,
        chunking_strategy=ChunkingStrategy.RECURSIVE
    )
    
    # Add sample documents
    documents = create_sample_documents()
    
    print("\nAdding documents to RAG system...")
    for doc_id, text in documents.items():
        chunk_ids = rag_system.add_document(
            document_id=doc_id,
            text=text,
            title=doc_id.replace("_", " ").title(),
            metadata={"category": "demo", "type": "educational"}
        )
        print(f"✅ Added {doc_id}: {len(chunk_ids)} chunks created")
    
    # Demonstrate search capabilities
    print("\nDemonstrating search capabilities...")
    
    test_queries = [
        "What is machine learning?",
        "How does supervised learning work?",
        "Data science workflow steps",
        "Neural networks and deep learning",
        "AI applications in healthcare"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = rag_system.search(query, k=3, include_chunk_context=True)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i} (Score: {result['score']:.4f}):")
            print(f"    Document: {result['document_id']}")
            print(f"    Chunk: {result['chunk_index'] + 1}/{result['total_chunks']}")
            print(f"    Size: {result['chunk_size']} chars")
            
            # Show text preview
            text_preview = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
            print(f"    Text: {text_preview}")
            
            # Show context if available
            if result.get('context'):
                context = result['context']
                if context['previous_chunks']:
                    print(f"    Previous context: {len(context['previous_chunks'])} chunks")
                if context['next_chunks']:
                    print(f"    Next context: {len(context['next_chunks'])} chunks")
    
    # Show system statistics
    print("\n" + "-" * 60)
    print("SYSTEM STATISTICS")
    print("-" * 60)
    
    stats = rag_system.get_system_statistics()
    print(f"Total chunks indexed: {stats['total_chunks']}")
    print(f"Index size: {stats['index_size_bytes'] / 1024:.1f} KB")
    print("Strategy distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        print(f"  {strategy}: {count} chunks")
    
    # Cleanup
    print(f"\n🧹 Cleaning up demo index...")
    if es_client.indices.exists(index="demo-chunking-rag"):
        es_client.indices.delete(index="demo-chunking-rag")
    print("✅ Cleanup complete")

def main():
    """Main demonstration function"""
    try:
        # Demonstrate chunking strategies
        demonstrate_chunking_strategies()
        
        # Demonstrate enhanced RAG system
        demonstrate_enhanced_rag()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\nKey takeaways:")
        print("• Different chunking strategies produce different results")
        print("• Recursive strategy adapts to document structure")
        print("• Enhanced RAG system automatically handles chunking")
        print("• Search results include chunk context and metadata")
        print("• System provides comprehensive statistics and monitoring")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise

if __name__ == "__main__":
    main()
