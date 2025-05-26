from elasticsearch import Elasticsearch
import logging
from local_embeddings import get_embedding_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to local Elasticsearch
try:
    es_client = Elasticsearch(
        [{"host": "localhost", "port": 9200, "scheme": "http"}],
        verify_certs=False,
        ssl_show_warn=False
    )
except Exception:
    # Fallback for older client versions
    es_client = Elasticsearch(["http://localhost:9200"])

# Initialize local embedding service
embedding_service = get_embedding_service()

# Create index with proper mapping
def create_index(index_name="rag-test-local"):
    """Create Elasticsearch index with proper mapping for local embeddings"""
    # Get embedding dimension from our local service
    embedding_dim = embedding_service.get_embedding_dimension()

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "title": {"type": "text"},
                "text_embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {"type": "object"}
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
    }

    # Delete index if it exists
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        logger.info(f"Deleted existing index: {index_name}")

    es_client.indices.create(index=index_name, body=mapping)
    logger.info(f"Created index: {index_name} with embedding dimension: {embedding_dim}")

def add_document(index_name: str, doc_id: str, text: str, title: str = "", metadata: dict = None):
    """Add a document to the index with local embeddings"""
    if metadata is None:
        metadata = {}

    # Generate embedding locally
    embedding = embedding_service.encode(text)

    document = {
        "text": text,
        "title": title,
        "text_embedding": embedding.tolist(),
        "metadata": metadata
    }

    response = es_client.index(index=index_name, id=doc_id, body=document)
    logger.info(f"Added document {doc_id} to index {index_name}")
    return response

def add_sample_documents(index_name: str):
    """Add sample documents for testing"""
    sample_docs = [
        {
            "id": "doc1",
            "title": "Introduction to RAG",
            "text": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It helps AI models access external knowledge to provide more accurate and up-to-date responses.",
            "metadata": {"category": "AI", "difficulty": "beginner"}
        },
        {
            "id": "doc2",
            "title": "Vector Databases",
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for similarity search and machine learning applications.",
            "metadata": {"category": "Database", "difficulty": "intermediate"}
        },
        {
            "id": "doc3",
            "title": "Elasticsearch Vector Search",
            "text": "Elasticsearch provides powerful vector search capabilities through dense vector fields and k-nearest neighbor (kNN) search. This enables semantic search and similarity matching.",
            "metadata": {"category": "Search", "difficulty": "intermediate"}
        },
        {
            "id": "doc4",
            "title": "Machine Learning Embeddings",
            "text": "Embeddings are dense vector representations of text, images, or other data. They capture semantic meaning and enable machines to understand relationships between different pieces of content.",
            "metadata": {"category": "ML", "difficulty": "beginner"}
        },
        {
            "id": "doc5",
            "title": "Local AI Development",
            "text": "Local AI development allows developers to build and test AI applications without relying on external APIs. This provides better privacy, control, and cost management for AI projects.",
            "metadata": {"category": "Development", "difficulty": "advanced"}
        }
    ]

    for doc in sample_docs:
        add_document(
            index_name=index_name,
            doc_id=doc["id"],
            text=doc["text"],
            title=doc["title"],
            metadata=doc["metadata"]
        )

    # Refresh index to make documents searchable
    es_client.indices.refresh(index=index_name)
    logger.info(f"Added {len(sample_docs)} sample documents to {index_name}")

def search_documents(index_name: str, query: str, k: int = 5):
    """Search documents using hybrid search (vector + text)"""
    # Generate query embedding
    query_embedding = embedding_service.encode(query)

    # Hybrid search combining vector similarity and text matching
    search_body = {
        "knn": {
            "field": "text_embedding",
            "query_vector": query_embedding.tolist(),
            "k": k,
            "num_candidates": k * 4
        },
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["text^2", "title^3"],  # Boost title matches
                "type": "best_fields"
            }
        },
        "size": k
    }

    response = es_client.search(index=index_name, body=search_body)
    return response

def print_search_results(response, query: str):
    """Print search results in a readable format"""
    hits = response['hits']['hits']
    total = response['hits']['total']['value']

    print(f"\nSearch results for: '{query}'")
    print(f"Found {total} total matches, showing top {len(hits)}:")
    print("-" * 80)

    for i, hit in enumerate(hits, 1):
        score = hit['_score']
        source = hit['_source']
        title = source.get('title', 'No title')
        text = source['text']
        metadata = source.get('metadata', {})

        print(f"{i}. {title}")
        print(f"   Score: {score:.4f}")
        print(f"   Category: {metadata.get('category', 'N/A')}")
        print(f"   Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print()

# Main function to test the setup
def main():
    """Main function to demonstrate the RAG system"""
    index_name = "rag-test-local"

    # Check if Elasticsearch is running
    try:
        if not es_client.ping():
            print("Could not connect to Elasticsearch. Make sure it's running.")
            print("Run: docker-compose up -d")
            return
    except Exception as e:
        print(f"Could not connect to Elasticsearch: {e}")
        print("Run: docker-compose up -d")
        return

    print("✓ Connected to Elasticsearch")

    try:
        # Create index
        create_index(index_name)
        print("✓ Created index with local embeddings")

        # Add sample documents
        add_sample_documents(index_name)
        print("✓ Added sample documents")

        # Test searches
        test_queries = [
            "How does RAG improve AI responses?",
            "What are vector databases?",
            "Local AI development benefits",
            "Machine learning embeddings explanation"
        ]

        for query in test_queries:
            response = search_documents(index_name, query)
            print_search_results(response, query)

    except Exception as e:
        logger.error(f"Error during setup or testing: {e}")
        raise

if __name__ == "__main__":
    main()
