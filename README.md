# Local RAG/Elasticsearch Testing Environment

A pure local testing environment for Retrieval-Augmented Generation (RAG) using Elasticsearch vector database and local machine learning models. No external API dependencies required.

## Features

- **Pure Local Setup**: No external API keys or cloud dependencies
- **Local Embeddings**: Uses sentence-transformers for text embeddings
- **Document Chunking**: Multiple strategies for handling large documents
- **Elasticsearch Vector Search**: Leverages Elasticsearch's dense vector capabilities
- **Hybrid Search**: Combines vector similarity with traditional text search
- **Enhanced RAG System**: Automatic chunking with context preservation
- **Comprehensive Testing**: Full test suite with performance benchmarks
- **Easy Setup**: Docker Compose for Elasticsearch, pip for Python dependencies

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Large Document  │───▶│ Document Chunker │───▶│   Text Chunks   │
│                 │    │ (Multi-Strategy) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Chunks   │───▶│ Local Embeddings │───▶│  Elasticsearch  │
│                 │    │ (sentence-trans) │    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│ Search Results  │◀───│  Hybrid Search   │◀───────────┘
│ + Context       │    │ (Vector + Text)  │
└─────────────────┘    └──────────────────┘
```

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Docker and Docker Compose
- 4GB+ RAM (for embedding models)

### 2. Setup

```bash
# Clone or navigate to the project directory
cd es-vector

# Start Elasticsearch
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Run the basic test
python rag_test.py

# Run unified workflow (basic + enhanced RAG comparison)
python unified_workflow.py
```

### 3. Run Tests

```bash
# Run comprehensive test suite
python test_rag_system.py

# Run document chunking tests
python test_document_chunking.py

# Or use pytest for detailed output
pytest test_rag_system.py -v
pytest test_document_chunking.py -v
```

## Quick Commands

```bash
# Using Makefile (recommended)
make start        # Start Elasticsearch
make demo         # Run basic RAG demo
make demo-chunking # Run document chunking demo
make workflow     # Run unified workflow (basic + enhanced comparison)
make test         # Run all tests
make test-chunking # Run chunking tests
make benchmark    # Run performance benchmarks
make stop         # Stop Elasticsearch

# Direct Python execution
python rag_test.py              # Basic RAG demo
python demo_chunking.py         # Document chunking demo
python unified_workflow.py      # Unified workflow comparison
python benchmark.py             # Performance benchmarks
```

## Project Structure

```
es-vector/
├── docker-compose.yml         # Elasticsearch container setup
├── requirements.txt           # Python dependencies
├── local_embeddings.py        # Local embedding service
├── document_chunker.py        # Document chunking strategies
├── enhanced_rag.py           # Enhanced RAG system with chunking
├── rag_test.py               # Basic RAG demonstration
├── demo_chunking.py          # Document chunking demonstration
├── unified_workflow.py       # Unified workflow (basic + enhanced comparison)
├── test_rag_system.py        # Basic RAG test suite
├── test_document_chunking.py # Document chunking test suite
├── benchmark.py              # Performance benchmarking
├── Makefile                  # Easy commands
└── README.md                 # This file
```

## Core Components

### Local Embeddings (`local_embeddings.py`)

- Uses `sentence-transformers` library
- Default model: `all-MiniLM-L6-v2` (384 dimensions, fast)
- Alternative: `all-mpnet-base-v2` (768 dimensions, higher quality)
- No external API calls required

### Document Chunking (`document_chunker.py`)

- **Multiple Strategies**: Fixed-size, sentence-based, paragraph-based, semantic sliding, recursive
- **Smart Overlap**: Configurable overlap between chunks for context preservation
- **Metadata Tracking**: Comprehensive metadata for each chunk including position, size, strategy
- **Adaptive Processing**: Recursive strategy automatically selects best approach per document
- **Size Management**: Configurable min/max chunk sizes with automatic merging of small chunks

### Enhanced RAG System (`enhanced_rag.py`)

- **Automatic Chunking**: Integrates document chunking with RAG pipeline
- **Context Preservation**: Maintains relationships between chunks
- **Flexible Search**: Supports chunk-level search with document context
- **Metadata Rich**: Comprehensive tracking of document and chunk information
- **Easy Management**: Simple API for adding, searching, and managing documents

### Unified Workflow (`unified_workflow.py`)

- **Complete Comparison**: Runs both basic and enhanced RAG systems
- **Performance Analysis**: Detailed performance metrics and comparison
- **Comprehensive Reporting**: Human-readable report with recommendations
- **JSON Output**: Machine-readable results for further analysis
- **Automatic Cleanup**: Manages test indices and cleanup

### Basic RAG System (`rag_test.py`)

- Creates Elasticsearch index with proper vector mapping
- Adds sample documents with embeddings
- Implements hybrid search (vector + text)
- Demonstrates end-to-end RAG workflow

### Test Suite (`test_rag_system.py`)

- Connection and setup tests
- Embedding consistency and similarity tests
- Search functionality and relevance tests
- Performance benchmarks

## Usage Examples

### Basic Search

```python
from rag_test import search_documents, print_search_results

# Search for documents
response = search_documents("rag-test-local", "machine learning embeddings")
print_search_results(response, "machine learning embeddings")
```

### Adding Custom Documents

```python
from rag_test import add_document

add_document(
    index_name="rag-test-local",
    doc_id="custom_doc_1",
    text="Your custom document text here",
    title="Custom Document",
    metadata={"category": "custom", "source": "user"}
)
```

### Custom Embedding Model

```python
from local_embeddings import LocalEmbeddingService

# Use a different model
embedding_service = LocalEmbeddingService("all-mpnet-base-v2")
embeddings = embedding_service.encode(["text1", "text2"])
```

## GitHub Actions CI/CD

This repository includes a comprehensive GitHub Actions workflow that runs both basic and enhanced RAG systems in a matrix configuration.

### Workflow Overview

The workflow (`.github/workflows/rag-testing.yml`) includes:

1. **Basic RAG Testing**: Tests the basic RAG system across multiple Python versions
2. **Enhanced RAG Matrix**: Tests document chunking with different strategies and chunk sizes
3. **Unified Workflow**: Runs comprehensive comparison between basic and enhanced systems
4. **Performance Benchmarking**: Optional performance testing
5. **Result Aggregation**: Combines all matrix results into a comprehensive report

### Matrix Configuration

**Enhanced RAG Matrix**:
- **Python Versions**: 3.9, 3.10, 3.11
- **Chunking Strategies**: fixed_size, sentence_based, paragraph_based, recursive
- **Chunk Sizes**: 500, 800, 1000 characters
- **Total Combinations**: 36 test configurations

### Triggering Workflows

```bash
# Automatic triggers
git push origin main          # Runs full workflow
git push origin develop       # Runs full workflow
# Pull requests to main        # Runs full workflow + PR comment

# Manual trigger with benchmarks
# Go to Actions tab → RAG System Testing Workflow → Run workflow
# Check "Run performance benchmarks" for additional performance testing
```

### Workflow Outputs

Each workflow run produces:

1. **Test Results**: Individual test results for each matrix combination
2. **Performance Metrics**: Timing and resource usage data
3. **Aggregated Report**: Markdown report comparing all configurations
4. **JSON Data**: Machine-readable results for further analysis
5. **PR Comments**: Automatic result summaries on pull requests

### Artifacts

The workflow saves results as artifacts:
- `basic-rag-results-py{version}`: Basic RAG test results
- `enhanced-rag-results-{strategy}-{size}-py{version}`: Enhanced RAG matrix results
- `unified-workflow-results`: Complete comparison results
- `aggregated-results`: Final analysis and recommendations
- `benchmark-results`: Performance benchmark data (when enabled)

### Example Matrix Result

```json
{
  "strategy": "recursive",
  "chunk_size": 800,
  "chunks_created": 5,
  "search_results": 3,
  "system_stats": {
    "total_chunks": 5,
    "index_size_bytes": 125300,
    "strategy_distribution": {"recursive": 5}
  }
}
```

## Configuration

### Elasticsearch Settings

The Docker Compose setup uses:
- Single node cluster (development only)
- Security disabled for simplicity
- Port 9200 exposed locally
- Persistent data volume

### Embedding Models

Available models (trade-off between speed and quality):

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | Development/Testing |
| all-mpnet-base-v2 | 768 | Medium | Better | Production |
| multi-qa-mpnet-base-dot-v1 | 768 | Medium | Best for Q&A | Question Answering |

## Performance

### Benchmarks (on MacBook Pro M1)

- **Index Creation**: ~2 seconds
- **Document Addition**: ~100ms per document
- **Search Query**: ~50-200ms
- **Embedding Generation**: ~10ms per sentence

### Memory Usage

- Elasticsearch: ~1GB RAM
- Embedding Model: ~500MB RAM
- Python Process: ~200MB RAM

## Troubleshooting

### Elasticsearch Connection Issues

```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# Restart Elasticsearch
docker-compose restart elasticsearch

# Check logs
docker-compose logs elasticsearch
```

### Memory Issues

```bash
# Increase Docker memory limit to 4GB+
# Or use smaller embedding model:
embedding_service = LocalEmbeddingService("all-MiniLM-L6-v2")
```

### Slow Performance

- Use smaller embedding model for faster processing
- Reduce `num_candidates` in search queries
- Add more RAM to Docker

## Development

### Adding New Features

1. Extend `local_embeddings.py` for new embedding models
2. Modify `rag_test.py` for new search capabilities
3. Add tests in `test_rag_system.py`

### Testing

```bash
# Run specific test
pytest test_rag_system.py::TestLocalRAGSystem::test_search_documents -v

# Run with coverage
pip install pytest-cov
pytest test_rag_system.py --cov=. --cov-report=html
```

## Next Steps

- Add support for different document types (PDF, HTML)
- Add more sophisticated ranking algorithms
- Create web interface for interactive testing
- Add support for multiple languages

## License

MIT License - feel free to use and modify for your projects.
