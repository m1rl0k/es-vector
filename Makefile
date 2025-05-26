.PHONY: help setup start stop test clean demo install-deps check-deps

# Default target
help:
	@echo "Local RAG/Elasticsearch Testing Environment"
	@echo "==========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  setup        - Complete setup (install deps + start services)"
	@echo "  install-deps - Install Python dependencies"
	@echo "  start        - Start Elasticsearch container"
	@echo "  stop         - Stop Elasticsearch container"
	@echo "  restart      - Restart Elasticsearch container"
	@echo "  demo         - Run the main RAG demonstration"
	@echo "  demo-chunking- Run document chunking demonstration"
	@echo "  workflow     - Run unified workflow (basic + enhanced RAG)"
	@echo "  aggregate    - Aggregate matrix results (for CI/CD testing)"
	@echo "  test         - Run the test suite"
	@echo "  test-chunking- Run document chunking tests"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  check-deps   - Check if dependencies are installed"
	@echo "  status       - Check system status"
	@echo "  clean        - Clean up containers and data"
	@echo "  logs         - Show Elasticsearch logs"
	@echo ""

# Complete setup
setup: install-deps start
	@echo "✓ Setup complete! Run 'make demo' to test the system."

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	pip3 install -r requirements.txt
	@echo "✓ Dependencies installed"

# Check if dependencies are installed
check-deps:
	@echo "Checking dependencies..."
	@python3 -c "import elasticsearch, sentence_transformers, numpy, pandas; print('✓ All dependencies available')" || \
	(echo "✗ Missing dependencies. Run 'make install-deps'" && exit 1)

# Start Elasticsearch
start:
	@echo "Starting Elasticsearch..."
	docker-compose up -d
	@echo "Waiting for Elasticsearch to be ready..."
	@timeout 60 bash -c 'until curl -s http://localhost:9200 > /dev/null; do sleep 2; done' || \
	(echo "✗ Elasticsearch failed to start" && exit 1)
	@echo "✓ Elasticsearch is running"

# Stop Elasticsearch
stop:
	@echo "Stopping Elasticsearch..."
	docker-compose down
	@echo "✓ Elasticsearch stopped"

# Restart Elasticsearch
restart: stop start

# Run the main demonstration
demo: check-deps
	@echo "Running RAG demonstration..."
	python3 rag_test.py

# Run document chunking demonstration
demo-chunking: check-deps
	@echo "Running document chunking demonstration..."
	python3 demo_chunking.py

# Run unified workflow (basic + enhanced RAG)
workflow: check-deps
	@echo "Running unified workflow (basic + enhanced RAG)..."
	python3 unified_workflow.py

# Aggregate matrix results (for CI/CD testing)
aggregate: check-deps
	@echo "Aggregating matrix results..."
	python3 aggregate_results.py

# Run tests
test: check-deps
	@echo "Running test suite..."
	python3 test_rag_system.py

# Run document chunking tests
test-chunking: check-deps
	@echo "Running document chunking tests..."
	python3 test_document_chunking.py

# Run tests with verbose output
test-verbose: check-deps
	@echo "Running test suite (verbose)..."
	pytest test_rag_system.py -v

# Check system status
status:
	@echo "System Status:"
	@echo "=============="
	@echo -n "Elasticsearch: "
	@curl -s http://localhost:9200 > /dev/null && echo "✓ Running" || echo "✗ Not running"
	@echo -n "Docker: "
	@docker --version > /dev/null 2>&1 && echo "✓ Available" || echo "✗ Not available"
	@echo -n "Python: "
	@python3 --version 2>&1 | head -1
	@echo -n "Dependencies: "
	@python3 -c "import elasticsearch, sentence_transformers; print('✓ Available')" 2>/dev/null || echo "✗ Missing"

# Show Elasticsearch logs
logs:
	docker-compose logs -f elasticsearch

# Clean up everything
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✓ Cleanup complete"

# Development helpers
dev-setup: setup
	@echo "Installing development dependencies..."
	pip3 install pytest pytest-cov jupyter ipykernel
	@echo "✓ Development environment ready"

# Quick health check
health:
	@echo "Health Check:"
	@echo "============"
	@curl -s http://localhost:9200/_cluster/health | python3 -m json.tool || echo "Elasticsearch not responding"

# Performance test
perf-test: check-deps
	@echo "Running performance tests..."
	@python3 -c "\
import time; \
from rag_test import *; \
print('Testing embedding generation...'); \
start = time.time(); \
embedding_service.encode(['test sentence'] * 100); \
print(f'100 embeddings: {time.time() - start:.2f}s')"
