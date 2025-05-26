"""
Enhanced RAG system with document chunking capabilities
Integrates document chunking with the existing RAG pipeline
"""
import logging
from typing import List, Dict, Any, Optional
from document_chunker import DocumentChunker, ChunkingStrategy, DocumentChunk
from rag_test import es_client, embedding_service, create_index, search_documents
import json

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    """Enhanced RAG system with document chunking support"""

    def __init__(self,
                 index_name: str = "enhanced-rag-local",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE):
        """
        Initialize enhanced RAG system

        Args:
            index_name: Elasticsearch index name
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            chunking_strategy: Default chunking strategy
        """
        self.index_name = index_name
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.default_strategy = chunking_strategy

        # Create index if it doesn't exist
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Ensure the Elasticsearch index exists with proper mapping"""
        try:
            if not es_client.indices.exists(index=self.index_name):
                create_index(self.index_name)
                logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def add_document(self,
                    document_id: str,
                    text: str,
                    title: str = "",
                    metadata: Optional[Dict[str, Any]] = None,
                    chunking_strategy: Optional[ChunkingStrategy] = None) -> List[str]:
        """
        Add a document to the RAG system with automatic chunking

        Args:
            document_id: Unique identifier for the document
            text: Document text content
            title: Document title
            metadata: Additional metadata
            chunking_strategy: Strategy to use for chunking (uses default if None)

        Returns:
            List of chunk IDs that were created
        """
        if metadata is None:
            metadata = {}

        strategy = chunking_strategy or self.default_strategy

        # Chunk the document
        chunks = self.chunker.chunk_document(text, document_id, strategy)

        if not chunks:
            logger.warning(f"No chunks created for document {document_id}")
            return []

        # Get chunking statistics
        stats = self.chunker.get_chunk_statistics(chunks)
        logger.info(f"Document {document_id}: {stats['total_chunks']} chunks, "
                   f"avg size: {stats['avg_chunk_size']:.0f} chars")

        # Index each chunk
        chunk_ids = []
        for chunk in chunks:
            try:
                # Prepare document for indexing
                doc_metadata = {
                    **metadata,
                    "document_id": document_id,
                    "document_title": title,
                    "chunk_metadata": chunk.metadata.__dict__
                }

                # Generate embedding for the chunk
                embedding = embedding_service.encode(chunk.text)

                # Prepare document for Elasticsearch
                es_document = {
                    "text": chunk.text,
                    "title": title,
                    "text_embedding": embedding.tolist(),
                    "metadata": doc_metadata
                }

                # Index the chunk
                response = es_client.index(
                    index=self.index_name,
                    id=chunk.metadata.chunk_id,
                    body=es_document
                )

                chunk_ids.append(chunk.metadata.chunk_id)
                logger.debug(f"Indexed chunk {chunk.metadata.chunk_id}")

            except Exception as e:
                logger.error(f"Error indexing chunk {chunk.metadata.chunk_id}: {e}")
                continue

        # Refresh index
        es_client.indices.refresh(index=self.index_name)

        logger.info(f"Successfully indexed {len(chunk_ids)} chunks for document {document_id}")
        return chunk_ids

    def search(self,
               query: str,
               k: int = 5,
               include_chunk_context: bool = True,
               min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks

        Args:
            query: Search query
            k: Number of results to return
            include_chunk_context: Whether to include neighboring chunks
            min_score: Minimum relevance score threshold

        Returns:
            List of search results with enhanced metadata
        """
        # Perform search
        response = search_documents(self.index_name, query, k)

        results = []
        for hit in response['hits']['hits']:
            if hit['_score'] < min_score:
                continue

            source = hit['_source']
            chunk_metadata = source['metadata']['chunk_metadata']

            result = {
                "chunk_id": hit['_id'],
                "score": hit['_score'],
                "text": source['text'],
                "title": source['title'],
                "document_id": source['metadata']['document_id'],
                "chunk_index": chunk_metadata['chunk_index'],
                "total_chunks": chunk_metadata['total_chunks'],
                "chunk_size": chunk_metadata['char_count'],
                "strategy": chunk_metadata['strategy'],
                "metadata": source['metadata']
            }

            # Add context from neighboring chunks if requested
            if include_chunk_context:
                context = self._get_chunk_context(
                    source['metadata']['document_id'],
                    chunk_metadata['chunk_index'],
                    chunk_metadata['total_chunks']
                )
                result['context'] = context

            results.append(result)

        return results

    def _get_chunk_context(self,
                          document_id: str,
                          chunk_index: int,
                          total_chunks: int,
                          context_window: int = 1) -> Dict[str, Any]:
        """Get context from neighboring chunks"""
        context = {
            "previous_chunks": [],
            "next_chunks": []
        }

        # Get previous chunks
        for i in range(max(0, chunk_index - context_window), chunk_index):
            chunk_id = f"{document_id}_chunk_{i}"
            try:
                chunk_doc = es_client.get(index=self.index_name, id=chunk_id)
                context["previous_chunks"].append({
                    "chunk_id": chunk_id,
                    "text": chunk_doc['_source']['text'][:200] + "..."  # Truncate for brevity
                })
            except:
                continue

        # Get next chunks
        for i in range(chunk_index + 1, min(total_chunks, chunk_index + context_window + 1)):
            chunk_id = f"{document_id}_chunk_{i}"
            try:
                chunk_doc = es_client.get(index=self.index_name, id=chunk_id)
                context["next_chunks"].append({
                    "chunk_id": chunk_id,
                    "text": chunk_doc['_source']['text'][:200] + "..."  # Truncate for brevity
                })
            except:
                continue

        return context

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        query = {
            "query": {
                "term": {
                    "metadata.document_id": document_id
                }
            },
            "size": 1000  # Adjust based on expected max chunks per document
        }

        response = es_client.search(index=self.index_name, body=query)

        chunks = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            chunk_metadata = source['metadata']['chunk_metadata']

            chunks.append({
                "chunk_id": hit['_id'],
                "text": source['text'],
                "chunk_index": chunk_metadata['chunk_index'],
                "chunk_size": chunk_metadata['char_count'],
                "strategy": chunk_metadata['strategy']
            })

        # Sort by chunk index
        chunks.sort(key=lambda x: x['chunk_index'])

        return chunks

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document"""
        query = {
            "query": {
                "term": {
                    "metadata.document_id": document_id
                }
            }
        }

        response = es_client.delete_by_query(index=self.index_name, body=query)
        deleted_count = response.get('deleted', 0)

        # Refresh index to make deletion visible immediately
        es_client.indices.refresh(index=self.index_name)

        logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
        return deleted_count

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        # Get index statistics
        stats_response = es_client.indices.stats(index=self.index_name)
        index_stats = stats_response['indices'][self.index_name]['total']

        # Get document count by strategy
        strategy_agg = {
            "aggs": {
                "strategies": {
                    "terms": {
                        "field": "metadata.chunk_metadata.strategy.keyword"
                    }
                }
            },
            "size": 0
        }

        agg_response = es_client.search(index=self.index_name, body=strategy_agg)
        strategy_counts = {}
        for bucket in agg_response['aggregations']['strategies']['buckets']:
            strategy_counts[bucket['key']] = bucket['doc_count']

        return {
            "total_chunks": index_stats['docs']['count'],
            "index_size_bytes": index_stats['store']['size_in_bytes'],
            "strategy_distribution": strategy_counts
        }
