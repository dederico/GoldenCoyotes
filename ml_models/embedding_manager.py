#!/usr/bin/env python3
"""
Embedding Manager
Centralized embedding management for the Business Dealer Intelligence System

This manager:
- Handles OpenAI embedding generation and caching
- Manages vector similarity searches and indexing
- Provides embedding-based content matching
- Supports multiple embedding models and dimensions
- Implements efficient batch processing and caching strategies

Following Task 7 from the PRP implementation blueprint.
"""

import os
import pickle
import json
import sqlite3
import hashlib
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Optional FAISS import for vector indexing
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config.intelligence_config import get_config
from config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    
    content_id: str
    content: str
    content_type: str
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    metadata: Dict[str, Any] = None


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    
    content_id: str
    content_type: str
    embedding_vector: List[float]
    embedding_model: str
    dimensions: int
    content_hash: str
    token_count: int
    generated_at: datetime
    cached: bool = False


@dataclass
class SimilarityResult:
    """Result of similarity search"""
    
    content_id: str
    content_type: str
    similarity_score: float
    embedding_vector: List[float]
    metadata: Dict[str, Any]
    distance: float


@dataclass
class EmbeddingStats:
    """Embedding generation statistics"""
    
    total_embeddings: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    total_tokens: int
    avg_generation_time: float
    last_updated: datetime


class EmbeddingManager:
    """
    Centralized embedding management system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Embedding Manager
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔢 Initializing Embedding Manager")
        
        # Initialize OpenAI client
        self._setup_openai_client()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize vector index
        self._setup_vector_index()
        
        # Performance metrics
        self.embeddings_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_tokens = 0
        self.total_generation_time = 0
        
        # Batch processing settings
        self.batch_size = self.ml_config.embedding_model.batch_size
        self.max_text_length = self.ml_config.embedding_model.max_text_length
        
        self.logger.info("✅ Embedding Manager initialized successfully")
    
    def _setup_openai_client(self):
        """Setup OpenAI client for embedding generation"""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.openai.api_key,
                timeout=self.config.openai.timeout,
                max_retries=self.config.openai.max_retries
            )
            
            # Test connection
            self.openai_client.models.list()
            self.logger.info("✅ OpenAI client initialized for embedding generation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for embedding caching"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for embedding caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for embedding caching: {e}")
            self.redis_enabled = False
    
    def _setup_database(self):
        """Setup database connection for embedding storage"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            
            # Ensure database exists
            if not os.path.exists(self.db_path):
                self.logger.warning(f"⚠️ Database not found at {self.db_path}")
            
            self.logger.info("✅ Database connection established for embedding storage")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_vector_index(self):
        """Setup vector index for similarity search"""
        self.vector_index = None
        self.index_to_content_id = {}
        self.content_id_to_index = {}
        
        if FAISS_AVAILABLE:
            try:
                # Initialize FAISS index
                embedding_dim = self.ml_config.embedding_model.embedding_dimension
                self.vector_index = faiss.IndexFlatL2(embedding_dim)
                self.logger.info("✅ FAISS vector index initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize FAISS index: {e}")
                self.vector_index = None
        else:
            self.logger.info("ℹ️ FAISS not available, using brute force similarity search")
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Generate embedding for content
        
        Args:
            request: Embedding request
            
        Returns:
            EmbeddingResult with embedding vector and metadata
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"🔢 Generating embedding for content {request.content_id}")
            
            # Generate content hash for caching
            content_hash = self._generate_content_hash(request.content)
            
            # Check cache first
            cached_result = await self._get_cached_embedding(request.content_id, content_hash)
            if cached_result:
                self.cache_hits += 1
                self.logger.info(f"🎯 Using cached embedding for content {request.content_id}")
                return cached_result
            
            self.cache_misses += 1
            
            # Preprocess content
            processed_content = self._preprocess_content(request.content, request.content_type)
            
            # Generate embedding using OpenAI
            embedding_response = self.openai_client.embeddings.create(
                model=request.model,
                input=processed_content,
                dimensions=request.dimensions
            )
            
            # Extract embedding vector
            embedding_vector = embedding_response.data[0].embedding
            token_count = embedding_response.usage.total_tokens
            
            # Create result
            result = EmbeddingResult(
                content_id=request.content_id,
                content_type=request.content_type,
                embedding_vector=embedding_vector,
                embedding_model=request.model,
                dimensions=len(embedding_vector),
                content_hash=content_hash,
                token_count=token_count,
                generated_at=datetime.now(),
                cached=False
            )
            
            # Cache the result
            await self._cache_embedding(request.content_id, result)
            
            # Store in database
            await self._store_embedding_in_db(result, request.metadata)
            
            # Update vector index
            await self._update_vector_index(result)
            
            # Update metrics
            self.embeddings_generated += 1
            self.total_tokens += token_count
            generation_time = (datetime.now() - start_time).total_seconds()
            self.total_generation_time += generation_time
            
            self.logger.info(f"✅ Embedding generated for content {request.content_id} in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating embedding: {e}")
            raise
    
    async def batch_generate_embeddings(
        self, 
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple content items in batch
        
        Args:
            requests: List of embedding requests
            
        Returns:
            List of EmbeddingResult objects
        """
        try:
            self.logger.info(f"🔢 Batch generating embeddings for {len(requests)} items")
            
            results = []
            
            # Process requests in batches
            for i in range(0, len(requests), self.batch_size):
                batch_requests = requests[i:i + self.batch_size]
                
                # Check cache for batch items
                batch_results = []
                uncached_requests = []
                
                for request in batch_requests:
                    content_hash = self._generate_content_hash(request.content)
                    cached_result = await self._get_cached_embedding(request.content_id, content_hash)
                    
                    if cached_result:
                        batch_results.append(cached_result)
                        self.cache_hits += 1
                    else:
                        uncached_requests.append(request)
                        self.cache_misses += 1
                
                # Generate embeddings for uncached items
                if uncached_requests:
                    # Prepare content for batch processing
                    batch_content = [
                        self._preprocess_content(req.content, req.content_type)
                        for req in uncached_requests
                    ]
                    
                    # Generate embeddings in batch
                    embedding_response = self.openai_client.embeddings.create(
                        model=uncached_requests[0].model,
                        input=batch_content,
                        dimensions=uncached_requests[0].dimensions
                    )
                    
                    # Process results
                    for j, request in enumerate(uncached_requests):
                        embedding_vector = embedding_response.data[j].embedding
                        token_count = embedding_response.usage.total_tokens // len(uncached_requests)
                        
                        result = EmbeddingResult(
                            content_id=request.content_id,
                            content_type=request.content_type,
                            embedding_vector=embedding_vector,
                            embedding_model=request.model,
                            dimensions=len(embedding_vector),
                            content_hash=self._generate_content_hash(request.content),
                            token_count=token_count,
                            generated_at=datetime.now(),
                            cached=False
                        )
                        
                        batch_results.append(result)
                        
                        # Cache result
                        await self._cache_embedding(request.content_id, result)
                        
                        # Store in database
                        await self._store_embedding_in_db(result, request.metadata)
                        
                        # Update vector index
                        await self._update_vector_index(result)
                        
                        # Update metrics
                        self.embeddings_generated += 1
                        self.total_tokens += token_count
                
                results.extend(batch_results)
                
                self.logger.info(f"✅ Processed batch {i//self.batch_size + 1}/{(len(requests) + self.batch_size - 1)//self.batch_size}")
            
            self.logger.info(f"✅ Batch embedding generation completed for {len(results)} items")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch embedding generation: {e}")
            raise
    
    async def find_similar_content(
        self,
        query_embedding: List[float],
        content_type: str = None,
        similarity_threshold: float = None,
        max_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find similar content using embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            content_type: Optional content type filter
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similar content with similarity scores
        """
        try:
            self.logger.info(f"🔍 Finding similar content with {len(query_embedding)} dim embedding")
            
            # Set default similarity threshold
            if similarity_threshold is None:
                similarity_threshold = self.ml_config.embedding_model.similarity_threshold
            
            # Use vector index if available
            if self.vector_index and self.vector_index.ntotal > 0:
                return await self._search_vector_index(
                    query_embedding, content_type, similarity_threshold, max_results
                )
            else:
                return await self._search_brute_force(
                    query_embedding, content_type, similarity_threshold, max_results
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error finding similar content: {e}")
            return []
    
    async def find_similar_by_content(
        self,
        content: str,
        content_type: str,
        similarity_threshold: float = None,
        max_results: int = 10
    ) -> List[SimilarityResult]:
        """
        Find similar content by generating embedding for query content
        
        Args:
            content: Query content
            content_type: Type of content
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similar content with similarity scores
        """
        try:
            # Generate embedding for query content
            query_request = EmbeddingRequest(
                content_id=f"query_{hashlib.md5(content.encode()).hexdigest()[:8]}",
                content=content,
                content_type=content_type,
                model=self.config.openai.embedding_model,
                dimensions=self.config.openai.embedding_dimensions
            )
            
            query_result = await self.generate_embedding(query_request)
            
            # Find similar content
            return await self.find_similar_content(
                query_result.embedding_vector,
                content_type,
                similarity_threshold,
                max_results
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar content by content: {e}")
            return []
    
    async def get_embedding_by_id(self, content_id: str) -> Optional[EmbeddingResult]:
        """
        Get embedding by content ID
        
        Args:
            content_id: ID of the content
            
        Returns:
            EmbeddingResult if found, None otherwise
        """
        try:
            # Check cache first
            cached_result = await self._get_cached_embedding_by_id(content_id)
            if cached_result:
                return cached_result
            
            # Check database
            return await self._get_embedding_from_db(content_id)
            
        except Exception as e:
            self.logger.error(f"❌ Error getting embedding by ID: {e}")
            return None
    
    async def delete_embedding(self, content_id: str) -> bool:
        """
        Delete embedding by content ID
        
        Args:
            content_id: ID of the content
            
        Returns:
            True if deleted successfully
        """
        try:
            self.logger.info(f"🗑️ Deleting embedding for content {content_id}")
            
            # Remove from cache
            await self._remove_from_cache(content_id)
            
            # Remove from database
            await self._remove_from_db(content_id)
            
            # Remove from vector index
            await self._remove_from_vector_index(content_id)
            
            self.logger.info(f"✅ Embedding deleted for content {content_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting embedding: {e}")
            return False
    
    async def rebuild_vector_index(self) -> bool:
        """
        Rebuild vector index from database
        
        Returns:
            True if rebuilt successfully
        """
        try:
            self.logger.info("🔄 Rebuilding vector index")
            
            if not FAISS_AVAILABLE:
                self.logger.warning("⚠️ FAISS not available, cannot rebuild vector index")
                return False
            
            # Clear existing index
            embedding_dim = self.ml_config.embedding_model.embedding_dimension
            self.vector_index = faiss.IndexFlatL2(embedding_dim)
            self.index_to_content_id = {}
            self.content_id_to_index = {}
            
            # Load all embeddings from database
            embeddings = await self._load_all_embeddings_from_db()
            
            if not embeddings:
                self.logger.info("ℹ️ No embeddings found in database")
                return True
            
            # Add embeddings to index
            vectors = []
            for embedding in embeddings:
                vectors.append(embedding['embedding_vector'])
                
                # Map index to content ID
                index = len(vectors) - 1
                self.index_to_content_id[index] = embedding['content_id']
                self.content_id_to_index[embedding['content_id']] = index
            
            # Add vectors to FAISS index
            vector_matrix = np.array(vectors).astype('float32')
            self.vector_index.add(vector_matrix)
            
            self.logger.info(f"✅ Vector index rebuilt with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error rebuilding vector index: {e}")
            return False
    
    def _preprocess_content(self, content: str, content_type: str) -> str:
        """Preprocess content for embedding generation"""
        # Basic preprocessing
        processed_content = content.strip()
        
        # Truncate if too long
        if len(processed_content) > self.max_text_length:
            processed_content = processed_content[:self.max_text_length]
            self.logger.warning(f"⚠️ Content truncated to {self.max_text_length} characters")
        
        # Content type specific preprocessing
        if content_type == "opportunity":
            processed_content = f"Business Opportunity: {processed_content}"
        elif content_type == "user_profile":
            processed_content = f"User Profile: {processed_content}"
        elif content_type == "content":
            processed_content = f"Content: {processed_content}"
        
        return processed_content
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _search_vector_index(
        self,
        query_embedding: List[float],
        content_type: str,
        similarity_threshold: float,
        max_results: int
    ) -> List[SimilarityResult]:
        """Search using FAISS vector index"""
        try:
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.vector_index.search(query_vector, max_results * 2)
            
            results = []
            for distance, index in zip(distances[0], indices[0]):
                if index == -1:  # No more results
                    break
                
                content_id = self.index_to_content_id.get(index)
                if not content_id:
                    continue
                
                # Convert L2 distance to cosine similarity
                similarity_score = 1 / (1 + distance)
                
                if similarity_score >= similarity_threshold:
                    # Get embedding details from database
                    embedding_data = await self._get_embedding_from_db(content_id)
                    
                    if embedding_data and (not content_type or embedding_data.content_type == content_type):
                        result = SimilarityResult(
                            content_id=content_id,
                            content_type=embedding_data.content_type,
                            similarity_score=similarity_score,
                            embedding_vector=embedding_data.embedding_vector,
                            metadata={},
                            distance=distance
                        )
                        results.append(result)
            
            # Sort by similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"❌ Error searching vector index: {e}")
            return []
    
    async def _search_brute_force(
        self,
        query_embedding: List[float],
        content_type: str,
        similarity_threshold: float,
        max_results: int
    ) -> List[SimilarityResult]:
        """Search using brute force similarity calculation"""
        try:
            # Load all embeddings from database
            embeddings = await self._load_all_embeddings_from_db(content_type)
            
            if not embeddings:
                return []
            
            # Calculate similarities
            query_vector = np.array(query_embedding).reshape(1, -1)
            similarities = []
            
            for embedding in embeddings:
                embedding_vector = np.array(embedding['embedding_vector']).reshape(1, -1)
                similarity = cosine_similarity(query_vector, embedding_vector)[0][0]
                
                if similarity >= similarity_threshold:
                    result = SimilarityResult(
                        content_id=embedding['content_id'],
                        content_type=embedding['content_type'],
                        similarity_score=similarity,
                        embedding_vector=embedding['embedding_vector'],
                        metadata=embedding.get('metadata', {}),
                        distance=1 - similarity
                    )
                    similarities.append(result)
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            self.logger.error(f"❌ Error in brute force search: {e}")
            return []
    
    async def _update_vector_index(self, result: EmbeddingResult):
        """Update vector index with new embedding"""
        try:
            if not self.vector_index:
                return
            
            # Add vector to index
            vector = np.array([result.embedding_vector]).astype('float32')
            self.vector_index.add(vector)
            
            # Update mappings
            index = self.vector_index.ntotal - 1
            self.index_to_content_id[index] = result.content_id
            self.content_id_to_index[result.content_id] = index
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating vector index: {e}")
    
    async def _remove_from_vector_index(self, content_id: str):
        """Remove embedding from vector index"""
        try:
            if not self.vector_index:
                return
            
            # Note: FAISS doesn't support direct removal, would need to rebuild
            # For now, just remove from mappings
            if content_id in self.content_id_to_index:
                index = self.content_id_to_index[content_id]
                del self.index_to_content_id[index]
                del self.content_id_to_index[content_id]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error removing from vector index: {e}")
    
    # Cache management methods
    async def _get_cached_embedding(self, content_id: str, content_hash: str) -> Optional[EmbeddingResult]:
        """Get cached embedding"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"embedding:{content_id}:{content_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return EmbeddingResult(
                    content_id=data['content_id'],
                    content_type=data['content_type'],
                    embedding_vector=data['embedding_vector'],
                    embedding_model=data['embedding_model'],
                    dimensions=data['dimensions'],
                    content_hash=data['content_hash'],
                    token_count=data['token_count'],
                    generated_at=datetime.fromisoformat(data['generated_at']),
                    cached=True
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _get_cached_embedding_by_id(self, content_id: str) -> Optional[EmbeddingResult]:
        """Get cached embedding by content ID"""
        if not self.redis_enabled:
            return None
        
        try:
            # Try to find cached embedding with any hash
            pattern = f"embedding:{content_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                cached_data = self.redis_client.get(keys[0])
                if cached_data:
                    data = json.loads(cached_data)
                    return EmbeddingResult(
                        content_id=data['content_id'],
                        content_type=data['content_type'],
                        embedding_vector=data['embedding_vector'],
                        embedding_model=data['embedding_model'],
                        dimensions=data['dimensions'],
                        content_hash=data['content_hash'],
                        token_count=data['token_count'],
                        generated_at=datetime.fromisoformat(data['generated_at']),
                        cached=True
                    )
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get by ID error: {e}")
        
        return None
    
    async def _cache_embedding(self, content_id: str, result: EmbeddingResult):
        """Cache embedding result"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"embedding:{content_id}:{result.content_hash}"
            ttl = self.config.redis.embedding_cache_ttl
            
            data = {
                'content_id': result.content_id,
                'content_type': result.content_type,
                'embedding_vector': result.embedding_vector,
                'embedding_model': result.embedding_model,
                'dimensions': result.dimensions,
                'content_hash': result.content_hash,
                'token_count': result.token_count,
                'generated_at': result.generated_at.isoformat()
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    async def _remove_from_cache(self, content_id: str):
        """Remove embedding from cache"""
        if not self.redis_enabled:
            return
        
        try:
            pattern = f"embedding:{content_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache remove error: {e}")
    
    # Database methods
    async def _store_embedding_in_db(self, result: EmbeddingResult, metadata: Dict[str, Any] = None):
        """Store embedding in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or update embedding
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_cache 
                (id, content_id, content_type, embedding_model, embedding_vector, 
                 content_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"emb_{result.content_id}_{result.content_hash[:8]}",
                result.content_id,
                result.content_type,
                result.embedding_model,
                json.dumps(result.embedding_vector),
                result.content_hash,
                result.generated_at.isoformat(),
                (result.generated_at + timedelta(seconds=self.config.redis.embedding_cache_ttl)).isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing embedding in database: {e}")
    
    async def _get_embedding_from_db(self, content_id: str) -> Optional[EmbeddingResult]:
        """Get embedding from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT content_id, content_type, embedding_model, embedding_vector, 
                       content_hash, created_at
                FROM embedding_cache
                WHERE content_id = ? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY created_at DESC
                LIMIT 1
            """, (content_id, datetime.now().isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return EmbeddingResult(
                    content_id=result[0],
                    content_type=result[1],
                    embedding_vector=json.loads(result[3]),
                    embedding_model=result[2],
                    dimensions=len(json.loads(result[3])),
                    content_hash=result[4],
                    token_count=0,  # Not stored in database
                    generated_at=datetime.fromisoformat(result[5]),
                    cached=False
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting embedding from database: {e}")
            return None
    
    async def _load_all_embeddings_from_db(self, content_type: str = None) -> List[Dict[str, Any]]:
        """Load all embeddings from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if content_type:
                cursor.execute("""
                    SELECT content_id, content_type, embedding_model, embedding_vector, 
                           content_hash, created_at
                    FROM embedding_cache
                    WHERE content_type = ? AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY created_at DESC
                """, (content_type, datetime.now().isoformat()))
            else:
                cursor.execute("""
                    SELECT content_id, content_type, embedding_model, embedding_vector, 
                           content_hash, created_at
                    FROM embedding_cache
                    WHERE expires_at IS NULL OR expires_at > ?
                    ORDER BY created_at DESC
                """, (datetime.now().isoformat(),))
            
            results = cursor.fetchall()
            conn.close()
            
            embeddings = []
            for result in results:
                embeddings.append({
                    'content_id': result[0],
                    'content_type': result[1],
                    'embedding_model': result[2],
                    'embedding_vector': json.loads(result[3]),
                    'content_hash': result[4],
                    'created_at': result[5]
                })
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ Error loading embeddings from database: {e}")
            return []
    
    async def _remove_from_db(self, content_id: str):
        """Remove embedding from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM embedding_cache WHERE content_id = ?", (content_id,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error removing embedding from database: {e}")
    
    # Management and utility methods
    def get_embedding_stats(self) -> EmbeddingStats:
        """Get embedding generation statistics"""
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_requests, 1)
        avg_generation_time = self.total_generation_time / max(self.embeddings_generated, 1)
        
        return EmbeddingStats(
            total_embeddings=self.embeddings_generated,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            cache_hit_rate=cache_hit_rate,
            total_tokens=self.total_tokens,
            avg_generation_time=avg_generation_time,
            last_updated=datetime.now()
        )
    
    def clear_embedding_cache(self, content_type: str = None) -> int:
        """Clear embedding cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if content_type:
                pattern = f"embedding:*:{content_type}:*"
            else:
                pattern = "embedding:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing embedding cache: {e}")
            return 0
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get embedding manager status"""
        stats = self.get_embedding_stats()
        
        return {
            "status": "operational",
            "embeddings_generated": stats.total_embeddings,
            "cache_performance": {
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "cache_hit_rate": stats.cache_hit_rate
            },
            "performance": {
                "total_tokens": stats.total_tokens,
                "avg_generation_time": stats.avg_generation_time
            },
            "configuration": {
                "embedding_model": self.config.openai.embedding_model,
                "embedding_dimensions": self.config.openai.embedding_dimensions,
                "batch_size": self.batch_size,
                "max_text_length": self.max_text_length,
                "redis_enabled": self.redis_enabled,
                "faiss_available": FAISS_AVAILABLE,
                "vector_index_size": self.vector_index.ntotal if self.vector_index else 0
            },
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the embedding manager
    async def test_embedding_manager():
        print("🔢 Testing Embedding Manager")
        print("=" * 50)
        
        try:
            manager = EmbeddingManager()
            
            # Test single embedding generation
            print("Generating single embedding...")
            request = EmbeddingRequest(
                content_id="test_content_1",
                content="This is a test content for embedding generation",
                content_type="text"
            )
            
            result = await manager.generate_embedding(request)
            print(f"Single Embedding Result: {result.content_id}, dims: {result.dimensions}")
            
            # Test batch embedding generation
            print("Generating batch embeddings...")
            batch_requests = [
                EmbeddingRequest(
                    content_id=f"test_content_{i}",
                    content=f"This is test content number {i}",
                    content_type="text"
                )
                for i in range(2, 5)
            ]
            
            batch_results = await manager.batch_generate_embeddings(batch_requests)
            print(f"Batch Results: {len(batch_results)} embeddings")
            
            # Test similarity search
            print("Testing similarity search...")
            similar_content = await manager.find_similar_content(
                result.embedding_vector,
                content_type="text",
                max_results=3
            )
            print(f"Similar Content: {len(similar_content)} matches")
            
            # Test similarity search by content
            print("Testing similarity search by content...")
            similar_by_content = await manager.find_similar_by_content(
                content="This is a test content",
                content_type="text",
                max_results=3
            )
            print(f"Similar by Content: {len(similar_by_content)} matches")
            
            # Test get embedding by ID
            print("Testing get embedding by ID...")
            embedding_by_id = await manager.get_embedding_by_id("test_content_1")
            print(f"Embedding by ID: {embedding_by_id.content_id if embedding_by_id else 'Not found'}")
            
            # Test rebuild vector index
            print("Testing rebuild vector index...")
            rebuild_success = await manager.rebuild_vector_index()
            print(f"Rebuild Vector Index: {'Success' if rebuild_success else 'Failed'}")
            
            # Test statistics
            stats = manager.get_embedding_stats()
            print(f"Statistics: {stats.total_embeddings} embeddings, {stats.cache_hit_rate:.2f} cache hit rate")
            
            # Test status
            status = manager.get_manager_status()
            print(f"Manager Status: {status['status']}")
            
            print("✅ Embedding Manager test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_embedding_manager())