#!/usr/bin/env python3
"""
Content Processor
OpenAI embeddings for multi-modal content analysis in the Business Dealer Intelligence System

This processor:
- Generates embeddings for text content using OpenAI's embedding models
- Performs content similarity analysis
- Extracts key insights from content using LLM analysis
- Manages embedding caching for performance optimization
- Supports multi-modal content analysis (text, images, structured data)

Following Task 4 from the PRP implementation blueprint.
"""

import os
import json
import sqlite3
import hashlib
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from .data_models import EmbeddingCache, Opportunity, UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    
    content_id: str
    content_type: str
    key_topics: List[str]
    sentiment_score: float
    complexity_score: float
    quality_score: float
    industry_tags: List[str]
    insights: List[str]
    processed_at: datetime


@dataclass
class SimilarityResult:
    """Content similarity result"""
    
    content_id_1: str
    content_id_2: str
    similarity_score: float
    matching_topics: List[str]
    analysis_method: str


class ContentProcessor:
    """
    Content processing engine that handles embeddings and multi-modal analysis
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Content Processor
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📄 Initializing Content Processor")
        
        # Initialize OpenAI client
        self._setup_openai_client()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize database connection
        self._setup_database()
        
        # Content processing metrics
        self.embeddings_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.analysis_requests = 0
        
        self.logger.info("✅ Content Processor initialized successfully")
    
    def _setup_openai_client(self):
        """Setup OpenAI client for embeddings and analysis"""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.openai.api_key,
                timeout=self.config.openai.timeout,
                max_retries=self.config.openai.max_retries
            )
            
            # Test connection
            self.openai_client.models.list()
            self.logger.info("✅ OpenAI client initialized for content processing")
            
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
            
            self.logger.info("✅ Database connection established for content processing")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    async def generate_embedding(
        self, 
        content: str, 
        content_type: str = "text",
        content_id: str = None
    ) -> Dict[str, Any]:
        """
        Generate embedding for content
        
        Args:
            content: Content to embed
            content_type: Type of content
            content_id: Optional content ID for caching
            
        Returns:
            Dictionary containing embedding and metadata
        """
        try:
            self.logger.info(f"📊 Generating embedding for {content_type} content")
            
            # Generate content hash for caching
            content_hash = self._generate_content_hash(content)
            
            # Check cache first
            if content_id:
                cached_embedding = await self._get_cached_embedding(content_id, content_hash)
                if cached_embedding:
                    self.cache_hits += 1
                    self.logger.info(f"🎯 Using cached embedding for content {content_id}")
                    return cached_embedding
            
            self.cache_misses += 1
            
            # Prepare content for embedding
            processed_content = self._preprocess_content(content, content_type)
            
            # Generate embedding using OpenAI
            embedding_response = self.openai_client.embeddings.create(
                model=self.config.openai.embedding_model,
                input=processed_content,
                dimensions=self.config.openai.embedding_dimensions
            )
            
            # Extract embedding vector
            embedding_vector = embedding_response.data[0].embedding
            
            # Create embedding result
            embedding_result = {
                "content_id": content_id or f"content_{content_hash[:8]}",
                "content_type": content_type,
                "embedding_vector": embedding_vector,
                "embedding_model": self.config.openai.embedding_model,
                "content_hash": content_hash,
                "dimensions": len(embedding_vector),
                "generated_at": datetime.now().isoformat(),
                "token_count": embedding_response.usage.total_tokens if hasattr(embedding_response, 'usage') else 0
            }
            
            # Cache the embedding
            if content_id:
                await self._cache_embedding(content_id, embedding_result)
            
            # Store in database
            await self._store_embedding_in_db(embedding_result)
            
            self.embeddings_generated += 1
            self.logger.info(f"✅ Embedding generated successfully for {content_type} content")
            
            return embedding_result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating embedding: {e}")
            return {"error": str(e), "content_type": content_type}
    
    async def analyze_content(
        self, 
        content: str, 
        content_type: str = "text",
        content_id: str = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis
        
        Args:
            content: Content to analyze
            content_type: Type of content
            content_id: Optional content ID
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"🔍 Analyzing {content_type} content")
            self.analysis_requests += 1
            
            # Generate embedding first
            embedding_result = await self.generate_embedding(content, content_type, content_id)
            
            if "error" in embedding_result:
                return embedding_result
            
            # Perform LLM-based analysis
            analysis_result = await self._perform_llm_analysis(content, content_type)
            
            # Combine results
            comprehensive_analysis = {
                "content_id": content_id or f"content_{embedding_result['content_hash'][:8]}",
                "content_type": content_type,
                "embedding_data": embedding_result,
                "analysis": analysis_result,
                "processed_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Content analysis completed")
            return comprehensive_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing content: {e}")
            return {"error": str(e), "content_type": content_type}
    
    async def find_similar_content(
        self, 
        query_content: str, 
        content_type: str = "text",
        similarity_threshold: float = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar content using embedding similarity
        
        Args:
            query_content: Content to find similarities for
            content_type: Type of content
            similarity_threshold: Minimum similarity score
            max_results: Maximum number of results
            
        Returns:
            List of similar content with similarity scores
        """
        try:
            self.logger.info(f"🔍 Finding similar content for {content_type}")
            
            # Generate embedding for query content
            query_embedding = await self.generate_embedding(query_content, content_type)
            
            if "error" in query_embedding:
                return []
            
            # Get similarity threshold
            if similarity_threshold is None:
                similarity_threshold = self.ml_config.embedding_model.similarity_threshold
            
            # Find similar embeddings in database
            similar_embeddings = await self._find_similar_embeddings(
                query_embedding["embedding_vector"],
                content_type,
                similarity_threshold,
                max_results
            )
            
            # Format results
            results = []
            for embedding_data, similarity_score in similar_embeddings:
                results.append({
                    "content_id": embedding_data["content_id"],
                    "content_type": embedding_data["content_type"],
                    "similarity_score": similarity_score,
                    "embedding_model": embedding_data["embedding_model"],
                    "matched_at": datetime.now().isoformat()
                })
            
            self.logger.info(f"✅ Found {len(results)} similar content items")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar content: {e}")
            return []
    
    async def batch_process_content(
        self, 
        content_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple content items in batch
        
        Args:
            content_items: List of content items to process
            
        Returns:
            List of processing results
        """
        try:
            self.logger.info(f"📦 Batch processing {len(content_items)} content items")
            
            # Process items in batches for efficiency
            batch_size = self.config.openai.embedding_batch_size
            results = []
            
            for i in range(0, len(content_items), batch_size):
                batch = content_items[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = []
                for item in batch:
                    task = self.analyze_content(
                        content=item.get("content", ""),
                        content_type=item.get("content_type", "text"),
                        content_id=item.get("content_id")
                    )
                    batch_tasks.append(task)
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results and exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Batch processing error: {result}")
                        results.append({"error": str(result)})
                    else:
                        results.append(result)
                
                self.logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(content_items) + batch_size - 1)//batch_size}")
            
            self.logger.info(f"✅ Batch processing completed for {len(content_items)} items")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch processing: {e}")
            return [{"error": str(e)}]
    
    async def _perform_llm_analysis(self, content: str, content_type: str) -> Dict[str, Any]:
        """Perform LLM-based content analysis"""
        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(content, content_type)
            
            # Get analysis from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.config.openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert content analyst for business intelligence."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=self.config.openai.temperature,
                max_tokens=self.config.openai.max_tokens
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            
            # Extract structured analysis
            analysis_result = self._parse_analysis_response(analysis_text)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in LLM analysis: {e}")
            return {
                "key_topics": [],
                "sentiment_score": 0.0,
                "complexity_score": 0.0,
                "quality_score": 0.0,
                "industry_tags": [],
                "insights": [f"Analysis failed: {str(e)}"]
            }
    
    def _create_analysis_prompt(self, content: str, content_type: str) -> str:
        """Create prompt for content analysis"""
        return f"""
        Analyze the following {content_type} content and provide a structured analysis:

        Content:
        {content}

        Please provide your analysis in the following JSON format:
        {{
            "key_topics": ["topic1", "topic2", "topic3"],
            "sentiment_score": 0.0,  // -1.0 to 1.0
            "complexity_score": 0.0,  // 0.0 to 1.0
            "quality_score": 0.0,     // 0.0 to 1.0
            "industry_tags": ["industry1", "industry2"],
            "insights": ["insight1", "insight2", "insight3"]
        }}

        Focus on:
        - Key topics and themes
        - Sentiment analysis (-1.0 negative to 1.0 positive)
        - Content complexity (0.0 simple to 1.0 complex)
        - Content quality (0.0 poor to 1.0 excellent)
        - Relevant industry tags
        - Actionable insights for business intelligence
        """
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try to extract JSON from response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                return json.loads(json_str)
            
            # Fallback parsing if JSON extraction fails
            return {
                "key_topics": [],
                "sentiment_score": 0.0,
                "complexity_score": 0.0,
                "quality_score": 0.0,
                "industry_tags": [],
                "insights": ["Could not parse analysis response"]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error parsing analysis response: {e}")
            return {
                "key_topics": [],
                "sentiment_score": 0.0,
                "complexity_score": 0.0,
                "quality_score": 0.0,
                "industry_tags": [],
                "insights": ["Analysis parsing failed"]
            }
    
    def _preprocess_content(self, content: str, content_type: str) -> str:
        """Preprocess content for embedding generation"""
        # Basic preprocessing
        processed_content = content.strip()
        
        # Truncate if too long
        max_length = self.config.openai.max_tokens * 4  # Approximate token limit
        if len(processed_content) > max_length:
            processed_content = processed_content[:max_length]
            self.logger.warning(f"⚠️ Content truncated to {max_length} characters")
        
        # Content type specific preprocessing
        if content_type == "opportunity":
            processed_content = f"Business Opportunity: {processed_content}"
        elif content_type == "user_profile":
            processed_content = f"User Profile: {processed_content}"
        elif content_type == "text":
            processed_content = f"Content: {processed_content}"
        
        return processed_content
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        return hashlib.md5(content.encode()).hexdigest()
    
    # Cache management methods
    async def _get_cached_embedding(self, content_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached embedding from Redis"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"embedding:{content_id}:{content_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_embedding(self, content_id: str, embedding_result: Dict[str, Any]):
        """Cache embedding in Redis"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"embedding:{content_id}:{embedding_result['content_hash']}"
            ttl = self.config.redis.embedding_cache_ttl
            
            # Store in cache
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(embedding_result, default=str)
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    async def _store_embedding_in_db(self, embedding_result: Dict[str, Any]):
        """Store embedding in database"""
        try:
            # Create EmbeddingCache model
            embedding_cache = EmbeddingCache(
                content_id=embedding_result["content_id"],
                content_type=embedding_result["content_type"],
                embedding_model=embedding_result["embedding_model"],
                embedding_vector=embedding_result["embedding_vector"],
                content_hash=embedding_result["content_hash"],
                expires_at=datetime.now() + timedelta(
                    seconds=self.config.redis.embedding_cache_ttl
                )
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_cache 
                (id, content_id, content_type, embedding_model, embedding_vector, content_hash, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_cache.id,
                embedding_cache.content_id,
                embedding_cache.content_type,
                embedding_cache.embedding_model,
                json.dumps(embedding_cache.embedding_vector),
                embedding_cache.content_hash,
                embedding_cache.created_at.isoformat(),
                embedding_cache.expires_at.isoformat() if embedding_cache.expires_at else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing embedding in database: {e}")
    
    async def _find_similar_embeddings(
        self, 
        query_vector: List[float], 
        content_type: str,
        similarity_threshold: float,
        max_results: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Find similar embeddings in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all embeddings of the same type
            cursor.execute("""
                SELECT content_id, content_type, embedding_model, embedding_vector, content_hash, created_at
                FROM embedding_cache
                WHERE content_type = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (content_type, datetime.now().isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            # Calculate similarities
            similarities = []
            query_vector_np = np.array(query_vector).reshape(1, -1)
            
            for row in results:
                try:
                    embedding_vector = json.loads(row[3])
                    embedding_vector_np = np.array(embedding_vector).reshape(1, -1)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_vector_np, embedding_vector_np)[0][0]
                    
                    if similarity >= similarity_threshold:
                        embedding_data = {
                            "content_id": row[0],
                            "content_type": row[1],
                            "embedding_model": row[2],
                            "embedding_vector": embedding_vector,
                            "content_hash": row[4],
                            "created_at": row[5]
                        }
                        similarities.append((embedding_data, similarity))
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing embedding: {e}")
                    continue
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar embeddings: {e}")
            return []
    
    # Performance and management methods
    def get_processor_status(self) -> Dict[str, Any]:
        """Get content processor status"""
        return {
            "status": "operational",
            "embeddings_generated": self.embeddings_generated,
            "analysis_requests": self.analysis_requests,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "configuration": {
                "embedding_model": self.config.openai.embedding_model,
                "embedding_dimensions": self.config.openai.embedding_dimensions,
                "chat_model": self.config.openai.chat_model,
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_embedding_cache(self, content_type: str = None) -> int:
        """Clear embedding cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if content_type:
                pattern = f"embedding:*:{content_type}*"
            else:
                pattern = "embedding:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing embedding cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the content processor
    async def test_content_processor():
        print("📄 Testing Content Processor")
        print("=" * 50)
        
        try:
            processor = ContentProcessor()
            
            # Test embedding generation
            sample_content = "This is a sample business opportunity in the technology sector"
            embedding_result = await processor.generate_embedding(
                content=sample_content,
                content_type="opportunity",
                content_id="test_opp_1"
            )
            print(f"Embedding Result: {embedding_result}")
            
            # Test content analysis
            analysis_result = await processor.analyze_content(
                content=sample_content,
                content_type="opportunity",
                content_id="test_opp_1"
            )
            print(f"Analysis Result: {analysis_result}")
            
            # Test similarity search
            similar_content = await processor.find_similar_content(
                query_content="Technology business opportunity",
                content_type="opportunity",
                max_results=5
            )
            print(f"Similar Content: {similar_content}")
            
            # Test status
            status = processor.get_processor_status()
            print(f"Processor Status: {status}")
            
            print("✅ Content Processor test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_content_processor())