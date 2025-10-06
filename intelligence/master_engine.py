#!/usr/bin/env python3
"""
Master Intelligence Engine
Central coordination hub for all intelligence operations in the Business Dealer Intelligence System

This engine coordinates:
- Behavior analysis and pattern recognition
- Opportunity matching and scoring
- Recommendation generation
- Content processing and analysis
- Caching and performance optimization

Following patterns from metalmex_orchestrator/orchestrators/master_orchestrator.py
"""

import os
import json
import redis
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from openai import OpenAI
from dataclasses import dataclass, field

from config.intelligence_config import get_config
from database.intelligence_schema import IntelligenceSchema
from intelligence.data_models2 import IntelligenceRequest, IntelligenceResponse

# Import engine components (these will be created in subsequent tasks)
# from .behavior_analyzer import BehaviorAnalyzer
# from .opportunity_matcher import OpportunityMatcher
# from .recommendation_engine import RecommendationEngine
# from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


@dataclass
class EngineStatus:
    """Status of intelligence engine components"""

    behavior_analyzer: bool = field(default=False)
    opportunity_matcher: bool = field(default=False)
    recommendation_engine: bool = field(default=False)
    content_processor: bool = field(default=False)
    redis_connection: bool = field(default=False)
    database_connection: bool = field(default=False)
    openai_connection: bool = field(default=False)


class MasterIntelligenceEngine:
    """
    Master intelligence engine that coordinates all intelligence operations
    PATTERN: Similar to MetalinMasterOrchestrator but focused on intelligence tasks
    """

    def __init__(self, config=None):
        """
        Initialize the Master Intelligence Engine

        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.start_time = datetime.now()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("🧠 Initializing Master Intelligence Engine")

        # Initialize connections
        self._setup_openai_client()
        self._setup_redis_client()
        self._setup_database()

        # Track engine status
        self.status = EngineStatus()

        # Initialize sub-engines (placeholder for now)
        self._initialize_sub_engines()

        # Performance metrics
        self.request_count = 0
        self.total_processing_time = 0
        self.cache_hits = 0
        self.cache_misses = 0

        self.logger.info("✅ Master Intelligence Engine initialized successfully")

    def _setup_openai_client(self):
        """Setup OpenAI client connection"""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.openai.api_key,
                timeout=self.config.openai.timeout,
                max_retries=self.config.openai.max_retries,
            )

            # Test connection
            self.openai_client.models.list()
            self.status.openai_connection = True
            self.logger.info("✅ OpenAI client initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            self.status.openai_connection = False
            raise

    def _setup_redis_client(self):
        """Setup Redis client connection"""
        try:
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout,
                socket_connect_timeout=self.config.redis.socket_connect_timeout,
                retry_on_timeout=self.config.redis.retry_on_timeout,
                decode_responses=True,
            )

            # Test connection
            self.redis_client.ping()
            self.status.redis_connection = True
            self.logger.info("✅ Redis client initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Redis client: {e}")
            self.status.redis_connection = False
            # Don't raise - system can work without Redis (with reduced performance)

    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_schema = IntelligenceSchema(
                self.config.database.intelligence_db_path
            )

            # Ensure database exists
            if not os.path.exists(self.config.database.intelligence_db_path):
                self.db_schema.create_database()
                self.logger.info("📊 Created new intelligence database")

            self.status.database_connection = True
            self.logger.info("✅ Database connection established successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            self.status.database_connection = False
            raise

    def _initialize_sub_engines(self):
        """Initialize intelligence sub-engines"""
        # Placeholder - these will be implemented in subsequent tasks
        self.behavior_analyzer = None  # BehaviorAnalyzer()
        self.opportunity_matcher = None  # OpportunityMatcher()
        self.recommendation_engine = None  # RecommendationEngine()
        self.content_processor = None  # ContentProcessor()

        # For now, mark as not available
        self.status.behavior_analyzer = False
        self.status.opportunity_matcher = False
        self.status.recommendation_engine = False
        self.status.content_processor = False

        self.logger.info("📋 Sub-engines initialized (placeholder mode)")

    async def process_user_request(
        self, user_id: str, request_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for intelligence operations
        PATTERN: Similar to process_unified_query in master orchestrator

        Args:
            user_id: ID of the user making the request
            request_type: Type of intelligence request
            context: Request context and parameters

        Returns:
            Dict containing the response data
        """
        start_time = time.time()
        self.request_count += 1

        try:
            self.logger.info(f"🔍 Processing request: {request_type} for user {user_id}")

            # Validate request
            IntelligenceRequest(
                user_id=user_id, request_type=request_type, context=context
            )

            # Check cache first
            cache_key = self._generate_cache_key(user_id, request_type, context)
            cached_result = await self._get_from_cache(cache_key)

            if cached_result:
                self.cache_hits += 1
                self.logger.info(f"🎯 Cache hit for key: {cache_key}")
                return cached_result

            self.cache_misses += 1

            # Route to appropriate handler
            if request_type == "recommendations":
                result = await self._get_recommendations(user_id, context)
            elif request_type == "opportunity_match":
                result = await self._match_opportunities(user_id, context)
            elif request_type == "behavior_analysis":
                result = await self._analyze_behavior(user_id, context)
            elif request_type == "analytics":
                result = await self._get_analytics(user_id, context)
            elif request_type == "prediction":
                result = await self._get_predictions(user_id, context)
            else:
                raise ValueError(f"Unknown request type: {request_type}")

            # Cache result
            await self._store_in_cache(cache_key, result, request_type)

            # Update metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            # Create response
            response = IntelligenceResponse(
                user_id=user_id,
                response_type=request_type,
                data=result,
                metadata={
                    "processing_time_ms": int(processing_time * 1000),
                    "cache_used": False,
                    "engine_status": self.status.__dict__,
                },
                processing_time_ms=int(processing_time * 1000),
            )

            self.logger.info(
                f"✅ Request processed successfully in {processing_time:.2f}s"
            )
            return response.dict()

        except Exception as e:
            self.logger.error(f"❌ Error processing request: {e}")
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "request_type": request_type,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            }

    async def _get_recommendations(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get personalized recommendations for a user
        ORCHESTRATION: Coordinate behavior analysis and matching
        """
        self.logger.info(f"📊 Generating recommendations for user {user_id}")

        try:
            # Placeholder implementation - will be replaced with actual engines
            # Step 1: Analyze user behavior patterns
            behavior_patterns = await self._analyze_user_behavior_patterns(user_id)

            # Step 2: Find matching opportunities
            opportunities = await self._find_matching_opportunities(
                user_id, behavior_patterns
            )

            # Step 3: Generate recommendations
            recommendations = await self._generate_recommendations(
                user_id, opportunities, behavior_patterns
            )

            return {
                "recommendations": recommendations,
                "behavior_insights": behavior_patterns,
                "match_count": len(opportunities),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"❌ Error generating recommendations: {e}")
            return {
                "recommendations": [],
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
            }

    async def _match_opportunities(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Match opportunities with user preferences"""
        self.logger.info(f"🎯 Matching opportunities for user {user_id}")

        # Placeholder implementation
        return {
            "matches": [],
            "total_opportunities": 0,
            "match_criteria": context.get("criteria", {}),
            "matched_at": datetime.now().isoformat(),
        }

    async def _analyze_behavior(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        self.logger.info(f"📈 Analyzing behavior for user {user_id}")

        # Placeholder implementation
        return {
            "behavior_patterns": {},
            "engagement_metrics": {},
            "insights": [],
            "analyzed_at": datetime.now().isoformat(),
        }

    async def _get_analytics(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get analytics data"""
        self.logger.info(f"📊 Getting analytics for user {user_id}")

        # Placeholder implementation
        return {
            "metrics": {},
            "trends": {},
            "insights": [],
            "generated_at": datetime.now().isoformat(),
        }

    async def _get_predictions(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get predictions for user behavior/outcomes"""
        self.logger.info(f"🔮 Generating predictions for user {user_id}")

        # Placeholder implementation
        return {
            "predictions": {},
            "confidence_scores": {},
            "model_version": "placeholder",
            "predicted_at": datetime.now().isoformat(),
        }

    # Helper methods for behavior analysis (placeholder implementations)
    async def _analyze_user_behavior_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        # TODO: Implement with BehaviorAnalyzer
        return {
            "engagement_score": 0.7,
            "preferred_categories": ["technology", "business"],
            "interaction_patterns": {"peak_hours": [9, 14, 17]},
            "confidence": 0.8,
        }

    async def _find_matching_opportunities(
        self, user_id: str, behavior_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find opportunities matching user patterns"""
        # TODO: Implement with OpportunityMatcher
        return [
            {
                "id": "opp_1",
                "title": "Sample Opportunity 1",
                "relevance_score": 0.85,
                "match_factors": ["industry", "location"],
            },
            {
                "id": "opp_2",
                "title": "Sample Opportunity 2",
                "relevance_score": 0.72,
                "match_factors": ["behavior", "timing"],
            },
        ]

    async def _generate_recommendations(
        self,
        user_id: str,
        opportunities: List[Dict[str, Any]],
        behavior_patterns: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate personalized recommendations"""
        # TODO: Implement with RecommendationEngine
        recommendations = []

        for opp in opportunities:
            rec = {
                "id": f"rec_{opp['id']}",
                "opportunity_id": opp["id"],
                "user_id": user_id,
                "score": opp["relevance_score"],
                "type": "personalized",
                "reasoning": f"Recommended based on {', '.join(opp['match_factors'])}",
                "generated_at": datetime.now().isoformat(),
            }
            recommendations.append(rec)

        return recommendations

    # Cache management methods
    def _generate_cache_key(
        self, user_id: str, request_type: str, context: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        context_hash = hash(json.dumps(context, sort_keys=True))
        return f"intelligence:{user_id}:{request_type}:{context_hash}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache"""
        if not self.status.redis_connection:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")

        return None

    async def _store_in_cache(
        self, cache_key: str, result: Dict[str, Any], request_type: str
    ):
        """Store result in cache"""
        if not self.status.redis_connection:
            return

        try:
            # Get TTL based on request type
            ttl = self._get_cache_ttl(request_type)

            # Store in cache
            self.redis_client.setex(cache_key, ttl, json.dumps(result, default=str))

        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")

    def _get_cache_ttl(self, request_type: str) -> int:
        """Get cache TTL for request type"""
        ttl_mapping = {
            "recommendations": self.config.redis.recommendation_cache_ttl,
            "behavior_analysis": self.config.redis.user_behavior_cache_ttl,
            "analytics": self.config.redis.analytics_cache_ttl,
            "opportunity_match": self.config.redis.recommendation_cache_ttl,
            "prediction": self.config.redis.analytics_cache_ttl,
        }

        return ttl_mapping.get(request_type, self.config.redis.default_ttl)

    # System management methods
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        uptime = datetime.now() - self.start_time
        avg_processing_time = self.total_processing_time / max(self.request_count, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)

        return {
            "status": "operational",
            "uptime_seconds": uptime.total_seconds(),
            "component_status": self.status.__dict__,
            "performance_metrics": {
                "total_requests": self.request_count,
                "avg_processing_time_ms": int(avg_processing_time * 1000),
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
            "configuration": {
                "environment": self.config.environment,
                "debug": self.config.debug,
                "openai_model": self.config.openai.chat_model,
                "embedding_model": self.config.openai.embedding_model,
                "redis_url": self.config.redis.url,
                "database_path": self.config.database.intelligence_db_path,
            },
            "last_updated": datetime.now().isoformat(),
        }

    def clear_cache(self, pattern: str = "intelligence:*") -> int:
        """Clear cache entries matching pattern"""
        if not self.status.redis_connection:
            return 0

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing cache: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "overall": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check OpenAI connection
        try:
            self.openai_client.models.list()
            health_status["components"]["openai"] = {
                "status": "healthy",
                "message": "Connection OK",
            }
        except Exception as e:
            health_status["components"]["openai"] = {
                "status": "unhealthy",
                "message": str(e),
            }
            health_status["overall"] = "degraded"

        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "message": "Connection OK",
            }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "message": str(e),
            }
            health_status["overall"] = "degraded"

        # Check Database connection
        try:
            if os.path.exists(self.config.database.intelligence_db_path):
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "message": "Database accessible",
                }
            else:
                health_status["components"]["database"] = {
                    "status": "unhealthy",
                    "message": "Database file not found",
                }
                health_status["overall"] = "degraded"
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "message": str(e),
            }
            health_status["overall"] = "degraded"

        return health_status

    def shutdown(self):
        """Gracefully shutdown the engine"""
        self.logger.info("🔄 Shutting down Master Intelligence Engine")

        # Close connections
        if self.status.redis_connection:
            self.redis_client.close()

        self.logger.info("✅ Master Intelligence Engine shut down successfully")


# Singleton instance
_master_engine = None


def get_master_engine(config=None) -> MasterIntelligenceEngine:
    """Get the global master intelligence engine instance"""
    global _master_engine
    if _master_engine is None:
        _master_engine = MasterIntelligenceEngine(config)
    return _master_engine


def reset_master_engine():
    """Reset the global master intelligence engine instance"""
    global _master_engine
    if _master_engine:
        _master_engine.shutdown()
    _master_engine = None


if __name__ == "__main__":
    # Test the master intelligence engine
    async def test_master_engine():
        print("🧠 Testing Master Intelligence Engine")
        print("=" * 50)

        try:
            # Initialize engine
            engine = MasterIntelligenceEngine()

            # Test status
            status = engine.get_engine_status()
            print(f"Engine Status: {status['status']}")
            print(f"Components: {status['component_status']}")

            # Test health check
            health = engine.health_check()
            print(f"Health Check: {health['overall']}")

            # Test request processing
            result = await engine.process_user_request(
                user_id="test_user",
                request_type="recommendations",
                context={"industry": "technology", "location": "san_francisco"},
            )

            print(f"Request Result: {result}")

            # Shutdown
            engine.shutdown()

            print("✅ Master Intelligence Engine test completed successfully!")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Run test
    asyncio.run(test_master_engine())
