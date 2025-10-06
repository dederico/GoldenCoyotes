#!/usr/bin/env python3
"""
Unit Tests for Master Intelligence Engine

Tests the core functionality of the Master Intelligence Engine including:
- Engine initialization and configuration
- User interaction processing 
- Opportunity matching
- Recommendation generation
- Behavior analysis
- Content processing
- Caching and performance
- Error handling and edge cases
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from ..intelligence.master_engine import MasterIntelligenceEngine
from ..core.data_models import (
    UserInteraction, 
    OpportunityType, 
    InteractionType, 
    ProcessingRequest,
    ProcessingResult,
    RecommendationRequest,
    EngineResponse
)
from ..config.intelligence_config import IntelligenceConfig
from ..config.ml_config import MLConfig


class TestMasterIntelligenceEngine:
    """Test suite for Master Intelligence Engine"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    @pytest.fixture
    def mock_config(self, temp_db):
        """Mock configuration for testing"""
        config = Mock(spec=IntelligenceConfig)
        config.database.intelligence_db_path = temp_db
        config.redis.url = "redis://localhost:6379"
        config.redis.cache_ttl = 300
        config.openai.api_key = "test_key"
        config.openai.model = "gpt-4"
        config.openai.timeout = 30
        config.openai.max_retries = 3
        return config

    @pytest.fixture
    def mock_ml_config(self):
        """Mock ML configuration for testing"""
        config = Mock(spec=MLConfig)
        config.models.scoring_model_path = "test_model.pkl"
        config.models.retrain_threshold = 0.8
        config.embedding_model.model_name = "text-embedding-3-large"
        config.embedding_model.embedding_dimension = 3072
        return config

    @pytest.fixture
    async def engine(self, mock_config, mock_ml_config):
        """Create test engine instance"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.from_url.return_value.ping.return_value = True
            engine = MasterIntelligenceEngine(mock_config, mock_ml_config)
            yield engine

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.config is not None
        assert engine.ml_config is not None
        assert engine.behavior_analyzer is not None
        assert engine.content_processor is not None
        assert engine.opportunity_matcher is not None
        assert engine.recommendation_engine is not None

    @pytest.mark.asyncio
    async def test_process_user_interaction_success(self, engine):
        """Test successful user interaction processing"""
        # Create test interaction
        interaction = UserInteraction(
            user_id="test_user",
            interaction_type=InteractionType.VIEW,
            opportunity_id="test_opportunity",
            timestamp=datetime.now(),
            interaction_data={"duration": 30},
            context_data={"device": "mobile"}
        )

        # Mock the behavior analyzer
        with patch.object(engine.behavior_analyzer, 'record_interaction') as mock_record:
            mock_record.return_value = True
            
            # Process interaction
            result = await engine.process_user_interaction(interaction)
            
            # Verify results
            assert result is True
            mock_record.assert_called_once_with(interaction)

    @pytest.mark.asyncio
    async def test_process_user_interaction_failure(self, engine):
        """Test user interaction processing failure"""
        # Create test interaction
        interaction = UserInteraction(
            user_id="test_user",
            interaction_type=InteractionType.VIEW,
            opportunity_id="test_opportunity",
            timestamp=datetime.now(),
            interaction_data={"duration": 30},
            context_data={"device": "mobile"}
        )

        # Mock the behavior analyzer to raise exception
        with patch.object(engine.behavior_analyzer, 'record_interaction') as mock_record:
            mock_record.side_effect = Exception("Database error")
            
            # Process interaction
            result = await engine.process_user_interaction(interaction)
            
            # Verify failure handling
            assert result is False

    @pytest.mark.asyncio
    async def test_process_business_data_success(self, engine):
        """Test successful business data processing"""
        # Create test request
        request = ProcessingRequest(
            request_id="test_request",
            user_id="test_user",
            data_type="opportunity",
            content="Test business opportunity",
            metadata={"priority": "high"}
        )

        # Mock the content processor
        with patch.object(engine.content_processor, 'process_content') as mock_process:
            mock_process.return_value = ProcessingResult(
                request_id="test_request",
                success=True,
                processed_content="Processed content",
                embeddings=[0.1, 0.2, 0.3],
                metadata={"processing_time": 0.5}
            )
            
            # Process data
            result = await engine.process_business_data(request)
            
            # Verify results
            assert result is not None
            assert result.success is True
            assert result.request_id == "test_request"
            mock_process.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_find_opportunities_success(self, engine):
        """Test successful opportunity finding"""
        # Mock the opportunity matcher
        with patch.object(engine.opportunity_matcher, 'find_opportunities') as mock_find:
            mock_opportunities = [
                {
                    "opportunity_id": "opp_1",
                    "title": "Test Opportunity 1",
                    "score": 0.9,
                    "match_reasons": ["high_relevance"]
                },
                {
                    "opportunity_id": "opp_2", 
                    "title": "Test Opportunity 2",
                    "score": 0.8,
                    "match_reasons": ["good_timing"]
                }
            ]
            mock_find.return_value = mock_opportunities
            
            # Find opportunities
            opportunities = await engine.find_opportunities("test_user")
            
            # Verify results
            assert len(opportunities) == 2
            assert opportunities[0]["opportunity_id"] == "opp_1"
            assert opportunities[0]["score"] == 0.9
            mock_find.assert_called_once_with("test_user", None)

    @pytest.mark.asyncio
    async def test_find_opportunities_with_filters(self, engine):
        """Test opportunity finding with filters"""
        filters = {
            "opportunity_type": OpportunityType.BUSINESS_EXPANSION,
            "min_score": 0.7
        }

        # Mock the opportunity matcher
        with patch.object(engine.opportunity_matcher, 'find_opportunities') as mock_find:
            mock_find.return_value = []
            
            # Find opportunities with filters
            opportunities = await engine.find_opportunities("test_user", filters)
            
            # Verify call
            mock_find.assert_called_once_with("test_user", filters)

    @pytest.mark.asyncio
    async def test_generate_recommendations_success(self, engine):
        """Test successful recommendation generation"""
        # Create test request
        request = RecommendationRequest(
            user_id="test_user",
            request_type="personalized",
            max_recommendations=5,
            context_data={"location": "office"}
        )

        # Mock the recommendation engine
        with patch.object(engine.recommendation_engine, 'generate_recommendations') as mock_generate:
            mock_recommendations = [
                {
                    "recommendation_id": "rec_1",
                    "title": "Test Recommendation 1",
                    "score": 0.9,
                    "explanation": "High relevance to user interests"
                }
            ]
            mock_generate.return_value = mock_recommendations
            
            # Generate recommendations
            recommendations = await engine.generate_recommendations(request)
            
            # Verify results
            assert len(recommendations) == 1
            assert recommendations[0]["recommendation_id"] == "rec_1"
            mock_generate.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_success(self, engine):
        """Test successful user behavior analysis"""
        # Mock the behavior analyzer
        with patch.object(engine.behavior_analyzer, 'analyze_behavior') as mock_analyze:
            mock_patterns = {
                "activity_level": 0.8,
                "preferred_times": ["09:00", "14:00"],
                "interests": ["technology", "finance"]
            }
            mock_analyze.return_value = mock_patterns
            
            # Analyze behavior
            patterns = await engine.analyze_user_behavior("test_user")
            
            # Verify results
            assert patterns["activity_level"] == 0.8
            assert len(patterns["preferred_times"]) == 2
            mock_analyze.assert_called_once_with("test_user", None)

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_with_timeframe(self, engine):
        """Test user behavior analysis with timeframe"""
        timeframe = "30d"

        # Mock the behavior analyzer
        with patch.object(engine.behavior_analyzer, 'analyze_behavior') as mock_analyze:
            mock_analyze.return_value = {}
            
            # Analyze behavior with timeframe
            patterns = await engine.analyze_user_behavior("test_user", timeframe)
            
            # Verify call
            mock_analyze.assert_called_once_with("test_user", timeframe)

    @pytest.mark.asyncio
    async def test_get_user_insights_success(self, engine):
        """Test successful user insights retrieval"""
        # Mock behavior analyzer and recommendation engine
        with patch.object(engine.behavior_analyzer, 'get_user_insights') as mock_insights:
            mock_insights.return_value = {
                "engagement_score": 0.85,
                "growth_potential": 0.7,
                "key_interests": ["ai", "automation"]
            }
            
            # Get insights
            insights = await engine.get_user_insights("test_user")
            
            # Verify results
            assert insights["engagement_score"] == 0.85
            assert "ai" in insights["key_interests"]
            mock_insights.assert_called_once_with("test_user")

    @pytest.mark.asyncio
    async def test_process_request_success(self, engine):
        """Test successful request processing"""
        # Create test request
        request = ProcessingRequest(
            request_id="test_request",
            user_id="test_user",
            data_type="interaction",
            content="Test interaction data",
            metadata={"source": "mobile_app"}
        )

        # Mock components
        with patch.object(engine, 'process_business_data') as mock_process:
            mock_result = ProcessingResult(
                request_id="test_request",
                success=True,
                processed_content="Processed",
                embeddings=[],
                metadata={}
            )
            mock_process.return_value = mock_result
            
            # Process request
            result = await engine.process_request(request)
            
            # Verify results
            assert result.success is True
            assert result.request_id == "test_request"

    @pytest.mark.asyncio
    async def test_get_engine_status(self, engine):
        """Test engine status retrieval"""
        # Get status
        status = engine.get_engine_status()
        
        # Verify status structure
        assert "status" in status
        assert "uptime" in status
        assert "requests_processed" in status
        assert "cache_performance" in status
        assert "component_status" in status
        assert "performance_metrics" in status

    @pytest.mark.asyncio
    async def test_cache_performance_tracking(self, engine):
        """Test cache performance tracking"""
        # Initial state
        initial_hits = engine.cache_hits
        initial_misses = engine.cache_misses
        
        # Mock cache operations
        with patch.object(engine, '_get_cached_result') as mock_get:
            mock_get.return_value = None  # Cache miss
            
            # This should increment cache misses
            await engine._get_cached_result("test_key")
            
            # Verify cache miss increment
            assert engine.cache_misses == initial_misses + 1

    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, engine):
        """Test error handling in processing"""
        # Create test request
        request = ProcessingRequest(
            request_id="test_request",
            user_id="test_user",
            data_type="invalid_type",
            content="Test content",
            metadata={}
        )

        # Mock to raise exception
        with patch.object(engine.content_processor, 'process_content') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            # Process request
            result = await engine.process_business_data(request)
            
            # Verify error handling
            assert result is not None
            assert result.success is False

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, engine):
        """Test concurrent request processing"""
        # Create multiple requests
        requests = [
            ProcessingRequest(
                request_id=f"request_{i}",
                user_id="test_user",
                data_type="opportunity",
                content=f"Content {i}",
                metadata={}
            ) for i in range(3)
        ]

        # Mock processing
        with patch.object(engine.content_processor, 'process_content') as mock_process:
            mock_process.return_value = ProcessingResult(
                request_id="test",
                success=True,
                processed_content="Processed",
                embeddings=[],
                metadata={}
            )
            
            # Process concurrently
            tasks = [engine.process_business_data(req) for req in requests]
            results = await asyncio.gather(*tasks)
            
            # Verify all processed
            assert len(results) == 3
            assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_redis_unavailable_handling(self, mock_config, mock_ml_config):
        """Test handling when Redis is unavailable"""
        with patch('redis.Redis') as mock_redis:
            mock_redis.from_url.return_value.ping.side_effect = Exception("Redis unavailable")
            
            # Should still initialize without Redis
            engine = MasterIntelligenceEngine(mock_config, mock_ml_config)
            
            # Should handle cache operations gracefully
            result = await engine._get_cached_result("test_key")
            assert result is None

    @pytest.mark.asyncio
    async def test_component_health_check(self, engine):
        """Test component health checking"""
        # Mock component health
        with patch.object(engine.behavior_analyzer, 'get_analyzer_status') as mock_status:
            mock_status.return_value = {"status": "healthy", "uptime": 100}
            
            # Check component health
            health = engine._check_component_health()
            
            # Verify health check
            assert "behavior_analyzer" in health
            assert health["behavior_analyzer"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_performance_metrics(self, engine):
        """Test performance metrics collection"""
        # Initial metrics
        initial_requests = engine.requests_processed
        
        # Create and process request
        request = ProcessingRequest(
            request_id="test_request",
            user_id="test_user",
            data_type="test",
            content="Test content",
            metadata={}
        )

        with patch.object(engine.content_processor, 'process_content') as mock_process:
            mock_process.return_value = ProcessingResult(
                request_id="test_request",
                success=True,
                processed_content="Processed",
                embeddings=[],
                metadata={}
            )
            
            # Process request
            await engine.process_business_data(request)
            
            # Verify metrics updated
            assert engine.requests_processed == initial_requests + 1

    @pytest.mark.asyncio
    async def test_integration_with_all_components(self, engine):
        """Test integration with all engine components"""
        # Create comprehensive test
        interaction = UserInteraction(
            user_id="test_user",
            interaction_type=InteractionType.CLICK,
            opportunity_id="test_opportunity",
            timestamp=datetime.now(),
            interaction_data={},
            context_data={}
        )

        request = RecommendationRequest(
            user_id="test_user",
            request_type="personalized",
            max_recommendations=3,
            context_data={}
        )

        # Mock all components
        with patch.object(engine.behavior_analyzer, 'record_interaction') as mock_record, \
             patch.object(engine.opportunity_matcher, 'find_opportunities') as mock_find, \
             patch.object(engine.recommendation_engine, 'generate_recommendations') as mock_recommend:
            
            mock_record.return_value = True
            mock_find.return_value = []
            mock_recommend.return_value = []
            
            # Test integration
            interaction_result = await engine.process_user_interaction(interaction)
            opportunities = await engine.find_opportunities("test_user")
            recommendations = await engine.generate_recommendations(request)
            
            # Verify all components called
            assert interaction_result is True
            assert isinstance(opportunities, list)
            assert isinstance(recommendations, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])