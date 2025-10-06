#!/usr/bin/env python3
"""
Intelligence API Endpoints
Flask API endpoints for intelligence operations

This API provides:
- User behavior analysis endpoints
- Content processing and recommendations
- Opportunity matching and scoring
- Real-time intelligence queries
- Machine learning model interactions

Following Task 10 from the PRP implementation blueprint.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import logging

from intelligence.master_engine import MasterIntelligenceEngine
from intelligence.data_models2 import (
    UserInteraction, OpportunityScore, Recommendation,
    BehaviorMetrics, IntelligenceRequest, IntelligenceResponse
)
# Import components dynamically to avoid circular import issues
# from intelligence.behavior_analyzer import BehaviorAnalyzer
# from intelligence.content_processor import ContentProcessor
# from intelligence.opportunity_matcher import OpportunityMatcher
# from intelligence.recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)

# Create Flask Blueprint
intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/api/intelligence')


def get_master_engine() -> MasterIntelligenceEngine:
    """Get the master intelligence engine instance"""
    if not hasattr(current_app, 'intelligence_master_engine'):
        current_app.intelligence_master_engine = MasterIntelligenceEngine()
    return current_app.intelligence_master_engine


def async_route(f):
    """Decorator to handle async route functions"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = f.__name__
    return wrapper


@intelligence_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for intelligence services"""
    try:
        master_engine = get_master_engine()
        status = master_engine.get_service_status()
        
        return jsonify({
            "status": "healthy",
            "service": "intelligence_api",
            "components": status,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@intelligence_bp.route('/analyze-behavior', methods=['POST'])
@async_route
async def analyze_user_behavior():
    """Analyze user behavior patterns"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("user_id is required")
        
        # Get optional parameters
        time_range = data.get('time_range', {})
        analysis_types = data.get('analysis_types', ['engagement', 'patterns', 'preferences'])
        
        master_engine = get_master_engine()
        
        # Create intelligence request
        intel_request = IntelligenceRequest(
            user_id=user_id,
            request_type='behavior_analysis',
            context={
                'time_range': time_range,
                'analysis_types': analysis_types
            }
        )
        
        # Process the request
        response = await master_engine.process_intelligence_request(intel_request)
        
        return jsonify({
            "success": True,
            "data": response.data,
            "metadata": response.metadata,
            "processing_time_ms": response.processing_time_ms,
            "timestamp": response.timestamp.isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error analyzing user behavior: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/recommendations', methods=['POST'])
@async_route
async def get_recommendations():
    """Get personalized recommendations for a user"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("user_id is required")
        
        # Get optional parameters
        recommendation_types = data.get('types', ['personalized', 'trending'])
        max_results = data.get('max_results', 10)
        context = data.get('context', {})
        
        master_engine = get_master_engine()
        
        # Create intelligence request
        intel_request = IntelligenceRequest(
            user_id=user_id,
            request_type='recommendations',
            context={
                'recommendation_types': recommendation_types,
                'max_results': max_results,
                'context': context
            }
        )
        
        # Process the request
        response = await master_engine.process_intelligence_request(intel_request)
        
        return jsonify({
            "success": True,
            "recommendations": response.data.get('recommendations', []),
            "metadata": response.metadata,
            "processing_time_ms": response.processing_time_ms,
            "timestamp": response.timestamp.isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error getting recommendations: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/opportunities/match', methods=['POST'])
@async_route
async def match_opportunities():
    """Match opportunities for a user"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        user_id = data.get('user_id')
        if not user_id:
            raise BadRequest("user_id is required")
        
        # Get optional parameters
        filters = data.get('filters', {})
        max_results = data.get('max_results', 20)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        
        master_engine = get_master_engine()
        
        # Create intelligence request
        intel_request = IntelligenceRequest(
            user_id=user_id,
            request_type='opportunity_match',
            context={
                'filters': filters,
                'max_results': max_results,
                'similarity_threshold': similarity_threshold
            }
        )
        
        # Process the request
        response = await master_engine.process_intelligence_request(intel_request)
        
        return jsonify({
            "success": True,
            "opportunities": response.data.get('opportunities', []),
            "match_scores": response.data.get('scores', []),
            "metadata": response.metadata,
            "processing_time_ms": response.processing_time_ms,
            "timestamp": response.timestamp.isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error matching opportunities: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/content/process', methods=['POST'])
@async_route
async def process_content():
    """Process content for analysis and embedding"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        content = data.get('content')
        content_type = data.get('content_type')
        
        if not content or not content_type:
            raise BadRequest("content and content_type are required")
        
        # Get optional parameters
        content_id = data.get('content_id', f"content_{datetime.now().timestamp()}")
        processing_options = data.get('processing_options', {})
        
        master_engine = get_master_engine()
        content_processor = master_engine.content_processor
        
        # Process the content
        result = await content_processor.process_content(
            content=content,
            content_type=content_type,
            content_id=content_id,
            **processing_options
        )
        
        return jsonify({
            "success": True,
            "content_id": result.content_id,
            "processing_results": result.processing_results,
            "embeddings_generated": result.embeddings_generated,
            "analysis_results": result.analysis_results,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp.isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error processing content: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/interactions', methods=['POST'])
@async_route
async def record_interaction():
    """Record user interaction"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Validate required fields
        required_fields = ['user_id', 'opportunity_id', 'interaction_type']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"{field} is required")
        
        # Create user interaction object
        interaction = UserInteraction(
            user_id=data['user_id'],
            opportunity_id=data['opportunity_id'],
            interaction_type=data['interaction_type'],
            duration=data.get('duration'),
            metadata=data.get('metadata', {})
        )
        
        master_engine = get_master_engine()
        behavior_analyzer = master_engine.behavior_analyzer
        
        # Record the interaction
        success = await behavior_analyzer.track_interaction(interaction)
        
        if success:
            return jsonify({
                "success": True,
                "interaction_id": interaction.id,
                "timestamp": interaction.timestamp.isoformat()
            }), 201
        else:
            return jsonify({"error": "Failed to record interaction"}), 500
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error recording interaction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/users/<user_id>/profile', methods=['GET'])
@async_route
async def get_user_profile(user_id):
    """Get user profile and behavior summary"""
    try:
        master_engine = get_master_engine()
        behavior_analyzer = master_engine.behavior_analyzer
        
        # Get user profile
        profile = await behavior_analyzer.get_user_profile(user_id)
        
        if not profile:
            raise NotFound(f"User profile not found for user_id: {user_id}")
        
        # Get behavior metrics
        behavior_summary = await behavior_analyzer.get_behavior_summary(user_id)
        
        return jsonify({
            "success": True,
            "profile": {
                "user_id": profile.user_id,
                "industry": profile.industry,
                "location": profile.location,
                "company_size": profile.company_size,
                "job_role": profile.job_role,
                "interests": profile.interests,
                "preferences": profile.preferences,
                "engagement_score": profile.engagement_score,
                "last_active": profile.last_active.isoformat() if profile.last_active else None,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            },
            "behavior_summary": behavior_summary,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except NotFound as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"❌ Error getting user profile: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/scores/<opportunity_id>', methods=['GET'])
@async_route
async def get_opportunity_scores(opportunity_id):
    """Get opportunity scores for all users or specific user"""
    try:
        user_id = request.args.get('user_id')
        
        master_engine = get_master_engine()
        
        if user_id:
            # Get score for specific user
            intel_request = IntelligenceRequest(
                user_id=user_id,
                request_type='opportunity_score',
                context={'opportunity_id': opportunity_id}
            )
            
            response = await master_engine.process_intelligence_request(intel_request)
            scores = response.data.get('scores', [])
        else:
            # Get scores for all users (limited result)
            # This would be implemented in master engine
            scores = []
        
        return jsonify({
            "success": True,
            "opportunity_id": opportunity_id,
            "scores": scores,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting opportunity scores: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/predictions', methods=['POST'])
@async_route
async def get_predictions():
    """Get ML predictions for user behavior or outcomes"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        user_id = data.get('user_id')
        prediction_type = data.get('prediction_type')
        
        if not user_id or not prediction_type:
            raise BadRequest("user_id and prediction_type are required")
        
        # Get optional parameters
        context = data.get('context', {})
        time_horizon = data.get('time_horizon', 30)  # days
        
        master_engine = get_master_engine()
        
        # Create intelligence request
        intel_request = IntelligenceRequest(
            user_id=user_id,
            request_type='prediction',
            context={
                'prediction_type': prediction_type,
                'context': context,
                'time_horizon': time_horizon
            }
        )
        
        # Process the request
        response = await master_engine.process_intelligence_request(intel_request)
        
        return jsonify({
            "success": True,
            "predictions": response.data.get('predictions', []),
            "confidence_scores": response.data.get('confidence_scores', []),
            "metadata": response.metadata,
            "processing_time_ms": response.processing_time_ms,
            "timestamp": response.timestamp.isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error getting predictions: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/similar-content', methods=['POST'])
@async_route
async def find_similar_content():
    """Find similar content using embeddings"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        content = data.get('content')
        content_type = data.get('content_type')
        
        if not content or not content_type:
            raise BadRequest("content and content_type are required")
        
        # Get optional parameters
        max_results = data.get('max_results', 10)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        
        master_engine = get_master_engine()
        content_processor = master_engine.content_processor
        
        # Find similar content
        similar_content = await content_processor.find_similar_content(
            content=content,
            content_type=content_type,
            max_results=max_results,
            similarity_threshold=similarity_threshold
        )
        
        return jsonify({
            "success": True,
            "similar_content": [
                {
                    "content_id": item.content_id,
                    "content_type": item.content_type,
                    "similarity_score": item.similarity_score,
                    "metadata": item.metadata
                }
                for item in similar_content
            ],
            "query_content_type": content_type,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error finding similar content: {e}")
        return jsonify({"error": "Internal server error"}), 500


@intelligence_bp.route('/status', methods=['GET'])
def get_intelligence_status():
    """Get detailed status of all intelligence components"""
    try:
        master_engine = get_master_engine()
        
        # Get status from all components
        status = {
            "service": "intelligence_api",
            "status": "operational",
            "components": master_engine.get_service_status(),
            "endpoints": {
                "health": "/api/intelligence/health",
                "analyze_behavior": "/api/intelligence/analyze-behavior",
                "recommendations": "/api/intelligence/recommendations",
                "match_opportunities": "/api/intelligence/opportunities/match",
                "process_content": "/api/intelligence/content/process",
                "record_interaction": "/api/intelligence/interactions",
                "user_profile": "/api/intelligence/users/{user_id}/profile",
                "opportunity_scores": "/api/intelligence/scores/{opportunity_id}",
                "predictions": "/api/intelligence/predictions",
                "similar_content": "/api/intelligence/similar-content"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting intelligence status: {e}")
        return jsonify({
            "service": "intelligence_api",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# Error handlers
@intelligence_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@intelligence_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@intelligence_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500