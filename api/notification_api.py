#!/usr/bin/env python3
"""
Notification API Endpoints
Flask API endpoints for notification management

This API provides:
- Smart notification prioritization and delivery
- User context analysis and preference management
- Multi-channel notification routing
- Notification history and analytics
- Real-time notification status tracking

Following Task 10 from the PRP implementation blueprint.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import logging

# Import components dynamically to avoid circular import issues
# from notification.smart_prioritizer import SmartPrioritizer
# from notification.context_analyzer import ContextAnalyzer
# from notification.delivery_optimizer import DeliveryOptimizer
# from notification.preference_manager import PreferenceManager
from intelligence.data_models2 import NotificationHistory, NotificationChannel

logger = logging.getLogger(__name__)

# Create Flask Blueprint
notification_bp = Blueprint('notification', __name__, url_prefix='/api/notifications')


def get_notification_components():
    """Get notification components instances"""
    if not hasattr(current_app, 'notification_components'):
        current_app.notification_components = {
            'smart_prioritizer': SmartPrioritizer(),
            'context_analyzer': ContextAnalyzer(),
            'delivery_optimizer': DeliveryOptimizer(),
            'preference_manager': PreferenceManager()
        }
    return current_app.notification_components


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


@notification_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for notification services"""
    try:
        components = get_notification_components()
        
        status = {
            "smart_prioritizer": "operational",
            "context_analyzer": "operational",
            "delivery_optimizer": "operational",
            "preference_manager": "operational"
        }
        
        return jsonify({
            "status": "healthy",
            "service": "notification_api",
            "components": status,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Notification health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@notification_bp.route('/send', methods=['POST'])
@async_route
async def send_notification():
    """Send a smart notification to a user"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Validate required fields
        required_fields = ['user_id', 'content', 'notification_type']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"{field} is required")
        
        user_id = data['user_id']
        content = data['content']
        notification_type = data['notification_type']
        
        # Get optional parameters
        priority = data.get('priority', 'medium')
        channels = data.get('channels', ['push', 'in_app'])
        metadata = data.get('metadata', {})
        schedule_time = data.get('schedule_time')  # Optional future scheduling
        
        components = get_notification_components()
        
        # Step 1: Analyze user context
        context_analyzer = components['context_analyzer']
        user_context = await context_analyzer.analyze_user_context(user_id)
        
        # Step 2: Calculate priority score
        smart_prioritizer = components['smart_prioritizer']
        priority_result = await smart_prioritizer.calculate_priority(
            user_id=user_id,
            notification_type=notification_type,
            content=content,
            user_context=user_context,
            base_priority=priority
        )
        
        # Step 3: Optimize delivery
        delivery_optimizer = components['delivery_optimizer']
        delivery_plan = await delivery_optimizer.optimize_delivery(
            user_id=user_id,
            notification_type=notification_type,
            priority_score=priority_result.priority_score,
            user_context=user_context,
            preferred_channels=channels,
            schedule_time=schedule_time
        )
        
        # Step 4: Execute delivery
        delivery_results = await delivery_optimizer.execute_delivery(delivery_plan)
        
        # Create notification history record
        notification_record = NotificationHistory(
            user_id=user_id,
            notification_type=notification_type,
            content=content,
            priority_score=priority_result.priority_score,
            delivery_channel=delivery_plan.primary_channel,
            metadata={
                **metadata,
                'delivery_plan_id': delivery_plan.plan_id,
                'context_score': user_context.availability_score,
                'priority_factors': priority_result.factors
            }
        )
        
        return jsonify({
            "success": True,
            "notification_id": notification_record.id,
            "priority_score": priority_result.priority_score,
            "delivery_plan": {
                "plan_id": delivery_plan.plan_id,
                "primary_channel": delivery_plan.primary_channel.value,
                "backup_channels": [ch.value for ch in delivery_plan.backup_channels],
                "delivery_time": delivery_plan.delivery_time.isoformat(),
                "estimated_delivery": delivery_plan.estimated_delivery.isoformat()
            },
            "delivery_results": [
                {
                    "channel": result.channel.value,
                    "status": result.status,
                    "delivered_at": result.delivered_at.isoformat() if result.delivered_at else None,
                    "error_message": result.error_message
                }
                for result in delivery_results
            ],
            "user_context": {
                "availability_score": user_context.availability_score,
                "attention_score": user_context.attention_score,
                "activity_level": user_context.activity_level.value
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error sending notification: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/batch', methods=['POST'])
@async_route
async def send_batch_notifications():
    """Send notifications to multiple users"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        notifications = data.get('notifications', [])
        if not notifications:
            raise BadRequest("notifications array is required")
        
        # Validate each notification
        for notif in notifications:
            required_fields = ['user_id', 'content', 'notification_type']
            for field in required_fields:
                if field not in notif:
                    raise BadRequest(f"{field} is required in all notifications")
        
        # Get configuration
        batch_size = data.get('batch_size', 10)
        delay_between_batches = data.get('delay_between_batches', 1)  # seconds
        
        components = get_notification_components()
        delivery_optimizer = components['delivery_optimizer']
        
        # Process notifications in batches
        results = []
        total_notifications = len(notifications)
        
        for i in range(0, total_notifications, batch_size):
            batch = notifications[i:i + batch_size]
            
            # Process batch
            batch_results = await delivery_optimizer.send_batch_notifications(batch)
            results.extend(batch_results)
            
            # Add delay between batches (except for the last batch)
            if i + batch_size < total_notifications:
                await asyncio.sleep(delay_between_batches)
        
        return jsonify({
            "success": True,
            "total_notifications": total_notifications,
            "batch_size": batch_size,
            "results": [
                {
                    "user_id": result.user_id,
                    "notification_id": result.notification_id,
                    "status": result.status,
                    "priority_score": result.priority_score,
                    "delivery_channel": result.delivery_channel.value,
                    "sent_at": result.sent_at.isoformat() if result.sent_at else None,
                    "error_message": result.error_message
                }
                for result in results
            ],
            "success_count": sum(1 for r in results if r.status == "delivered"),
            "failed_count": sum(1 for r in results if r.status == "failed"),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error sending batch notifications: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/users/<user_id>/context', methods=['GET'])
@async_route
async def get_user_context(user_id):
    """Get current user context for notification timing"""
    try:
        components = get_notification_components()
        context_analyzer = components['context_analyzer']
        
        # Analyze user context
        user_context = await context_analyzer.analyze_user_context(user_id)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "context": {
                "activity_level": user_context.activity_level.value,
                "device_context": user_context.device_context,
                "location_context": user_context.location_context,
                "time_context": user_context.time_context,
                "availability_score": user_context.availability_score,
                "attention_score": user_context.attention_score,
                "behavior_patterns": user_context.behavior_patterns,
                "last_updated": user_context.last_updated.isoformat(),
                "expires_at": user_context.expires_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting user context for {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/users/<user_id>/preferences', methods=['GET'])
@async_route
async def get_user_preferences(user_id):
    """Get user notification preferences"""
    try:
        components = get_notification_components()
        preference_manager = components['preference_manager']
        
        # Get user preferences
        preferences = await preference_manager.get_user_preferences(user_id)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "preferences": {
                "notification_enabled": preferences.notification_enabled,
                "channel_preferences": {
                    channel.value: pref for channel, pref in preferences.channel_preferences.items()
                },
                "quiet_hours": {
                    "enabled": preferences.quiet_hours.enabled,
                    "start_time": preferences.quiet_hours.start_time,
                    "end_time": preferences.quiet_hours.end_time,
                    "timezone": preferences.quiet_hours.timezone
                },
                "frequency_limits": preferences.frequency_limits,
                "content_filters": preferences.content_filters,
                "custom_settings": preferences.custom_settings,
                "last_updated": preferences.last_updated.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting preferences for {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/users/<user_id>/preferences', methods=['PUT'])
@async_route
async def update_user_preferences(user_id):
    """Update user notification preferences"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        components = get_notification_components()
        preference_manager = components['preference_manager']
        
        # Update preferences
        success = await preference_manager.update_user_preferences(user_id, data)
        
        if success:
            # Get updated preferences
            updated_preferences = await preference_manager.get_user_preferences(user_id)
            
            return jsonify({
                "success": True,
                "user_id": user_id,
                "message": "Preferences updated successfully",
                "preferences": {
                    "notification_enabled": updated_preferences.notification_enabled,
                    "channel_preferences": {
                        channel.value: pref for channel, pref in updated_preferences.channel_preferences.items()
                    },
                    "quiet_hours": {
                        "enabled": updated_preferences.quiet_hours.enabled,
                        "start_time": updated_preferences.quiet_hours.start_time,
                        "end_time": updated_preferences.quiet_hours.end_time,
                        "timezone": updated_preferences.quiet_hours.timezone
                    },
                    "frequency_limits": updated_preferences.frequency_limits,
                    "content_filters": updated_preferences.content_filters,
                    "custom_settings": updated_preferences.custom_settings,
                    "last_updated": updated_preferences.last_updated.isoformat()
                },
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({"error": "Failed to update preferences"}), 500
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error updating preferences for {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/users/<user_id>/history', methods=['GET'])
@async_route
async def get_notification_history(user_id):
    """Get notification history for a user"""
    try:
        # Get query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        notification_type = request.args.get('notification_type')
        
        components = get_notification_components()
        delivery_optimizer = components['delivery_optimizer']
        
        # Get notification history
        history = await delivery_optimizer.get_notification_history(
            user_id=user_id,
            limit=limit,
            offset=offset,
            start_date=start_date,
            end_date=end_date,
            notification_type=notification_type
        )
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "history": [
                {
                    "id": notif.id,
                    "notification_type": notif.notification_type,
                    "content": notif.content,
                    "priority_score": notif.priority_score,
                    "sent_at": notif.sent_at.isoformat(),
                    "opened_at": notif.opened_at.isoformat() if notif.opened_at else None,
                    "clicked_at": notif.clicked_at.isoformat() if notif.clicked_at else None,
                    "dismissed_at": notif.dismissed_at.isoformat() if notif.dismissed_at else None,
                    "delivery_channel": notif.delivery_channel.value,
                    "metadata": notif.metadata
                }
                for notif in history
            ],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": len(history)
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting notification history for {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/priority/calculate', methods=['POST'])
@async_route
async def calculate_priority():
    """Calculate notification priority score"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Validate required fields
        required_fields = ['user_id', 'notification_type', 'content']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"{field} is required")
        
        user_id = data['user_id']
        notification_type = data['notification_type']
        content = data['content']
        base_priority = data.get('base_priority', 'medium')
        context_data = data.get('context', {})
        
        components = get_notification_components()
        
        # Get user context
        context_analyzer = components['context_analyzer']
        if context_data:
            # Use provided context
            from notification.context_analyzer import UserContext, ActivityLevel
            user_context = UserContext(
                user_id=user_id,
                activity_level=ActivityLevel(context_data.get('activity_level', 'medium')),
                device_context=context_data.get('device_context', {}),
                location_context=context_data.get('location_context', {}),
                time_context=context_data.get('time_context', {}),
                availability_score=context_data.get('availability_score', 0.5),
                attention_score=context_data.get('attention_score', 0.5),
                preferences=context_data.get('preferences', {}),
                behavior_patterns=context_data.get('behavior_patterns', {}),
                last_updated=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=15)
            )
        else:
            # Analyze context
            user_context = await context_analyzer.analyze_user_context(user_id)
        
        # Calculate priority
        smart_prioritizer = components['smart_prioritizer']
        priority_result = await smart_prioritizer.calculate_priority(
            user_id=user_id,
            notification_type=notification_type,
            content=content,
            user_context=user_context,
            base_priority=base_priority
        )
        
        return jsonify({
            "success": True,
            "priority_result": {
                "priority_score": priority_result.priority_score,
                "priority_level": priority_result.priority_level,
                "factors": priority_result.factors,
                "reasoning": priority_result.reasoning,
                "recommended_timing": priority_result.recommended_timing.isoformat(),
                "confidence": priority_result.confidence,
                "calculated_at": priority_result.calculated_at.isoformat()
            },
            "user_context": {
                "availability_score": user_context.availability_score,
                "attention_score": user_context.attention_score,
                "activity_level": user_context.activity_level.value
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error calculating priority: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/delivery/optimize', methods=['POST'])
@async_route
async def optimize_delivery():
    """Optimize notification delivery plan"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        # Validate required fields
        required_fields = ['user_id', 'notification_type', 'priority_score']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"{field} is required")
        
        user_id = data['user_id']
        notification_type = data['notification_type']
        priority_score = data['priority_score']
        preferred_channels = data.get('preferred_channels', ['push', 'in_app'])
        schedule_time = data.get('schedule_time')
        
        components = get_notification_components()
        
        # Get user context
        context_analyzer = components['context_analyzer']
        user_context = await context_analyzer.analyze_user_context(user_id)
        
        # Optimize delivery
        delivery_optimizer = components['delivery_optimizer']
        delivery_plan = await delivery_optimizer.optimize_delivery(
            user_id=user_id,
            notification_type=notification_type,
            priority_score=priority_score,
            user_context=user_context,
            preferred_channels=[NotificationChannel(ch) for ch in preferred_channels],
            schedule_time=datetime.fromisoformat(schedule_time) if schedule_time else None
        )
        
        return jsonify({
            "success": True,
            "delivery_plan": {
                "plan_id": delivery_plan.plan_id,
                "user_id": delivery_plan.user_id,
                "primary_channel": delivery_plan.primary_channel.value,
                "backup_channels": [ch.value for ch in delivery_plan.backup_channels],
                "delivery_time": delivery_plan.delivery_time.isoformat(),
                "estimated_delivery": delivery_plan.estimated_delivery.isoformat(),
                "retry_strategy": delivery_plan.retry_strategy,
                "optimization_factors": delivery_plan.optimization_factors,
                "confidence_score": delivery_plan.confidence_score,
                "created_at": delivery_plan.created_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error optimizing delivery: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/analytics', methods=['GET'])
@async_route
async def get_notification_analytics():
    """Get notification analytics and performance metrics"""
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        user_id = request.args.get('user_id')
        notification_type = request.args.get('notification_type')
        
        if not start_date:
            # Default to last 7 days
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        components = get_notification_components()
        delivery_optimizer = components['delivery_optimizer']
        
        # Get analytics data
        analytics = await delivery_optimizer.get_notification_analytics(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            notification_type=notification_type
        )
        
        return jsonify({
            "success": True,
            "analytics": analytics,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting notification analytics: {e}")
        return jsonify({"error": "Internal server error"}), 500


@notification_bp.route('/status', methods=['GET'])
def get_notification_status():
    """Get detailed status of all notification components"""
    try:
        components = get_notification_components()
        
        status = {
            "service": "notification_api",
            "status": "operational",
            "components": {
                "smart_prioritizer": "operational",
                "context_analyzer": "operational",
                "delivery_optimizer": "operational",
                "preference_manager": "operational"
            },
            "endpoints": {
                "health": "/api/notifications/health",
                "send_notification": "/api/notifications/send",
                "batch_notifications": "/api/notifications/batch",
                "user_context": "/api/notifications/users/{user_id}/context",
                "user_preferences": "/api/notifications/users/{user_id}/preferences",
                "notification_history": "/api/notifications/users/{user_id}/history",
                "calculate_priority": "/api/notifications/priority/calculate",
                "optimize_delivery": "/api/notifications/delivery/optimize",
                "analytics": "/api/notifications/analytics"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting notification status: {e}")
        return jsonify({
            "service": "notification_api",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# Error handlers
@notification_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@notification_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@notification_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500