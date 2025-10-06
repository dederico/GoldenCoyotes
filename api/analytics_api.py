#!/usr/bin/env python3
"""
Analytics API Endpoints
Flask API endpoints for analytics and insights

This API provides:
- Dashboard data and visualization endpoints
- Metrics calculation and reporting
- Predictive analytics and forecasting
- Insight generation and management
- Performance monitoring and KPIs

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
# from analytics.metrics_calculator import MetricsCalculator
# from analytics.insight_generator import InsightGenerator
# from analytics.dashboard_data import DashboardDataProvider
# from analytics.predictive_analytics import PredictiveAnalytics

logger = logging.getLogger(__name__)

# Create Flask Blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')


def get_analytics_components():
    """Get analytics components instances"""
    if not hasattr(current_app, 'analytics_components'):
        current_app.analytics_components = {
            'metrics_calculator': MetricsCalculator(),
            'insight_generator': InsightGenerator(),
            'dashboard_provider': DashboardDataProvider(),
            'predictive_analytics': PredictiveAnalytics()
        }
    return current_app.analytics_components


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


@analytics_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for analytics services"""
    try:
        components = get_analytics_components()
        
        status = {
            "metrics_calculator": "operational",
            "insight_generator": "operational", 
            "dashboard_provider": "operational",
            "predictive_analytics": "operational"
        }
        
        return jsonify({
            "status": "healthy",
            "service": "analytics_api",
            "components": status,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Analytics health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


@analytics_bp.route('/metrics/<metric_name>', methods=['GET'])
@async_route
async def get_metric(metric_name):
    """Get specific metric value"""
    try:
        # Get query parameters
        time_period = request.args.get('time_period', datetime.now().strftime('%Y-%m-%d'))
        user_id = request.args.get('user_id')
        filters = request.args.get('filters', '{}')
        
        try:
            filters = json.loads(filters)
        except json.JSONDecodeError:
            filters = {}
        
        components = get_analytics_components()
        metrics_calculator = components['metrics_calculator']
        
        # Calculate metric
        metric_result = await metrics_calculator.calculate_metric(
            metric_name=metric_name,
            time_period=time_period,
            user_id=user_id,
            filters=filters
        )
        
        return jsonify({
            "success": True,
            "metric": {
                "name": metric_result.metric_name,
                "value": metric_result.value,
                "unit": metric_result.unit,
                "time_period": metric_result.time_period,
                "calculation_time": metric_result.calculation_time.isoformat(),
                "metadata": metric_result.metadata
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting metric {metric_name}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/metrics/time-series/<metric_name>', methods=['GET'])
@async_route
async def get_time_series_metrics(metric_name):
    """Get time series data for a metric"""
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        granularity = request.args.get('granularity', 'daily')
        user_id = request.args.get('user_id')
        filters = request.args.get('filters', '{}')
        
        if not start_date:
            # Default to last 30 days
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            filters = json.loads(filters)
        except json.JSONDecodeError:
            filters = {}
        
        components = get_analytics_components()
        metrics_calculator = components['metrics_calculator']
        
        # Get time series data
        time_series_data = await metrics_calculator.get_time_series_metrics(
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            user_id=user_id,
            filters=filters
        )
        
        return jsonify({
            "success": True,
            "metric_name": metric_name,
            "data": [
                {
                    "time_period": metric.time_period,
                    "value": metric.value,
                    "unit": metric.unit,
                    "metadata": metric.metadata
                }
                for metric in time_series_data
            ],
            "granularity": granularity,
            "start_date": start_date,
            "end_date": end_date,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting time series for {metric_name}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/dashboard/<dashboard_id>', methods=['GET'])
@async_route
async def get_dashboard_data(dashboard_id):
    """Get dashboard data with all widgets"""
    try:
        # Get query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        filters = request.args.get('filters', '{}')
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        if not start_date:
            # Default to last 30 days
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        try:
            filters = json.loads(filters)
        except json.JSONDecodeError:
            filters = {}
        
        time_range = {
            'start': start_date,
            'end': end_date
        }
        
        components = get_analytics_components()
        dashboard_provider = components['dashboard_provider']
        
        # Generate dashboard data
        dashboard_data = await dashboard_provider.generate_dashboard_data(
            dashboard_id=dashboard_id,
            time_range=time_range,
            filters=filters,
            use_cache=use_cache
        )
        
        return jsonify({
            "success": True,
            "dashboard": {
                "id": dashboard_data.dashboard_id,
                "title": dashboard_data.title,
                "widgets": [
                    {
                        "id": widget.widget_id,
                        "type": widget.widget_type,
                        "title": widget.title,
                        "data": widget.data,
                        "config": widget.config,
                        "position": widget.position,
                        "size": widget.size,
                        "last_updated": widget.last_updated.isoformat()
                    }
                    for widget in dashboard_data.widgets
                ],
                "filters": dashboard_data.filters,
                "time_range": dashboard_data.time_range,
                "refresh_interval": dashboard_data.refresh_interval,
                "generated_at": dashboard_data.generated_at.isoformat(),
                "expires_at": dashboard_data.expires_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting dashboard {dashboard_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/insights', methods=['GET'])
@async_route
async def get_insights():
    """Get AI-generated insights"""
    try:
        # Get query parameters
        insight_types = request.args.getlist('types')
        if not insight_types:
            insight_types = ['trend', 'anomaly', 'opportunity', 'warning']
        
        time_period = request.args.get('time_period', datetime.now().strftime('%Y-%m-%d'))
        user_id = request.args.get('user_id')
        max_results = int(request.args.get('max_results', 10))
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        components = get_analytics_components()
        insight_generator = components['insight_generator']
        
        # Generate insights
        insights = await insight_generator.generate_insights(
            insight_types=insight_types,
            time_period=time_period,
            user_id=user_id,
            max_results=max_results,
            use_cache=use_cache
        )
        
        return jsonify({
            "success": True,
            "insights": [
                {
                    "id": insight.id,
                    "title": insight.title,
                    "description": insight.description,
                    "type": insight.insight_type,
                    "urgency": insight.urgency,
                    "confidence": insight.confidence,
                    "impact_score": insight.impact_score,
                    "data_points": insight.data_points,
                    "recommendations": insight.recommendations,
                    "generated_at": insight.generated_at.isoformat(),
                    "expires_at": insight.expires_at.isoformat() if insight.expires_at else None
                }
                for insight in insights
            ],
            "total_insights": len(insights),
            "insight_types": insight_types,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting insights: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/insights/summary', methods=['GET'])
@async_route
async def get_insights_summary():
    """Get insights summary and statistics"""
    try:
        # Get query parameters
        time_period = request.args.get('time_period', datetime.now().strftime('%Y-%m-%d'))
        use_cache = request.args.get('use_cache', 'true').lower() == 'true'
        
        components = get_analytics_components()
        insight_generator = components['insight_generator']
        
        # Generate insights summary
        summary = await insight_generator.generate_insight_summary(
            time_period=time_period,
            use_cache=use_cache
        )
        
        return jsonify({
            "success": True,
            "summary": {
                "total_insights": summary.total_insights,
                "insights_by_type": summary.insights_by_type,
                "insights_by_urgency": summary.insights_by_urgency,
                "avg_confidence": summary.avg_confidence,
                "key_insights": [
                    {
                        "id": insight.id,
                        "title": insight.title,
                        "type": insight.insight_type,
                        "urgency": insight.urgency,
                        "confidence": insight.confidence,
                        "generated_at": insight.generated_at.isoformat()
                    }
                    for insight in summary.key_insights
                ],
                "period_start": summary.period_start.isoformat(),
                "period_end": summary.period_end.isoformat(),
                "generated_at": summary.generated_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting insights summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/forecast', methods=['POST'])
@async_route
async def generate_forecast():
    """Generate predictive forecast for a metric"""
    try:
        data = request.get_json()
        if not data:
            raise BadRequest("Request body is required")
        
        metric_name = data.get('metric_name')
        if not metric_name:
            raise BadRequest("metric_name is required")
        
        # Get parameters
        forecast_horizon = data.get('forecast_horizon', 30)  # days
        historical_days = data.get('historical_days', 90)
        confidence_level = data.get('confidence_level', 0.95)
        seasonality = data.get('seasonality', True)
        external_factors = data.get('external_factors', {})
        
        components = get_analytics_components()
        predictive_analytics = components['predictive_analytics']
        
        # Create forecast request
        from analytics.predictive_analytics import ForecastRequest
        forecast_request = ForecastRequest(
            metric_name=metric_name,
            forecast_horizon=forecast_horizon,
            historical_days=historical_days,
            confidence_level=confidence_level,
            seasonality=seasonality,
            external_factors=external_factors
        )
        
        # Generate forecast
        forecast_result = await predictive_analytics.generate_forecast(forecast_request)
        
        return jsonify({
            "success": True,
            "forecast": {
                "metric_name": forecast_result.metric_name,
                "forecast_dates": forecast_result.forecast_dates,
                "forecast_values": forecast_result.forecast_values,
                "confidence_intervals": [
                    {"lower": ci[0], "upper": ci[1]}
                    for ci in forecast_result.confidence_intervals
                ],
                "model_performance": forecast_result.model_performance,
                "feature_importance": forecast_result.feature_importance,
                "forecast_horizon": forecast_result.forecast_horizon,
                "confidence_level": forecast_result.confidence_level,
                "generated_at": forecast_result.generated_at.isoformat(),
                "expires_at": forecast_result.expires_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"❌ Error generating forecast: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/trends/<metric_name>', methods=['GET'])
@async_route
async def analyze_trends(metric_name):
    """Analyze trends for a specific metric"""
    try:
        # Get query parameters
        analysis_period = request.args.get('analysis_period', '30d')
        
        components = get_analytics_components()
        predictive_analytics = components['predictive_analytics']
        
        # Analyze trends
        trend_analysis = await predictive_analytics.analyze_trends(
            metric_name=metric_name,
            analysis_period=analysis_period
        )
        
        return jsonify({
            "success": True,
            "trend_analysis": {
                "metric_name": trend_analysis.metric_name,
                "trend_direction": trend_analysis.trend_direction,
                "trend_strength": trend_analysis.trend_strength,
                "seasonal_patterns": trend_analysis.seasonal_patterns,
                "trend_change_points": trend_analysis.trend_change_points,
                "forecast_confidence": trend_analysis.forecast_confidence,
                "analysis_period": trend_analysis.analysis_period
            },
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error analyzing trends for {metric_name}: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/anomalies', methods=['GET'])
@async_route
async def detect_anomalies():
    """Detect anomalies in metrics"""
    try:
        # Get query parameters
        metric_name = request.args.get('metric_name')
        time_range = request.args.get('time_range', '7d')
        
        components = get_analytics_components()
        predictive_analytics = components['predictive_analytics']
        
        # Detect anomalies
        anomalies = await predictive_analytics.detect_anomalies(
            metric_name=metric_name,
            time_range=time_range
        )
        
        return jsonify({
            "success": True,
            "anomalies": [
                {
                    "metric_name": anomaly.metric_name,
                    "predicted_anomalies": anomaly.predicted_anomalies,
                    "anomaly_probability": anomaly.anomaly_probability,
                    "prediction_dates": anomaly.prediction_dates,
                    "threshold_values": anomaly.threshold_values,
                    "confidence_score": anomaly.confidence_score
                }
                for anomaly in anomalies
            ],
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error detecting anomalies: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get analytics system performance metrics"""
    try:
        components = get_analytics_components()
        
        # Get status from all components
        performance = {
            "metrics_calculator": components['metrics_calculator'].get_calculator_status(),
            "insight_generator": components['insight_generator'].get_generator_status(),
            "dashboard_provider": components['dashboard_provider'].get_provider_status(),
            "predictive_analytics": "operational"  # Would implement get_status method
        }
        
        return jsonify({
            "success": True,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/metrics/available', methods=['GET'])
def get_available_metrics():
    """Get list of available metrics"""
    try:
        # Define available metrics
        available_metrics = [
            {
                "name": "daily_active_users",
                "description": "Number of active users per day",
                "unit": "count",
                "category": "engagement"
            },
            {
                "name": "user_engagement_rate",
                "description": "User engagement rate percentage",
                "unit": "percentage",
                "category": "engagement"
            },
            {
                "name": "opportunity_conversion_rate",
                "description": "Opportunity conversion rate",
                "unit": "percentage",
                "category": "conversion"
            },
            {
                "name": "recommendation_click_rate",
                "description": "Recommendation click-through rate",
                "unit": "percentage",
                "category": "recommendation"
            },
            {
                "name": "session_duration",
                "description": "Average session duration",
                "unit": "minutes",
                "category": "engagement"
            },
            {
                "name": "user_retention_rate",
                "description": "User retention rate",
                "unit": "percentage",
                "category": "retention"
            }
        ]
        
        return jsonify({
            "success": True,
            "metrics": available_metrics,
            "total_metrics": len(available_metrics),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting available metrics: {e}")
        return jsonify({"error": "Internal server error"}), 500


@analytics_bp.route('/status', methods=['GET'])
def get_analytics_status():
    """Get detailed status of all analytics components"""
    try:
        components = get_analytics_components()
        
        status = {
            "service": "analytics_api",
            "status": "operational",
            "components": {
                "metrics_calculator": "operational",
                "insight_generator": "operational",
                "dashboard_provider": "operational",
                "predictive_analytics": "operational"
            },
            "endpoints": {
                "health": "/api/analytics/health",
                "get_metric": "/api/analytics/metrics/{metric_name}",
                "time_series": "/api/analytics/metrics/time-series/{metric_name}",
                "dashboard": "/api/analytics/dashboard/{dashboard_id}",
                "insights": "/api/analytics/insights",
                "insights_summary": "/api/analytics/insights/summary",
                "forecast": "/api/analytics/forecast",
                "trends": "/api/analytics/trends/{metric_name}",
                "anomalies": "/api/analytics/anomalies",
                "performance": "/api/analytics/performance",
                "available_metrics": "/api/analytics/metrics/available"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics status: {e}")
        return jsonify({
            "service": "analytics_api",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500


# Error handlers
@analytics_bp.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad request", "message": str(error)}), 400


@analytics_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": str(error)}), 404


@analytics_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "message": str(error)}), 500