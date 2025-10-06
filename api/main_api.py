#!/usr/bin/env python3
"""
Main Flask API for Business Dealer Intelligence System
RESTful API endpoints for the complete business intelligence service

This API provides:
- Health checking and system status
- User interaction processing
- Opportunity discovery and matching
- Recommendation generation
- Analytics and insights
- Notification management
- System administration

All endpoints support JSON request/response format.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

# Import API blueprints
from .intelligence_api import intelligence_bp
from .analytics_api import analytics_bp  
from .notification_api import notification_bp

logger = logging.getLogger(__name__)


def create_app(config, components):
    """Create and configure Flask application"""
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Store config and components in app context
    app.config['INTELLIGENCE_CONFIG'] = config
    app.config['COMPONENTS'] = components
    
    # Register API blueprints
    app.register_blueprint(intelligence_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(notification_bp)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found',
            'status': 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An internal server error occurred',
            'status': 500
        }), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'Business Dealer Intelligence',
                'version': '1.0.0'
            }
            
            # Check component health
            components = app.config['COMPONENTS']
            component_status = {}
            
            for name, component in components.items():
                try:
                    if hasattr(component, 'get_status'):
                        component_status[name] = component.get_status()
                    else:
                        component_status[name] = {'status': 'operational'}
                except Exception as e:
                    component_status[name] = {'status': 'error', 'error': str(e)}
            
            status['components'] = component_status
            return jsonify(status), 200
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }), 500
    
    # Root endpoint with service information
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint with service information"""
        return jsonify({
            'service': 'Business Dealer Intelligence System',
            'version': '1.0.0',
            'description': 'AI-powered business intelligence and recommendation system',
            'endpoints': {
                'health': '/health',
                'dashboard': '/dashboard',
                'opportunities': '/api/opportunities',
                'recommendations': '/api/recommendations',
                'analytics': '/api/analytics',
                'notifications': '/api/notifications',
                'users': '/api/users'
            },
            'documentation': '/docs',
            'timestamp': datetime.now().isoformat()
        })
    
    # Dashboard endpoint
    @app.route('/dashboard', methods=['GET'])
    def dashboard():
        """Simple dashboard view"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Dealer Intelligence Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
                .header { text-align: center; margin-bottom: 30px; }
                .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
                .status-card { background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }
                .status-card h3 { margin: 0 0 10px 0; color: #333; }
                .status-card p { margin: 5px 0; color: #666; }
                .status-healthy { border-left-color: #28a745; }
                .status-warning { border-left-color: #ffc107; }
                .status-error { border-left-color: #dc3545; }
                .api-section { margin-top: 30px; }
                .api-endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; }
                .api-endpoint code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
                .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 Business Dealer Intelligence Dashboard</h1>
                    <p>AI-Powered Business Intelligence & Recommendations</p>
                </div>
                
                <div class="status-grid">
                    <div class="status-card status-healthy">
                        <h3>🚀 Service Status</h3>
                        <p><strong>Status:</strong> Running</p>
                        <p><strong>Version:</strong> 1.0.0</p>
                        <p><strong>Uptime:</strong> Active</p>
                    </div>
                    
                    <div class="status-card status-healthy">
                        <h3>🧠 Intelligence Engine</h3>
                        <p><strong>Status:</strong> Operational</p>
                        <p><strong>Components:</strong> 4 Active</p>
                        <p><strong>Processing:</strong> Real-time</p>
                    </div>
                    
                    <div class="status-card status-healthy">
                        <h3>📊 Analytics</h3>
                        <p><strong>Status:</strong> Operational</p>
                        <p><strong>Metrics:</strong> Active</p>
                        <p><strong>Insights:</strong> AI-Generated</p>
                    </div>
                    
                    <div class="status-card status-healthy">
                        <h3>🔔 Notifications</h3>
                        <p><strong>Status:</strong> Operational</p>
                        <p><strong>Channels:</strong> Multi-channel</p>
                        <p><strong>Delivery:</strong> Optimized</p>
                    </div>
                </div>
                
                <div class="api-section">
                    <h2>🔗 API Endpoints</h2>
                    
                    <div class="api-endpoint">
                        <h4>Health Check</h4>
                        <p><code>GET /health</code> - Check service health and component status</p>
                    </div>
                    
                    <div class="api-endpoint">
                        <h4>Opportunities</h4>
                        <p><code>GET /api/opportunities</code> - Get business opportunities</p>
                        <p><code>POST /api/opportunities/match</code> - Match opportunities for user</p>
                    </div>
                    
                    <div class="api-endpoint">
                        <h4>Recommendations</h4>
                        <p><code>GET /api/recommendations</code> - Get personalized recommendations</p>
                        <p><code>POST /api/recommendations/generate</code> - Generate new recommendations</p>
                    </div>
                    
                    <div class="api-endpoint">
                        <h4>Analytics</h4>
                        <p><code>GET /api/analytics/metrics</code> - Get business metrics</p>
                        <p><code>GET /api/analytics/insights</code> - Get AI-generated insights</p>
                    </div>
                    
                    <div class="api-endpoint">
                        <h4>Notifications</h4>
                        <p><code>GET /api/notifications</code> - Get notification queue</p>
                        <p><code>POST /api/notifications/send</code> - Send notification</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Business Dealer Intelligence System v1.0.0 | Real-time Dashboard | {{ timestamp }}</p>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(() => location.reload(), 30000);
            </script>
        </body>
        </html>
        """
        return render_template_string(dashboard_html, timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # API Routes
    @app.route('/api/opportunities', methods=['GET'])
    def get_opportunities():
        """Get business opportunities"""
        try:
            user_id = request.args.get('user_id', 'demo_user')
            max_results = int(request.args.get('max_results', 10))
            
            # Demo data for now
            opportunities = [
                {
                    'id': 'opp_001',
                    'title': 'Strategic Partnership Opportunity',
                    'description': 'High-value partnership opportunity in the technology sector',
                    'score': 0.92,
                    'type': 'partnership',
                    'estimated_value': 250000,
                    'probability': 0.78,
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'opp_002',
                    'title': 'Market Expansion Opportunity',
                    'description': 'Opportunity to expand into emerging markets',
                    'score': 0.85,
                    'type': 'market_entry',
                    'estimated_value': 180000,
                    'probability': 0.65,
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'opp_003',
                    'title': 'Cost Reduction Initiative',
                    'description': 'Opportunity to reduce operational costs through automation',
                    'score': 0.76,
                    'type': 'cost_reduction',
                    'estimated_value': 120000,
                    'probability': 0.82,
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            return jsonify({
                'opportunities': opportunities[:max_results],
                'total': len(opportunities),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting opportunities: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/recommendations', methods=['GET'])
    def get_recommendations():
        """Get personalized recommendations"""
        try:
            user_id = request.args.get('user_id', 'demo_user')
            max_results = int(request.args.get('max_results', 5))
            
            # Demo data for now
            recommendations = [
                {
                    'id': 'rec_001',
                    'title': 'Invest in AI-Powered Analytics',
                    'description': 'Based on your current business patterns, investing in AI analytics could increase efficiency by 35%',
                    'score': 0.89,
                    'type': 'investment',
                    'urgency': 'high',
                    'expected_impact': 'high',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'rec_002',
                    'title': 'Expand Digital Marketing',
                    'description': 'Your target audience shows 40% higher engagement with digital channels',
                    'score': 0.82,
                    'type': 'marketing',
                    'urgency': 'medium',
                    'expected_impact': 'medium',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'rec_003',
                    'title': 'Optimize Supply Chain',
                    'description': 'Supply chain optimization could reduce costs by 15-20%',
                    'score': 0.75,
                    'type': 'optimization',
                    'urgency': 'low',
                    'expected_impact': 'high',
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            return jsonify({
                'recommendations': recommendations[:max_results],
                'total': len(recommendations),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analytics/metrics', methods=['GET'])
    def get_metrics():
        """Get business metrics"""
        try:
            time_period = request.args.get('period', '30d')
            
            # Demo metrics
            metrics = {
                'daily_active_users': {
                    'value': 1250,
                    'change': '+12.5%',
                    'trend': 'increasing'
                },
                'user_engagement_rate': {
                    'value': 0.73,
                    'change': '+5.2%',
                    'trend': 'increasing'
                },
                'opportunity_conversion_rate': {
                    'value': 0.28,
                    'change': '+8.1%',
                    'trend': 'increasing'
                },
                'recommendation_click_rate': {
                    'value': 0.45,
                    'change': '+3.7%',
                    'trend': 'stable'
                },
                'revenue_growth': {
                    'value': 0.15,
                    'change': '+2.3%',
                    'trend': 'increasing'
                }
            }
            
            return jsonify({
                'metrics': metrics,
                'period': time_period,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analytics/insights', methods=['GET'])
    def get_insights():
        """Get AI-generated insights"""
        try:
            # Demo insights
            insights = [
                {
                    'id': 'insight_001',
                    'title': 'User Engagement Peak Detected',
                    'description': 'User engagement has increased by 25% during afternoon hours (2-4 PM)',
                    'type': 'trend',
                    'urgency': 'medium',
                    'confidence': 0.87,
                    'recommendations': [
                        'Schedule important notifications during peak hours',
                        'Increase content delivery during 2-4 PM window',
                        'Consider targeted campaigns during high engagement periods'
                    ],
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'insight_002',
                    'title': 'Opportunity Matching Accuracy Improved',
                    'description': 'The AI matching algorithm has improved accuracy by 18% over the last month',
                    'type': 'performance',
                    'urgency': 'low',
                    'confidence': 0.92,
                    'recommendations': [
                        'Continue current matching strategy',
                        'Collect more user feedback to further improve accuracy',
                        'Consider expanding matching criteria'
                    ],
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'insight_003',
                    'title': 'Seasonal Pattern Identified',
                    'description': 'Business opportunities show 40% higher success rates during Q4',
                    'type': 'seasonal',
                    'urgency': 'high',
                    'confidence': 0.83,
                    'recommendations': [
                        'Prepare Q4 opportunity pipeline',
                        'Increase resource allocation for Q4',
                        'Develop Q4-specific marketing strategy'
                    ],
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            return jsonify({
                'insights': insights,
                'total': len(insights),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/notifications', methods=['GET'])
    def get_notifications():
        """Get notification queue"""
        try:
            user_id = request.args.get('user_id', 'demo_user')
            
            # Demo notifications
            notifications = [
                {
                    'id': 'notif_001',
                    'title': 'New High-Value Opportunity',
                    'message': 'A new partnership opportunity worth $250K has been identified',
                    'type': 'opportunity',
                    'priority': 'high',
                    'status': 'pending',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'notif_002',
                    'title': 'Weekly Analytics Report',
                    'message': 'Your weekly business intelligence report is ready',
                    'type': 'report',
                    'priority': 'medium',
                    'status': 'delivered',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 'notif_003',
                    'title': 'AI Insight Alert',
                    'message': 'New trend detected: User engagement peak at 2-4 PM',
                    'type': 'insight',
                    'priority': 'medium',
                    'status': 'pending',
                    'created_at': datetime.now().isoformat()
                }
            ]
            
            return jsonify({
                'notifications': notifications,
                'total': len(notifications),
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting notifications: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/users/<user_id>/profile', methods=['GET'])
    def get_user_profile(user_id):
        """Get user profile and preferences"""
        try:
            # Demo user profile
            profile = {
                'user_id': user_id,
                'name': 'Demo User',
                'email': 'demo@example.com',
                'preferences': {
                    'notification_channels': ['email', 'push'],
                    'business_interests': ['technology', 'finance', 'marketing'],
                    'notification_frequency': 'daily',
                    'opportunity_types': ['partnership', 'investment', 'market_entry']
                },
                'activity_summary': {
                    'total_opportunities_viewed': 45,
                    'total_recommendations_clicked': 23,
                    'last_active': datetime.now().isoformat(),
                    'engagement_score': 0.78
                },
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': datetime.now().isoformat()
            }
            
            return jsonify(profile)
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/docs', methods=['GET'])
    def api_documentation():
        """API documentation"""
        docs = {
            'title': 'Business Dealer Intelligence API',
            'version': '1.0.0',
            'description': 'RESTful API for AI-powered business intelligence and recommendations',
            'base_url': request.base_url.replace('/docs', ''),
            'endpoints': [
                {
                    'path': '/health',
                    'method': 'GET',
                    'description': 'Health check and system status',
                    'response': 'Service health information'
                },
                {
                    'path': '/api/opportunities',
                    'method': 'GET',
                    'description': 'Get business opportunities',
                    'parameters': ['user_id', 'max_results'],
                    'response': 'List of business opportunities'
                },
                {
                    'path': '/api/recommendations',
                    'method': 'GET',
                    'description': 'Get personalized recommendations',
                    'parameters': ['user_id', 'max_results'],
                    'response': 'List of personalized recommendations'
                },
                {
                    'path': '/api/analytics/metrics',
                    'method': 'GET',
                    'description': 'Get business metrics',
                    'parameters': ['period'],
                    'response': 'Business metrics and KPIs'
                },
                {
                    'path': '/api/analytics/insights',
                    'method': 'GET',
                    'description': 'Get AI-generated insights',
                    'parameters': [],
                    'response': 'AI-generated business insights'
                },
                {
                    'path': '/api/notifications',
                    'method': 'GET',
                    'description': 'Get notification queue',
                    'parameters': ['user_id'],
                    'response': 'List of notifications'
                },
                {
                    'path': '/api/users/{user_id}/profile',
                    'method': 'GET',
                    'description': 'Get user profile',
                    'parameters': ['user_id'],
                    'response': 'User profile and preferences'
                }
            ]
        }
        
        return jsonify(docs)
    
    return app