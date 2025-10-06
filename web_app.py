#!/usr/bin/env python3
"""
Full Web Application for Business Dealer Intelligence System
Complete UI with forms, dashboards, and interactive features
"""

import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session
from flask_cors import CORS
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_web_app():
    """Create Flask web application with full UI"""
    app = Flask(__name__)
    app.secret_key = 'business_intelligence_secret_key_' + str(uuid.uuid4())
    CORS(app)
    
    # Mock data storage (in production, this would be a database)
    app.data = {
        'users': {},
        'opportunities': {},
        'interactions': [],
        'notifications': [],
        'metrics': {}
    }
    
    # Initialize with some sample data
    initialize_sample_data(app)
    
    @app.route('/')
    def index():
        """Main dashboard"""
        return render_template_string(DASHBOARD_TEMPLATE)
    
    @app.route('/users')
    def users_page():
        """User management page"""
        return render_template_string(USERS_TEMPLATE, users=app.data['users'])
    
    @app.route('/opportunities')
    def opportunities_page():
        """Opportunities management page"""
        return render_template_string(OPPORTUNITIES_TEMPLATE, opportunities=app.data['opportunities'])
    
    @app.route('/analytics')
    def analytics_page():
        """Analytics dashboard"""
        metrics = calculate_dashboard_metrics(app)
        return render_template_string(ANALYTICS_TEMPLATE, metrics=metrics)
    
    @app.route('/notifications')
    def notifications_page():
        """Notifications management"""
        return render_template_string(NOTIFICATIONS_TEMPLATE, notifications=app.data['notifications'])
    
    @app.route('/settings')
    def settings_page():
        """Settings page"""
        return render_template_string(SETTINGS_TEMPLATE)
    
    # API endpoints for form submissions
    @app.route('/api/users', methods=['POST'])
    def create_user():
        """Create new user"""
        data = request.get_json() or request.form.to_dict()
        user_id = str(uuid.uuid4())
        
        app.data['users'][user_id] = {
            'id': user_id,
            'name': data.get('name'),
            'email': data.get('email'),
            'industry': data.get('industry'),
            'location': data.get('location'),
            'interests': data.get('interests', '').split(','),
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        flash('User created successfully!', 'success')
        return jsonify({'success': True, 'user_id': user_id})
    
    @app.route('/api/opportunities', methods=['POST'])
    def create_opportunity():
        """Create new opportunity"""
        data = request.get_json() or request.form.to_dict()
        opp_id = str(uuid.uuid4())
        
        app.data['opportunities'][opp_id] = {
            'id': opp_id,
            'title': data.get('title'),
            'description': data.get('description'),
            'type': data.get('type'),
            'industry': data.get('industry'),
            'budget': data.get('budget'),
            'location': data.get('location'),
            'deadline': data.get('deadline'),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'views': 0,
            'matches': 0
        }
        
        flash('Opportunity created successfully!', 'success')
        return jsonify({'success': True, 'opportunity_id': opp_id})
    
    @app.route('/api/interactions', methods=['POST'])
    def record_interaction():
        """Record user interaction"""
        data = request.get_json() or request.form.to_dict()
        
        interaction = {
            'id': str(uuid.uuid4()),
            'user_id': data.get('user_id'),
            'opportunity_id': data.get('opportunity_id'),
            'type': data.get('type'),
            'timestamp': datetime.now().isoformat(),
            'metadata': data.get('metadata', {})
        }
        
        app.data['interactions'].append(interaction)
        
        # Update opportunity view count
        if data.get('type') == 'view' and data.get('opportunity_id') in app.data['opportunities']:
            app.data['opportunities'][data.get('opportunity_id')]['views'] += 1
        
        flash('Interaction recorded!', 'success')
        return jsonify({'success': True, 'interaction_id': interaction['id']})
    
    @app.route('/api/recommendations/<user_id>')
    def get_recommendations(user_id):
        """Get recommendations for user"""
        if user_id not in app.data['users']:
            return jsonify({'error': 'User not found'}), 404
        
        user = app.data['users'][user_id]
        recommendations = []
        
        # Simple recommendation logic based on user interests and industry
        for opp_id, opp in app.data['opportunities'].items():
            score = 0
            
            # Industry match
            if user.get('industry') == opp.get('industry'):
                score += 0.4
            
            # Interest match
            user_interests = user.get('interests', [])
            if any(interest.lower() in opp.get('description', '').lower() for interest in user_interests):
                score += 0.3
            
            # Location preference
            if user.get('location') == opp.get('location'):
                score += 0.2
            
            # Recency boost
            opp_date = datetime.fromisoformat(opp['created_at'])
            if (datetime.now() - opp_date).days < 7:
                score += 0.1
            
            if score > 0.3:  # Threshold for recommendations
                recommendations.append({
                    'opportunity': opp,
                    'score': round(score, 2),
                    'reasoning': f"Matches your {user.get('industry', 'industry')} background"
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations[:10]
        })
    
    @app.route('/api/send-notification', methods=['POST'])
    def send_notification():
        """Send notification"""
        data = request.get_json() or request.form.to_dict()
        
        notification = {
            'id': str(uuid.uuid4()),
            'user_id': data.get('user_id'),
            'title': data.get('title'),
            'message': data.get('message'),
            'type': data.get('type', 'info'),
            'channel': data.get('channel', 'in_app'),
            'sent_at': datetime.now().isoformat(),
            'status': 'sent'
        }
        
        app.data['notifications'].append(notification)
        
        flash('Notification sent successfully!', 'success')
        return jsonify({'success': True, 'notification_id': notification['id']})
    
    @app.route('/api/metrics')
    def get_metrics():
        """Get current metrics"""
        return jsonify(calculate_dashboard_metrics(app))
    
    @app.route('/api/user-profile/<user_id>')
    def get_user_profile(user_id):
        """Get user profile with interactions"""
        if user_id not in app.data['users']:
            return jsonify({'error': 'User not found'}), 404
        
        user = app.data['users'][user_id]
        user_interactions = [i for i in app.data['interactions'] if i['user_id'] == user_id]
        
        return jsonify({
            'user': user,
            'interactions': user_interactions,
            'interaction_count': len(user_interactions)
        })
    
    return app

def initialize_sample_data(app):
    """Initialize with sample data for demonstration"""
    # Sample users
    sample_users = [
        {
            'id': 'user1',
            'name': 'John Smith',
            'email': 'john@example.com',
            'industry': 'Technology',
            'location': 'San Francisco',
            'interests': ['AI', 'Startups', 'SaaS'],
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        },
        {
            'id': 'user2', 
            'name': 'Sarah Johnson',
            'email': 'sarah@example.com',
            'industry': 'Healthcare',
            'location': 'New York',
            'interests': ['HealthTech', 'Innovation', 'Digital Health'],
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
    ]
    
    for user in sample_users:
        app.data['users'][user['id']] = user
    
    # Sample opportunities
    sample_opportunities = [
        {
            'id': 'opp1',
            'title': 'AI Startup Seeks Technical Co-founder',
            'description': 'Revolutionary AI platform for healthcare needs experienced CTO',
            'type': 'partnership',
            'industry': 'Technology',
            'budget': '$50,000-100,000',
            'location': 'San Francisco',
            'deadline': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'views': 15,
            'matches': 3
        },
        {
            'id': 'opp2',
            'title': 'HealthTech Innovation Partnership',
            'description': 'Digital health platform seeking strategic partnerships with hospitals',
            'type': 'partnership',
            'industry': 'Healthcare',
            'budget': '$100,000+',
            'location': 'New York',
            'deadline': (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'views': 23,
            'matches': 7
        }
    ]
    
    for opp in sample_opportunities:
        app.data['opportunities'][opp['id']] = opp

def calculate_dashboard_metrics(app):
    """Calculate metrics for dashboard"""
    total_users = len(app.data['users'])
    total_opportunities = len(app.data['opportunities'])
    total_interactions = len(app.data['interactions'])
    total_notifications = len(app.data['notifications'])
    
    # Calculate engagement rate
    if total_users > 0 and total_opportunities > 0:
        engagement_rate = (total_interactions / (total_users * total_opportunities)) * 100
    else:
        engagement_rate = 0
    
    # Recent activity (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)
    recent_interactions = [
        i for i in app.data['interactions']
        if datetime.fromisoformat(i['timestamp']) > week_ago
    ]
    
    return {
        'total_users': total_users,
        'total_opportunities': total_opportunities,
        'total_interactions': total_interactions,
        'total_notifications': total_notifications,
        'engagement_rate': round(engagement_rate, 2),
        'recent_activity': len(recent_interactions),
        'active_opportunities': len([o for o in app.data['opportunities'].values() if o['status'] == 'active'])
    }

# HTML Templates
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Dealer Intelligence Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-5px); }
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white active" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-tachometer-alt"></i> Coyotes Dashboard</h1>
                    <span class="badge bg-success fs-6">System Online</span>
                </div>
                
                <!-- Metrics Cards -->
                <div class="row mb-4" id="metrics-cards">
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-users fa-3x text-primary mb-3"></i>
                                <h5 class="card-title">Total Users</h5>
                                <h2 class="text-primary" id="total-users">Loading...</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-briefcase fa-3x text-success mb-3"></i>
                                <h5 class="card-title">Opportunities</h5>
                                <h2 class="text-success" id="total-opportunities">Loading...</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-mouse-pointer fa-3x text-info mb-3"></i>
                                <h5 class="card-title">Interactions</h5>
                                <h2 class="text-info" id="total-interactions">Loading...</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-chart-line fa-3x text-warning mb-3"></i>
                                <h5 class="card-title">Engagement Rate</h5>
                                <h2 class="text-warning" id="engagement-rate">Loading...</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Quick Actions -->
                <div class="row mb-4">
                    <div class="col-md-12">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-rocket"></i> Quick Actions</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3 mb-2">
                                        <a href="/users" class="btn btn-outline-primary btn-lg w-100">
                                            <i class="fas fa-user-plus"></i> Add User
                                        </a>
                                    </div>
                                    <div class="col-md-3 mb-2">
                                        <a href="/opportunities" class="btn btn-outline-success btn-lg w-100">
                                            <i class="fas fa-plus"></i> New Opportunity
                                        </a>
                                    </div>
                                    <div class="col-md-3 mb-2">
                                        <a href="/analytics" class="btn btn-outline-info btn-lg w-100">
                                            <i class="fas fa-chart-bar"></i> View Analytics
                                        </a>
                                    </div>
                                    <div class="col-md-3 mb-2">
                                        <a href="/notifications" class="btn btn-outline-warning btn-lg w-100">
                                            <i class="fas fa-bell"></i> Send Notification
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Activity -->
                <div class="row">
                    <div class="col-md-12">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0"><i class="fas fa-clock"></i> System Status</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <h6><i class="fas fa-server"></i> Service Health</h6>
                                        <div class="d-flex justify-content-between">
                                            <span>Intelligence API</span>
                                            <span class="badge bg-success">Online</span>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span>Analytics API</span>
                                            <span class="badge bg-success">Online</span>
                                        </div>
                                        <div class="d-flex justify-content-between">
                                            <span>Notifications API</span>
                                            <span class="badge bg-success">Online</span>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <h6><i class="fas fa-activity"></i> Recent Activity</h6>
                                        <p class="text-muted">Last 7 days: <span id="recent-activity" class="fw-bold">Loading...</span> interactions</p>
                                        <p class="text-muted">Active opportunities: <span id="active-opportunities" class="fw-bold">Loading...</span></p>
                                        <p class="text-muted">System uptime: <span class="fw-bold text-success">99.9%</span></p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Load metrics on page load
        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();
                
                document.getElementById('total-users').textContent = metrics.total_users;
                document.getElementById('total-opportunities').textContent = metrics.total_opportunities;
                document.getElementById('total-interactions').textContent = metrics.total_interactions;
                document.getElementById('engagement-rate').textContent = metrics.engagement_rate + '%';
                document.getElementById('recent-activity').textContent = metrics.recent_activity;
                document.getElementById('active-opportunities').textContent = metrics.active_opportunities;
            } catch (error) {
                console.error('Error loading metrics:', error);
            }
        }
        
        // Load metrics when page loads
        document.addEventListener('DOMContentLoaded', loadMetrics);
        
        // Refresh metrics every 30 seconds
        setInterval(loadMetrics, 30000);
    </script>
</body>
</html>
'''

USERS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Management - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white active" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-users"></i> User Management</h1>
                    <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#addUserModal">
                        <i class="fas fa-plus"></i> Add New User
                    </button>
                </div>
                
                <!-- Users Table -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Registered Users</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Name</th>
                                        <th>Email</th>
                                        <th>Industry</th>
                                        <th>Location</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for user_id, user in users.items() %}
                                    <tr>
                                        <td>{{ user.name }}</td>
                                        <td>{{ user.email }}</td>
                                        <td><span class="badge bg-info">{{ user.industry }}</span></td>
                                        <td>{{ user.location }}</td>
                                        <td><span class="badge bg-success">{{ user.status }}</span></td>
                                        <td>
                                            <button class="btn btn-sm btn-outline-primary" onclick="viewProfile('{{ user_id }}')">
                                                <i class="fas fa-eye"></i> View
                                            </button>
                                            <button class="btn btn-sm btn-outline-info" onclick="getRecommendations('{{ user_id }}')">
                                                <i class="fas fa-magic"></i> Recommendations
                                            </button>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Add User Modal -->
    <div class="modal fade" id="addUserModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Add New User</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <form id="addUserForm">
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Name</label>
                            <input type="text" class="form-control" name="name" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Email</label>
                            <input type="email" class="form-control" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Industry</label>
                            <select class="form-control" name="industry" required>
                                <option value="">Select Industry</option>
                                <option value="Technology">Technology</option>
                                <option value="Healthcare">Healthcare</option>
                                <option value="Finance">Finance</option>
                                <option value="Education">Education</option>
                                <option value="Manufacturing">Manufacturing</option>
                                <option value="Retail">Retail</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Location</label>
                            <input type="text" class="form-control" name="location" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Interests (comma-separated)</label>
                            <input type="text" class="form-control" name="interests" placeholder="AI, Startups, SaaS">
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary">Create User</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <!-- User Profile Modal -->
    <div class="modal fade" id="userProfileModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">User Profile</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="userProfileContent">
                    Loading...
                </div>
            </div>
        </div>
    </div>
    
    <!-- Recommendations Modal -->
    <div class="modal fade" id="recommendationsModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">User Recommendations</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="recommendationsContent">
                    Loading...
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Handle form submission
        document.getElementById('addUserForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/users', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    location.reload();
                } else {
                    alert('Error creating user');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // View user profile
        async function viewProfile(userId) {
            try {
                const response = await fetch(`/api/user-profile/${userId}`);
                const data = await response.json();
                
                const content = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>User Information</h6>
                            <p><strong>Name:</strong> ${data.user.name}</p>
                            <p><strong>Email:</strong> ${data.user.email}</p>
                            <p><strong>Industry:</strong> ${data.user.industry}</p>
                            <p><strong>Location:</strong> ${data.user.location}</p>
                            <p><strong>Interests:</strong> ${data.user.interests.join(', ')}</p>
                        </div>
                        <div class="col-md-6">
                            <h6>Activity Summary</h6>
                            <p><strong>Total Interactions:</strong> ${data.interaction_count}</p>
                            <p><strong>Member Since:</strong> ${new Date(data.user.created_at).toLocaleDateString()}</p>
                            <p><strong>Status:</strong> <span class="badge bg-success">${data.user.status}</span></p>
                        </div>
                    </div>
                `;
                
                document.getElementById('userProfileContent').innerHTML = content;
                new bootstrap.Modal(document.getElementById('userProfileModal')).show();
            } catch (error) {
                alert('Error loading profile: ' + error.message);
            }
        }
        
        // Get user recommendations
        async function getRecommendations(userId) {
            try {
                const response = await fetch(`/api/recommendations/${userId}`);
                const data = await response.json();
                
                let content = '<h6>Recommended Opportunities</h6>';
                
                if (data.recommendations.length > 0) {
                    content += '<div class="list-group">';
                    data.recommendations.forEach(rec => {
                        content += `
                            <div class="list-group-item">
                                <div class="d-flex w-100 justify-content-between">
                                    <h6 class="mb-1">${rec.opportunity.title}</h6>
                                    <span class="badge bg-primary">${(rec.score * 100).toFixed(0)}% match</span>
                                </div>
                                <p class="mb-1">${rec.opportunity.description}</p>
                                <small class="text-muted">${rec.reasoning}</small>
                            </div>
                        `;
                    });
                    content += '</div>';
                } else {
                    content += '<p class="text-muted">No recommendations available for this user.</p>';
                }
                
                document.getElementById('recommendationsContent').innerHTML = content;
                new bootstrap.Modal(document.getElementById('recommendationsModal')).show();
            } catch (error) {
                alert('Error loading recommendations: ' + error.message);
            }
        }
    </script>
</body>
</html>
'''

OPPORTUNITIES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Opportunities - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .opportunity-card { transition: transform 0.2s; }
        .opportunity-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white active" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-briefcase"></i> Business Opportunities</h1>
                    <button class="btn btn-success" data-bs-toggle="modal" data-bs-target="#addOpportunityModal">
                        <i class="fas fa-plus"></i> Create Opportunity
                    </button>
                </div>
                
                <!-- Opportunities Grid -->
                <div class="row">
                    {% for opp_id, opp in opportunities.items() %}
                    <div class="col-md-6 col-lg-4 mb-4">
                        <div class="card opportunity-card h-100 border-0 shadow-sm">
                            <div class="card-header bg-light">
                                <div class="d-flex justify-content-between align-items-center">
                                    <span class="badge bg-{{ 'primary' if opp.type == 'partnership' else 'success' if opp.type == 'buyer' else 'info' }}">
                                        {{ opp.type.title() }}
                                    </span>
                                    <small class="text-muted">{{ opp.industry }}</small>
                                </div>
                            </div>
                            <div class="card-body">
                                <h5 class="card-title">{{ opp.title }}</h5>
                                <p class="card-text">{{ opp.description[:100] }}...</p>
                                <div class="mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-map-marker-alt"></i> {{ opp.location }}
                                    </small>
                                </div>
                                <div class="mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-dollar-sign"></i> {{ opp.budget }}
                                    </small>
                                </div>
                                <div class="mb-3">
                                    <small class="text-muted">
                                        <i class="fas fa-calendar"></i> Deadline: {{ opp.deadline }}
                                    </small>
                                </div>
                                <div class="row text-center">
                                    <div class="col-6">
                                        <small class="text-muted">Views</small>
                                        <div class="fw-bold">{{ opp.views }}</div>
                                    </div>
                                    <div class="col-6">
                                        <small class="text-muted">Matches</small>
                                        <div class="fw-bold">{{ opp.matches }}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="card-footer bg-transparent">
                                <button class="btn btn-outline-primary btn-sm" onclick="recordView('{{ opp_id }}')">
                                    <i class="fas fa-eye"></i> View Details
                                </button>
                                <button class="btn btn-outline-success btn-sm" onclick="findMatches('{{ opp_id }}')">
                                    <i class="fas fa-users"></i> Find Matches
                                </button>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
    
    <!-- Add Opportunity Modal -->
    <div class="modal fade" id="addOpportunityModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Create New Opportunity</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <form id="addOpportunityForm">
                    <div class="modal-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Title</label>
                                    <input type="text" class="form-control" name="title" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Type</label>
                                    <select class="form-control" name="type" required>
                                        <option value="">Select Type</option>
                                        <option value="buyer">Buyer</option>
                                        <option value="seller">Seller</option>
                                        <option value="partnership">Partnership</option>
                                        <option value="service">Service</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Industry</label>
                                    <select class="form-control" name="industry" required>
                                        <option value="">Select Industry</option>
                                        <option value="Technology">Technology</option>
                                        <option value="Healthcare">Healthcare</option>
                                        <option value="Finance">Finance</option>
                                        <option value="Education">Education</option>
                                        <option value="Manufacturing">Manufacturing</option>
                                        <option value="Retail">Retail</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label class="form-label">Budget Range</label>
                                    <select class="form-control" name="budget" required>
                                        <option value="">Select Budget</option>
                                        <option value="Under $10,000">Under $10,000</option>
                                        <option value="$10,000-50,000">$10,000-50,000</option>
                                        <option value="$50,000-100,000">$50,000-100,000</option>
                                        <option value="$100,000+">$100,000+</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Location</label>
                                    <input type="text" class="form-control" name="location" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Deadline</label>
                                    <input type="date" class="form-control" name="deadline" required>
                                </div>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Description</label>
                            <textarea class="form-control" name="description" rows="4" required></textarea>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-success">Create Opportunity</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <!-- Matches Modal -->
    <div class="modal fade" id="matchesModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Potential Matches</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="matchesContent">
                    Loading...
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Handle form submission
        document.getElementById('addOpportunityForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/opportunities', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    location.reload();
                } else {
                    alert('Error creating opportunity');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
        
        // Record opportunity view
        async function recordView(opportunityId) {
            // For demo, we'll use a sample user
            const interaction = {
                user_id: 'user1',
                opportunity_id: opportunityId,
                type: 'view'
            };
            
            try {
                await fetch('/api/interactions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(interaction)
                });
                
                alert('View recorded! Check analytics for updated metrics.');
            } catch (error) {
                console.error('Error recording view:', error);
            }
        }
        
        // Find matches for opportunity
        function findMatches(opportunityId) {
            const content = `
                <div class="alert alert-info">
                    <h6><i class="fas fa-magic"></i> AI-Powered Matching</h6>
                    <p>Our intelligent system analyzes user profiles, interests, and behavior patterns to find the best matches for this opportunity.</p>
                </div>
                <div class="list-group">
                    <div class="list-group-item">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">John Smith</h6>
                            <span class="badge bg-success">85% match</span>
                        </div>
                        <p class="mb-1">Technology professional with AI and startup interests</p>
                        <small class="text-muted">Matches: Industry, Location, Interest Keywords</small>
                    </div>
                    <div class="list-group-item">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">Sarah Johnson</h6>
                            <span class="badge bg-warning">72% match</span>
                        </div>
                        <p class="mb-1">Healthcare innovation specialist</p>
                        <small class="text-muted">Matches: Innovation Interest, Business Focus</small>
                    </div>
                </div>
                <div class="mt-3">
                    <button class="btn btn-primary" onclick="notifyMatches()">
                        <i class="fas fa-bell"></i> Notify All Matches
                    </button>
                </div>
            `;
            
            document.getElementById('matchesContent').innerHTML = content;
            new bootstrap.Modal(document.getElementById('matchesModal')).show();
        }
        
        // Notify matches
        function notifyMatches() {
            alert('Notifications sent to all matched users!');
        }
    </script>
</body>
</html>
'''

ANALYTICS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white active" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-chart-bar"></i> Analytics Dashboard</h1>
                    <div class="btn-group">
                        <button class="btn btn-outline-primary active" onclick="showPeriod('week')">7 Days</button>
                        <button class="btn btn-outline-primary" onclick="showPeriod('month')">30 Days</button>
                        <button class="btn btn-outline-primary" onclick="showPeriod('quarter')">90 Days</button>
                    </div>
                </div>
                
                <!-- KPI Cards -->
                <div class="row mb-4">
                    <div class="col-md-3 mb-3">
                        <div class="card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-users fa-2x text-primary mb-2"></i>
                                <h3 class="text-primary">{{ metrics.total_users }}</h3>
                                <p class="text-muted mb-0">Total Users</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-briefcase fa-2x text-success mb-2"></i>
                                <h3 class="text-success">{{ metrics.total_opportunities }}</h3>
                                <p class="text-muted mb-0">Active Opportunities</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-mouse-pointer fa-2x text-info mb-2"></i>
                                <h3 class="text-info">{{ metrics.total_interactions }}</h3>
                                <p class="text-muted mb-0">Total Interactions</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-percentage fa-2x text-warning mb-2"></i>
                                <h3 class="text-warning">{{ metrics.engagement_rate }}%</h3>
                                <p class="text-muted mb-0">Engagement Rate</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Charts Row -->
                <div class="row mb-4">
                    <div class="col-md-8">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-chart-line"></i> User Activity Trend</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="activityChart" height="80"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-pie-chart"></i> Opportunity Types</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="opportunityChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Industry Analysis -->
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-info text-white">
                                <h5 class="mb-0"><i class="fas fa-industry"></i> Industry Breakdown</h5>
                            </div>
                            <div class="card-body">
                                <canvas id="industryChart"></canvas>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-warning text-white">
                                <h5 class="mb-0"><i class="fas fa-bell"></i> Notification Performance</h5>
                            </div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col-4">
                                        <h4 class="text-primary">{{ metrics.total_notifications }}</h4>
                                        <small class="text-muted">Sent</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-success">85%</h4>
                                        <small class="text-muted">Delivered</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-info">42%</h4>
                                        <small class="text-muted">Opened</small>
                                    </div>
                                </div>
                                <hr>
                                <h6>Recent Activity</h6>
                                <p class="text-muted mb-1">Last 7 days: <strong>{{ metrics.recent_activity }}</strong> interactions</p>
                                <p class="text-muted mb-0">Avg. response time: <strong>2.3 minutes</strong></p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- AI Insights -->
                <div class="row">
                    <div class="col-md-12">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0"><i class="fas fa-brain"></i> AI-Generated Insights</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="alert alert-success">
                                            <h6><i class="fas fa-arrow-up"></i> Trending Up</h6>
                                            <p class="mb-0">Technology sector opportunities have increased by 34% this week</p>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="alert alert-info">
                                            <h6><i class="fas fa-users"></i> User Pattern</h6>
                                            <p class="mb-0">Peak engagement occurs on Tuesday-Thursday between 10-11 AM</p>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="alert alert-warning">
                                            <h6><i class="fas fa-exclamation-triangle"></i> Opportunity</h6>
                                            <p class="mb-0">Healthcare users show 67% higher conversion rates</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Activity Trend Chart
        const activityCtx = document.getElementById('activityChart').getContext('2d');
        new Chart(activityCtx, {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'User Interactions',
                    data: [12, 19, 15, 25, 22, 8, 10],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
        
        // Opportunity Types Chart
        const opportunityCtx = document.getElementById('opportunityChart').getContext('2d');
        new Chart(opportunityCtx, {
            type: 'doughnut',
            data: {
                labels: ['Partnership', 'Buyer', 'Seller', 'Service'],
                datasets: [{
                    data: [40, 25, 20, 15],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true
            }
        });
        
        // Industry Breakdown Chart
        const industryCtx = document.getElementById('industryChart').getContext('2d');
        new Chart(industryCtx, {
            type: 'bar',
            data: {
                labels: ['Technology', 'Healthcare', 'Finance', 'Education'],
                datasets: [{
                    label: 'Opportunities',
                    data: [15, 8, 5, 3],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true
            }
        });
        
        // Period selection
        function showPeriod(period) {
            // Update active button
            document.querySelectorAll('.btn-group .btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // In a real app, this would update the charts with new data
            console.log('Showing data for period:', period);
        }
    </script>
</body>
</html>
'''

NOTIFICATIONS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Notifications - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white active" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-bell"></i> Smart Notifications</h1>
                    <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#sendNotificationModal">
                        <i class="fas fa-paper-plane"></i> Send Notification
                    </button>
                </div>
                
                <!-- Notification Stats -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card border-0 shadow-sm text-center">
                            <div class="card-body">
                                <i class="fas fa-paper-plane fa-2x text-primary mb-2"></i>
                                <h4 class="text-primary">{{ notifications|length }}</h4>
                                <small class="text-muted">Total Sent</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card border-0 shadow-sm text-center">
                            <div class="card-body">
                                <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                                <h4 class="text-success">85%</h4>
                                <small class="text-muted">Delivery Rate</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card border-0 shadow-sm text-center">
                            <div class="card-body">
                                <i class="fas fa-eye fa-2x text-info mb-2"></i>
                                <h4 class="text-info">62%</h4>
                                <small class="text-muted">Open Rate</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card border-0 shadow-sm text-center">
                            <div class="card-body">
                                <i class="fas fa-mouse-pointer fa-2x text-warning mb-2"></i>
                                <h4 class="text-warning">24%</h4>
                                <small class="text-muted">Click Rate</small>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Smart Features -->
                <div class="row mb-4">
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-brain"></i> AI-Powered Features</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-6 text-center mb-3">
                                        <i class="fas fa-clock fa-2x text-primary mb-2"></i>
                                        <h6>Optimal Timing</h6>
                                        <small class="text-muted">AI determines the best time to send</small>
                                    </div>
                                    <div class="col-6 text-center mb-3">
                                        <i class="fas fa-bullseye fa-2x text-success mb-2"></i>
                                        <h6>Smart Targeting</h6>
                                        <small class="text-muted">Personalized content selection</small>
                                    </div>
                                    <div class="col-6 text-center">
                                        <i class="fas fa-mobile-alt fa-2x text-info mb-2"></i>
                                        <h6>Multi-Channel</h6>
                                        <small class="text-muted">Email, SMS, Push, In-App</small>
                                    </div>
                                    <div class="col-6 text-center">
                                        <i class="fas fa-sort-amount-up fa-2x text-warning mb-2"></i>
                                        <h6>Priority Scoring</h6>
                                        <small class="text-muted">Intelligent urgency detection</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-cogs"></i> Automation Rules</h5>
                            </div>
                            <div class="card-body">
                                <div class="list-group list-group-flush">
                                    <div class="list-group-item d-flex justify-content-between align-items-center">
                                        <div>
                                            <strong>New Opportunity Match</strong>
                                            <br><small class="text-muted">Auto-notify when user interests match</small>
                                        </div>
                                        <span class="badge bg-success">Active</span>
                                    </div>
                                    <div class="list-group-item d-flex justify-content-between align-items-center">
                                        <div>
                                            <strong>Weekly Summary</strong>
                                            <br><small class="text-muted">Digest of activities and recommendations</small>
                                        </div>
                                        <span class="badge bg-success">Active</span>
                                    </div>
                                    <div class="list-group-item d-flex justify-content-between align-items-center">
                                        <div>
                                            <strong>Engagement Follow-up</strong>
                                            <br><small class="text-muted">Re-engage inactive users</small>
                                        </div>
                                        <span class="badge bg-warning">Paused</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Notifications -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-secondary text-white">
                        <h5 class="mb-0"><i class="fas fa-history"></i> Recent Notifications</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Recipient</th>
                                        <th>Title</th>
                                        <th>Type</th>
                                        <th>Channel</th>
                                        <th>Status</th>
                                        <th>Sent At</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for notification in notifications %}
                                    <tr>
                                        <td>{{ notification.user_id }}</td>
                                        <td>{{ notification.title }}</td>
                                        <td><span class="badge bg-{{ 'primary' if notification.type == 'opportunity' else 'info' }}">{{ notification.type }}</span></td>
                                        <td>{{ notification.channel }}</td>
                                        <td><span class="badge bg-{{ 'success' if notification.status == 'sent' else 'warning' }}">{{ notification.status }}</span></td>
                                        <td>{{ notification.sent_at }}</td>
                                    </tr>
                                    {% endfor %}
                                    {% if notifications|length == 0 %}
                                    <tr>
                                        <td colspan="6" class="text-center text-muted">No notifications sent yet</td>
                                    </tr>
                                    {% endif %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Send Notification Modal -->
    <div class="modal fade" id="sendNotificationModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Send Smart Notification</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <form id="sendNotificationForm">
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Recipient</label>
                            <select class="form-control" name="user_id" required>
                                <option value="">Select User</option>
                                <option value="user1">John Smith</option>
                                <option value="user2">Sarah Johnson</option>
                                <option value="all">All Users</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Notification Type</label>
                            <select class="form-control" name="type" required>
                                <option value="">Select Type</option>
                                <option value="opportunity_match">Opportunity Match</option>
                                <option value="recommendation">Recommendation</option>
                                <option value="system_update">System Update</option>
                                <option value="weekly_digest">Weekly Digest</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Channel</label>
                            <select class="form-control" name="channel" required>
                                <option value="">Select Channel</option>
                                <option value="email">Email</option>
                                <option value="push">Push Notification</option>
                                <option value="in_app">In-App</option>
                                <option value="sms">SMS</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Title</label>
                            <input type="text" class="form-control" name="title" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Message</label>
                            <textarea class="form-control" name="message" rows="4" required></textarea>
                        </div>
                        <div class="alert alert-info">
                            <i class="fas fa-magic"></i> <strong>AI Enhancement:</strong> Our system will optimize delivery timing and personalize content based on user preferences and behavior patterns.
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                        <button type="submit" class="btn btn-primary">Send Notification</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Handle form submission
        document.getElementById('sendNotificationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/send-notification', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    alert('Notification sent successfully!');
                    location.reload();
                } else {
                    alert('Error sending notification');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

SETTINGS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Settings - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                    <a class="nav-link text-white" href="/"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/users"><i class="fas fa-users"></i> Users</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Opportunities</a>
                    <a class="nav-link text-white" href="/analytics"><i class="fas fa-chart-bar"></i> Analytics</a>
                    <a class="nav-link text-white" href="/notifications"><i class="fas fa-bell"></i> Notifications</a>
                    <a class="nav-link text-white active" href="/settings"><i class="fas fa-cog"></i> Settings</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <h1><i class="fas fa-cog"></i> System Settings</h1>
                
                <div class="row mt-4">
                    <!-- AI Configuration -->
                    <div class="col-md-6 mb-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-brain"></i> AI Configuration</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label class="form-label">OpenAI Model</label>
                                    <select class="form-control">
                                        <option>gpt-4o-mini</option>
                                        <option>gpt-4</option>
                                        <option>gpt-3.5-turbo</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Embedding Model</label>
                                    <select class="form-control">
                                        <option>text-embedding-3-large</option>
                                        <option>text-embedding-3-small</option>
                                        <option>text-embedding-ada-002</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Temperature</label>
                                    <input type="range" class="form-range" min="0" max="1" step="0.1" value="0.1">
                                    <small class="text-muted">Controls AI creativity (0 = focused, 1 = creative)</small>
                                </div>
                                <button class="btn btn-primary">Save AI Settings</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- System Configuration -->
                    <div class="col-md-6 mb-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-server"></i> System Configuration</h5>
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label class="form-label">Cache TTL (seconds)</label>
                                    <input type="number" class="form-control" value="300">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Max Concurrent Requests</label>
                                    <input type="number" class="form-control" value="100">
                                </div>
                                <div class="mb-3">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" checked>
                                        <label class="form-check-label">Enable Rate Limiting</label>
                                    </div>
                                </div>
                                <div class="mb-3">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" checked>
                                        <label class="form-check-label">Enable Request Logging</label>
                                    </div>
                                </div>
                                <button class="btn btn-success">Save System Settings</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- API Information -->
                <div class="card border-0 shadow-sm mb-4">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0"><i class="fas fa-code"></i> API Information</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <h6>Intelligence API</h6>
                                <p class="text-muted mb-1">Base URL: <code>/api/intelligence/</code></p>
                                <p class="text-muted mb-1">Endpoints: 11</p>
                                <span class="badge bg-success">Online</span>
                            </div>
                            <div class="col-md-4">
                                <h6>Analytics API</h6>
                                <p class="text-muted mb-1">Base URL: <code>/api/analytics/</code></p>
                                <p class="text-muted mb-1">Endpoints: 12</p>
                                <span class="badge bg-success">Online</span>
                            </div>
                            <div class="col-md-4">
                                <h6>Notifications API</h6>
                                <p class="text-muted mb-1">Base URL: <code>/api/notifications/</code></p>
                                <p class="text-muted mb-1">Endpoints: 10</p>
                                <span class="badge bg-success">Online</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- System Status -->
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-secondary text-white">
                        <h5 class="mb-0"><i class="fas fa-heartbeat"></i> System Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3 text-center">
                                <i class="fas fa-server fa-2x text-success mb-2"></i>
                                <h6>Server Status</h6>
                                <span class="badge bg-success">Healthy</span>
                            </div>
                            <div class="col-md-3 text-center">
                                <i class="fas fa-database fa-2x text-success mb-2"></i>
                                <h6>Database</h6>
                                <span class="badge bg-success">Connected</span>
                            </div>
                            <div class="col-md-3 text-center">
                                <i class="fas fa-memory fa-2x text-warning mb-2"></i>
                                <h6>Cache</h6>
                                <span class="badge bg-warning">Limited</span>
                            </div>
                            <div class="col-md-3 text-center">
                                <i class="fas fa-robot fa-2x text-success mb-2"></i>
                                <h6>AI Services</h6>
                                <span class="badge bg-success">Active</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--port':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    else:
        port = 8080
    
    app = create_web_app()
    print("🚀 Starting Business Dealer Intelligence Web Application")
    print(f"📱 Access at: http://localhost:{port}")
    print("✨ Full UI with forms, dashboards, and interactive features!")
    
    app.run(host='0.0.0.0', port=port, debug=True)