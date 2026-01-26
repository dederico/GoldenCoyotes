#!/usr/bin/env python3
"""
Golden Coyotes Platform - Complete Multi-User Business Intelligence Platform
Real user authentication, database persistence, AI matching, and email notifications
"""

print("🚀 Iniciando Golden Coyotes Platform...", flush=True)
print("📦 Cargando dependencias (esto puede tardar 30-60 segundos)...\n", flush=True)

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from functools import wraps

print("  ✓ Dependencias básicas cargadas", flush=True)

from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session
from flask_cors import CORS

print("  ✓ Flask cargado", flush=True)

from database_setup import DatabaseManager
print("  ✓ DatabaseManager cargado", flush=True)

from email_service import EmailService
print("  ✓ EmailService cargado", flush=True)

from ai_matching_engine import AIMatchingEngine
from admin_templates import (
    ADMIN_LOGIN_TEMPLATE, ADMIN_DASHBOARD_TEMPLATE,
    ADMIN_USERS_TEMPLATE, ADMIN_OPPORTUNITIES_TEMPLATE, ADMIN_LOGS_TEMPLATE
)

print("  ✓ Todos los módulos cargados\n", flush=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldenCoyotesPlatform:
    """Complete multi-user platform for business networking"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = os.getenv('SECRET_KEY', 'golden_coyotes_secret_' + str(uuid.uuid4()))
        CORS(self.app)
        
        # Initialize components
        self.db = DatabaseManager()
        self.email_service = EmailService()
        self.ai_matcher = AIMatchingEngine(self.db)
        
        # Setup routes
        self.setup_routes()
    
    def require_login(self, f):
        """Decorator to require user login"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    
    def require_admin(self, min_level=1):
        """Decorator to require admin login"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if 'user_id' not in session:
                    return redirect(url_for('admin_login'))
                
                if not session.get('is_admin') or session.get('admin_level', 0) < min_level:
                    flash('Access denied. Admin privileges required.', 'error')
                    return redirect(url_for('admin_login'))
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def setup_routes(self):
        """Setup all Flask routes"""
        
        @self.app.route('/')
        def index():
            """Landing page or dashboard"""
            if 'user_id' in session:
                return redirect(url_for('dashboard'))
            return render_template_string(LANDING_TEMPLATE)
        
        @self.app.route('/register', methods=['GET', 'POST'])
        def register():
            """User registration"""
            if request.method == 'POST':
                data = request.get_json() or request.form.to_dict()
                ref_code = data.get('ref') or request.args.get('ref')
                
                # Validate required fields
                required_fields = ['email', 'password', 'name', 'industry', 'location']
                for field in required_fields:
                    if not data.get(field):
                        return jsonify({'error': f'{field} is required'}), 400
                
                # Create user
                user_id = self.db.create_user(
                    email=data['email'],
                    password=data['password'],
                    name=data['name'],
                    industry=data['industry'],
                    location=data['location'],
                    bio=data.get('bio', ''),
                    skills=data.get('skills', ''),
                    interests=data.get('interests', ''),
                    company=data.get('company', ''),
                    position=data.get('position', ''),
                    phone=data.get('phone', '')
                )
                
                if user_id:
                    if ref_code and ref_code != user_id:
                        inviter = self.db.get_user(ref_code)
                        if inviter:
                            self.db.create_connection(
                                ref_code,
                                user_id,
                                message="Invitación aceptada automáticamente",
                                status="accepted",
                                accepted_at=datetime.now().isoformat()
                            )

                    # Send welcome email
                    self.email_service.send_welcome_email(data['email'], data['name'])
                    
                    # Auto-login
                    session['user_id'] = user_id
                    session['user_name'] = data['name']
                    session['user_email'] = data['email']
                    
                    flash('Welcome to Golden Coyotes! Your account has been created.', 'success')
                    return jsonify({'success': True, 'redirect': '/dashboard'})
                else:
                    return jsonify({'error': 'Email already exists or registration failed'}), 400
            
            ref_code = request.args.get('ref', '')
            return render_template_string(REGISTER_TEMPLATE, ref_code=ref_code)
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """User login"""
            if request.method == 'POST':
                data = request.get_json() or request.form.to_dict()
                
                user = self.db.authenticate_user(data.get('email'), data.get('password'))
                
                if user:
                    session['user_id'] = user['id']
                    session['user_name'] = user['name']
                    session['user_email'] = user['email']
                    session['user_role'] = user.get('user_role', 'user')
                    session['admin_level'] = user.get('admin_level', 0)
                    session['is_admin'] = user.get('user_role') == 'admin'
                    
                    # Redirect admin users to admin panel
                    if user.get('user_role') == 'admin':
                        flash(f'Welcome back, Admin {user["name"]}!', 'success')
                        return jsonify({'success': True, 'redirect': '/admin/dashboard'})
                    else:
                        flash(f'Welcome back, {user["name"]}!', 'success')
                        return jsonify({'success': True, 'redirect': '/dashboard'})
                else:
                    return jsonify({'error': 'Invalid email or password'}), 401
            
            return render_template_string(LOGIN_TEMPLATE)
        
        @self.app.route('/logout')
        def logout():
            """User logout"""
            session.clear()
            flash('You have been logged out.', 'info')
            return redirect(url_for('index'))
        
        @self.app.route('/dashboard')
        @self.require_login
        def dashboard():
            """User dashboard"""
            user = self.db.get_user(session['user_id'])
            user_opportunities = self.db.get_opportunities(user_id=session['user_id'], limit=10)
            ai_matches = self.ai_matcher.calculate_opportunity_matches(session['user_id'], limit=5)
            
            # Get connections count
            connections = self.db.get_user_connections(session['user_id'])
            
            metrics = {
                'my_opportunities': len(user_opportunities),
                'connections': len(connections),
                'ai_matches': len(ai_matches),
                'profile_completion': self._calculate_profile_completion(user)
            }
            
            return render_template_string(DASHBOARD_TEMPLATE, 
                                        user=user, 
                                        opportunities=user_opportunities,
                                        ai_matches=ai_matches,
                                        metrics=metrics)
        
        @self.app.route('/opportunities')
        @self.require_login
        def opportunities():
            """Browse all opportunities"""
            all_opportunities = self.db.get_opportunities(limit=50)
            return render_template_string(OPPORTUNITIES_TEMPLATE, opportunities=all_opportunities)
        
        @self.app.route('/my-opportunities')
        @self.require_login
        def my_opportunities():
            """User's own opportunities"""
            user_opportunities = self.db.get_opportunities(user_id=session['user_id'])
            return render_template_string(MY_OPPORTUNITIES_TEMPLATE, opportunities=user_opportunities)
        
        @self.app.route('/create-opportunity', methods=['GET', 'POST'])
        @self.require_login
        def create_opportunity():
            """Create new opportunity"""
            if request.method == 'POST':
                data = request.get_json() or request.form.to_dict()
                
                opp_id = self.db.create_opportunity(
                    user_id=session['user_id'],
                    title=data['title'],
                    description=data['description'],
                    opp_type=data['type'],
                    industry=data.get('industry'),
                    budget_min=data.get('budget_min'),
                    budget_max=data.get('budget_max'),
                    location=data.get('location'),
                    deadline=data.get('deadline'),
                    requirements=data.get('requirements'),
                    tags=data.get('tags')
                )
                
                if opp_id:
                    # Find and notify potential matches
                    self._notify_opportunity_matches(opp_id)
                    
                    flash('Opportunity created successfully!', 'success')
                    return jsonify({'success': True, 'opportunity_id': opp_id})
                else:
                    return jsonify({'error': 'Failed to create opportunity'}), 500
            
            return render_template_string(CREATE_OPPORTUNITY_TEMPLATE)
        
        @self.app.route('/network')
        @self.require_login
        def network():
            """User's network and connections"""
            connections = self.db.get_user_connections(session['user_id'])
            suggested_users = self.ai_matcher.calculate_user_matches(session['user_id'], limit=10)
            
            # Get existing connection user IDs to filter suggestions
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT connected_user_id FROM connections 
                WHERE user_id = ? 
                UNION 
                SELECT user_id FROM connections 
                WHERE connected_user_id = ?
            ''', (session['user_id'], session['user_id']))
            connected_user_ids = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            return render_template_string(NETWORK_TEMPLATE, 
                                        connections=connections,
                                        suggested_users=suggested_users,
                                        connected_user_ids=connected_user_ids,
                                        current_user_id=session['user_id'])
        
        @self.app.route('/profile')
        @self.require_login
        def profile():
            """User profile"""
            user = self.db.get_user(session['user_id'])
            return render_template_string(PROFILE_TEMPLATE, user=user)
        
        @self.app.route('/messages')
        @self.require_login
        def messages():
            """User messages"""
            user_messages = self.db.get_user_messages(session['user_id'])
            connections = self.db.get_user_connections(session['user_id'])
            return render_template_string(MESSAGES_TEMPLATE, 
                                        messages=user_messages,
                                        connections=connections)
        
        @self.app.route('/conversation/<user_id>')
        @self.require_login
        def conversation(user_id):
            """View conversation with specific user"""
            conversation_messages = self.db.get_conversation(session['user_id'], user_id)
            other_user = self.db.get_user(user_id)
            return render_template_string(CONVERSATION_TEMPLATE,
                                        messages=conversation_messages,
                                        other_user=other_user,
                                        current_user_id=session['user_id'])
        
        # API Endpoints
        @self.app.route('/api/matches/<user_id>')
        @self.require_login
        def api_get_matches(user_id):
            """Get AI matches for user"""
            if user_id != session['user_id']:
                return jsonify({'error': 'Unauthorized'}), 403
            
            matches = self.ai_matcher.calculate_opportunity_matches(user_id, limit=10)
            return jsonify({'matches': matches})
        
        @self.app.route('/api/connect', methods=['POST'])
        @self.require_login
        def api_connect():
            """Send connection request"""
            try:
                data = request.get_json()
                logger.info(f"Connection request data: {data}")
                logger.info(f"Session user_id: {session.get('user_id')}")
                
                if not data or not data.get('target_user_id'):
                    return jsonify({'error': 'target_user_id is required'}), 400
                
                # Check if users are the same
                if data['target_user_id'] == session['user_id']:
                    return jsonify({'error': 'Cannot connect to yourself'}), 400
                
                conn_id = self.db.create_connection(
                    user_id=session['user_id'],
                    target_user_id=data['target_user_id'],
                    message=data.get('message', '')
                )
                
                logger.info(f"Connection creation result: {conn_id}")
                
                if conn_id:
                    # Send email notification
                    target_user = self.db.get_user(data['target_user_id'])
                    if target_user:
                        try:
                            self.email_service.send_connection_request_notification(
                                target_user['email'],
                                target_user['name'],
                                session['user_name'],
                                f"http://localhost:8080/profile/{session['user_id']}",
                                data.get('message', '')
                            )
                        except Exception as e:
                            logger.warning(f"Email notification failed: {e}")
                    
                    return jsonify({'success': True, 'connection_id': conn_id})
                else:
                    return jsonify({'error': 'Connection already exists or creation failed'}), 400
                    
            except Exception as e:
                logger.error(f"Connection API error: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Internal server error: {str(e)}'}), 500
        
        @self.app.route('/api/send-message', methods=['POST'])
        @self.require_login
        def api_send_message():
            """Send message to another user"""
            data = request.get_json()
            
            required_fields = ['recipient_id', 'subject', 'content']
            for field in required_fields:
                if not data.get(field):
                    return jsonify({'error': f'{field} is required'}), 400
            
            message_id = self.db.send_message(
                sender_id=session['user_id'],
                recipient_id=data['recipient_id'],
                subject=data['subject'],
                content=data['content'],
                opportunity_id=data.get('opportunity_id')
            )
            
            if message_id:
                # Send email notification
                recipient = self.db.get_user(data['recipient_id'])
                if recipient:
                    self.email_service.send_message_notification(
                        recipient['email'],
                        recipient['name'],
                        session['user_name'],
                        data['subject'],
                        f"http://localhost:8080/conversation/{session['user_id']}"
                    )
                
                return jsonify({'success': True, 'message_id': message_id})
            else:
                return jsonify({'error': 'Failed to send message'}), 500

        @self.app.route('/api/invite-friends', methods=['POST'])
        @self.require_login
        def api_invite_friends():
            """Send invitations to friends via email"""
            try:
                data = request.get_json()

                if not data or not data.get('emails'):
                    return jsonify({'error': 'emails is required'}), 400

                emails = data['emails']
                personal_message = data.get('message', '')

                # Get current user info
                current_user = self.db.get_user(session['user_id'])
                if not current_user:
                    return jsonify({'error': 'User not found'}), 404

                # Send invitations
                sent_count = 0
                failed_emails = []

                for email in emails:
                    try:
                        # Create invitation link with referral code
                        base_url = request.host_url.rstrip('/')
                        invite_link = f"{base_url}/register?ref={session['user_id']}"

                        # Send invitation email
                        success = self.email_service.send_invitation_email(
                            recipient_email=email,
                            inviter_name=current_user['name'],
                            inviter_company=current_user.get('company', ''),
                            personal_message=personal_message,
                            invite_link=invite_link
                        )

                        if success:
                            sent_count += 1
                        else:
                            failed_emails.append(email)

                    except Exception as e:
                        logger.error(f"Failed to send invitation to {email}: {e}")
                        failed_emails.append(email)

                if sent_count > 0:
                    return jsonify({
                        'success': True,
                        'sent_count': sent_count,
                        'failed_count': len(failed_emails),
                        'failed_emails': failed_emails
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to send invitations',
                        'failed_emails': failed_emails
                    }), 500

            except Exception as e:
                logger.error(f"Error in invite-friends endpoint: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/messages/<user_id>')
        @self.require_login
        def api_get_messages(user_id):
            """Get messages for current user"""
            if user_id != session['user_id']:
                return jsonify({'error': 'Unauthorized'}), 403
            
            messages = self.db.get_user_messages(user_id)
            return jsonify({'messages': messages})
        
        @self.app.route('/api/conversation/<other_user_id>')
        @self.require_login
        def api_get_conversation(other_user_id):
            """Get conversation between current user and another user"""
            conversation = self.db.get_conversation(session['user_id'], other_user_id)
            return jsonify({'conversation': conversation})
        
        @self.app.route('/api/update-profile', methods=['POST'])
        @self.require_login
        def api_update_profile():
            """Update user profile"""
            data = request.get_json()
            
            # Update user in database
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE users 
                    SET industry=?, location=?, bio=?, skills=?, interests=?, company=?, position=?, phone=?
                    WHERE id=?
                ''', (
                    data.get('industry'),
                    data.get('location'), 
                    data.get('bio'),
                    data.get('skills'),
                    data.get('interests'),
                    data.get('company'),
                    data.get('position'),
                    data.get('phone'),
                    session['user_id']
                ))
                
                conn.commit()
                flash('Profile updated successfully!', 'success')
                return jsonify({'success': True})
                
            except Exception as e:
                logger.error(f"Error updating profile: {e}")
                conn.rollback()
                return jsonify({'error': 'Failed to update profile'}), 500
            finally:
                conn.close()
        
        # Admin Routes
        @self.app.route('/admin/login', methods=['GET', 'POST'])
        def admin_login():
            """Admin login page"""
            if request.method == 'POST':
                data = request.get_json() or request.form.to_dict()
                
                user = self.db.authenticate_user(data.get('email'), data.get('password'))
                
                if user and user.get('user_role') == 'admin':
                    session['user_id'] = user['id']
                    session['user_name'] = user['name']
                    session['user_email'] = user['email']
                    session['user_role'] = user['user_role']
                    session['admin_level'] = user['admin_level']
                    session['is_admin'] = True
                    
                    # Log admin login
                    self.db.log_admin_action(
                        admin_id=user['id'],
                        action='admin_login',
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get('User-Agent')
                    )
                    
                    flash(f'Welcome, Admin {user["name"]}!', 'success')
                    return jsonify({'success': True, 'redirect': '/admin/dashboard'})
                else:
                    return jsonify({'error': 'Invalid credentials or insufficient privileges'}), 401
            
            return render_template_string(ADMIN_LOGIN_TEMPLATE)
        
        @self.app.route('/admin/dashboard')
        @self.require_admin(min_level=1)
        def admin_dashboard():
            """Admin dashboard with platform statistics"""
            stats = self.db.get_platform_statistics()
            recent_users = self.db.get_all_users(limit=10)
            recent_logs = self.db.get_admin_logs(limit=20)
            
            return render_template_string(ADMIN_DASHBOARD_TEMPLATE, 
                                        stats=stats,
                                        recent_users=recent_users,
                                        recent_logs=recent_logs)
        
        @self.app.route('/admin/users')
        @self.require_admin(min_level=1)
        def admin_users():
            """Admin user management"""
            users = self.db.get_all_users(limit=100)
            return render_template_string(ADMIN_USERS_TEMPLATE, users=users)
        
        @self.app.route('/admin/opportunities')
        @self.require_admin(min_level=1)
        def admin_opportunities():
            """Admin opportunity management"""
            opportunities = self.db.get_opportunities(limit=100)
            return render_template_string(ADMIN_OPPORTUNITIES_TEMPLATE, opportunities=opportunities)
        
        @self.app.route('/admin/logs')
        @self.require_admin(min_level=2)
        def admin_logs():
            """Admin activity logs"""
            logs = self.db.get_admin_logs(limit=200)
            return render_template_string(ADMIN_LOGS_TEMPLATE, logs=logs)
        
        @self.app.route('/admin/api/user-status', methods=['POST'])
        @self.require_admin(min_level=2)
        def admin_update_user_status():
            """Update user status (admin action)"""
            data = request.get_json()
            
            if not data.get('user_id') or not data.get('status'):
                return jsonify({'error': 'user_id and status are required'}), 400
            
            success = self.db.update_user_status(data['user_id'], data['status'])
            
            if success:
                # Log admin action
                self.db.log_admin_action(
                    admin_id=session['user_id'],
                    action='user_status_update',
                    target_type='user',
                    target_id=data['user_id'],
                    details=f"Status changed to {data['status']}",
                    ip_address=request.remote_addr
                )
                
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Failed to update user status'}), 500
    
    def _calculate_profile_completion(self, user):
        """Calculate profile completion percentage"""
        fields = ['name', 'email', 'industry', 'location', 'bio', 'skills', 'interests', 'company', 'position']
        completed = sum(1 for field in fields if user.get(field))
        return int((completed / len(fields)) * 100)
    
    def _notify_opportunity_matches(self, opportunity_id):
        """Notify users about new opportunity matches"""
        # Get opportunity details
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM opportunities WHERE id = ?', (opportunity_id,))
            result = cursor.fetchone()
            
            if not result:
                return
            
            columns = [description[0] for description in cursor.description]
            opportunity = dict(zip(columns, result))
            
            # Get all users except the creator
            cursor.execute('SELECT * FROM users WHERE id != ? AND status = "active"', (opportunity['user_id'],))
            users = cursor.fetchall()
            user_columns = [description[0] for description in cursor.description]
            users = [dict(zip(user_columns, row)) for row in users]
            
            # Calculate matches and send notifications
            for user in users:
                try:
                    user_profile = self.ai_matcher._build_user_profile(user)
                    score = self.ai_matcher._calculate_match_score(user_profile, opportunity)
                    
                    if score > 0.6:  # High match threshold for notifications
                        self.email_service.send_opportunity_match_notification(
                            user['email'],
                            user['name'],
                            opportunity['title'],
                            score,
                            f"http://localhost:8080/opportunity/{opportunity_id}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error notifying user {user.get('id')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in opportunity notification: {e}")
        finally:
            conn.close()
    
    def run(self, host='0.0.0.0', port=8080, debug=True):
        """Run the Flask application"""
        print("🚀 Starting Golden Coyotes Platform")
        print(f"📱 Access at: http://localhost:{port}")
        print("✨ Complete multi-user business networking platform!")
        print("\n🔧 Features:")
        print("- User registration and authentication")
        print("- Database persistence (SQLite)")
        print("- AI-powered matching")
        print("- Email notifications")
        print("- User networks and connections")
        print("- Personal dashboards")
        
        self.app.run(host=host, port=port, debug=debug)

# HTML Templates

LANDING_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Coyotes - Business Networking Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; color: white; }
        .feature-card { transition: transform 0.3s; }
        .feature-card:hover { transform: translateY(-10px); }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: rgba(0,0,0,0.1);">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-brain"></i> Golden Coyotes</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/login">Login</a>
                <a class="nav-link" href="/register">Register</a>
            </div>
        </div>
    </nav>
    
    <div class="hero-section d-flex align-items-center">
        <div class="container text-center">
            <h1 class="display-4 mb-4">🐺 Welcome to Golden Coyotes</h1>
            <p class="lead mb-5">The AI-powered business networking platform that connects opportunities with the right people</p>
            
            <div class="row justify-content-center mb-5">
                <div class="col-md-8">
                    <div class="row">
                        <div class="col-md-4 mb-3">
                            <div class="text-center">
                                <i class="fas fa-brain fa-3x mb-3"></i>
                                <h5>AI Matching</h5>
                                <p>Smart algorithms find perfect matches</p>
                            </div>
                        </div>
                        <div class="col-md-4 mb-3">
                            <div class="text-center">
                                <i class="fas fa-network-wired fa-3x mb-3"></i>
                                <h5>Professional Network</h5>
                                <p>Build meaningful business connections</p>
                            </div>
                        </div>
                        <div class="col-md-4 mb-3">
                            <div class="text-center">
                                <i class="fas fa-rocket fa-3x mb-3"></i>
                                <h5>Opportunities</h5>
                                <p>Discover and create business opportunities</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="text-center">
                <a href="/register" class="btn btn-light btn-lg me-3">
                    <i class="fas fa-user-plus"></i> Join Now
                </a>
                <a href="/login" class="btn btn-outline-light btn-lg">
                    <i class="fas fa-sign-in-alt"></i> Login
                </a>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .login-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    </style>
</head>
<body class="d-flex align-items-center">
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6 col-lg-4">
                <div class="login-card p-5">
                    <div class="text-center mb-4">
                        <h2><i class="fas fa-brain"></i> Golden Coyotes</h2>
                        <p class="text-muted">Welcome back to your network</p>
                    </div>
                    
                    <form id="loginForm">
                        <div class="mb-3">
                            <label class="form-label">Email</label>
                            <input type="email" class="form-control" name="email" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Password</label>
                            <input type="password" class="form-control" name="password" required>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mb-3">
                            <i class="fas fa-sign-in-alt"></i> Login
                        </button>
                    </form>
                    
                    <div class="text-center">
                        <p>Don't have an account? <a href="/register">Register here</a></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('loginForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    window.location.href = result.redirect;
                } else {
                    alert(result.error);
                }
            } catch (error) {
                alert('Login failed: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px 0; }
        .register-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8 col-lg-6">
                <div class="register-card p-5">
                    <div class="text-center mb-4">
                        <h2><i class="fas fa-brain"></i> Join Golden Coyotes</h2>
                        <p class="text-muted">Start building your professional network</p>
                    </div>
                    
                    <form id="registerForm">
                        <input type="hidden" name="ref" value="{{ ref_code }}">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Full Name *</label>
                                <input type="text" class="form-control" name="name" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Email *</label>
                                <input type="email" class="form-control" name="email" required>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Password *</label>
                                <input type="password" class="form-control" name="password" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Industry *</label>
                                <select class="form-control" name="industry" required>
                                    <option value="">Select Industry</option>
                                    <option value="Technology">Technology</option>
                                    <option value="Healthcare">Healthcare</option>
                                    <option value="Finance">Finance</option>
                                    <option value="Education">Education</option>
                                    <option value="Manufacturing">Manufacturing</option>
                                    <option value="Retail">Retail</option>
                                    <option value="Consulting">Consulting</option>
                                    <option value="Real Estate">Real Estate</option>
                                    <option value="Media">Media</option>
                                    <option value="Other">Other</option>
                                </select>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Location *</label>
                                <input type="text" class="form-control" name="location" placeholder="City, State" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Company</label>
                                <input type="text" class="form-control" name="company">
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Position</label>
                                <input type="text" class="form-control" name="position" placeholder="Your job title">
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Phone</label>
                                <input type="tel" class="form-control" name="phone">
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Skills</label>
                            <input type="text" class="form-control" name="skills" placeholder="AI, Marketing, Sales, etc. (comma-separated)">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Interests</label>
                            <input type="text" class="form-control" name="interests" placeholder="Startups, Innovation, Networking, etc. (comma-separated)">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Bio</label>
                            <textarea class="form-control" name="bio" rows="3" placeholder="Tell us about yourself and your professional background..."></textarea>
                        </div>
                        
                        <button type="submit" class="btn btn-primary w-100 mb-3">
                            <i class="fas fa-user-plus"></i> Create Account
                        </button>
                    </form>
                    
                    <div class="text-center">
                        <p>Already have an account? <a href="/login">Login here</a></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('registerForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    window.location.href = result.redirect;
                } else {
                    alert(result.error);
                }
            } catch (error) {
                alert('Registration failed: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-5px); }
        .match-card { border-left: 4px solid #28a745; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                <a class="nav-link text-white active" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1>🏠 Welcome back, {{ user.name }}!</h1>
                    <span class="badge bg-success fs-6">Profile {{ metrics.profile_completion }}% Complete</span>
                </div>
                
                <!-- Quick Stats -->
                <div class="row mb-4">
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-briefcase fa-3x text-primary mb-3"></i>
                                <h5 class="card-title">My Opportunities</h5>
                                <h2 class="text-primary">{{ metrics.my_opportunities }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-users fa-3x text-success mb-3"></i>
                                <h5 class="card-title">Connections</h5>
                                <h2 class="text-success">{{ metrics.connections }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-magic fa-3x text-info mb-3"></i>
                                <h5 class="card-title">AI Matches</h5>
                                <h2 class="text-info">{{ metrics.ai_matches }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-chart-line fa-3x text-warning mb-3"></i>
                                <h5 class="card-title">Profile Score</h5>
                                <h2 class="text-warning">{{ metrics.profile_completion }}%</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row">
                    <!-- AI Matches -->
                    <div class="col-md-8">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-magic"></i> AI-Powered Opportunity Matches</h5>
                            </div>
                            <div class="card-body">
                                {% if ai_matches %}
                                    {% for match in ai_matches %}
                                    <div class="card match-card mb-3">
                                        <div class="card-body">
                                            <div class="d-flex justify-content-between align-items-start">
                                                <div>
                                                    <h6 class="card-title">{{ match.opportunity.title }}</h6>
                                                    <p class="card-text text-muted">{{ match.opportunity.description[:150] }}...</p>
                                                    <small class="text-muted">
                                                        <i class="fas fa-industry"></i> {{ match.opportunity.industry }} | 
                                                        <i class="fas fa-map-marker-alt"></i> {{ match.opportunity.location }}
                                                    </small>
                                                </div>
                                                <div class="text-end">
                                                    <span class="badge bg-success">{{ (match.score * 100)|int }}% Match</span>
                                                    <br><br>
                                                    <button class="btn btn-outline-primary btn-sm">View Details</button>
                                                </div>
                                            </div>
                                            <hr>
                                            <small class="text-muted"><strong>Why this matches:</strong> {{ match.reasoning }}</small>
                                        </div>
                                    </div>
                                    {% endfor %}
                                {% else %}
                                    <div class="text-center text-muted py-4">
                                        <i class="fas fa-search fa-3x mb-3"></i>
                                        <p>No AI matches yet. Complete your profile to get better matches!</p>
                                        <a href="/profile" class="btn btn-primary">Complete Profile</a>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Quick Actions -->
                    <div class="col-md-4">
                        <div class="card border-0 shadow-sm mb-4">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-rocket"></i> Quick Actions</h5>
                            </div>
                            <div class="card-body">
                                <div class="d-grid gap-2">
                                    <a href="/create-opportunity" class="btn btn-outline-primary">
                                        <i class="fas fa-plus"></i> Create Opportunity
                                    </a>
                                    <a href="/opportunities" class="btn btn-outline-success">
                                        <i class="fas fa-search"></i> Browse Opportunities
                                    </a>
                                    <a href="/network" class="btn btn-outline-info">
                                        <i class="fas fa-users"></i> Expand Network
                                    </a>
                                    <a href="/profile" class="btn btn-outline-warning">
                                        <i class="fas fa-user-edit"></i> Update Profile
                                    </a>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Recent Opportunities -->
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0"><i class="fas fa-history"></i> My Recent Opportunities</h5>
                            </div>
                            <div class="card-body">
                                {% if opportunities %}
                                    {% for opp in opportunities[:3] %}
                                    <div class="mb-3">
                                        <h6>{{ opp.title }}</h6>
                                        <small class="text-muted">{{ opp.type }} | {{ opp.created_at }}</small>
                                    </div>
                                    {% endfor %}
                                    <a href="/my-opportunities" class="btn btn-sm btn-outline-secondary">View All</a>
                                {% else %}
                                    <p class="text-muted">No opportunities yet. <a href="/create-opportunity">Create your first one!</a></p>
                                {% endif %}
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

OPPORTUNITIES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Browse Opportunities - Golden Coyotes</title>
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
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white active" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-briefcase"></i> Browse All Opportunities</h1>
                    <a href="/create-opportunity" class="btn btn-success">
                        <i class="fas fa-plus"></i> Create Opportunity
                    </a>
                </div>
                
                <!-- Opportunities Grid -->
                <div class="row">
                    {% for opp in opportunities %}
                    <div class="col-md-6 col-lg-4 mb-4">
                        <div class="card opportunity-card h-100 border-0 shadow-sm">
                            <div class="card-header bg-light">
                                <div class="d-flex justify-content-between align-items-center">
                                    <span class="badge bg-primary">{{ opp.type.title() }}</span>
                                    <small class="text-muted">{{ opp.industry }}</small>
                                </div>
                            </div>
                            <div class="card-body">
                                <h5 class="card-title">{{ opp.title }}</h5>
                                <p class="card-text">{{ opp.description[:150] }}...</p>
                                <div class="mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-map-marker-alt"></i> {{ opp.location }}
                                    </small>
                                </div>
                                <div class="mb-2">
                                    <small class="text-muted">
                                        <i class="fas fa-user"></i> by {{ opp.creator_name }}
                                    </small>
                                </div>
                                <div class="mb-3">
                                    <small class="text-muted">
                                        <i class="fas fa-calendar"></i> {{ opp.created_at[:10] }}
                                    </small>
                                </div>
                            </div>
                            <div class="card-footer bg-transparent">
                                <button class="btn btn-outline-primary btn-sm" onclick="contactUser('{{ opp.user_id }}')">
                                    <i class="fas fa-envelope"></i> Contact
                                </button>
                                <button class="btn btn-outline-success btn-sm" onclick="connectUser('{{ opp.user_id }}')">
                                    <i class="fas fa-user-plus"></i> Connect
                                </button>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                
                {% if not opportunities %}
                <div class="text-center text-muted py-5">
                    <i class="fas fa-search fa-3x mb-3"></i>
                    <h4>No opportunities yet</h4>
                    <p>Be the first to create an opportunity!</p>
                    <a href="/create-opportunity" class="btn btn-primary">Create First Opportunity</a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function contactUser(userId) {
            alert('Contact feature coming soon! User ID: ' + userId);
        }
        
        function connectUser(userId) {
            if (confirm('Send connection request to this user?')) {
                fetch('/api/connect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        target_user_id: userId,
                        message: 'Hi! I saw your opportunity and would like to connect.'
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Connection request sent!');
                    } else {
                        alert('Error: ' + data.error);
                    }
                });
            }
        }
    </script>
</body>
</html>
'''

MY_OPPORTUNITIES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Opportunities - Golden Coyotes</title>
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
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white active" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-list"></i> My Opportunities</h1>
                    <a href="/create-opportunity" class="btn btn-success">
                        <i class="fas fa-plus"></i> Create New
                    </a>
                </div>
                
                <!-- Opportunities Table -->
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        {% if opportunities %}
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Title</th>
                                        <th>Type</th>
                                        <th>Industry</th>
                                        <th>Location</th>
                                        <th>Created</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for opp in opportunities %}
                                    <tr>
                                        <td>
                                            <strong>{{ opp.title }}</strong>
                                            <br><small class="text-muted">{{ opp.description[:100] }}...</small>
                                        </td>
                                        <td><span class="badge bg-primary">{{ opp.type.title() }}</span></td>
                                        <td>{{ opp.industry }}</td>
                                        <td>{{ opp.location }}</td>
                                        <td>{{ opp.created_at[:10] }}</td>
                                        <td>
                                            {% if opp.is_active %}
                                            <span class="badge bg-success">Active</span>
                                            {% else %}
                                            <span class="badge bg-secondary">Inactive</span>
                                            {% endif %}
                                        </td>
                                        <td>
                                            <button class="btn btn-sm btn-outline-primary" onclick="editOpportunity('{{ opp.id }}')">
                                                <i class="fas fa-edit"></i> Edit
                                            </button>
                                            <button class="btn btn-sm btn-outline-danger" onclick="deleteOpportunity('{{ opp.id }}')">
                                                <i class="fas fa-trash"></i> Delete
                                            </button>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <div class="text-center text-muted py-5">
                            <i class="fas fa-plus-circle fa-3x mb-3"></i>
                            <h4>No opportunities yet</h4>
                            <p>Create your first opportunity to start connecting with potential partners!</p>
                            <a href="/create-opportunity" class="btn btn-primary">Create First Opportunity</a>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function editOpportunity(oppId) {
            alert('Edit feature coming soon! Opportunity ID: ' + oppId);
        }
        
        function deleteOpportunity(oppId) {
            if (confirm('Are you sure you want to delete this opportunity?')) {
                alert('Delete feature coming soon! Opportunity ID: ' + oppId);
            }
        }
    </script>
</body>
</html>
'''

CREATE_OPPORTUNITY_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Opportunity - Golden Coyotes</title>
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
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white active" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="row justify-content-center">
                    <div class="col-md-8">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h4 class="mb-0"><i class="fas fa-plus"></i> Create New Opportunity</h4>
                            </div>
                            <div class="card-body">
                                <form id="createOpportunityForm">
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Title *</label>
                                            <input type="text" class="form-control" name="title" required>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Type *</label>
                                            <select class="form-control" name="type" required>
                                                <option value="">Select Type</option>
                                                <option value="partnership">Partnership</option>
                                                <option value="buyer">Looking to Buy</option>
                                                <option value="seller">Looking to Sell</option>
                                                <option value="service">Service Needed</option>
                                                <option value="investment">Investment</option>
                                                <option value="collaboration">Collaboration</option>
                                            </select>
                                        </div>
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Industry</label>
                                            <select class="form-control" name="industry">
                                                <option value="">Select Industry</option>
                                                <option value="Technology">Technology</option>
                                                <option value="Healthcare">Healthcare</option>
                                                <option value="Finance">Finance</option>
                                                <option value="Education">Education</option>
                                                <option value="Manufacturing">Manufacturing</option>
                                                <option value="Retail">Retail</option>
                                                <option value="Consulting">Consulting</option>
                                                <option value="Real Estate">Real Estate</option>
                                                <option value="Media">Media</option>
                                                <option value="Other">Other</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Location</label>
                                            <input type="text" class="form-control" name="location" placeholder="City, State">
                                        </div>
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Budget Min ($)</label>
                                            <input type="number" class="form-control" name="budget_min" placeholder="0">
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Budget Max ($)</label>
                                            <input type="number" class="form-control" name="budget_max" placeholder="100000">
                                        </div>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Deadline</label>
                                        <input type="date" class="form-control" name="deadline">
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Description *</label>
                                        <textarea class="form-control" name="description" rows="5" required 
                                                  placeholder="Describe your opportunity, what you're looking for, and any specific requirements..."></textarea>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Requirements</label>
                                        <textarea class="form-control" name="requirements" rows="3" 
                                                  placeholder="Specific skills, experience, or qualifications needed..."></textarea>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Tags</label>
                                        <input type="text" class="form-control" name="tags" 
                                               placeholder="startup, AI, marketing, etc. (comma-separated)">
                                    </div>
                                    
                                    <div class="d-grid">
                                        <button type="submit" class="btn btn-success btn-lg">
                                            <i class="fas fa-rocket"></i> Create Opportunity
                                        </button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('createOpportunityForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/create-opportunity', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Opportunity created successfully!');
                    window.location.href = '/my-opportunities';
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error creating opportunity: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

NETWORK_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .user-card { transition: transform 0.2s; }
        .user-card:hover { transform: translateY(-2px); }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4"><h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3><nav class="nav flex-column">
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white active" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <h1><i class="fas fa-users"></i> Professional Network</h1>

                <!-- Invite Friends Section -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card border-0 shadow-sm" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            <div class="card-body text-white p-4">
                                <div class="row align-items-center">
                                    <div class="col-md-6">
                                        <h4><i class="fas fa-user-plus"></i> Grow Your Network</h4>
                                        <p class="mb-0">Invite your friends and colleagues to join Golden Coyotes</p>
                                    </div>
                                    <div class="col-md-6 text-md-end">
                                        <button class="btn btn-light btn-lg me-2 mb-2" onclick="openInviteModal()">
                                            <i class="fas fa-envelope"></i> Invite by Email
                                        </button>
                                        <button class="btn btn-outline-light btn-lg me-2 mb-2" onclick="shareOnLinkedIn()">
                                            <i class="fab fa-linkedin"></i> Share on LinkedIn
                                        </button>
                                        <button class="btn btn-outline-light btn-lg mb-2" onclick="shareOnFacebook()">
                                            <i class="fab fa-facebook"></i> Share on Facebook
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <!-- My Connections -->
                    <div class="col-md-6 mb-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0"><i class="fas fa-handshake"></i> My Connections ({{ connections|length }})</h5>
                            </div>
                            <div class="card-body">
                                {% if connections %}
                                    {% for connection in connections %}
                                    <div class="d-flex justify-content-between align-items-center mb-3 p-3 bg-light rounded">
                                        <div>
                                            <h6>{{ connection.name }}</h6>
                                            <small class="text-muted">{{ connection.industry }} • {{ connection.location }}</small>
                                        </div>
                                        <button class="btn btn-sm btn-outline-primary" onclick="messageUser('{{ connection.id }}')">
                                            <i class="fas fa-envelope"></i> Message
                                        </button>
                                    </div>
                                    {% endfor %}
                                {% else %}
                                    <div class="text-center text-muted py-4">
                                        <i class="fas fa-users fa-3x mb-3"></i>
                                        <p>No connections yet. Start building your network!</p>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Suggested Connections -->
                    <div class="col-md-6 mb-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-magic"></i> AI-Suggested Connections</h5>
                            </div>
                            <div class="card-body">
                                {% if suggested_users %}
                                    {% for match in suggested_users %}
                                    <div class="user-card mb-3 p-3 border rounded">
                                        <div class="d-flex justify-content-between align-items-start">
                                            <div>
                                                <h6>{{ match.user.name }}</h6>
                                                <p class="text-muted mb-1">{{ match.user.industry }} • {{ match.user.location }}</p>
                                                <small class="text-muted">{{ match.reasoning }}</small>
                                                {% if match.common_interests %}
                                                <br><small class="text-success">Common interests: {{ match.common_interests|join(', ') }}</small>
                                                {% endif %}
                                            </div>
                                            <div class="text-end">
                                                <span class="badge bg-success">{{ (match.score * 100)|int }}% Match</span>
                                                <br><br>
                                                {% if match.user.id == current_user_id %}
                                                <span class="text-muted small">That's you!</span>
                                                {% elif match.user.id in connected_user_ids %}
                                                <button class="btn btn-sm btn-outline-primary" onclick="messageUser('{{ match.user.id }}')">
                                                    <i class="fas fa-envelope"></i> Message
                                                </button>
                                                <br><small class="text-success">Already connected</small>
                                                {% else %}
                                                <button class="btn btn-sm btn-outline-success" onclick="connectUser('{{ match.user.id }}')">
                                                    <i class="fas fa-user-plus"></i> Connect
                                                </button>
                                                {% endif %}
                                            </div>
                                        </div>
                                    </div>
                                    {% endfor %}
                                {% else %}
                                    <div class="text-center text-muted py-4">
                                        <i class="fas fa-search fa-3x mb-3"></i>
                                        <p>No suggestions available. Complete your profile for better matches!</p>
                                        <a href="/profile" class="btn btn-primary">Update Profile</a>
                                    </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Invite Friends Modal -->
    <div class="modal fade" id="inviteModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header bg-primary text-white">
                    <h5 class="modal-title"><i class="fas fa-envelope"></i> Invite Friends by Email</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <form id="inviteForm">
                        <div class="mb-3">
                            <label class="form-label">Email Addresses</label>
                            <textarea class="form-control" id="emailAddresses" rows="4"
                                      placeholder="Enter email addresses (one per line or comma-separated)&#10;ejemplo@email.com, amigo@email.com"
                                      required></textarea>
                            <small class="text-muted">You can enter multiple email addresses separated by commas or new lines</small>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Personal Message (Optional)</label>
                            <textarea class="form-control" id="inviteMessage" rows="3"
                                      placeholder="Add a personal message to your invitation..."></textarea>
                        </div>
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> Your friends will receive an email invitation to join Golden Coyotes and connect with you.
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="sendInvitations()">
                        <i class="fas fa-paper-plane"></i> Send Invitations
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let inviteModal;

        document.addEventListener('DOMContentLoaded', function() {
            inviteModal = new bootstrap.Modal(document.getElementById('inviteModal'));
        });

        function openInviteModal() {
            inviteModal.show();
        }

        function sendInvitations() {
            const emailsText = document.getElementById('emailAddresses').value.trim();
            const message = document.getElementById('inviteMessage').value.trim();

            if (!emailsText) {
                alert('Please enter at least one email address');
                return;
            }

            // Parse email addresses (split by comma, newline, or semicolon)
            const emails = emailsText
                .split(/[,;\\n]/)
                .map(e => e.trim())
                .filter(e => e.length > 0);

            if (emails.length === 0) {
                alert('Please enter valid email addresses');
                return;
            }

            // Validate email format
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            const invalidEmails = emails.filter(e => !emailRegex.test(e));

            if (invalidEmails.length > 0) {
                alert('Invalid email addresses found: ' + invalidEmails.join(', '));
                return;
            }

            // Send invitations
            fetch('/api/invite-friends', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    emails: emails,
                    message: message
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`✅ Successfully sent ${data.sent_count} invitation(s)!`);
                    inviteModal.hide();
                    document.getElementById('inviteForm').reset();
                } else {
                    alert('Error: ' + (data.error || 'Failed to send invitations'));
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Network error occurred. Please try again.');
            });
        }

        function shareOnLinkedIn() {
            const inviteUrl = encodeURIComponent(window.location.origin + '/register?ref=' + '{{ session.user_id }}');
            const text = encodeURIComponent('Join me on Golden Coyotes - A professional networking platform for business opportunities!');
            const linkedInUrl = 'https://www.linkedin.com/sharing/share-offsite/?url=' + inviteUrl;
            window.open(linkedInUrl, '_blank', 'width=600,height=400');
        }

        function shareOnFacebook() {
            const inviteUrl = encodeURIComponent(window.location.origin + '/register?ref=' + '{{ session.user_id }}');
            const facebookUrl = 'https://www.facebook.com/sharer/sharer.php?u=' + inviteUrl;
            window.open(facebookUrl, '_blank', 'width=600,height=400');
        }

        function connectUser(userId) {
            const message = prompt('Add a personal message (optional):');
            if (message !== null) {
                fetch('/api/connect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        target_user_id: userId,
                        message: message || 'Hi! I would like to connect with you on Golden Coyotes.'
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Connection request sent successfully!');
                        location.reload();
                    } else {
                        if (data.error.includes('already exists')) {
                            alert('You are already connected to this user or have a pending request.');
                        } else if (data.error.includes('Cannot connect to yourself')) {
                            alert('You cannot send a connection request to yourself.');
                        } else {
                            alert('Error: ' + data.error);
                        }
                    }
                });
            }
        }
        
        function messageUser(userId) {
            window.location.href = '/conversation/' + userId;
        }
    </script>
</body>
</html>
'''

PROFILE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profile - Golden Coyotes</title>
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
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white active" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="row">
                    <div class="col-md-8">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h4 class="mb-0"><i class="fas fa-user-edit"></i> Update Profile</h4>
                            </div>
                            <div class="card-body">
                                <form id="updateProfileForm">
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Industry</label>
                                            <select class="form-control" name="industry">
                                                <option value="">Select Industry</option>
                                                <option value="Technology" {% if user.industry == 'Technology' %}selected{% endif %}>Technology</option>
                                                <option value="Healthcare" {% if user.industry == 'Healthcare' %}selected{% endif %}>Healthcare</option>
                                                <option value="Finance" {% if user.industry == 'Finance' %}selected{% endif %}>Finance</option>
                                                <option value="Education" {% if user.industry == 'Education' %}selected{% endif %}>Education</option>
                                                <option value="Manufacturing" {% if user.industry == 'Manufacturing' %}selected{% endif %}>Manufacturing</option>
                                                <option value="Retail" {% if user.industry == 'Retail' %}selected{% endif %}>Retail</option>
                                                <option value="Consulting" {% if user.industry == 'Consulting' %}selected{% endif %}>Consulting</option>
                                                <option value="Real Estate" {% if user.industry == 'Real Estate' %}selected{% endif %}>Real Estate</option>
                                                <option value="Media" {% if user.industry == 'Media' %}selected{% endif %}>Media</option>
                                                <option value="Other" {% if user.industry == 'Other' %}selected{% endif %}>Other</option>
                                            </select>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Location</label>
                                            <input type="text" class="form-control" name="location" value="{{ user.location or '' }}">
                                        </div>
                                    </div>
                                    
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Company</label>
                                            <input type="text" class="form-control" name="company" value="{{ user.company or '' }}">
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <label class="form-label">Position</label>
                                            <input type="text" class="form-control" name="position" value="{{ user.position or '' }}">
                                        </div>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Phone</label>
                                        <input type="tel" class="form-control" name="phone" value="{{ user.phone or '' }}">
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Skills</label>
                                        <input type="text" class="form-control" name="skills" value="{{ user.skills or '' }}" 
                                               placeholder="AI, Marketing, Sales, etc. (comma-separated)">
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Interests</label>
                                        <input type="text" class="form-control" name="interests" value="{{ user.interests or '' }}" 
                                               placeholder="Startups, Innovation, Networking, etc. (comma-separated)">
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label class="form-label">Bio</label>
                                        <textarea class="form-control" name="bio" rows="4" 
                                                  placeholder="Tell us about yourself and your professional background...">{{ user.bio or '' }}</textarea>
                                    </div>
                                    
                                    <button type="submit" class="btn btn-primary">
                                        <i class="fas fa-save"></i> Update Profile
                                    </button>
                                </form>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0"><i class="fas fa-user"></i> Profile Overview</h5>
                            </div>
                            <div class="card-body">
                                <div class="text-center mb-3">
                                    <i class="fas fa-user-circle fa-4x text-muted"></i>
                                    <h5 class="mt-2">{{ user.name }}</h5>
                                    <p class="text-muted">{{ user.email }}</p>
                                </div>
                                
                                <hr>
                                
                                <div class="mb-2">
                                    <strong>Industry:</strong> {{ user.industry or 'Not specified' }}
                                </div>
                                <div class="mb-2">
                                    <strong>Location:</strong> {{ user.location or 'Not specified' }}
                                </div>
                                <div class="mb-2">
                                    <strong>Company:</strong> {{ user.company or 'Not specified' }}
                                </div>
                                <div class="mb-2">
                                    <strong>Position:</strong> {{ user.position or 'Not specified' }}
                                </div>
                                <div class="mb-2">
                                    <strong>Member Since:</strong> {{ user.created_at[:10] }}
                                </div>
                                
                                {% if user.skills %}
                                <hr>
                                <div class="mb-2">
                                    <strong>Skills:</strong>
                                    <div class="mt-1">
                                        {% for skill in user.skills.split(',') %}
                                        <span class="badge bg-secondary me-1">{{ skill.strip() }}</span>
                                        {% endfor %}
                                    </div>
                                </div>
                                {% endif %}
                                
                                {% if user.interests %}
                                <div class="mb-2">
                                    <strong>Interests:</strong>
                                    <div class="mt-1">
                                        {% for interest in user.interests.split(',') %}
                                        <span class="badge bg-info me-1">{{ interest.strip() }}</span>
                                        {% endfor %}
                                    </div>
                                </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('updateProfileForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            try {
                const response = await fetch('/api/update-profile', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Profile updated successfully!');
                    location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error updating profile: ' + error.message);
            }
        });
    </script>
</body>
</html>
'''

MESSAGES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Messages - Golden Coyotes</title>
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
                <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                <a class="nav-link text-white active" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                <hr>
                <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
            </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <h1><i class="fas fa-envelope"></i> Messages</h1>
                
                <div class="row">
                    <!-- Message List -->
                    <div class="col-md-4">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5><i class="fas fa-inbox"></i> Conversations</h5>
                            </div>
                            <div class="card-body p-0">
                                {% for connection in connections %}
                                <div class="p-3 border-bottom cursor-pointer" onclick="openConversation('{{ connection.id }}')">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h6 class="mb-1">{{ connection.name }}</h6>
                                            <small class="text-muted">{{ connection.industry }} • {{ connection.location }}</small>
                                        </div>
                                        <i class="fas fa-chevron-right text-muted"></i>
                                    </div>
                                </div>
                                {% endfor %}
                                
                                {% if not connections %}
                                <div class="p-3 text-center text-muted">
                                    <i class="fas fa-users fa-2x mb-2"></i>
                                    <p>No connections yet</p>
                                    <a href="/network" class="btn btn-sm btn-primary">Find Connections</a>
                                </div>
                                {% endif %}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Recent Messages -->
                    <div class="col-md-8">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-light">
                                <h5><i class="fas fa-comments"></i> Recent Messages</h5>
                            </div>
                            <div class="card-body">
                                {% for message in messages[:10] %}
                                <div class="p-3 border-bottom">
                                    <div class="d-flex justify-content-between align-items-start">
                                        <div>
                                            <h6 class="mb-1">
                                                {% if message.sender_id == session.user_id %}
                                                To: {{ message.recipient_name }}
                                                {% else %}
                                                From: {{ message.sender_name }}
                                                {% endif %}
                                            </h6>
                                            <p class="mb-1"><strong>{{ message.subject }}</strong></p>
                                            <p class="mb-1">{{ message.content[:100] }}{% if message.content|length > 100 %}...{% endif %}</p>
                                            <small class="text-muted">{{ message.created_at }}</small>
                                        </div>
                                        <div>
                                            {% if message.sender_id != session.user_id %}
                                            <button class="btn btn-sm btn-outline-primary" onclick="openConversation('{{ message.sender_id }}')">
                                                <i class="fas fa-reply"></i> Reply
                                            </button>
                                            {% endif %}
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                                
                                {% if not messages %}
                                <div class="text-center text-muted py-4">
                                    <i class="fas fa-envelope-open fa-3x mb-3"></i>
                                    <h5>No messages yet</h5>
                                    <p>Start a conversation with someone from your network!</p>
                                </div>
                                {% endif %}
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

CONVERSATION_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .message-bubble { max-width: 70%; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px; }
        .message-sent { background-color: #007bff; color: white; margin-left: auto; }
        .message-received { background-color: #e9ecef; color: #333; }
        .chat-container { height: 500px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-2 sidebar text-white p-4">
                <h3 class="mb-4"><i class="fas fa-brain"></i> Golden Coyotes</h3>
                <nav class="nav flex-column">
                    <a class="nav-link text-white" href="/dashboard"><i class="fas fa-tachometer-alt"></i> Dashboard</a>
                    <a class="nav-link text-white" href="/opportunities"><i class="fas fa-briefcase"></i> Browse Opportunities</a>
                    <a class="nav-link text-white" href="/my-opportunities"><i class="fas fa-list"></i> My Opportunities</a>
                    <a class="nav-link text-white" href="/create-opportunity"><i class="fas fa-plus"></i> Create Opportunity</a>
                    <a class="nav-link text-white" href="/network"><i class="fas fa-users"></i> Network</a>
                    <a class="nav-link text-white active" href="/messages"><i class="fas fa-envelope"></i> Messages</a>
                    <a class="nav-link text-white" href="/profile"><i class="fas fa-user"></i> Profile</a>
                    <hr>
                    <a class="nav-link text-white" href="/logout"><i class="fas fa-sign-out-alt"></i> Logout</a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1><i class="fas fa-comments"></i> Conversation with {{ other_user.name }}</h1>
                    <a href="/messages" class="btn btn-outline-secondary">
                        <i class="fas fa-arrow-left"></i> Back to Messages
                    </a>
                </div>
                
                <div class="card border-0 shadow-sm">
                    <div class="card-header bg-primary text-white">
                        <h5><i class="fas fa-user"></i> {{ other_user.name }} - {{ other_user.industry }}</h5>
                    </div>
                    
                    <!-- Messages -->
                    <div class="card-body chat-container" id="chatContainer">
                        {% for message in messages %}
                        <div class="d-flex {% if message.sender_id == current_user_id %}justify-content-end{% else %}justify-content-start{% endif %}">
                            <div class="message-bubble {% if message.sender_id == current_user_id %}message-sent{% else %}message-received{% endif %}">
                                <strong>{{ message.subject }}</strong><br>
                                {{ message.content }}<br>
                                <small class="opacity-75">{{ message.created_at }}</small>
                            </div>
                        </div>
                        {% endfor %}
                        
                        {% if not messages %}
                        <div class="text-center text-muted py-4">
                            <i class="fas fa-comments fa-3x mb-3"></i>
                            <h5>Start a conversation</h5>
                            <p>Send your first message to {{ other_user.name }}</p>
                        </div>
                        {% endif %}
                    </div>
                    
                    <!-- Message Form -->
                    <div class="card-footer">
                        <form id="messageForm">
                            <div class="row g-2">
                                <div class="col-md-3">
                                    <input type="text" class="form-control" id="subject" placeholder="Subject" required>
                                </div>
                                <div class="col-md-7">
                                    <input type="text" class="form-control" id="content" placeholder="Type your message..." required>
                                </div>
                                <div class="col-md-2">
                                    <button type="submit" class="btn btn-primary w-100">
                                        <i class="fas fa-paper-plane"></i> Send
                                    </button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('messageForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const subject = document.getElementById('subject').value;
            const content = document.getElementById('content').value;
            
            if (!subject || !content) {
                alert('Please fill in both subject and message');
                return;
            }
            
            try {
                const response = await fetch('/api/send-message', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        recipient_id: '{{ other_user.id }}',
                        subject: subject,
                        content: content
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Clear form
                    document.getElementById('subject').value = '';
                    document.getElementById('content').value = '';
                    
                    // Reload page to show new message
                    location.reload();
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Error sending message: ' + error.message);
            }
        });
        
        // Auto-scroll to bottom of chat
        const chatContainer = document.getElementById('chatContainer');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    platform = GoldenCoyotesPlatform()
    platform.run()
