#!/usr/bin/env python3
"""
Golden Coyotes Admin Panel - AFTER FEEDBACK Implementation
Panel de administración mejorado para monitorear la plataforma Golden Coyotes
según el feedback de Junio 2025

Funcionalidades de Admin:
- Dashboard con métricas según los 4 cuadrantes
- Monitoreo de redes "Friends & Family"
- Análisis de oportunidades públicas vs dirigidas
- Sistema de invitaciones y networking
- Métricas PUSH & PULL
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

def create_admin_app():
    """Create Flask admin application with enhanced UI for feedback requirements"""
    app = Flask(__name__)
    app.secret_key = 'admin_golden_coyotes_after_feedback_' + str(uuid.uuid4())
    CORS(app)
    
    # Enhanced mock data storage following feedback structure
    app.data = {
        'users': {},
        'opportunities_public': {},
        'opportunities_directed': {},
        'user_networks': {},  # Friends & Family networks
        'invitations': {},
        'company_access': {},  # PUSH & PULL data
        'ai_recommendations': {},
        'interactions': [],
        'notifications': [],
        'metrics': {}
    }
    
    # Initialize with enhanced sample data
    initialize_enhanced_sample_data(app)
    
    @app.route('/')
    def admin_dashboard():
        """Admin dashboard with feedback-based metrics"""
        metrics = calculate_enhanced_metrics(app)
        return render_template_string(ADMIN_DASHBOARD_TEMPLATE, metrics=metrics)
    
    @app.route('/users-networks')
    def users_networks():
        """Monitor user networks and Friends & Family connections"""
        networks_data = get_networks_analytics(app)
        return render_template_string(USERS_NETWORKS_TEMPLATE, data=networks_data)
    
    @app.route('/opportunities-analysis')
    def opportunities_analysis():
        """Analysis of public vs directed opportunities"""
        opportunities_data = get_opportunities_analytics(app)
        return render_template_string(OPPORTUNITIES_ANALYSIS_TEMPLATE, data=opportunities_data)
    
    @app.route('/push-pull-monitor')
    def push_pull_monitor():
        """Monitor PUSH & PULL company access system"""
        push_pull_data = get_push_pull_analytics(app)
        return render_template_string(PUSH_PULL_MONITOR_TEMPLATE, data=push_pull_data)
    
    @app.route('/invitations-system')
    def invitations_system():
        """Monitor invitation system and network growth"""
        invitations_data = get_invitations_analytics(app)
        return render_template_string(INVITATIONS_SYSTEM_TEMPLATE, data=invitations_data)
    
    @app.route('/ai-recommendations')
    def ai_recommendations():
        """Monitor AI recommendation system performance"""
        ai_data = get_ai_analytics(app)
        return render_template_string(AI_RECOMMENDATIONS_TEMPLATE, data=ai_data)
    
    @app.route('/quadrants-performance')
    def quadrants_performance():
        """Performance metrics for the 4 quadrants"""
        quadrants_data = get_quadrants_analytics(app)
        return render_template_string(QUADRANTS_PERFORMANCE_TEMPLATE, data=quadrants_data)
    
    # API endpoints for admin operations
    @app.route('/api/admin/user-activity/<user_id>')
    def admin_user_activity(user_id):
        """Get detailed user activity"""
        activity = get_user_detailed_activity(app, user_id)
        return jsonify(activity)
    
    @app.route('/api/admin/opportunity-insights/<opp_id>')
    def admin_opportunity_insights(opp_id):
        """Get detailed opportunity insights"""
        insights = get_opportunity_detailed_insights(app, opp_id)
        return jsonify(insights)
    
    @app.route('/api/admin/network-analysis')
    def admin_network_analysis():
        """Get network analysis data"""
        analysis = perform_network_analysis(app)
        return jsonify(analysis)
    
    @app.route('/api/admin/export-data')
    def admin_export_data():
        """Export platform data for analysis"""
        export_data = prepare_data_export(app)
        return jsonify(export_data)
    
    return app

def initialize_enhanced_sample_data(app):
    """Initialize with enhanced sample data following feedback structure"""
    
    # Enhanced user data with Friends & Family networks
    sample_users = {
        'user1': {
            'id': 'user1',
            'name': 'Juan Carlos Pérez',
            'email': 'juan@example.com',
            'phone': '+52-555-1234',
            'industry_preferences': ['Tecnología', 'Fintech'],
            'friends_family_network': ['user2', 'user3'],
            'invited_by': None,
            'invitations_sent': 5,
            'invitations_accepted': 3,
            'quadrant_usage': {
                'subir_oportunidad': 12,
                'oportunidad_dirigida': 8,
                'buscar_oportunidad': 25,
                'mis_dirigidas': 15
            },
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'status': 'active'
        },
        'user2': {
            'id': 'user2',
            'name': 'María Elena García',
            'email': 'maria@example.com',
            'phone': '+52-555-5678',
            'industry_preferences': ['Salud', 'E-commerce'],
            'friends_family_network': ['user1', 'user4'],
            'invited_by': 'user1',
            'invitations_sent': 3,
            'invitations_accepted': 2,
            'quadrant_usage': {
                'subir_oportunidad': 6,
                'oportunidad_dirigida': 4,
                'buscar_oportunidad': 18,
                'mis_dirigidas': 7
            },
            'created_at': (datetime.now() - timedelta(days=5)).isoformat(),
            'last_activity': (datetime.now() - timedelta(hours=2)).isoformat(),
            'status': 'active'
        }
    }
    
    app.data['users'] = sample_users
    
    # Public opportunities (Cuadrante 1)
    sample_public_opportunities = {
        'pub_opp1': {
            'id': 'pub_opp1',
            'user_id': 'user1',
            'title': 'Fintech necesita desarrollador blockchain',
            'description': 'Startup de pagos digitales busca experto en blockchain para desarrollo de wallet cripto',
            'industry': 'Fintech',
            'type': 'servicio',
            'is_public': True,
            'views': 23,
            'interests_marked': 8,
            'contacts_made': 3,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=30)).isoformat(),
            'status': 'active'
        }
    }
    
    app.data['opportunities_public'] = sample_public_opportunities
    
    # Directed opportunities (Cuadrante 2)
    sample_directed_opportunities = {
        'dir_opp1': {
            'id': 'dir_opp1',
            'user_id': 'user1',
            'title': 'Inversión privada para e-commerce',
            'description': 'Buscamos inversionista para expandir tienda online de productos orgánicos',
            'industry': 'E-commerce',
            'type': 'producto',
            'is_public': False,
            'directed_to': ['user2'],
            'viewed_by': ['user2'],
            'responses': 1,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
    }
    
    app.data['opportunities_directed'] = sample_directed_opportunities
    
    # PUSH & PULL company access data
    sample_company_access = {
        'comp_acc1': {
            'id': 'comp_acc1',
            'user_id': 'user1',
            'company_name': 'BBVA México',
            'industry': 'Fintech',
            'access_level': 'Director de Innovación',
            'contact_person': 'Roberto Martínez',
            'opportunities_matched': 5,
            'successful_connections': 2,
            'created_at': datetime.now().isoformat()
        }
    }
    
    app.data['company_access'] = sample_company_access
    
    # Invitations tracking
    sample_invitations = {
        'inv1': {
            'id': 'inv1',
            'sender_id': 'user1',
            'recipient_phone': '+52-555-9999',
            'recipient_name': 'Pedro López',
            'platform': 'whatsapp',
            'status': 'sent',
            'sent_at': datetime.now().isoformat(),
            'accepted_at': None
        }
    }
    
    app.data['invitations'] = sample_invitations

def calculate_enhanced_metrics(app):
    """Calculate enhanced metrics based on feedback requirements"""
    total_users = len(app.data['users'])
    total_public_opportunities = len(app.data['opportunities_public'])
    total_directed_opportunities = len(app.data['opportunities_directed'])
    total_company_access = len(app.data['company_access'])
    total_invitations = len(app.data['invitations'])
    
    # Calculate network growth
    network_sizes = []
    for user in app.data['users'].values():
        network_sizes.append(len(user.get('friends_family_network', [])))
    
    avg_network_size = sum(network_sizes) / len(network_sizes) if network_sizes else 0
    
    # Calculate quadrant usage
    quadrant_usage = {
        'subir_oportunidad': 0,
        'oportunidad_dirigida': 0,
        'buscar_oportunidad': 0,
        'mis_dirigidas': 0
    }
    
    for user in app.data['users'].values():
        for quadrant, usage in user.get('quadrant_usage', {}).items():
            quadrant_usage[quadrant] += usage
    
    # Calculate conversion rates
    invitations_sent = sum([user.get('invitations_sent', 0) for user in app.data['users'].values()])
    invitations_accepted = sum([user.get('invitations_accepted', 0) for user in app.data['users'].values()])
    invitation_conversion_rate = (invitations_accepted / invitations_sent * 100) if invitations_sent > 0 else 0
    
    return {
        'total_users': total_users,
        'total_public_opportunities': total_public_opportunities,
        'total_directed_opportunities': total_directed_opportunities,
        'total_company_access': total_company_access,
        'total_invitations': total_invitations,
        'avg_network_size': round(avg_network_size, 1),
        'quadrant_usage': quadrant_usage,
        'invitation_conversion_rate': round(invitation_conversion_rate, 1),
        'active_networks': len([u for u in app.data['users'].values() if u.get('friends_family_network')]),
        'push_pull_matches': sum([ca.get('opportunities_matched', 0) for ca in app.data['company_access'].values()])
    }

def get_networks_analytics(app):
    """Get analytics for Friends & Family networks"""
    networks = []
    for user in app.data['users'].values():
        network_data = {
            'user': user,
            'network_size': len(user.get('friends_family_network', [])),
            'invitations_pending': user.get('invitations_sent', 0) - user.get('invitations_accepted', 0),
            'network_activity': sum(user.get('quadrant_usage', {}).values()),
            'connection_quality': 'Alta' if len(user.get('friends_family_network', [])) > 5 else 'Media'
        }
        networks.append(network_data)
    
    return {
        'networks': networks,
        'total_connections': sum([n['network_size'] for n in networks]),
        'avg_network_size': sum([n['network_size'] for n in networks]) / len(networks) if networks else 0
    }

def get_opportunities_analytics(app):
    """Get analytics for opportunities (public vs directed)"""
    public_opps = list(app.data['opportunities_public'].values())
    directed_opps = list(app.data['opportunities_directed'].values())
    
    public_performance = sum([opp.get('contacts_made', 0) for opp in public_opps])
    directed_performance = sum([opp.get('responses', 0) for opp in directed_opps])
    
    return {
        'public_opportunities': public_opps,
        'directed_opportunities': directed_opps,
        'public_performance': public_performance,
        'directed_performance': directed_performance,
        'total_views': sum([opp.get('views', 0) for opp in public_opps]),
        'conversion_rate': round((public_performance + directed_performance) / max(len(public_opps) + len(directed_opps), 1) * 100, 2)
    }

def get_push_pull_analytics(app):
    """Get PUSH & PULL system analytics"""
    company_access = list(app.data['company_access'].values())
    
    return {
        'company_access': company_access,
        'total_companies': len(company_access),
        'total_matches': sum([ca.get('opportunities_matched', 0) for ca in company_access]),
        'successful_connections': sum([ca.get('successful_connections', 0) for ca in company_access]),
        'industries_covered': len(set([ca.get('industry') for ca in company_access]))
    }

def get_invitations_analytics(app):
    """Get invitations system analytics"""
    invitations = list(app.data['invitations'].values())
    
    platform_breakdown = {}
    for inv in invitations:
        platform = inv.get('platform', 'unknown')
        platform_breakdown[platform] = platform_breakdown.get(platform, 0) + 1
    
    return {
        'invitations': invitations,
        'platform_breakdown': platform_breakdown,
        'total_sent': len(invitations),
        'accepted': len([inv for inv in invitations if inv.get('accepted_at')]),
        'pending': len([inv for inv in invitations if inv.get('status') == 'sent'])
    }

def get_ai_analytics(app):
    """Get AI recommendations analytics"""
    return {
        'recommendations_generated': 150,
        'click_through_rate': 23.5,
        'conversion_rate': 12.8,
        'user_satisfaction': 4.2
    }

def get_quadrants_analytics(app):
    """Get 4 quadrants performance analytics"""
    quadrant_metrics = {}
    
    for user in app.data['users'].values():
        for quadrant, usage in user.get('quadrant_usage', {}).items():
            if quadrant not in quadrant_metrics:
                quadrant_metrics[quadrant] = {'total_usage': 0, 'users': 0}
            quadrant_metrics[quadrant]['total_usage'] += usage
            quadrant_metrics[quadrant]['users'] += 1 if usage > 0 else 0
    
    return {
        'quadrant_metrics': quadrant_metrics,
        'most_used': max(quadrant_metrics.keys(), key=lambda x: quadrant_metrics[x]['total_usage']) if quadrant_metrics else None,
        'least_used': min(quadrant_metrics.keys(), key=lambda x: quadrant_metrics[x]['total_usage']) if quadrant_metrics else None
    }

# Mock functions for other admin operations
def get_user_detailed_activity(app, user_id):
    """Get detailed user activity"""
    return {'activity': 'mock_activity_data'}

def get_opportunity_detailed_insights(app, opp_id):
    """Get detailed opportunity insights"""
    return {'insights': 'mock_insights_data'}

def perform_network_analysis(app):
    """Perform network analysis"""
    return {'analysis': 'mock_network_analysis'}

def prepare_data_export(app):
    """Prepare data for export"""
    return {'export': 'mock_export_data'}

# ==================== HTML TEMPLATES ====================

ADMIN_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard - Golden Coyotes After Feedback</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .sidebar { min-height: 100vh; background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); }
        .content { background-color: #f8f9fa; min-height: 100vh; }
        .metric-card { transition: transform 0.2s; }
        .metric-card:hover { transform: translateY(-5px); }
        .quadrant-metric { 
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Enhanced Sidebar -->
            <div class="col-md-2 sidebar text-white p-4">
                <h3 class="mb-4">
                    <i class="fas fa-crown"></i> 
                    Admin Panel
                </h3>
                <p class="small opacity-75 mb-4">Golden Coyotes - After Feedback</p>
                
                <nav class="nav flex-column">
                    <a class="nav-link text-white active" href="/">
                        <i class="fas fa-tachometer-alt"></i> Dashboard Principal
                    </a>
                    <a class="nav-link text-white" href="/users-networks">
                        <i class="fas fa-users"></i> Redes Friends & Family
                    </a>
                    <a class="nav-link text-white" href="/opportunities-analysis">
                        <i class="fas fa-briefcase"></i> Análisis Oportunidades
                    </a>
                    <a class="nav-link text-white" href="/quadrants-performance">
                        <i class="fas fa-th-large"></i> 4 Cuadrantes
                    </a>
                    <a class="nav-link text-white" href="/push-pull-monitor">
                        <i class="fas fa-building"></i> PUSH & PULL
                    </a>
                    <a class="nav-link text-white" href="/invitations-system">
                        <i class="fas fa-share-alt"></i> Sistema Invitaciones
                    </a>
                    <a class="nav-link text-white" href="/ai-recommendations">
                        <i class="fas fa-robot"></i> IA & Recomendaciones
                    </a>
                </nav>
            </div>
            
            <!-- Main Content -->
            <div class="col-md-10 content p-4">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <div>
                        <h1><i class="fas fa-chart-line"></i> Dashboard Administrativo</h1>
                        <p class="text-muted">Monitoreo de la plataforma Golden Coyotes basado en feedback Junio 2025</p>
                    </div>
                    <span class="badge bg-success fs-6">Sistema Online</span>
                </div>
                
                <!-- KPI Cards Row 1 -->
                <div class="row mb-4">
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-users fa-3x text-primary mb-3"></i>
                                <h3 class="text-primary">{{ metrics.total_users }}</h3>
                                <p class="mb-0 small">Usuarios Totales</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-globe fa-3x text-success mb-3"></i>
                                <h3 class="text-success">{{ metrics.total_public_opportunities }}</h3>
                                <p class="mb-0 small">Oportunidades Públicas</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-share fa-3x text-info mb-3"></i>
                                <h3 class="text-info">{{ metrics.total_directed_opportunities }}</h3>
                                <p class="mb-0 small">Oportunidades Dirigidas</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-network-wired fa-3x text-warning mb-3"></i>
                                <h3 class="text-warning">{{ metrics.avg_network_size }}</h3>
                                <p class="mb-0 small">Tamaño Promedio Red</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-building fa-3x text-danger mb-3"></i>
                                <h3 class="text-danger">{{ metrics.total_company_access }}</h3>
                                <p class="mb-0 small">PUSH & PULL</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card h-100 border-0 shadow-sm">
                            <div class="card-body text-center">
                                <i class="fas fa-percentage fa-3x text-secondary mb-3"></i>
                                <h3 class="text-secondary">{{ metrics.invitation_conversion_rate }}%</h3>
                                <p class="mb-0 small">Conversión Invitaciones</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 4 Cuadrantes Performance -->
                <div class="row mb-4">
                    <div class="col-12">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-primary text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-th-large"></i> 
                                    Rendimiento de los 4 Cuadrantes (Según Feedback)
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3">
                                        <div class="card quadrant-metric border-0 h-100">
                                            <div class="card-body text-center">
                                                <i class="fas fa-upload fa-2x mb-2"></i>
                                                <h4>{{ metrics.quadrant_usage.subir_oportunidad }}</h4>
                                                <p class="mb-0">Subir Oportunidad</p>
                                                <small class="opacity-75">Cuadrante 1</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card quadrant-metric border-0 h-100" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                                            <div class="card-body text-center">
                                                <i class="fas fa-share fa-2x mb-2"></i>
                                                <h4>{{ metrics.quadrant_usage.oportunidad_dirigida }}</h4>
                                                <p class="mb-0">Oportunidad Dirigida</p>
                                                <small class="opacity-75">Cuadrante 2</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card quadrant-metric border-0 h-100" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                                            <div class="card-body text-center">
                                                <i class="fas fa-search fa-2x mb-2"></i>
                                                <h4>{{ metrics.quadrant_usage.buscar_oportunidad }}</h4>
                                                <p class="mb-0">Buscar Oportunidad</p>
                                                <small class="opacity-75">Cuadrante 3</small>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="card quadrant-metric border-0 h-100" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                            <div class="card-body text-center">
                                                <i class="fas fa-envelope fa-2x mb-2"></i>
                                                <h4>{{ metrics.quadrant_usage.mis_dirigidas }}</h4>
                                                <p class="mb-0">Mis Dirigidas</p>
                                                <small class="opacity-75">Cuadrante 4</small>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Additional Metrics -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-success text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-users-cog"></i> 
                                    Métricas Friends & Family
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col-4">
                                        <h4 class="text-primary">{{ metrics.active_networks }}</h4>
                                        <small class="text-muted">Redes Activas</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-success">{{ metrics.avg_network_size }}</h4>
                                        <small class="text-muted">Promedio Contactos</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-info">{{ metrics.total_invitations }}</h4>
                                        <small class="text-muted">Invitaciones</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-warning text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-building"></i> 
                                    Sistema PUSH & PULL
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row text-center">
                                    <div class="col-4">
                                        <h4 class="text-primary">{{ metrics.total_company_access }}</h4>
                                        <small class="text-muted">Empresas Registradas</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-success">{{ metrics.push_pull_matches }}</h4>
                                        <small class="text-muted">Matches Realizados</small>
                                    </div>
                                    <div class="col-4">
                                        <h4 class="text-info">89%</h4>
                                        <small class="text-muted">Tasa Éxito</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- System Status -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card border-0 shadow-sm">
                            <div class="card-header bg-secondary text-white">
                                <h5 class="mb-0">
                                    <i class="fas fa-server"></i> 
                                    Estado del Sistema - Implementación After Feedback
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3 text-center">
                                        <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                                        <h6>4 Cuadrantes</h6>
                                        <span class="badge bg-success">Implementado</span>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <i class="fas fa-users fa-2x text-success mb-2"></i>
                                        <h6>Friends & Family</h6>
                                        <span class="badge bg-success">Activo</span>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <i class="fas fa-building fa-2x text-success mb-2"></i>
                                        <h6>PUSH & PULL</h6>
                                        <span class="badge bg-success">Funcionando</span>
                                    </div>
                                    <div class="col-md-3 text-center">
                                        <i class="fas fa-robot fa-2x text-warning mb-2"></i>
                                        <h6>IA Recomendaciones</h6>
                                        <span class="badge bg-warning">En Mejora</span>
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
</body>
</html>
'''

# Plantillas adicionales para las otras páginas del admin
USERS_NETWORKS_TEMPLATE = '''
<h2>Análisis de Redes Friends & Family</h2>
<p>Monitoreo detallado de las redes de contactos según el feedback</p>
'''

OPPORTUNITIES_ANALYSIS_TEMPLATE = '''
<h2>Análisis de Oportunidades Públicas vs Dirigidas</h2>
<p>Comparativa de rendimiento entre cuadrantes 1 y 2</p>
'''

PUSH_PULL_MONITOR_TEMPLATE = '''
<h2>Monitor del Sistema PUSH & PULL</h2>
<p>Seguimiento de empresas y conexiones empresariales</p>
'''

INVITATIONS_SYSTEM_TEMPLATE = '''
<h2>Sistema de Invitaciones</h2>
<p>Análisis de invitaciones por WhatsApp, Facebook, LinkedIn, Instagram</p>
'''

AI_RECOMMENDATIONS_TEMPLATE = '''
<h2>Rendimiento de IA y Recomendaciones</h2>
<p>Métricas del sistema de recomendaciones inteligentes</p>
'''

QUADRANTS_PERFORMANCE_TEMPLATE = '''
<h2>Rendimiento de los 4 Cuadrantes</h2>
<p>Análisis detallado del uso de cada cuadrante según feedback</p>
'''

if __name__ == "__main__":
    import sys
    
    port = 8081  # Puerto diferente para admin
    if len(sys.argv) > 1 and sys.argv[1] == '--port':
        port = int(sys.argv[2]) if len(sys.argv) > 2 else port
    
    app = create_admin_app()
    print("🚀 Starting Golden Coyotes Admin Panel - After Feedback")
    print(f"👑 ADMIN: http://localhost:{port}")
    print("✨ Panel de administración mejorado según feedback Junio 2025")
    print("\n📊 Características:")
    print("   - Monitoreo de 4 cuadrantes")
    print("   - Análisis redes Friends & Family")
    print("   - Métricas PUSH & PULL")
    print("   - Sistema de invitaciones")
    print("   - IA y recomendaciones")
    
    app.run(host='0.0.0.0', port=port, debug=True)