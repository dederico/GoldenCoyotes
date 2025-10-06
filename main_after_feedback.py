#!/usr/bin/env python3
"""
Golden Coyotes Platform - AFTER FEEDBACK - Unified Entry Point
Punto de entrada unificado para deployment en Render.com

Este archivo maneja ambos servicios:
- Vistas de USUARIO (golden_coyotes_after_feedback.py) - Puerto 5001
- Vistas de ADMIN (web_app_after_feedback.py) - Puerto 8081

Para deployment en Render.com, este archivo permite ejecutar ambos servicios
en una sola instancia para evitar costos adicionales.
"""

import os
import sys
import threading
import time
from flask import Flask, redirect, render_template_string
import logging

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_unified_app():
    """Create unified Flask app that redirects to user and admin interfaces"""
    from flask import request
    import requests as http_requests

    app = Flask(__name__)
    app.secret_key = os.getenv('SECRET_KEY', 'golden_coyotes_unified_' + str(time.time()))

    @app.route('/')
    def index():
        """Main landing page with links to both interfaces"""
        return render_template_string(UNIFIED_LANDING_TEMPLATE)

    @app.route('/user')
    @app.route('/user/<path:path>')
    def user_proxy(path=''):
        """Proxy requests to user service"""
        return redirect(f'http://localhost:5001/{path}', code=302)

    @app.route('/admin')
    @app.route('/admin/<path:path>')
    def admin_proxy(path=''):
        """Proxy requests to admin service"""
        return redirect(f'http://localhost:8081/{path}', code=302)

    @app.route('/health')
    def health():
        """Health check endpoint for Render.com"""
        return {'status': 'healthy', 'services': ['user_app', 'admin_app']}, 200

    return app

def start_user_service():
    """Start the user service in a separate thread"""
    try:
        from golden_coyotes_after_feedback import GoldenCoyotesAfterFeedback
        user_app = GoldenCoyotesAfterFeedback()
        
        logger.info("🚀 Starting User Service on port 5001")
        user_app.run(host='0.0.0.0', port=5001, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Error starting user service: {e}")

def start_admin_service():
    """Start the admin service in a separate thread"""
    try:
        from web_app_after_feedback import create_admin_app
        admin_app = create_admin_app()
        
        logger.info("👑 Starting Admin Service on port 8081")
        admin_app.run(host='0.0.0.0', port=8081, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Error starting admin service: {e}")

def start_unified_service():
    """Start the unified landing service"""
    try:
        unified_app = create_unified_app()
        
        # Get port from environment (Render.com sets PORT)
        port = int(os.getenv('PORT', 10000))
        
        logger.info(f"🌐 Starting Unified Landing Service on port {port}")
        unified_app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        logger.error(f"❌ Error starting unified service: {e}")

def main():
    """Main entry point"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Golden Coyotes - After Feedback Deployment        ║
    ║                                                              ║
    ║  🎯 Implementación basada en feedback Junio 2025            ║
    ║  👥 Vistas de Usuario + Panel de Administrador              ║
    ║  🚀 Optimizado para deployment en Render.com               ║
    ║  💰 Una sola instancia = sin costos adicionales            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Determine deployment mode
    is_render_deployment = os.getenv('RENDER') is not None
    run_mode = os.getenv('RUN_MODE', 'unified')
    
    if is_render_deployment or run_mode == 'unified':
        logger.info("🌐 Running in UNIFIED mode for Render.com deployment")
        
        # Start user and admin services in background threads
        user_thread = threading.Thread(target=start_user_service, daemon=True)
        admin_thread = threading.Thread(target=start_admin_service, daemon=True)
        
        user_thread.start()
        admin_thread.start()
        
        # Wait a bit for services to start
        time.sleep(2)
        
        # Start unified landing service (this blocks)
        start_unified_service()
        
    else:
        # Development mode - choose which service to run
        if len(sys.argv) > 1:
            service = sys.argv[1]
            
            if service == 'user':
                logger.info("👥 Running USER service only")
                start_user_service()
                
            elif service == 'admin':
                logger.info("👑 Running ADMIN service only")  
                start_admin_service()
                
            elif service == 'unified':
                logger.info("🌐 Running UNIFIED service only")
                start_unified_service()
                
            else:
                print("❌ Invalid service. Use: user, admin, or unified")
                sys.exit(1)
        else:
            # Default: run all services
            logger.info("🚀 Running ALL services (development mode)")
            
            user_thread = threading.Thread(target=start_user_service, daemon=True)
            admin_thread = threading.Thread(target=start_admin_service, daemon=True)
            
            user_thread.start()
            admin_thread.start()
            
            time.sleep(2)
            start_unified_service()

# HTML template for unified landing page
UNIFIED_LANDING_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Coyotes - After Feedback</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            min-height: 100vh; 
        }
        .service-card { 
            transition: all 0.3s ease; 
            cursor: pointer;
            border: none;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .service-card:hover { 
            transform: translateY(-10px); 
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        .user-card { 
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
        }
        .admin-card { 
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
    </style>
</head>
<body>
    <div class="hero d-flex align-items-center">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <h1 class="display-3 fw-bold mb-4">
                        <i class="fas fa-users-cog text-warning"></i>
                        Golden Coyotes
                    </h1>
                    <h2 class="h3 mb-4">After Feedback Implementation</h2>
                    <p class="lead mb-5">
                        Plataforma de networking empresarial implementada según el feedback de Junio 2025
                    </p>
                    
                    <div class="row justify-content-center">
                        <!-- User Interface -->
                        <div class="col-md-5 mb-4">
                            <div class="card service-card user-card h-100" onclick="window.location.href='/user'">
                                <div class="card-body text-center p-5">
                                    <i class="fas fa-users fa-5x mb-4"></i>
                                    <h3 class="card-title">Interfaz de Usuario</h3>
                                    <p class="card-text mb-4">
                                        Accede a los 4 cuadrantes: Subir Oportunidad, Oportunidad Dirigida, 
                                        Buscar Oportunidades y Mis Dirigidas
                                    </p>
                                    <div class="row text-center small">
                                        <div class="col-6">
                                            <i class="fas fa-upload"></i>
                                            <div>Subir</div>
                                        </div>
                                        <div class="col-6">
                                            <i class="fas fa-share"></i>
                                            <div>Dirigida</div>
                                        </div>
                                    </div>
                                    <div class="row text-center small mt-2">
                                        <div class="col-6">
                                            <i class="fas fa-search"></i>
                                            <div>Buscar</div>
                                        </div>
                                        <div class="col-6">
                                            <i class="fas fa-envelope"></i>
                                            <div>Dirigidas</div>
                                        </div>
                                    </div>
                                    <div class="mt-4">
                                        <span class="badge bg-light text-dark">Puerto 5001</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Admin Interface -->
                        <div class="col-md-5 mb-4">
                            <div class="card service-card admin-card h-100" onclick="window.location.href='/admin'">
                                <div class="card-body text-center p-5">
                                    <i class="fas fa-crown fa-5x mb-4"></i>
                                    <h3 class="card-title">Panel de Administrador</h3>
                                    <p class="card-text mb-4">
                                        Monitorea métricas, redes Friends & Family, sistema PUSH & PULL, 
                                        e invitaciones
                                    </p>
                                    <div class="row text-center small">
                                        <div class="col-6">
                                            <i class="fas fa-chart-bar"></i>
                                            <div>Métricas</div>
                                        </div>
                                        <div class="col-6">
                                            <i class="fas fa-network-wired"></i>
                                            <div>Redes</div>
                                        </div>
                                    </div>
                                    <div class="row text-center small mt-2">
                                        <div class="col-6">
                                            <i class="fas fa-building"></i>
                                            <div>PUSH & PULL</div>
                                        </div>
                                        <div class="col-6">
                                            <i class="fas fa-robot"></i>
                                            <div>IA</div>
                                        </div>
                                    </div>
                                    <div class="mt-4">
                                        <span class="badge bg-light text-dark">Puerto 8081</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Features Overview -->
                    <div class="row mt-5">
                        <div class="col-12">
                            <div class="card bg-light text-dark">
                                <div class="card-body">
                                    <h4 class="card-title">
                                        <i class="fas fa-star text-warning"></i>
                                        Características Implementadas (Feedback Junio 2025)
                                    </h4>
                                    <div class="row mt-4">
                                        <div class="col-md-3 text-center">
                                            <i class="fas fa-th-large fa-2x text-primary mb-2"></i>
                                            <h6>4 Cuadrantes</h6>
                                            <small class="text-muted">Según estructura del feedback</small>
                                        </div>
                                        <div class="col-md-3 text-center">
                                            <i class="fas fa-users fa-2x text-success mb-2"></i>
                                            <h6>Friends & Family</h6>
                                            <small class="text-muted">Redes de confianza</small>
                                        </div>
                                        <div class="col-md-3 text-center">
                                            <i class="fas fa-share-alt fa-2x text-info mb-2"></i>
                                            <h6>Invitaciones Sociales</h6>
                                            <small class="text-muted">WhatsApp, FB, LinkedIn</small>
                                        </div>
                                        <div class="col-md-3 text-center">
                                            <i class="fas fa-building fa-2x text-warning mb-2"></i>
                                            <h6>PUSH & PULL</h6>
                                            <small class="text-muted">Acceso empresarial</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Deployment Info -->
                    <div class="row mt-4">
                        <div class="col-12">
                            <div class="alert alert-success">
                                <i class="fas fa-cloud"></i>
                                <strong>Deployment Render.com:</strong> 
                                Ambos servicios funcionan en una sola instancia para optimizar costos.
                                Base de datos incluida para persistencia completa.
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
    main()