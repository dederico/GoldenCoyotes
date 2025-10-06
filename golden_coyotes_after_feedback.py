#!/usr/bin/env python3
"""
Golden Coyotes Platform - AFTER FEEDBACK Implementation
Plataforma de networking basada en economía colaborativa "Friends & Family"
Implementación basada en el feedback del documento de Junio 2025

Funcionalidades principales:
- 4 Cuadrantes según feedback
- Sistema de invitaciones WhatsApp/Redes Sociales
- Networking "Friends & Family"
- Oportunidades públicas vs dirigidas
- Sistema PUSH & PULL
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session
from flask_cors import CORS

from database_setup import DatabaseManager
from email_service import EmailService
from ai_matching_engine import AIMatchingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldenCoyotesAfterFeedback:
    """Golden Coyotes Platform - Implementación basada en feedback"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = os.getenv('SECRET_KEY', 'golden_coyotes_after_feedback_' + str(uuid.uuid4()))
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
    
    def setup_routes(self):
        """Setup all routes according to feedback"""
        
        # ==================== PÁGINA PRINCIPAL ====================
        @self.app.route('/')
        def home():
            """Página inicial con video explicativo"""
            return render_template_string(HOME_TEMPLATE)
        
        # ==================== REGISTRO Y LOGIN ====================
        @self.app.route('/register', methods=['GET', 'POST'])
        def register():
            """Registro de usuario con términos y políticas"""
            if request.method == 'POST':
                data = request.form.to_dict()  # Usar form data para registro
                
                # Validar campos requeridos
                if not data.get('name') or not data.get('email') or not data.get('password'):
                    flash('Por favor completa todos los campos requeridos', 'error')
                    return render_template_string(REGISTER_TEMPLATE)
                
                # Mock user creation - en producción guardar en base de datos real
                user_data = {
                    'id': 'user_' + data.get('email').split('@')[0],
                    'name': data.get('name'),
                    'email': data.get('email'),
                    'phone': data.get('phone', ''),
                    'industry_preferences': data.get('industry_preferences', '').split(',') if data.get('industry_preferences') else [],
                    'accepted_terms': data.get('accepted_terms') == 'on',
                    'accepted_privacy': data.get('accepted_privacy') == 'on',
                    'safe_contact_agreement': data.get('safe_contact_agreement') == 'on',
                    'created_at': datetime.now().isoformat()
                }
                
                flash('¡Registro exitoso! Ahora puedes iniciar sesión.', 'success')
                return redirect(url_for('login'))
            
            return render_template_string(REGISTER_TEMPLATE)
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login de usuario"""
            if request.method == 'POST':
                data = request.form.to_dict()  # Siempre usar form data para login
                email = data.get('email')
                password = data.get('password')
                
                # Mock authentication - en producción usar base de datos real
                if email and password:
                    # Crear usuario mock si no existe
                    mock_user = {
                        'id': 'user_' + email.split('@')[0],
                        'name': email.split('@')[0].title(),
                        'email': email
                    }
                    
                    session['user_id'] = mock_user['id']
                    session['user_name'] = mock_user['name']
                    flash('¡Bienvenido a Golden Coyotes!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Por favor ingresa email y contraseña', 'error')
            
            return render_template_string(LOGIN_TEMPLATE)
        
        # ==================== DASHBOARD PRINCIPAL (4 CUADRANTES) ====================
        @self.app.route('/dashboard')
        @self.require_login
        def dashboard():
            """Dashboard principal con los 4 cuadrantes según feedback"""
            user_id = session['user_id']
            
            # Obtener contadores para notificaciones
            counters = {
                'oportunidades_publicas': len(self.get_public_opportunities(user_id)),
                'oportunidades_dirigidas': len(self.get_directed_opportunities(user_id)),
                'mis_oportunidades': len(self.get_my_opportunities(user_id)),
                'invitaciones_pendientes': len(self.get_pending_invitations(user_id))
            }
            
            return render_template_string(DASHBOARD_TEMPLATE, counters=counters, user_name=session['user_name'])
        
        # ==================== CUADRANTE 1: SUBO OPORTUNIDAD ====================
        @self.app.route('/subir-oportunidad', methods=['GET', 'POST'])
        @self.require_login
        def subir_oportunidad():
            """Cuadrante 1: Subir oportunidad pública"""
            if request.method == 'POST':
                data = request.form.to_dict()
                
                opportunity_data = {
                    'user_id': session['user_id'],
                    'title': data.get('title'),
                    'description': data.get('description'),
                    'industry': data.get('industry'),
                    'type': data.get('type'),  # producto o servicio
                    'is_public': True,  # Siempre pública en cuadrante 1
                    'media_files': data.get('media_files', []),
                    'created_at': datetime.now().isoformat()
                }
                
                try:
                    opp_id = self.create_opportunity(opportunity_data)
                    flash('¡Oportunidad publicada exitosamente!', 'success')
                    return redirect(url_for('dashboard'))
                except Exception as e:
                    flash(f'Error al crear oportunidad: {e}', 'error')
            
            return render_template_string(SUBIR_OPORTUNIDAD_TEMPLATE)
        
        # ==================== CUADRANTE 2: OPORTUNIDAD DIRIGIDA ====================
        @self.app.route('/oportunidad-dirigida', methods=['GET', 'POST'])
        @self.require_login
        def oportunidad_dirigida():
            """Cuadrante 2: Enviar oportunidad a contactos específicos"""
            user_id = session['user_id']
            my_contacts = self.get_user_contacts(user_id)
            
            if request.method == 'POST':
                data = request.form.to_dict()
                
                opportunity_data = {
                    'user_id': user_id,
                    'title': data.get('title'),
                    'description': data.get('description'),
                    'industry': data.get('industry'),
                    'type': data.get('type'),
                    'is_public': False,  # Privada/dirigida
                    'directed_to': data.get('selected_contacts', []),
                    'media_files': data.get('media_files', []),
                    'created_at': datetime.now().isoformat()
                }
                
                try:
                    opp_id = self.create_opportunity(opportunity_data)
                    # Notificar a los contactos seleccionados
                    self.notify_directed_opportunity(opp_id, opportunity_data['directed_to'])
                    flash('¡Oportunidad enviada a tus contactos!', 'success')
                    return redirect(url_for('dashboard'))
                except Exception as e:
                    flash(f'Error al enviar oportunidad: {e}', 'error')
            
            return render_template_string(OPORTUNIDAD_DIRIGIDA_TEMPLATE, contacts=my_contacts)
        
        # ==================== CUADRANTE 3: BUSCO OPORTUNIDAD GENERAL ====================
        @self.app.route('/buscar-oportunidad')
        @self.require_login
        def buscar_oportunidad():
            """Cuadrante 3: Ver oportunidades públicas con opción IA"""
            user_id = session['user_id']
            view_type = request.args.get('view', 'ai')  # 'ai' o 'all'
            
            if view_type == 'ai':
                # Oportunidades recomendadas por IA
                opportunities = self.get_ai_recommended_opportunities(user_id)
                title = "Oportunidades Recomendadas por IA"
            else:
                # Todas las oportunidades públicas
                opportunities = self.get_public_opportunities(user_id)
                title = "Todas las Oportunidades Públicas"
            
            return render_template_string(BUSCAR_OPORTUNIDAD_TEMPLATE, 
                                       opportunities=opportunities, 
                                       title=title, 
                                       view_type=view_type)
        
        # ==================== CUADRANTE 4: OPORTUNIDADES DIRIGIDAS A MÍ ====================
        @self.app.route('/mis-oportunidades-dirigidas')
        @self.require_login
        def mis_oportunidades_dirigidas():
            """Cuadrante 4: Ver oportunidades que me enviaron directamente"""
            user_id = session['user_id']
            directed_opportunities = self.get_directed_opportunities(user_id)
            
            return render_template_string(MIS_OPORTUNIDADES_DIRIGIDAS_TEMPLATE, 
                                       opportunities=directed_opportunities)
        
        # ==================== FUNCIONALIDAD PUSH & PULL ====================
        @self.app.route('/push-pull', methods=['GET', 'POST'])
        @self.require_login
        def push_pull():
            """Área PUSH & PULL - empresas donde tengo acceso"""
            user_id = session['user_id']
            
            if request.method == 'POST':
                data = request.form.to_dict()
                
                company_access = {
                    'user_id': user_id,
                    'company_name': data.get('company_name'),
                    'industry': data.get('industry'),
                    'access_level': data.get('access_level'),
                    'contact_person': data.get('contact_person'),
                    'description': data.get('description'),
                    'created_at': datetime.now().isoformat()
                }
                
                try:
                    self.create_company_access(company_access)
                    flash('¡Acceso a empresa registrado!', 'success')
                except Exception as e:
                    flash(f'Error: {e}', 'error')
            
            my_company_access = self.get_user_company_access(user_id)
            return render_template_string(PUSH_PULL_TEMPLATE, companies=my_company_access)
        
        # ==================== INVITACIONES ====================
        @self.app.route('/invitar-contactos')
        @self.require_login
        def invitar_contactos():
            """Sistema de invitaciones a WhatsApp, Facebook, LinkedIn, Instagram"""
            return render_template_string(INVITAR_CONTACTOS_TEMPLATE)
        
        # ==================== API ENDPOINTS ====================
        @self.app.route('/api/mark-interest', methods=['POST'])
        @self.require_login
        def mark_interest():
            """Marcar oportunidad como de interés"""
            data = request.get_json()
            user_id = session['user_id']
            opportunity_id = data.get('opportunity_id')
            
            try:
                self.mark_opportunity_interest(user_id, opportunity_id)
                return jsonify({'success': True})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/contact-opportunity-owner', methods=['POST'])
        @self.require_login
        def contact_opportunity_owner():
            """Contactar al dueño de la oportunidad"""
            data = request.get_json()
            user_id = session['user_id']
            opportunity_id = data.get('opportunity_id')
            message = data.get('message')
            
            try:
                self.send_contact_message(user_id, opportunity_id, message)
                return jsonify({'success': True, 'message': 'Mensaje enviado'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/logout')
        @self.require_login
        def logout():
            """Cerrar sesión"""
            session.clear()
            flash('¡Sesión cerrada exitosamente!', 'info')
            return redirect(url_for('home'))
    
    # ==================== MÉTODOS DE DATOS ====================
    def get_public_opportunities(self, user_id):
        """Obtener oportunidades públicas de la red del usuario"""
        # Mock data - en producción sería consulta a BD
        return [
            {
                'id': 'opp1',
                'title': 'Startup FinTech busca socio técnico',
                'description': 'Plataforma de pagos digitales necesita CTO',
                'industry': 'Fintech',
                'type': 'servicio',
                'owner_name': 'Ana García',
                'created_at': '2025-01-15'
            }
        ]
    
    def get_directed_opportunities(self, user_id):
        """Obtener oportunidades dirigidas al usuario"""
        return [
            {
                'id': 'opp2',
                'title': 'Inversión en e-commerce',
                'description': 'Tienda online necesita capital de trabajo',
                'industry': 'E-commerce',
                'type': 'producto',
                'sender_name': 'Carlos López',
                'sent_at': '2025-01-14'
            }
        ]
    
    def get_my_opportunities(self, user_id):
        """Obtener las oportunidades creadas por el usuario"""
        return []
    
    def get_pending_invitations(self, user_id):
        """Obtener invitaciones pendientes"""
        return []
    
    def get_user_contacts(self, user_id):
        """Obtener contactos del usuario"""
        return [
            {'id': 'c1', 'name': 'Juan Pérez', 'relationship': 'Amigo'},
            {'id': 'c2', 'name': 'María Silva', 'relationship': 'Familiar'}
        ]
    
    def get_ai_recommended_opportunities(self, user_id):
        """Oportunidades recomendadas por IA"""
        return self.get_public_opportunities(user_id)  # Mock
    
    def get_user_company_access(self, user_id):
        """Obtener empresas donde el usuario tiene acceso"""
        return []
    
    def create_opportunity(self, opportunity_data):
        """Crear nueva oportunidad"""
        # Mock implementation
        return str(uuid.uuid4())
    
    def create_company_access(self, company_data):
        """Registrar acceso a empresa"""
        # Mock implementation
        pass
    
    def notify_directed_opportunity(self, opp_id, contact_ids):
        """Notificar oportunidad dirigida a contactos"""
        # Mock implementation
        pass
    
    def mark_opportunity_interest(self, user_id, opportunity_id):
        """Marcar oportunidad como de interés"""
        # Mock implementation
        pass
    
    def send_contact_message(self, user_id, opportunity_id, message):
        """Enviar mensaje al dueño de oportunidad"""
        # Mock implementation
        pass
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the application"""
        print("🚀 Starting Golden Coyotes - After Feedback Implementation")
        print(f"📱 USUARIOS: http://localhost:{port}")
        print("✨ Implementación basada en feedback Junio 2025")
        print("\n🎯 Funcionalidades:")
        print("   - 4 Cuadrantes según feedback")
        print("   - Sistema Friends & Family")
        print("   - Oportunidades públicas vs dirigidas")
        print("   - PUSH & PULL empresarial")
        print("   - Invitaciones redes sociales")
        
        self.app.run(host=host, port=port, debug=debug)

# ==================== TEMPLATES HTML ====================

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Coyotes - Conectando Oportunidades de Negocio</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            min-height: 80vh; 
        }
        .feature-card { 
            transition: transform 0.3s; 
            border: none; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        }
        .feature-card:hover { 
            transform: translateY(-5px); 
        }
        .video-container { 
            position: relative; 
            width: 100%; 
            height: 400px; 
            background: #000; 
            border-radius: 15px; 
            overflow: hidden; 
        }
        .video-placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            background: linear-gradient(45deg, #667eea, #764ba2);
        }
    </style>
</head>
<body>
    <!-- Hero Section -->
    <section class="hero d-flex align-items-center">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-lg-6">
                    <h1 class="display-4 fw-bold mb-4">
                        <i class="fas fa-users-cog text-warning"></i>
                        Golden Coyotes
                    </h1>
                    <h2 class="h3 mb-4">Conectando Oportunidades de Negocio</h2>
                    <p class="lead mb-4">
                        Una comunidad social basada en economía colaborativa donde conectas 
                        con tu red "Friends & Family" para descubrir y compartir oportunidades de negocio.
                    </p>
                    <div class="d-grid gap-2 d-md-flex">
                        <a href="{{ url_for('register') }}" class="btn btn-warning btn-lg px-4 me-md-2">
                            <i class="fas fa-rocket"></i> Comenzar Ahora
                        </a>
                        <a href="{{ url_for('login') }}" class="btn btn-outline-light btn-lg px-4">
                            <i class="fas fa-sign-in-alt"></i> Iniciar Sesión
                        </a>
                    </div>
                </div>
                <div class="col-lg-6">
                    <!-- Video Explicativo -->
                    <div class="video-container">
                        <div class="video-placeholder">
                            <div class="text-center">
                                <i class="fas fa-play-circle fa-5x text-white mb-3"></i>
                                <h4 class="text-white">Video Explicativo</h4>
                                <p class="text-white-50">¿Qué es Golden Coyotes y cómo funciona?</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section class="py-5">
        <div class="container">
            <div class="row text-center mb-5">
                <div class="col">
                    <h2 class="h1 mb-3">¿Cómo funciona Golden Coyotes?</h2>
                    <p class="lead text-muted">Tu red de contactos seguros para oportunidades de negocio</p>
                </div>
            </div>
            
            <div class="row g-4">
                <div class="col-md-6 col-lg-3">
                    <div class="feature-card card h-100 text-center p-4">
                        <div class="card-body">
                            <i class="fas fa-upload fa-3x text-primary mb-3"></i>
                            <h5>Subir Oportunidad</h5>
                            <p class="text-muted">Comparte oportunidades públicas con toda tu red</p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 col-lg-3">
                    <div class="feature-card card h-100 text-center p-4">
                        <div class="card-body">
                            <i class="fas fa-share fa-3x text-success mb-3"></i>
                            <h5>Oportunidad Dirigida</h5>
                            <p class="text-muted">Envía oportunidades específicas a contactos seleccionados</p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 col-lg-3">
                    <div class="feature-card card h-100 text-center p-4">
                        <div class="card-body">
                            <i class="fas fa-search fa-3x text-info mb-3"></i>
                            <h5>Buscar Oportunidades</h5>
                            <p class="text-muted">Explora oportunidades de tu red con IA inteligente</p>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6 col-lg-3">
                    <div class="feature-card card h-100 text-center p-4">
                        <div class="card-body">
                            <i class="fas fa-envelope fa-3x text-warning mb-3"></i>
                            <h5>Mis Dirigidas</h5>
                            <p class="text-muted">Revisa oportunidades que te enviaron directamente</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- CTA Section -->
    <section class="bg-light py-5">
        <div class="container text-center">
            <h2 class="h1 mb-4">¿Listo para conectar oportunidades?</h2>
            <p class="lead mb-4">Únete a la comunidad de networking empresarial más segura</p>
            <a href="{{ url_for('register') }}" class="btn btn-primary btn-lg">
                <i class="fas fa-user-plus"></i> Registrarse Gratis
            </a>
        </div>
    </section>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Registro - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .register-container { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .register-card { 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
        }
    </style>
</head>
<body class="register-container">
    <div class="container d-flex align-items-center justify-content-center min-vh-100 py-5">
        <div class="row w-100">
            <div class="col-md-8 col-lg-6 mx-auto">
                <div class="register-card p-5">
                    <div class="text-center mb-4">
                        <h1 class="h2">
                            <i class="fas fa-users-cog text-primary"></i>
                            Únete a Golden Coyotes
                        </h1>
                        <p class="text-muted">Crea tu perfil y comienza a conectar oportunidades</p>
                    </div>

                    <form id="registerForm" method="POST">
                        <!-- Datos Personales -->
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Nombre Completo *</label>
                                <input type="text" class="form-control" name="name" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Email *</label>
                                <input type="email" class="form-control" name="email" required>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Teléfono *</label>
                                <input type="tel" class="form-control" name="phone" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Contraseña *</label>
                                <input type="password" class="form-control" name="password" required>
                            </div>
                        </div>

                        <!-- Preferencias de Industria -->
                        <div class="mb-3">
                            <label class="form-label">Industrias de Interés (separadas por coma)</label>
                            <input type="text" class="form-control" name="industry_preferences" 
                                   placeholder="Tecnología, Fintech, E-commerce, Salud">
                            <small class="text-muted">Esto ayudará a la IA a recomendarte oportunidades relevantes</small>
                        </div>

                        <!-- Términos y Políticas -->
                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="accepted_terms" value="true" required>
                                <label class="form-check-label">
                                    He leído y acepto los <a href="#" class="text-primary">Términos y Condiciones</a>
                                </label>
                            </div>
                        </div>

                        <div class="mb-3">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="accepted_privacy" value="true" required>
                                <label class="form-check-label">
                                    Acepto las <a href="#" class="text-primary">Políticas de Privacidad y Protección de Datos</a>
                                </label>
                            </div>
                        </div>

                        <div class="mb-4">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" name="safe_contact_agreement" value="true" required>
                                <label class="form-check-label">
                                    Entiendo y acepto el concepto de <strong>"Contacto Seguro"</strong> basado en Friends & Family
                                </label>
                            </div>
                            <small class="text-muted">
                                Solo podrás conectar con personas que invites o que te inviten directamente, 
                                creando una red de confianza.
                            </small>
                        </div>

                        <div class="d-grid mb-3">
                            <button type="submit" class="btn btn-primary btn-lg">
                                <i class="fas fa-rocket"></i> Crear Mi Cuenta
                            </button>
                        </div>
                    </form>

                    <div class="text-center">
                        <p class="mb-0">¿Ya tienes cuenta? 
                            <a href="{{ url_for('login') }}" class="text-primary">Inicia sesión aquí</a>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iniciar Sesión - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .login-container { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
        }
        .login-card { 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 40px rgba(0,0,0,0.1); 
        }
    </style>
</head>
<body class="login-container">
    <div class="container d-flex align-items-center justify-content-center min-vh-100">
        <div class="row w-100">
            <div class="col-md-6 col-lg-4 mx-auto">
                <div class="login-card p-5">
                    <div class="text-center mb-4">
                        <h1 class="h2">
                            <i class="fas fa-users-cog text-primary"></i>
                            Golden Coyotes
                        </h1>
                        <p class="text-muted">Accede a tu red de oportunidades</p>
                    </div>

                    {% with messages = get_flashed_messages(with_categories=true) %}
                        {% if messages %}
                            {% for category, message in messages %}
                                <div class="alert alert-{{ 'danger' if category == 'error' else category }}">
                                    {{ message }}
                                </div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}

                    <form method="POST">
                        <div class="mb-3">
                            <label class="form-label">Email</label>
                            <input type="email" class="form-control" name="email" required>
                        </div>
                        
                        <div class="mb-4">
                            <label class="form-label">Contraseña</label>
                            <input type="password" class="form-control" name="password" required>
                        </div>

                        <div class="d-grid mb-3">
                            <button type="submit" class="btn btn-primary btn-lg">
                                <i class="fas fa-sign-in-alt"></i> Iniciar Sesión
                            </button>
                        </div>
                    </form>

                    <div class="text-center">
                        <p class="mb-0">¿No tienes cuenta? 
                            <a href="{{ url_for('register') }}" class="text-primary">Regístrate aquí</a>
                        </p>
                        <p class="mt-2">
                            <a href="{{ url_for('home') }}" class="text-muted">← Volver al inicio</a>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .dashboard-container { 
            background: #f8f9fa; 
            min-height: 100vh; 
        }
        .quadrant-card { 
            height: 250px; 
            transition: all 0.3s ease; 
            cursor: pointer; 
            position: relative;
            overflow: hidden;
        }
        .quadrant-card:hover { 
            transform: translateY(-10px); 
            box-shadow: 0 15px 30px rgba(0,0,0,0.2); 
        }
        .notification-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: #dc3545;
            color: white;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }
        .quadrant-1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .quadrant-2 { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .quadrant-3 { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .quadrant-4 { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    </style>
</head>
<body class="dashboard-container">
    <!-- Header -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-users-cog"></i> Golden Coyotes
            </a>
            <div class="navbar-nav ms-auto">
                <span class="navbar-text me-3">¡Hola, {{ user_name }}!</span>
                <a href="{{ url_for('invitar_contactos') }}" class="btn btn-outline-light btn-sm me-2">
                    <i class="fas fa-user-plus"></i> Invitar
                </a>
                <a href="{{ url_for('push_pull') }}" class="btn btn-outline-warning btn-sm me-2">
                    <i class="fas fa-building"></i> Push & Pull
                </a>
                <a href="{{ url_for('logout') }}" class="btn btn-outline-danger btn-sm">
                    <i class="fas fa-sign-out-alt"></i> Salir
                </a>
            </div>
        </div>
    </nav>

    <!-- Dashboard Principal -->
    <div class="container py-5">
        <div class="text-center mb-5">
            <h1 class="h2 text-dark">Mi Dashboard de Oportunidades</h1>
            <p class="text-muted">Selecciona un cuadrante para comenzar</p>
        </div>

        <!-- Los 4 Cuadrantes -->
        <div class="row g-4">
            <!-- Cuadrante 1: Subo Oportunidad -->
            <div class="col-md-6">
                <div class="card quadrant-card quadrant-1 text-white border-0" onclick="location.href='{{ url_for('subir_oportunidad') }}'">
                    {% if counters.mis_oportunidades > 0 %}
                        <div class="notification-badge">{{ counters.mis_oportunidades }}</div>
                    {% endif %}
                    <div class="card-body d-flex flex-column justify-content-center align-items-center text-center">
                        <i class="fas fa-upload fa-4x mb-3"></i>
                        <h3>Subo Oportunidad</h3>
                        <p class="mb-0">Comparte una oportunidad pública con toda tu red</p>
                        <small class="opacity-75 mt-2">
                            • Selecciona industria<br>
                            • Producto o Servicio<br>
                            • Incluye fotos/videos
                        </small>
                    </div>
                </div>
            </div>

            <!-- Cuadrante 2: Oportunidad Dirigida -->
            <div class="col-md-6">
                <div class="card quadrant-card quadrant-2 text-white border-0" onclick="location.href='{{ url_for('oportunidad_dirigida') }}'">
                    <div class="card-body d-flex flex-column justify-content-center align-items-center text-center">
                        <i class="fas fa-share fa-4x mb-3"></i>
                        <h3>Oportunidad Dirigida</h3>
                        <p class="mb-0">Envía una oportunidad a contactos específicos</p>
                        <small class="opacity-75 mt-2">
                            • Selecciona contactos<br>
                            • Envío privado<br>
                            • Notificación directa
                        </small>
                    </div>
                </div>
            </div>

            <!-- Cuadrante 3: Busco Oportunidad General -->
            <div class="col-md-6">
                <div class="card quadrant-card quadrant-3 text-white border-0" onclick="location.href='{{ url_for('buscar_oportunidad') }}'">
                    {% if counters.oportunidades_publicas > 0 %}
                        <div class="notification-badge">{{ counters.oportunidades_publicas }}</div>
                    {% endif %}
                    <div class="card-body d-flex flex-column justify-content-center align-items-center text-center">
                        <i class="fas fa-search fa-4x mb-3"></i>
                        <h3>Busco Oportunidades</h3>
                        <p class="mb-0">Explora oportunidades de tu red</p>
                        <small class="opacity-75 mt-2">
                            • Recomendaciones IA<br>
                            • Filtrar por industria<br>
                            • Marcar interesantes
                        </small>
                    </div>
                </div>
            </div>

            <!-- Cuadrante 4: Mis Oportunidades Dirigidas -->
            <div class="col-md-6">
                <div class="card quadrant-card quadrant-4 text-white border-0" onclick="location.href='{{ url_for('mis_oportunidades_dirigidas') }}'">
                    {% if counters.oportunidades_dirigidas > 0 %}
                        <div class="notification-badge">{{ counters.oportunidades_dirigidas }}</div>
                    {% endif %}
                    <div class="card-body d-flex flex-column justify-content-center align-items-center text-center">
                        <i class="fas fa-envelope fa-4x mb-3"></i>
                        <h3>Mis Dirigidas</h3>
                        <p class="mb-0">Oportunidades que me enviaron directamente</p>
                        <small class="opacity-75 mt-2">
                            • Enviadas por contactos<br>
                            • Revisión privada<br>
                            • Marcar interés
                        </small>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Stats -->
        <div class="row mt-5">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="fas fa-chart-bar text-primary"></i> 
                            Tu Actividad Reciente
                        </h5>
                        <div class="row text-center">
                            <div class="col-3">
                                <h4 class="text-primary">{{ counters.oportunidades_publicas }}</h4>
                                <small class="text-muted">Oportunidades Disponibles</small>
                            </div>
                            <div class="col-3">
                                <h4 class="text-success">{{ counters.oportunidades_dirigidas }}</h4>
                                <small class="text-muted">Dirigidas a Ti</small>
                            </div>
                            <div class="col-3">
                                <h4 class="text-info">{{ counters.mis_oportunidades }}</h4>
                                <small class="text-muted">Tus Oportunidades</small>
                            </div>
                            <div class="col-3">
                                <h4 class="text-warning">{{ counters.invitaciones_pendientes }}</h4>
                                <small class="text-muted">Invitaciones Pendientes</small>
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

# Continuaré con el resto de templates...
SUBIR_OPORTUNIDAD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Subir Oportunidad - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-primary">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Subir Oportunidad Pública</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">
                            <i class="fas fa-upload"></i> 
                            Cuadrante 1: Crear Oportunidad Pública
                        </h4>
                        <p class="mb-0 mt-2 opacity-75">
                            Esta oportunidad será visible para toda tu red de contactos
                        </p>
                    </div>
                    <div class="card-body">
                        <form method="POST" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label class="form-label">Industria *</label>
                                <select class="form-control" name="industry" required>
                                    <option value="">Selecciona una industria</option>
                                    <option value="Tecnología">Tecnología</option>
                                    <option value="Fintech">Fintech</option>
                                    <option value="E-commerce">E-commerce</option>
                                    <option value="Salud">Salud</option>
                                    <option value="Educación">Educación</option>
                                    <option value="Manufactura">Manufactura</option>
                                    <option value="Servicios">Servicios</option>
                                    <option value="Inmobiliaria">Inmobiliaria</option>
                                    <option value="Alimentaria">Alimentaria</option>
                                    <option value="Otros">Otros</option>
                                </select>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Tipo *</label>
                                <div class="row">
                                    <div class="col-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="type" value="producto" required>
                                            <label class="form-check-label">
                                                <i class="fas fa-box text-primary"></i> Producto
                                            </label>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="type" value="servicio" required>
                                            <label class="form-check-label">
                                                <i class="fas fa-cogs text-success"></i> Servicio
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Nombre de la Oportunidad *</label>
                                <input type="text" class="form-control" name="title" 
                                       placeholder="ej: Startup busca socio técnico para plataforma de pagos" required>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Descripción *</label>
                                <textarea class="form-control" name="description" rows="4" required
                                          placeholder="Describe detalladamente la oportunidad: qué necesitas, qué ofreces, objetivos, etc."></textarea>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Fotos o Videos (opcional)</label>
                                <input type="file" class="form-control" name="media_files" multiple 
                                       accept="image/*,video/*">
                                <small class="text-muted">
                                    Puedes subir imágenes o videos que ayuden a explicar tu oportunidad
                                </small>
                            </div>

                            <div class="alert alert-info">
                                <i class="fas fa-info-circle"></i>
                                <strong>Oportunidad Pública:</strong> Todos tus contactos de confianza podrán ver esta oportunidad.
                                Si quieres dirigirla a contactos específicos, usa el "Cuadrante 2".
                            </div>

                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary btn-lg">
                                    <i class="fas fa-rocket"></i> Publicar Oportunidad
                                </button>
                                <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                                    Cancelar
                                </a>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# Plantillas adicionales para completar la implementación...
OPORTUNIDAD_DIRIGIDA_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oportunidad Dirigida - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-success">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Oportunidad Dirigida</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card shadow">
                    <div class="card-header bg-success text-white">
                        <h4 class="mb-0">
                            <i class="fas fa-share"></i> 
                            Cuadrante 2: Enviar Oportunidad a Contactos Específicos
                        </h4>
                        <p class="mb-0 mt-2 opacity-75">
                            Esta oportunidad será enviada solo a los contactos que selecciones
                        </p>
                    </div>
                    <div class="card-body">
                        {% with messages = get_flashed_messages(with_categories=true) %}
                            {% if messages %}
                                {% for category, message in messages %}
                                    <div class="alert alert-{{ 'danger' if category == 'error' else category }} alert-dismissible fade show">
                                        {{ message }}
                                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                    </div>
                                {% endfor %}
                            {% endif %}
                        {% endwith %}

                        <form method="POST" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label class="form-label">Industria *</label>
                                <select class="form-control" name="industry" required>
                                    <option value="">Selecciona una industria</option>
                                    <option value="Tecnología">Tecnología</option>
                                    <option value="Fintech">Fintech</option>
                                    <option value="E-commerce">E-commerce</option>
                                    <option value="Salud">Salud</option>
                                    <option value="Educación">Educación</option>
                                    <option value="Manufactura">Manufactura</option>
                                    <option value="Servicios">Servicios</option>
                                    <option value="Inmobiliaria">Inmobiliaria</option>
                                    <option value="Alimentaria">Alimentaria</option>
                                    <option value="Otros">Otros</option>
                                </select>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Tipo *</label>
                                <div class="row">
                                    <div class="col-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="type" value="producto" required>
                                            <label class="form-check-label">
                                                <i class="fas fa-box text-primary"></i> Producto
                                            </label>
                                        </div>
                                    </div>
                                    <div class="col-6">
                                        <div class="form-check">
                                            <input class="form-check-input" type="radio" name="type" value="servicio" required>
                                            <label class="form-check-label">
                                                <i class="fas fa-cogs text-success"></i> Servicio
                                            </label>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Nombre de la Oportunidad *</label>
                                <input type="text" class="form-control" name="title" 
                                       placeholder="ej: Buscamos socio inversionista para expansión" required>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Descripción *</label>
                                <textarea class="form-control" name="description" rows="4" required
                                          placeholder="Describe la oportunidad específica para tus contactos seleccionados..."></textarea>
                            </div>

                            <!-- Selector de Contactos -->
                            <div class="mb-3">
                                <label class="form-label">Seleccionar Contactos *</label>
                                <p class="text-muted small">Elige a cuáles de tus contactos de confianza enviar esta oportunidad:</p>
                                <div class="row">
                                    {% if contacts %}
                                        {% for contact in contacts %}
                                            <div class="col-md-6 mb-2">
                                                <div class="form-check">
                                                    <input class="form-check-input" type="checkbox" name="selected_contacts" value="{{ contact.id }}">
                                                    <label class="form-check-label">
                                                        <i class="fas fa-user-friends text-info"></i>
                                                        <strong>{{ contact.name }}</strong>
                                                        <br><small class="text-muted">{{ contact.relationship }}</small>
                                                    </label>
                                                </div>
                                            </div>
                                        {% endfor %}
                                    {% else %}
                                        <div class="col-12">
                                            <div class="alert alert-info">
                                                <i class="fas fa-info-circle"></i>
                                                Aún no tienes contactos en tu red Friends & Family. 
                                                <a href="{{ url_for('invitar_contactos') }}" class="alert-link">Invita contactos aquí</a>
                                            </div>
                                        </div>
                                    {% endif %}
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Fotos o Videos (opcional)</label>
                                <input type="file" class="form-control" name="media_files" multiple 
                                       accept="image/*,video/*">
                                <small class="text-muted">
                                    Archivos que ayuden a explicar tu oportunidad a tus contactos
                                </small>
                            </div>

                            <div class="alert alert-warning">
                                <i class="fas fa-share"></i>
                                <strong>Oportunidad Dirigida:</strong> Solo los contactos seleccionados recibirán 
                                esta oportunidad de forma privada. Si quieres que toda tu red la vea, 
                                usa el <a href="{{ url_for('subir_oportunidad') }}" class="alert-link">Cuadrante 1</a>.
                            </div>

                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-success btn-lg">
                                    <i class="fas fa-paper-plane"></i> Enviar a Contactos Seleccionados
                                </button>
                                <a href="{{ url_for('dashboard') }}" class="btn btn-outline-secondary">
                                    Cancelar
                                </a>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

BUSCAR_OPORTUNIDAD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Buscar Oportunidades - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .opportunity-card { 
            transition: transform 0.2s; 
            cursor: pointer;
        }
        .opportunity-card:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        .ai-badge { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white;
        }
        .interest-btn.active {
            background: #28a745;
            color: white;
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-info">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Buscar Oportunidades</span>
        </div>
    </nav>

    <div class="container py-4">
        <!-- Header con opciones según feedback -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h4 class="card-title">
                            <i class="fas fa-search"></i> 
                            {{ title }}
                        </h4>
                        <p class="card-text text-muted">Explora oportunidades de tu red Friends & Family</p>
                        
                        <!-- Toggle entre IA y Todas (según feedback) -->
                        <div class="btn-group w-100" role="group">
                            <input type="radio" class="btn-check" name="view_type" id="ai_view" {% if view_type == 'ai' %}checked{% endif %}>
                            <label class="btn btn-outline-primary" for="ai_view" onclick="changeView('ai')">
                                <i class="fas fa-robot"></i> 
                                Recomendaciones IA
                                <small class="d-block">Oportunidades que la IA identificó para ti</small>
                            </label>

                            <input type="radio" class="btn-check" name="view_type" id="all_view" {% if view_type == 'all' %}checked{% endif %}>
                            <label class="btn btn-outline-secondary" for="all_view" onclick="changeView('all')">
                                <i class="fas fa-list"></i> 
                                Todas las Oportunidades
                                <small class="d-block">Toda tu red + redes extendidas</small>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Filtros -->
        <div class="row mb-4">
            <div class="col-md-4">
                <select class="form-control" id="industry_filter">
                    <option value="">Todas las industrias</option>
                    <option value="Tecnología">Tecnología</option>
                    <option value="Fintech">Fintech</option>
                    <option value="E-commerce">E-commerce</option>
                    <option value="Salud">Salud</option>
                    <option value="Educación">Educación</option>
                </select>
            </div>
            <div class="col-md-4">
                <select class="form-control" id="type_filter">
                    <option value="">Producto y Servicio</option>
                    <option value="producto">Solo Productos</option>
                    <option value="servicio">Solo Servicios</option>
                </select>
            </div>
            <div class="col-md-4">
                <input type="text" class="form-control" id="search_filter" placeholder="Buscar por palabra clave...">
            </div>
        </div>

        <!-- Lista de Oportunidades -->
        <div class="row" id="opportunities-container">
            {% if opportunities %}
                {% for opp in opportunities %}
                <div class="col-lg-6 mb-4 opportunity-item" 
                     data-industry="{{ opp.industry }}" 
                     data-type="{{ opp.type }}">
                    <div class="card opportunity-card h-100 border-0 shadow-sm">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <div>
                                <span class="badge bg-{{ 'primary' if opp.type == 'producto' else 'success' }}">
                                    <i class="fas fa-{{ 'box' if opp.type == 'producto' else 'cogs' }}"></i>
                                    {{ opp.type.title() }}
                                </span>
                                <span class="badge bg-secondary ms-2">{{ opp.industry }}</span>
                            </div>
                            {% if view_type == 'ai' %}
                                <span class="ai-badge badge">
                                    <i class="fas fa-brain"></i> 85% Match
                                </span>
                            {% endif %}
                        </div>
                        
                        <div class="card-body">
                            <h5 class="card-title">{{ opp.title }}</h5>
                            <p class="card-text text-muted">{{ opp.description[:150] }}{% if opp.description|length > 150 %}...{% endif %}</p>
                            
                            <div class="row text-center mt-3">
                                <div class="col-4">
                                    <small class="text-muted">Publicado por</small>
                                    <div class="fw-bold">{{ opp.owner_name }}</div>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted">Fecha</small>
                                    <div class="fw-bold">{{ opp.created_at[:10] }}</div>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted">Vistas</small>
                                    <div class="fw-bold">{{ opp.get('views', 0) }}</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card-footer bg-transparent">
                            <div class="d-grid gap-2">
                                <button class="btn btn-outline-warning interest-btn" 
                                        onclick="markInterest('{{ opp.id }}', this)">
                                    <i class="fas fa-star"></i> Marcar Interés
                                </button>
                                <button class="btn btn-primary" 
                                        onclick="contactOwner('{{ opp.id }}', '{{ opp.owner_name }}')">
                                    <i class="fas fa-comment"></i> Contactar a {{ opp.owner_name }}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body text-center py-5">
                            <i class="fas fa-search fa-5x text-muted mb-4"></i>
                            <h4>No hay oportunidades disponibles</h4>
                            <p class="text-muted">Tu red aún no ha publicado oportunidades, o necesitas expandir tu red Friends & Family.</p>
                            <div class="mt-4">
                                <a href="{{ url_for('invitar_contactos') }}" class="btn btn-primary me-2">
                                    <i class="fas fa-user-plus"></i> Invitar Contactos
                                </a>
                                <a href="{{ url_for('subir_oportunidad') }}" class="btn btn-success">
                                    <i class="fas fa-plus"></i> Crear Primera Oportunidad
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            {% endif %}
        </div>

        <!-- Ventaja del Networking según feedback -->
        {% if view_type == 'ai' %}
        <div class="row mt-4">
            <div class="col-12">
                <div class="alert alert-info">
                    <h5><i class="fas fa-lightbulb"></i> Ventaja del Networking IA</h5>
                    <p class="mb-0">
                        <strong>La IA puede identificar oportunidades de personas que NO están en tu red directa, 
                        pero SÍ están en la red de tus contactos.</strong> ¡Esto te permite hacer networking puro 
                        contactando a tu contacto para conectar con esa tercera persona!
                    </p>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <!-- Modal para contactar dueño -->
    <div class="modal fade" id="contactModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Contactar Dueño de Oportunidad</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label">Tu mensaje:</label>
                        <textarea class="form-control" id="contact_message" rows="4" 
                                  placeholder="Hola, vi tu oportunidad y me interesa..."></textarea>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                    <button type="button" class="btn btn-primary" onclick="sendContactMessage()">
                        <i class="fas fa-paper-plane"></i> Enviar Mensaje
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentOpportunityId = null;

        // Cambiar vista entre IA y Todas
        function changeView(viewType) {
            window.location.href = `?view=${viewType}`;
        }

        // Marcar oportunidad como de interés
        function markInterest(oppId, button) {
            fetch('/api/mark-interest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({opportunity_id: oppId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    button.classList.add('active');
                    button.innerHTML = '<i class="fas fa-star"></i> ¡Marcado!';
                }
            });
        }

        // Contactar dueño de oportunidad
        function contactOwner(oppId, ownerName) {
            currentOpportunityId = oppId;
            document.querySelector('.modal-title').innerText = `Contactar a ${ownerName}`;
            new bootstrap.Modal(document.getElementById('contactModal')).show();
        }

        // Enviar mensaje de contacto
        function sendContactMessage() {
            const message = document.getElementById('contact_message').value;
            if (!message.trim()) return;

            fetch('/api/contact-opportunity-owner', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    opportunity_id: currentOpportunityId,
                    message: message
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('¡Mensaje enviado exitosamente!');
                    bootstrap.Modal.getInstance(document.getElementById('contactModal')).hide();
                    document.getElementById('contact_message').value = '';
                }
            });
        }

        // Filtros en tiempo real
        document.getElementById('industry_filter').addEventListener('change', filterOpportunities);
        document.getElementById('type_filter').addEventListener('change', filterOpportunities);
        document.getElementById('search_filter').addEventListener('input', filterOpportunities);

        function filterOpportunities() {
            const industry = document.getElementById('industry_filter').value;
            const type = document.getElementById('type_filter').value;
            const search = document.getElementById('search_filter').value.toLowerCase();

            document.querySelectorAll('.opportunity-item').forEach(item => {
                const matchIndustry = !industry || item.dataset.industry === industry;
                const matchType = !type || item.dataset.type === type;
                const matchSearch = !search || 
                    item.textContent.toLowerCase().includes(search);

                item.style.display = (matchIndustry && matchType && matchSearch) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
'''

MIS_OPORTUNIDADES_DIRIGIDAS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mis Oportunidades Dirigidas - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .directed-opp-card { 
            transition: transform 0.2s; 
            border-left: 4px solid #28a745;
        }
        .directed-opp-card:hover { 
            transform: translateY(-5px); 
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        .sender-info {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            border-radius: 10px;
        }
        .interest-btn.active {
            background: #ffc107;
            color: #000;
        }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-success">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Mis Oportunidades Dirigidas</span>
        </div>
    </nav>

    <div class="container py-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h4 class="card-title">
                            <i class="fas fa-envelope-open"></i> 
                            Cuadrante 4: Oportunidades Enviadas Directamente a Ti
                        </h4>
                        <p class="card-text text-muted">
                            Estas oportunidades fueron enviadas específicamente para ti por tus contactos de confianza
                        </p>
                        
                        <!-- Stats rápidos -->
                        <div class="row text-center mt-3">
                            <div class="col-md-4">
                                <h5 class="text-primary">{{ opportunities|length }}</h5>
                                <small class="text-muted">Total Recibidas</small>
                            </div>
                            <div class="col-md-4">
                                <h5 class="text-success">{{ opportunities|selectattr('interesse_marcado')|list|length }}</h5>
                                <small class="text-muted">Marcadas con Interés</small>
                            </div>
                            <div class="col-md-4">
                                <h5 class="text-info">{{ opportunities|selectattr('respondida')|list|length }}</h5>
                                <small class="text-muted">Respondidas</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Lista de Oportunidades Dirigidas -->
        <div class="row">
            {% if opportunities %}
                {% for opp in opportunities %}
                <div class="col-lg-6 mb-4">
                    <div class="card directed-opp-card h-100 border-0 shadow-sm">
                        <!-- Header con info del remitente -->
                        <div class="card-header sender-info">
                            <div class="d-flex justify-content-between align-items-center">
                                <div class="d-flex align-items-center">
                                    <i class="fas fa-user-circle fa-2x me-2"></i>
                                    <div>
                                        <strong>{{ opp.sender_name }}</strong>
                                        <br><small class="opacity-75">Te envió esta oportunidad</small>
                                    </div>
                                </div>
                                <div class="text-end">
                                    <small class="opacity-75">{{ opp.sent_at[:10] }}</small>
                                    <br><span class="badge bg-light text-dark">
                                        <i class="fas fa-{{ 'box' if opp.type == 'producto' else 'cogs' }}"></i>
                                        {{ opp.type.title() }}
                                    </span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h5 class="card-title mb-0">{{ opp.title }}</h5>
                                <span class="badge bg-secondary">{{ opp.industry }}</span>
                            </div>
                            
                            <p class="card-text text-muted">{{ opp.description }}</p>
                            
                            <!-- Indicador de por qué te la enviaron -->
                            <div class="alert alert-light border-start border-success border-3">
                                <small>
                                    <i class="fas fa-lightbulb text-warning"></i>
                                    <strong>¿Por qué te la enviaron?</strong> 
                                    Probablemente {{ opp.sender_name }} conoce tu experiencia en {{ opp.industry }} 
                                    o sabe que puedes estar interesado en este tipo de {{ opp.type }}.
                                </small>
                            </div>
                        </div>
                        
                        <div class="card-footer bg-transparent">
                            <div class="row g-2">
                                <div class="col-6">
                                    <button class="btn btn-outline-warning w-100 interest-btn" 
                                            onclick="markInterest('{{ opp.id }}', this)">
                                        <i class="fas fa-star"></i> Marcar Interés
                                    </button>
                                </div>
                                <div class="col-6">
                                    <button class="btn btn-success w-100" 
                                            onclick="respondToSender('{{ opp.id }}', '{{ opp.sender_name }}')">
                                        <i class="fas fa-reply"></i> Responder
                                    </button>
                                </div>
                                <div class="col-12 mt-2">
                                    <button class="btn btn-outline-primary w-100" 
                                            onclick="viewFullOpportunity('{{ opp.id }}')">
                                        <i class="fas fa-eye"></i> Ver Detalles Completos
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="col-12">
                    <div class="card border-0 shadow-sm">
                        <div class="card-body text-center py-5">
                            <i class="fas fa-inbox fa-5x text-muted mb-4"></i>
                            <h4>No tienes oportunidades dirigidas</h4>
                            <p class="text-muted">
                                Aún no has recibido oportunidades específicas de tus contactos. 
                                Esto sucede cuando alguien de tu red Friends & Family tiene una oportunidad 
                                que cree que te puede interesar específicamente.
                            </p>
                            
                            <!-- Sugerencias para recibir más -->
                            <div class="alert alert-info text-start mt-4">
                                <h6><i class="fas fa-tips"></i> Consejos para recibir más oportunidades dirigidas:</h6>
                                <ul class="mb-0">
                                    <li>Expande tu red invitando más contactos de confianza</li>
                                    <li>Participa activamente creando tus propias oportunidades</li>
                                    <li>Actualiza tu perfil con tus intereses e industrias</li>
                                    <li>Interactúa con las oportunidades de tu red</li>
                                </ul>
                            </div>
                            
                            <div class="mt-4">
                                <a href="{{ url_for('invitar_contactos') }}" class="btn btn-primary me-2">
                                    <i class="fas fa-user-plus"></i> Invitar Contactos
                                </a>
                                <a href="{{ url_for('buscar_oportunidad') }}" class="btn btn-success me-2">
                                    <i class="fas fa-search"></i> Explorar Oportunidades
                                </a>
                                <a href="{{ url_for('subir_oportunidad') }}" class="btn btn-outline-primary">
                                    <i class="fas fa-plus"></i> Crear Oportunidad
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            {% endif %}
        </div>

        <!-- Información sobre el concepto -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="alert alert-info">
                    <h5><i class="fas fa-info-circle"></i> ¿Qué son las Oportunidades Dirigidas?</h5>
                    <p class="mb-2">
                        Las oportunidades dirigidas son aquellas que tus contactos de la red Friends & Family 
                        decidieron enviarte específicamente porque:
                    </p>
                    <ul class="mb-0">
                        <li><strong>Conocen tu perfil profesional</strong> y saben que puede interesarte</li>
                        <li><strong>Confían en tu experiencia</strong> en esa área o industria</li>
                        <li><strong>Quieren hacer networking</strong> conectándote con la oportunidad</li>
                        <li><strong>Creen que tienes el contacto</strong> que falta para cerrar el negocio</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal para responder al remitente -->
    <div class="modal fade" id="respondModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Responder a Contacto</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label">Tu respuesta:</label>
                        <textarea class="form-control" id="response_message" rows="4" 
                                  placeholder="Hola, gracias por enviarme esta oportunidad..."></textarea>
                    </div>
                    <div class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="interested_check">
                            <label class="form-check-label" for="interested_check">
                                ✅ Estoy interesado en esta oportunidad
                            </label>
                        </div>
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="have_contact_check">
                            <label class="form-check-label" for="have_contact_check">
                                🤝 Tengo un contacto que podría interesarle
                            </label>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancelar</button>
                    <button type="button" class="btn btn-success" onclick="sendResponse()">
                        <i class="fas fa-paper-plane"></i> Enviar Respuesta
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal para detalles completos -->
    <div class="modal fade" id="detailsModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Detalles de la Oportunidad</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="opportunity-details">
                    <!-- Se llena dinámicamente -->
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentOpportunityId = null;

        // Marcar oportunidad como de interés
        function markInterest(oppId, button) {
            fetch('/api/mark-interest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({opportunity_id: oppId})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    button.classList.add('active');
                    button.innerHTML = '<i class="fas fa-star"></i> ¡Marcado!';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error al marcar interés');
            });
        }

        // Responder al remitente
        function respondToSender(oppId, senderName) {
            currentOpportunityId = oppId;
            document.querySelector('#respondModal .modal-title').innerText = `Responder a ${senderName}`;
            new bootstrap.Modal(document.getElementById('respondModal')).show();
        }

        // Enviar respuesta
        function sendResponse() {
            const message = document.getElementById('response_message').value;
            const interested = document.getElementById('interested_check').checked;
            const haveContact = document.getElementById('have_contact_check').checked;

            if (!message.trim()) {
                alert('Por favor escribe un mensaje');
                return;
            }

            fetch('/api/respond-directed-opportunity', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    opportunity_id: currentOpportunityId,
                    message: message,
                    interested: interested,
                    have_contact: haveContact
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('¡Respuesta enviada exitosamente!');
                    bootstrap.Modal.getInstance(document.getElementById('respondModal')).hide();
                    // Limpiar formulario
                    document.getElementById('response_message').value = '';
                    document.getElementById('interested_check').checked = false;
                    document.getElementById('have_contact_check').checked = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error al enviar respuesta');
            });
        }

        // Ver detalles completos
        function viewFullOpportunity(oppId) {
            fetch(`/api/opportunity-details/${oppId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const opp = data.opportunity;
                    document.getElementById('opportunity-details').innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h6>Información Básica</h6>
                                <p><strong>Título:</strong> ${opp.title}</p>
                                <p><strong>Industria:</strong> ${opp.industry}</p>
                                <p><strong>Tipo:</strong> ${opp.type}</p>
                                <p><strong>Remitente:</strong> ${opp.sender_name}</p>
                            </div>
                            <div class="col-md-6">
                                <h6>Detalles</h6>
                                <p><strong>Enviado:</strong> ${opp.sent_at}</p>
                                <p><strong>Estado:</strong> ${opp.status}</p>
                            </div>
                            <div class="col-12">
                                <h6>Descripción Completa</h6>
                                <p>${opp.description}</p>
                            </div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('opportunity-details').innerHTML = 
                    '<p class="text-danger">Error al cargar detalles</p>';
            });
            
            new bootstrap.Modal(document.getElementById('detailsModal')).show();
        }
    </script>
</body>
</html>
'''

PUSH_PULL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PUSH & PULL - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-warning">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">PUSH & PULL Empresarial</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body">
                        <h4><i class="fas fa-building"></i> Sistema PUSH & PULL</h4>
                        <p class="text-muted">Registra empresas donde tienes acceso para que otros sepan que puedes ayudarles</p>
                        <button class="btn btn-warning" data-bs-toggle="modal" data-bs-target="#addCompanyModal">
                            <i class="fas fa-plus"></i> Agregar Empresa
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            {% for company in companies %}
            <div class="col-md-6 mb-3">
                <div class="card">
                    <div class="card-body">
                        <h5>{{ company.company_name }}</h5>
                        <p><strong>Industria:</strong> {{ company.industry }}</p>
                        <p><strong>Nivel de acceso:</strong> {{ company.access_level }}</p>
                        <p><strong>Contacto:</strong> {{ company.contact_person }}</p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <!-- Modal agregar empresa -->
    <div class="modal fade" id="addCompanyModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <form method="POST">
                    <div class="modal-header">
                        <h5>Agregar Empresa</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">Nombre de la Empresa</label>
                            <input type="text" class="form-control" name="company_name" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Industria</label>
                            <select class="form-control" name="industry" required>
                                <option>Tecnología</option>
                                <option>Fintech</option>
                                <option>E-commerce</option>
                                <option>Salud</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Nivel de Acceso</label>
                            <input type="text" class="form-control" name="access_level" placeholder="ej: Director, Gerente">
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Persona de Contacto</label>
                            <input type="text" class="form-control" name="contact_person">
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="submit" class="btn btn-warning">Agregar</button>
                    </div>
                </form>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

INVITAR_CONTACTOS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Invitar Contactos - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .social-card { transition: transform 0.2s; }
        .social-card:hover { transform: translateY(-5px); }
        .whatsapp-card { background: linear-gradient(45deg, #25d366, #128c7e); }
        .facebook-card { background: linear-gradient(45deg, #1877f2, #42a5f5); }
        .linkedin-card { background: linear-gradient(45deg, #0a66c2, #1e88e5); }
        .instagram-card { background: linear-gradient(45deg, #e4405f, #fd1d1d, #fcb045); }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-primary">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Invitar Contactos</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="row mb-4">
            <div class="col-12">
                <div class="card border-0 shadow-sm">
                    <div class="card-body text-center">
                        <h4><i class="fas fa-users"></i> Expande tu Red Friends & Family</h4>
                        <p class="text-muted">Invita contactos de confianza para crear una red de oportunidades</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- WhatsApp -->
            <div class="col-md-6 mb-4">
                <div class="card social-card whatsapp-card text-white border-0 shadow">
                    <div class="card-body text-center p-4">
                        <i class="fab fa-whatsapp fa-4x mb-3"></i>
                        <h3>WhatsApp</h3>
                        <p>Invita desde tu lista de contactos de WhatsApp</p>
                        <button class="btn btn-light" onclick="inviteWhatsApp()">
                            <i class="fas fa-share"></i> Invitar por WhatsApp
                        </button>
                    </div>
                </div>
            </div>

            <!-- Facebook -->
            <div class="col-md-6 mb-4">
                <div class="card social-card facebook-card text-white border-0 shadow">
                    <div class="card-body text-center p-4">
                        <i class="fab fa-facebook fa-4x mb-3"></i>
                        <h3>Facebook</h3>
                        <p>Comparte con tus amigos de Facebook</p>
                        <button class="btn btn-light" onclick="inviteFacebook()">
                            <i class="fas fa-share"></i> Compartir en Facebook
                        </button>
                    </div>
                </div>
            </div>

            <!-- LinkedIn -->
            <div class="col-md-6 mb-4">
                <div class="card social-card linkedin-card text-white border-0 shadow">
                    <div class="card-body text-center p-4">
                        <i class="fab fa-linkedin fa-4x mb-3"></i>
                        <h3>LinkedIn</h3>
                        <p>Conecta con tu red profesional</p>
                        <button class="btn btn-light" onclick="inviteLinkedIn()">
                            <i class="fas fa-share"></i> Compartir en LinkedIn
                        </button>
                    </div>
                </div>
            </div>

            <!-- Instagram -->
            <div class="col-md-6 mb-4">
                <div class="card social-card instagram-card text-white border-0 shadow">
                    <div class="card-body text-center p-4">
                        <i class="fab fa-instagram fa-4x mb-3"></i>
                        <h3>Instagram</h3>
                        <p>Comparte en tus historias de Instagram</p>
                        <button class="btn btn-light" onclick="inviteInstagram()">
                            <i class="fas fa-share"></i> Compartir en Instagram
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Invitación por email -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        <h5><i class="fas fa-envelope"></i> Invitación Manual</h5>
                        <form>
                            <div class="row">
                                <div class="col-md-8">
                                    <input type="email" class="form-control" placeholder="email@ejemplo.com">
                                </div>
                                <div class="col-md-4">
                                    <button type="submit" class="btn btn-primary w-100">Enviar Invitación</button>
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
        const inviteMessage = "¡Únete a Golden Coyotes! Una plataforma para conectar oportunidades de negocio con tu red Friends & Family. http://localhost:5002";

        function inviteWhatsApp() {
            window.open(`https://wa.me/?text=${encodeURIComponent(inviteMessage)}`, '_blank');
        }

        function inviteFacebook() {
            window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent('http://localhost:5002')}`, '_blank');
        }

        function inviteLinkedIn() {
            window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent('http://localhost:5002')}`, '_blank');
        }

        function inviteInstagram() {
            alert('Copia este mensaje para compartir en Instagram:\\n\\n' + inviteMessage);
            navigator.clipboard.writeText(inviteMessage);
        }
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    import sys
    
    port = 5001  # Puerto diferente para no conflictar
    if len(sys.argv) > 1:
        if sys.argv[1] == '--port':
            port = int(sys.argv[2]) if len(sys.argv) > 2 else port
    
    app = GoldenCoyotesAfterFeedback()
    app.run(host='0.0.0.0', port=port, debug=True)