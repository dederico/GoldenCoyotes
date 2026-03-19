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

from flask import Flask, abort, render_template_string, request, jsonify, redirect, url_for, flash, session
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
                ref_code = data.get('ref') or request.args.get('ref')
                
                # Validar campos requeridos
                if not data.get('name') or not data.get('email') or not data.get('password'):
                    flash('Por favor completa todos los campos requeridos', 'error')
                    return render_template_string(REGISTER_TEMPLATE)
                
                user_id = self.db.create_user(
                    email=data.get('email'),
                    password=data.get('password'),
                    name=data.get('name'),
                    phone=data.get('phone', ''),
                    interests=data.get('industry_preferences', '')
                )

                if not user_id:
                    flash('El email ya existe o el registro falló', 'error')
                    return render_template_string(REGISTER_TEMPLATE, ref_code=ref_code or '')

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

                session['user_id'] = user_id
                session['user_name'] = data.get('name')
                flash('¡Registro exitoso! Bienvenido a Golden Coyotes.', 'success')
                return redirect(url_for('dashboard'))
            
            ref_code = request.args.get('ref', '')
            return render_template_string(REGISTER_TEMPLATE, ref_code=ref_code)
        
        @self.app.route('/login', methods=['GET', 'POST'])
        def login():
            """Login de usuario"""
            if request.method == 'POST':
                data = request.form.to_dict()  # Siempre usar form data para login
                email = data.get('email')
                password = data.get('password')
                
                if email and password:
                    user = self.db.authenticate_user(email, password)
                    if user:
                        session['user_id'] = user['id']
                        session['user_name'] = user['name']
                        flash('¡Bienvenido a Golden Coyotes!', 'success')
                        return redirect(url_for('dashboard'))
                    flash('Credenciales incorrectas', 'error')
                else:
                    flash('Por favor ingresa email y contraseña', 'error')
            
            return render_template_string(LOGIN_TEMPLATE)
        
        # ==================== DASHBOARD PRINCIPAL (4 CUADRANTES) ====================
        @self.app.route('/dashboard')
        @self.require_login
        def dashboard():
            """Dashboard principal con los 4 cuadrantes según feedback"""
            user_id = session['user_id']
            ai_opportunities = self.get_ai_recommended_opportunities(user_id)
            ai_opportunity_ids = [opp['id'] for opp in ai_opportunities if opp.get('id')]
            
            # Obtener contadores para notificaciones
            counters = {
                'oportunidades_publicas': self.db.count_unviewed_opportunities(user_id, ai_opportunity_ids),
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
                    'expiration_date': data.get('expiration_date'),
                    'image_url': data.get('image_url'),
                    'is_public': True,  # Siempre pública en cuadrante 1
                    'media_files': data.get('media_files', []),
                    'created_at': datetime.now().isoformat()
                }

                try:
                    opp_id = self.create_opportunity(opportunity_data)
                    flash('¡Oportunidad publicada exitosamente! Podrás verla en "Mis Oportunidades".', 'success')
                    return redirect(url_for('mis_oportunidades'))
                except Exception as e:
                    flash(f'Error al crear oportunidad: {e}', 'error')

            # Calcular fechas para el formulario
            today = datetime.now()
            min_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
            default_date = (today + timedelta(days=30)).strftime('%Y-%m-%d')

            return render_template_string(SUBIR_OPORTUNIDAD_TEMPLATE,
                                        min_date=min_date,
                                        default_date=default_date)
        
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
                    'expiration_date': data.get('expiration_date'),
                    'image_url': data.get('image_url'),
                    'is_public': False,  # Privada/dirigida
                    'directed_to': data.get('selected_contacts', []),
                    'media_files': data.get('media_files', []),
                    'created_at': datetime.now().isoformat()
                }

                try:
                    opp_id = self.create_opportunity(opportunity_data)
                    # Notificar a los contactos seleccionados
                    self.notify_directed_opportunity(opp_id, opportunity_data['directed_to'])
                    flash('¡Oportunidad enviada exitosamente a tus contactos! Podrás verla en "Mis Oportunidades".', 'success')
                    return redirect(url_for('mis_oportunidades'))
                except Exception as e:
                    flash(f'Error al enviar oportunidad: {e}', 'error')

            # Calcular fechas para el formulario
            today = datetime.now()
            min_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
            default_date = (today + timedelta(days=30)).strftime('%Y-%m-%d')

            return render_template_string(OPORTUNIDAD_DIRIGIDA_TEMPLATE,
                                        contacts=my_contacts,
                                        min_date=min_date,
                                        default_date=default_date)
        
        # ==================== CUADRANTE 3: BUSCO OPORTUNIDAD GENERAL ====================
        @self.app.route('/buscar-oportunidad')
        @self.require_login
        def buscar_oportunidad():
            """Cuadrante 3: Ver oportunidades públicas con opción IA"""
            user_id = session['user_id']
            view_type = request.args.get('view', 'ai')  # 'ai' o 'all'
            selected_industry = (request.args.get('industry') or '').strip()
            selected_type = (request.args.get('type') or '').strip()
            search_term = (request.args.get('search') or '').strip()
            
            if view_type == 'ai':
                # La IA recomienda usando el perfil del usuario; los filtros refinan la lista resultante.
                opportunities = self.get_ai_recommended_opportunities(user_id)
                title = "Recomendaciones de GOLDEN"
                explanation = "GOLDEN, nuestro agente IA, usa la información de tu perfil al registrarte para proponerte oportunidades. Los filtros que selecciones aquí refinan ese resultado."
            else:
                # Todas las oportunidades públicas
                opportunities = self.get_public_opportunities(user_id)
                title = "Todas las Oportunidades Públicas"
                explanation = "Esta vista muestra las oportunidades disponibles en tu red. Los filtros que selecciones aquí refinan lo que ves."

            opportunities = self.filter_opportunities_for_display(
                opportunities,
                industry=selected_industry,
                opportunity_type=selected_type,
                search_term=search_term
            )
            opportunities = self.attach_review_statuses(
                user_id,
                opportunities,
                exclude_discarded=True
            )
            if view_type == 'ai':
                opportunities = self.attach_golden_messages(opportunities)
            
            return render_template_string(BUSCAR_OPORTUNIDAD_TEMPLATE, 
                                       opportunities=opportunities, 
                                       title=title, 
                                       view_type=view_type,
                                       explanation=explanation,
                                       selected_industry=selected_industry,
                                       selected_type=selected_type,
                                       search_term=search_term,
                                       page_mode='browse')

        @self.app.route('/mis-oportunidades-interes')
        @self.require_login
        def mis_oportunidades_interes():
            """Ver oportunidades marcadas con interés."""
            user_id = session['user_id']
            selected_industry = (request.args.get('industry') or '').strip()
            selected_type = (request.args.get('type') or '').strip()
            search_term = (request.args.get('search') or '').strip()

            opportunities = self.get_public_opportunities(user_id)
            opportunities = self.filter_opportunities_for_display(
                opportunities,
                industry=selected_industry,
                opportunity_type=selected_type,
                search_term=search_term
            )
            opportunities = self.attach_review_statuses(
                user_id,
                opportunities,
                exclude_discarded=True
            )
            opportunities = self.filter_opportunities_by_review_status(opportunities, 'interested')

            return render_template_string(
                BUSCAR_OPORTUNIDAD_TEMPLATE,
                opportunities=opportunities,
                title='Mis Oportunidades de Interés',
                view_type='interest',
                explanation='Aquí se concentran únicamente las oportunidades que marcaste en amarillo para consultarlas después sin perder las demás.',
                selected_industry=selected_industry,
                selected_type=selected_type,
                search_term=search_term,
                page_mode='interest'
            )

        @self.app.route('/mis-oportunidades-standby')
        @self.require_login
        def mis_oportunidades_standby():
            """Ver oportunidades dejadas en stand by."""
            user_id = session['user_id']
            selected_industry = (request.args.get('industry') or '').strip()
            selected_type = (request.args.get('type') or '').strip()
            search_term = (request.args.get('search') or '').strip()

            opportunities = self.get_public_opportunities(user_id)
            opportunities = self.filter_opportunities_for_display(
                opportunities,
                industry=selected_industry,
                opportunity_type=selected_type,
                search_term=search_term
            )
            opportunities = self.attach_review_statuses(
                user_id,
                opportunities,
                exclude_discarded=True
            )
            opportunities = self.filter_opportunities_by_review_status(opportunities, 'standby')

            return render_template_string(
                BUSCAR_OPORTUNIDAD_TEMPLATE,
                opportunities=opportunities,
                title='Mis Oportunidades en Stand By',
                view_type='standby',
                explanation='Aquí se guardan las oportunidades que dejaste en stand by para retomarlas después sin mezclarlas con tus intereses prioritarios.',
                selected_industry=selected_industry,
                selected_type=selected_type,
                search_term=search_term,
                page_mode='standby'
            )
        
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
            base_url = request.host_url.rstrip('/')
            invite_link = f"{base_url}/register?ref={session['user_id']}"
            inviter_name = session.get('user_name', 'Golden Coyotes')
            return render_template_string(
                INVITAR_CONTACTOS_TEMPLATE,
                invite_link=invite_link,
                inviter_name=inviter_name
            )
        
        # ==================== API ENDPOINTS ====================
        @self.app.route('/api/send-invitation-email', methods=['POST'])
        @self.require_login
        def send_invitation_email():
            """Enviar invitación por correo"""
            import logging
            logger = logging.getLogger(__name__)

            data = request.get_json() or request.form.to_dict()
            recipient_email = (data.get('email') or '').strip()
            personal_message = (data.get('message') or '').strip()

            logger.info(f"📧 Attempting to send invitation email to: {recipient_email}")

            if not recipient_email:
                logger.warning(f"❌ Email invitation failed: No email provided")
                return jsonify({'success': False, 'error': 'Email requerido'}), 400

            inviter = self.db.get_user(session['user_id'])
            inviter_name = inviter['name'] if inviter else session.get('user_name', 'Golden Coyotes')
            inviter_company = inviter.get('company', '') if inviter else ''
            invite_link = f"{request.host_url.rstrip('/')}/register?ref={session['user_id']}"

            logger.info(f"📤 Sending invitation from {inviter_name} to {recipient_email}")

            # Verificar configuración SMTP antes de enviar
            smtp_config = {
                'server': self.email_service.smtp_server,
                'port': self.email_service.smtp_port,
                'email': self.email_service.email_address,
                'password_set': bool(self.email_service.email_password and self.email_service.email_password != 'your-app-password')
            }
            logger.debug(f"SMTP Configuration: {smtp_config}")

            if not smtp_config['password_set']:
                error_msg = 'Las credenciales SMTP no están configuradas. Por favor configura EMAIL_ADDRESS y EMAIL_PASSWORD en variables de entorno.'
                logger.error(f"❌ {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'details': 'Revisa las variables de entorno: EMAIL_ADDRESS, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT'
                }), 500

            success = self.email_service.send_invitation_email(
                recipient_email=recipient_email,
                inviter_name=inviter_name,
                inviter_company=inviter_company,
                personal_message=personal_message,
                invite_link=invite_link
            )

            if success:
                logger.info(f"✅ Invitation email sent successfully to {recipient_email}")
                return jsonify({'success': True, 'message': f'Invitación enviada a {recipient_email}'})

            error_msg = 'No se pudo enviar el correo. Error de autenticación SMTP o conexión. Revisa los logs del servidor para más detalles.'
            logger.error(f"❌ Failed to send invitation email to {recipient_email}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'details': 'Revisa los logs del servidor para detalles específicos del error SMTP'
            }), 500
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

        @self.app.route('/api/opportunity-review-action', methods=['POST'])
        @self.require_login
        def opportunity_review_action():
            """Guardar acción de revisión: interés, stand by o descartar."""
            data = request.get_json() or {}
            user_id = session['user_id']
            opportunity_id = data.get('opportunity_id')
            action = data.get('action')

            if action not in {'interested', 'standby', 'discarded'}:
                return jsonify({'success': False, 'error': 'Acción inválida'}), 400

            if not opportunity_id:
                return jsonify({'success': False, 'error': 'opportunity_id requerido'}), 400

            try:
                success = self.set_opportunity_review_action(user_id, opportunity_id, action)
                if not success:
                    return jsonify({'success': False, 'error': 'No se pudo guardar la acción'}), 500
                return jsonify({'success': True, 'action': action})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
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

        # ==================== API ENDPOINTS PARA CONEXIONES ====================
        @self.app.route('/api/connections', methods=['GET'])
        @self.require_login
        def get_connections():
            """Obtener todas las conexiones del usuario"""
            user_id = session['user_id']
            status = request.args.get('status', 'accepted')

            try:
                connections = self.db.get_user_connections(user_id, status=status)
                return jsonify({'success': True, 'connections': connections})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/connections/pending', methods=['GET'])
        @self.require_login
        def get_pending_connections():
            """Obtener solicitudes de conexión pendientes"""
            user_id = session['user_id']

            try:
                pending = self.db.get_pending_connection_requests(user_id)
                return jsonify({'success': True, 'pending_requests': pending, 'count': len(pending)})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/connections/request', methods=['POST'])
        @self.require_login
        def request_connection():
            """Solicitar conexión con otro usuario"""
            data = request.get_json()
            user_id = session['user_id']
            target_user_id = data.get('target_user_id')
            message = data.get('message', '')

            if not target_user_id:
                return jsonify({'success': False, 'error': 'target_user_id requerido'}), 400

            if target_user_id == user_id:
                return jsonify({'success': False, 'error': 'No puedes conectarte contigo mismo'}), 400

            try:
                # Verificar que el usuario destino existe
                target_user = self.db.get_user(target_user_id)
                if not target_user:
                    return jsonify({'success': False, 'error': 'Usuario no encontrado'}), 404

                # Crear solicitud de conexión
                connection_id = self.db.create_connection(user_id, target_user_id, message, status='pending')

                if connection_id:
                    return jsonify({
                        'success': True,
                        'message': 'Solicitud de conexión enviada',
                        'connection_id': connection_id
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'La conexión ya existe o hubo un error'
                    }), 400
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/connections/accept', methods=['POST'])
        @self.require_login
        def accept_connection():
            """Aceptar una solicitud de conexión"""
            data = request.get_json()
            user_id = session['user_id']
            connection_id = data.get('connection_id')

            if not connection_id:
                return jsonify({'success': False, 'error': 'connection_id requerido'}), 400

            try:
                success = self.db.accept_connection(connection_id, user_id)

                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Conexión aceptada exitosamente'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No se pudo aceptar la conexión. Verifica que la solicitud existe y te pertenece.'
                    }), 400
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/connections/reject', methods=['POST'])
        @self.require_login
        def reject_connection():
            """Rechazar una solicitud de conexión"""
            data = request.get_json()
            user_id = session['user_id']
            connection_id = data.get('connection_id')

            if not connection_id:
                return jsonify({'success': False, 'error': 'connection_id requerido'}), 400

            try:
                success = self.db.reject_connection(connection_id, user_id)

                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Conexión rechazada'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No se pudo rechazar la conexión. Verifica que la solicitud existe y te pertenece.'
                    }), 400
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        # ==================== NUEVAS RUTAS FEEDBACK ====================
        @self.app.route('/mis-oportunidades')
        @self.require_login
        def mis_oportunidades():
            """Ver todas las oportunidades que he publicado"""
            user_id = session['user_id']
            user_opportunities = self.db.get_opportunities(user_id=user_id)
            return render_template_string(MIS_OPORTUNIDADES_TEMPLATE, opportunities=user_opportunities)

        @self.app.route('/oportunidades/<opportunity_id>')
        @self.require_login
        def opportunity_detail(opportunity_id):
            """Ver detalle completo de una oportunidad accesible para el usuario."""
            user_id = session['user_id']
            opportunity = self.get_accessible_opportunity(user_id, opportunity_id)

            if not opportunity:
                abort(404)

            is_owner = opportunity['user_id'] == user_id
            if not is_owner:
                self.db.record_opportunity_view(user_id, opportunity_id)
                opportunity['views'] = (opportunity.get('views') or 0) + 1
            return render_template_string(
                OPPORTUNITY_DETAIL_TEMPLATE,
                opportunity=opportunity,
                is_owner=is_owner,
                back_url=request.referrer or url_for('dashboard')
            )

        @self.app.route('/opportunities-status')
        @self.require_login
        def opportunities_status():
            """Ver estado de oportunidades de mi red con vigencia"""
            user_id = session['user_id']

            # CAMBIO CRÍTICO: Solo mostrar oportunidades de mi red + mis propias oportunidades
            network_opportunities = self.db.get_opportunities(network_only=True, requesting_user_id=user_id, limit=50)
            my_opportunities = self.db.get_opportunities(user_id=user_id, limit=50)

            # Combinar ambas listas sin duplicados
            all_opportunities = my_opportunities.copy()
            existing_ids = {opp['id'] for opp in my_opportunities}

            for opp in network_opportunities:
                if opp['id'] not in existing_ids:
                    all_opportunities.append(opp)

            # Calcular estado de vigencia para cada oportunidad
            for opp in all_opportunities:
                if opp.get('expiration_date'):
                    exp_date = datetime.strptime(opp['expiration_date'], '%Y-%m-%d')
                    days_left = (exp_date - datetime.now()).days
                    opp['days_left'] = days_left
                    opp['is_expired'] = days_left < 0
                    opp['is_expiring_soon'] = 0 <= days_left <= 7
                else:
                    opp['days_left'] = None
                    opp['is_expired'] = False
                    opp['is_expiring_soon'] = False

            return render_template_string(OPPORTUNITIES_STATUS_TEMPLATE, opportunities=all_opportunities)

        # ==================== VISTA DE MIS CONTACTOS ====================
        @self.app.route('/mis-contactos')
        @self.require_login
        def mis_contactos():
            """Ver y gestionar mis contactos y solicitudes de conexión"""
            user_id = session['user_id']

            # Obtener contactos aceptados
            my_contacts = self.db.get_user_connections(user_id, status='accepted')

            # Obtener solicitudes pendientes (que me enviaron a mí)
            pending_requests = self.db.get_pending_connection_requests(user_id)

            return render_template_string(
                MIS_CONTACTOS_TEMPLATE,
                contacts=my_contacts,
                pending_requests=pending_requests,
                total_contacts=len(my_contacts),
                pending_count=len(pending_requests)
            )

        # ==================== ADMIN: SEED DATABASE ====================
        @self.app.route('/admin/seed-demo-data')
        def seed_demo_data():
            """Poblar base de datos con datos de demostración (temporal)"""
            try:
                import seed_database
                seed_database.seed_database()
                return jsonify({
                    'success': True,
                    'message': 'Base de datos poblada con datos de demostración exitosamente'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        @self.app.route('/logout')
        @self.require_login
        def logout():
            """Cerrar sesión"""
            session.clear()
            flash('¡Sesión cerrada exitosamente!', 'info')
            return redirect(url_for('home'))
    
    # ==================== MÉTODOS DE DATOS ====================
    def get_public_opportunities(self, user_id):
        """Obtener oportunidades públicas de la red del usuario (solo de contactos)"""
        # CAMBIO CRÍTICO: Ahora filtra solo oportunidades de mi red de contactos
        return self.db.get_opportunities(network_only=True, requesting_user_id=user_id, limit=50)

    def get_directed_opportunities(self, user_id):
        """Obtener oportunidades dirigidas específicamente al usuario"""
        # Buscar oportunidades donde el user_id esté en el campo tags (dirigidas a él)
        all_opportunities = self.db.get_opportunities(network_only=True, requesting_user_id=user_id, limit=50)

        # Filtrar solo las que me mencionan en tags
        directed = []
        for opp in all_opportunities:
            tags = opp.get('tags', '')
            if tags and user_id in tags.split(','):
                directed.append(opp)

        return directed

    def get_my_opportunities(self, user_id):
        """Obtener las oportunidades creadas por el usuario"""
        return self.db.get_opportunities(user_id=user_id)

    def get_pending_invitations(self, user_id):
        """Obtener invitaciones pendientes de conexión"""
        return self.db.get_pending_connection_requests(user_id)

    def get_user_contacts(self, user_id):
        """Obtener contactos aceptados del usuario"""
        connections = self.db.get_user_connections(user_id, status='accepted')

        # Formatear para vista
        contacts = []
        for conn in connections:
            contacts.append({
                'id': conn['id'],
                'name': conn['name'],
                'email': conn['email'],
                'company': conn.get('company', ''),
                'position': conn.get('position', ''),
                'industry': conn.get('industry', ''),
                'relationship': 'Contacto de confianza'
            })

        return contacts

    def get_ai_recommended_opportunities(self, user_id):
        """Oportunidades recomendadas por IA (solo de mi red)"""
        # Obtener oportunidades de mi red y usar AI matcher para scoring
        network_opportunities = self.get_public_opportunities(user_id)

        # Si hay AI matching engine, usar para recomendar
        if hasattr(self, 'ai_matcher') and self.ai_matcher and hasattr(self.ai_matcher, 'calculate_opportunity_matches'):
            try:
                matches = self.ai_matcher.calculate_opportunity_matches(user_id, limit=len(network_opportunities) or 10)
                matched_by_id = {
                    match['opportunity']['id']: {
                        **match['opportunity'],
                        'match_score': match.get('score'),
                        'match_reasoning': match.get('reasoning')
                    }
                    for match in matches if match.get('opportunity', {}).get('id')
                }

                ordered = []
                used_ids = set()
                for match in matches:
                    opp = match.get('opportunity')
                    if opp and opp.get('id') in matched_by_id and opp['id'] not in used_ids:
                        ordered.append(matched_by_id[opp['id']])
                        used_ids.add(opp['id'])

                for opp in network_opportunities:
                    if opp.get('id') not in used_ids:
                        ordered.append(opp)

                return ordered
            except Exception:
                return network_opportunities

        return network_opportunities
    
    def get_user_company_access(self, user_id):
        """Obtener empresas donde el usuario tiene acceso"""
        return []

    def filter_opportunities_for_display(self, opportunities, industry='', opportunity_type='', search_term=''):
        """Filter opportunities according to the current module selections."""
        filtered = opportunities

        if industry:
            filtered = [opp for opp in filtered if (opp.get('industry') or '') == industry]

        if opportunity_type:
            filtered = [opp for opp in filtered if (opp.get('type') or '') == opportunity_type]

        if search_term:
            search_lower = search_term.lower()
            filtered = [
                opp for opp in filtered
                if search_lower in (opp.get('title') or '').lower()
                or search_lower in (opp.get('description') or '').lower()
                or search_lower in (opp.get('industry') or '').lower()
            ]

        return filtered

    def filter_opportunities_by_review_status(self, opportunities, review_filter=''):
        """Filter opportunities by saved review status."""
        if not review_filter:
            return opportunities
        return [opp for opp in opportunities if (opp.get('review_status') or '') == review_filter]

    def attach_review_statuses(self, user_id, opportunities, exclude_discarded=False):
        """Attach the user's review status to each opportunity."""
        opportunity_ids = [opp['id'] for opp in opportunities if opp.get('id')]
        statuses = self.db.get_opportunity_review_statuses(user_id, opportunity_ids)

        enriched = []
        for opp in opportunities:
            opp['review_status'] = statuses.get(opp.get('id'))
            if exclude_discarded and opp['review_status'] == 'discarded':
                continue
            enriched.append(opp)

        return enriched

    def set_opportunity_review_action(self, user_id, opportunity_id, action):
        """Persist review action for a user."""
        return self.db.set_opportunity_review_status(user_id, opportunity_id, action)

    def attach_golden_messages(self, opportunities):
        """Attach a short GOLDEN explanation to each AI recommendation."""
        enriched = []
        for opp in opportunities:
            reasoning = (opp.get('match_reasoning') or '').strip()
            if reasoning:
                opp['golden_message'] = f"GOLDEN detectó esta oportunidad para ti por: {reasoning}."
            else:
                industry = opp.get('industry') or 'tu perfil'
                opp['golden_message'] = f"GOLDEN detectó esta oportunidad para ti por afinidad con {industry} y el contexto de tu red."
            enriched.append(opp)
        return enriched

    def get_accessible_opportunity(self, user_id, opportunity_id):
        """Obtener una oportunidad validando que el usuario pueda verla."""
        opportunity = self.db.get_opportunity_by_id(opportunity_id)
        if not opportunity:
            return None

        if opportunity['user_id'] == user_id:
            return opportunity

        tags = (opportunity.get('tags') or '').strip()
        directed_user_ids = [tag.strip() for tag in tags.split(',') if tag.strip()]
        is_directed = bool(directed_user_ids)

        if is_directed:
            return opportunity if user_id in directed_user_ids else None

        return opportunity if self.db.is_connected(user_id, opportunity['user_id']) else None
    
    def create_opportunity(self, opportunity_data):
        """Crear nueva oportunidad"""
        expiration_date = opportunity_data.get('expiration_date')
        if not expiration_date:
            expiration_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

        opp_id = self.db.create_opportunity(
            user_id=opportunity_data.get('user_id'),
            title=opportunity_data.get('title'),
            description=opportunity_data.get('description'),
            opp_type=opportunity_data.get('type'),
            industry=opportunity_data.get('industry'),
            expiration_date=expiration_date,
            image_url=opportunity_data.get('image_url'),
            tags=','.join(opportunity_data.get('directed_to', [])) if 'directed_to' in opportunity_data else None
        )
        return opp_id
    
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
        return self.set_opportunity_review_action(user_id, opportunity_id, 'interested')
    
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
                        GOLDEN, nuestro agente IA, buscará matches relevantes para ti dentro de tu red y redes extendidas.
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
                                <p class="text-white-50">Conoce cómo GOLDEN, nuestro agente IA, te ayuda a encontrar matches.</p>
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
                            <p class="text-muted">Explora oportunidades de tu red con GOLDEN, nuestro agente IA</p>
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
                        <input type="hidden" name="ref" value="{{ ref_code }}">
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
                            <small class="text-muted">Esto ayudará a GOLDEN, nuestro agente IA, a recomendarte oportunidades relevantes</small>
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
                <a href="{{ url_for('mis_oportunidades_interes') }}" class="btn btn-outline-warning btn-sm me-2">
                    <i class="fas fa-star"></i> Mis Intereses
                </a>
                <a href="{{ url_for('mis_oportunidades_standby') }}" class="btn btn-outline-secondary btn-sm me-2">
                    <i class="fas fa-clock"></i> Stand By
                </a>
                <a href="{{ url_for('mis_oportunidades') }}" class="btn btn-outline-info btn-sm me-2">
                    <i class="fas fa-list"></i> Mis Oportunidades
                </a>
                <a href="{{ url_for('opportunities_status') }}" class="btn btn-outline-warning btn-sm me-2">
                    <i class="fas fa-chart-line"></i> Estado
                </a>
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
                            • El contador rojo = pendientes por abrir<br>
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
                                <label class="form-label">Vigencia de la Oportunidad *</label>
                                <input type="date" class="form-control" name="expiration_date" required
                                       min="{{ min_date }}"
                                       value="{{ default_date }}">
                                <small class="text-muted">
                                    Fecha hasta la cual esta oportunidad estará vigente. Por defecto: 30 días.
                                </small>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Imagen de la Oportunidad (opcional)</label>
                                <input type="url" class="form-control" name="image_url"
                                       placeholder="https://ejemplo.com/imagen.jpg">
                                <small class="text-muted">
                                    Pega la URL de una imagen que represente tu oportunidad (ej: desde Imgur, Dropbox, Google Drive público)
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

                            <div class="mb-3">
                                <label class="form-label">Vigencia de la Oportunidad *</label>
                                <input type="date" class="form-control" name="expiration_date" required
                                       min="{{ min_date }}"
                                       value="{{ default_date }}">
                                <small class="text-muted">
                                    Fecha hasta la cual esta oportunidad estará vigente. Por defecto: 30 días.
                                </small>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Imagen de la Oportunidad (opcional)</label>
                                <input type="url" class="form-control" name="image_url"
                                       placeholder="https://ejemplo.com/imagen.jpg">
                                <small class="text-muted">
                                    Pega la URL de una imagen que represente tu oportunidad (ej: desde Imgur, Dropbox, Google Drive público)
                                </small>
                            </div>

                            <!-- Selector de Contactos -->
                            <div class="mb-3">
                                <label class="form-label">Seleccionar Contactos *</label>
                                <p class="text-muted small">Elige a cuáles de tus contactos de confianza enviar esta oportunidad:</p>
                                {% if contacts %}
                                    <div class="border rounded p-3" style="max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
                                        <div class="row">
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
                                        </div>
                                    </div>
                                {% else %}
                                    <div class="alert alert-info">
                                        <i class="fas fa-info-circle"></i>
                                        Aún no tienes contactos en tu red Friends & Family.
                                        <a href="{{ url_for('invitar_contactos') }}" class="alert-link">Invita contactos aquí</a>
                                    </div>
                                {% endif %}
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
        .opportunities-scroll-box {
            max-height: 70vh;
            overflow-y: auto;
            padding-right: 8px;
        }
        .ai-badge { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: white;
        }
        .interest-btn.active {
            background: #28a745;
            color: white;
        }
        .standby-btn.active {
            background: #6c757d;
            color: white;
        }
        .discard-btn.active {
            background: #dc3545;
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
                        <div class="alert alert-info mb-3">
                            <small>{{ explanation }}</small>
                        </div>
                        <div class="alert alert-light border mb-3">
                            <small>
                                <strong>Acciones:</strong> estrella amarilla = interés, reloj = stand by, bote rojo = descartar y quitar de esta bandeja.
                            </small>
                        </div>
                        <div class="alert alert-info mb-0">
                            <small>
                                <strong>Contador rojo del dashboard:</strong> representa las oportunidades recomendadas por IA que todavía no has abierto.
                                Cuando abres una oportunidad, el contador se ajusta automáticamente.
                            </small>
                        </div>
                        
                        {% if page_mode == 'browse' %}
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
                        <div class="mt-3">
                            <a href="{{ url_for('mis_oportunidades_interes') }}" class="btn btn-outline-warning">
                                <i class="fas fa-star"></i> Ver Mis Oportunidades de Interés
                            </a>
                            <a href="{{ url_for('mis_oportunidades_standby') }}" class="btn btn-outline-secondary ms-2">
                                <i class="fas fa-clock"></i> Ver Mis Stand By
                            </a>
                        </div>
                        {% else %}
                        <div class="mt-3">
                            <a href="{{ url_for('buscar_oportunidad', view='ai') }}" class="btn btn-outline-primary me-2">
                                <i class="fas fa-robot"></i> Volver a Recomendaciones de GOLDEN
                            </a>
                            <a href="{{ url_for('buscar_oportunidad', view='all') }}" class="btn btn-outline-secondary">
                                <i class="fas fa-list"></i> Volver a Todas las Oportunidades
                            </a>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <!-- Filtros -->
        <form class="row mb-4" method="GET" action="{{ url_for('buscar_oportunidad') if page_mode == 'browse' else url_for('mis_oportunidades_interes') }}">
            {% if page_mode == 'browse' %}
            <input type="hidden" name="view" value="{{ view_type }}">
            {% endif %}
            <div class="col-md-4">
                <select class="form-control" id="industry_filter" name="industry">
                    <option value="">Todas las industrias</option>
                    <option value="Tecnología" {% if selected_industry == 'Tecnología' %}selected{% endif %}>Tecnología</option>
                    <option value="Fintech" {% if selected_industry == 'Fintech' %}selected{% endif %}>Fintech</option>
                    <option value="E-commerce" {% if selected_industry == 'E-commerce' %}selected{% endif %}>E-commerce</option>
                    <option value="Salud" {% if selected_industry == 'Salud' %}selected{% endif %}>Salud</option>
                    <option value="Educación" {% if selected_industry == 'Educación' %}selected{% endif %}>Educación</option>
                </select>
            </div>
            <div class="col-md-4">
                <select class="form-control" id="type_filter" name="type">
                    <option value="">Producto y Servicio</option>
                    <option value="producto" {% if selected_type == 'producto' %}selected{% endif %}>Solo Productos</option>
                    <option value="servicio" {% if selected_type == 'servicio' %}selected{% endif %}>Solo Servicios</option>
                </select>
            </div>
            <div class="col-md-4 d-flex gap-2">
                <input type="text" class="form-control" id="search_filter" name="search" placeholder="Buscar por palabra clave..." value="{{ search_term }}">
                <button type="submit" class="btn btn-primary">
                    <i class="fas fa-filter"></i>
                </button>
            </div>
        </form>

        {% if view_type == 'all' %}
        <div class="alert alert-info mb-4">
            <h5><i class="fas fa-info-circle"></i> Sobre "Todas las Oportunidades"</h5>
            <p class="mb-0">
                En esta sección ves todas las oportunidades disponibles dentro de tu red. Usa los filtros para acotar la búsqueda y luego revisa cada una en la bandeja: puedes marcar interés, dejarla en stand by o descartarla si no aplica para ti.
            </p>
        </div>
        {% endif %}

        <!-- Lista de Oportunidades -->
        {% if view_type == 'all' %}
        <div class="card border-0 shadow-sm mb-4">
            <div class="card-header bg-white">
                <h5 class="mb-0"><i class="fas fa-list"></i> Bandeja de oportunidades con scroll</h5>
                <small class="text-muted">Explora la lista completa, baja con scroll y decide si te interesa, la dejas en stand by o la descartas.</small>
            </div>
            <div class="card-body">
                <div class="opportunities-scroll-box">
        {% endif %}
        <div class="row" id="opportunities-container">
            {% if opportunities %}
                {% for opp in opportunities %}
                <div class="col-lg-6 mb-4 opportunity-item" 
                     data-opportunity-id="{{ opp.id }}"
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
                                    <i class="fas fa-brain"></i> {{ ((opp.match_score or 0.85) * 100)|round|int }}% Match
                                </span>
                            {% endif %}
                        </div>
                        
                        <div class="card-body">
                            <h5 class="card-title">{{ opp.title }}</h5>
                            <p class="card-text text-muted">{{ opp.description[:150] }}{% if opp.description|length > 150 %}...{% endif %}</p>
                            {% if view_type == 'ai' %}
                            <div class="alert alert-warning py-2 mb-3">
                                <small>
                                    <i class="fas fa-robot"></i>
                                    <strong>GOLDEN dice:</strong> {{ opp.golden_message }}
                                </small>
                            </div>
                            {% endif %}
                            
                            <div class="row text-center mt-3">
                                <div class="col-4">
                                    <small class="text-muted">Publicado por</small>
                                    <div class="fw-bold">{{ opp.owner_name or opp.creator_name }}</div>
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
                                <a href="{{ url_for('opportunity_detail', opportunity_id=opp.id) }}" class="btn btn-outline-info">
                                    <i class="fas fa-eye"></i> Abrir oportunidad
                                </a>
                                <button class="btn btn-outline-warning interest-btn {% if opp.review_status == 'interested' %}active{% endif %}" 
                                        onclick="setReviewAction('{{ opp.id }}', 'interested', this)">
                                    <i class="fas fa-star"></i> {{ 'Me interesa' if opp.review_status == 'interested' else 'Marcar Interés' }}
                                </button>
                                <button class="btn btn-outline-secondary standby-btn {% if opp.review_status == 'standby' %}active{% endif %}" 
                                        onclick="setReviewAction('{{ opp.id }}', 'standby', this)">
                                    <i class="fas fa-clock"></i> {{ 'En stand by' if opp.review_status == 'standby' else 'Dejar en Stand By' }}
                                </button>
                                <button class="btn btn-outline-danger discard-btn {% if opp.review_status == 'discarded' %}active{% endif %}" 
                                        onclick="setReviewAction('{{ opp.id }}', 'discarded', this)">
                                    <i class="fas fa-trash"></i> {{ 'Descartada' if opp.review_status == 'discarded' else 'Descartar' }}
                                </button>
                                <button class="btn btn-primary" 
                                        onclick="contactOwner('{{ opp.id }}', '{{ opp.owner_name or opp.creator_name }}')">
                                    <i class="fas fa-comment"></i> Contactar a {{ opp.owner_name or opp.creator_name }}
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
        {% if view_type == 'all' %}
                </div>
            </div>
        </div>
        {% endif %}

        <div class="row mt-4">
            <div class="col-12">
                <div class="alert alert-info">
                    {% if view_type == 'ai' %}
                    <h5><i class="fas fa-lightbulb"></i> Ventaja del Networking IA</h5>
                    <p class="mb-0">
                        <strong>La IA puede identificar oportunidades de personas que NO están en tu red directa, 
                        pero SÍ están en la red de tus contactos.</strong> Esto te permite hacer networking más fino
                        partiendo de afinidad con tu perfil.
                    </p>
                    {% elif view_type == 'interest' %}
                    <h5><i class="fas fa-star"></i> Tu Bandeja de Interés</h5>
                    <p class="mb-0">
                        <strong>Aquí concentras solo las oportunidades que marcaste en amarillo.</strong> Esta vista existe
                        para que puedas retomarlas después sin perder de vista el resto de tus oportunidades.
                    </p>
                    {% elif view_type == 'standby' %}
                    <h5><i class="fas fa-clock"></i> Tu Bandeja en Stand By</h5>
                    <p class="mb-0">
                        <strong>Aquí concentras las oportunidades que dejaste pendientes para revisar después.</strong>
                        Es una bandeja intermedia entre lo prioritario y lo descartado.
                    </p>
                    {% else %}
                    <h5><i class="fas fa-network-wired"></i> Vista Completa de Oportunidades</h5>
                    <p class="mb-0">
                        <strong>Aquí ves todas las oportunidades disponibles en tu red.</strong> Esta vista te ayuda a
                        explorar manualmente, comparar opciones y detectar oportunidades que quizá la IA no priorizó,
                        pero que sí pueden servirte por contexto, memoria o un contacto reciente.
                    </p>
                    {% endif %}
                </div>
            </div>
        </div>
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

        function setReviewAction(oppId, action, button) {
            fetch('/api/opportunity-review-action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({opportunity_id: oppId, action: action})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const card = button.closest('.opportunity-item');
                    const buttons = card.querySelectorAll('.interest-btn, .standby-btn, .discard-btn');
                    buttons.forEach(btn => btn.classList.remove('active'));

                    const labels = {
                        interested: '<i class="fas fa-star"></i> Me interesa',
                        standby: '<i class="fas fa-clock"></i> En stand by',
                        discarded: '<i class="fas fa-trash"></i> Descartada'
                    };

                    button.classList.add('active');
                    button.innerHTML = labels[action];

                    if (action === 'discarded') {
                        card.remove();
                    }
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
                                    <a href="{{ url_for('opportunity_detail', opportunity_id=opp.id) }}" class="btn btn-outline-primary w-100">
                                        <i class="fas fa-eye"></i> Ver Detalles Completos
                                    </a>
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
                        <form id="emailInviteForm" onsubmit="sendInviteEmail(event)">
                            <div class="row">
                                <div class="col-md-8 mb-2">
                                    <input type="email" class="form-control" name="email" placeholder="email@ejemplo.com" required>
                                </div>
                                <div class="col-md-8 mb-2">
                                    <textarea class="form-control" name="message" rows="2" placeholder="Mensaje personal (opcional)"></textarea>
                                </div>
                                <div class="col-md-4 mb-2">
                                    <button type="submit" class="btn btn-primary w-100">Enviar Invitación</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal de resultado -->
    <div class="modal fade" id="inviteResultModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="inviteResultTitle">Resultado</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body" id="inviteResultBody"></div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cerrar</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        var inviteLink = "{{ invite_link }}";
        var inviterName = "{{ inviter_name }}";
        var inviteMessage = inviterName + ' te invita a Golden Coyotes! Una plataforma para conectar oportunidades de negocio con tu red Friends & Family. ' + inviteLink;

        var inviteResultModal = new bootstrap.Modal(document.getElementById('inviteResultModal'));

        function showInviteResult(title, message, isError) {
            isError = isError || false;
            var titleEl = document.getElementById('inviteResultTitle');
            var bodyEl = document.getElementById('inviteResultBody');
            titleEl.textContent = title;
            // Convertir saltos de linea a <br> para mejor visualizacion
            bodyEl.innerHTML = message.replace(/\\n/g, '<br>');
            titleEl.className = isError ? 'modal-title text-danger' : 'modal-title text-success';
            inviteResultModal.show();
        }

        async function tryNativeShare() {
            if (navigator.share) {
                try {
                    await navigator.share({
                        title: 'Invitación a Golden Coyotes',
                        text: inviteMessage,
                        url: inviteLink
                    });
                    return true;
                } catch (err) {
                    return false;
                }
            }
            return false;
        }

        function inviteWhatsApp() {
            tryNativeShare().then(function(shared) {
                if (!shared) {
                    window.open('https://wa.me/?text=' + encodeURIComponent(inviteMessage), '_blank');
                }
            });
        }

        function inviteFacebook() {
            var facebookUrl = 'https://www.facebook.com/sharer/sharer.php?u=' + encodeURIComponent(inviteLink);
            window.open(facebookUrl, '_blank', 'width=600,height=500');
        }

        function inviteLinkedIn() {
            // Intentar usar Web Share API primero (funciona mejor en moviles)
            tryNativeShare().then(function(shared) {
                if (shared) {
                    return;
                }
                // Si no funciona, abrir mensajes de LinkedIn y copiar el mensaje
                var linkedInMessagesUrl = 'https://www.linkedin.com/messaging/';
                window.open(linkedInMessagesUrl, '_blank');

                // Copiar mensaje al portapapeles
                if (navigator.clipboard) {
                    navigator.clipboard.writeText(inviteMessage).then(function() {
                        showInviteResult(
                            'LinkedIn - Mensaje copiado',
                            'Se abrio LinkedIn Mensajes y se copio el mensaje al portapapeles.\\n\\nPega el mensaje en un chat con tus amigos para invitarlos.',
                            false
                        );
                    }).catch(function() {
                        showInviteResult(
                            'LinkedIn abierto',
                            'Se abrio LinkedIn Mensajes.\\n\\nCopia y pega este mensaje:\\n\\n' + inviteMessage,
                            false
                        );
                    });
                } else {
                    showInviteResult(
                        'LinkedIn abierto',
                        'Se abrio LinkedIn Mensajes.\\n\\nCopia y pega este mensaje:\\n\\n' + inviteMessage,
                        false
                    );
                }
            });
        }

        function inviteInstagram() {
            tryNativeShare().then(function(shared) {
                if (shared) {
                    return;
                }
                var instagramUrl = 'https://www.instagram.com/direct/inbox/';
                window.open(instagramUrl, '_blank');
                if (navigator.clipboard) {
                    navigator.clipboard.writeText(inviteMessage);
                }
                showInviteResult('Mensaje copiado', 'Se copio el mensaje al portapapeles. Pegalo en Instagram Direct.', false);
            });
        }

        function sendInviteEmail(e) {
            e.preventDefault();
            var form = document.getElementById('emailInviteForm');
            if (!form) {
                alert('Formulario no encontrado.');
                return false;
            }
            var formData = new FormData(form);
            var payload = Object.fromEntries(formData);

            // Mostrar indicador de carga
            var submitBtn = form.querySelector('button[type="submit"]');
            var originalBtnText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Enviando...';

            fetch('/api/send-invitation-email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(function(response) {
                return response.json();
            })
            .then(function(result) {
                console.log('Server response:', result);

                if (result.success) {
                    form.reset();
                    var message = result.message || 'El correo se envio correctamente.';
                    showInviteResult('Invitacion enviada', message, false);
                } else {
                    var errorMessage = result.error || 'No se pudo enviar el correo.';
                    if (result.details) {
                        errorMessage += '\\n\\n' + result.details;
                    }
                    console.error('Email error:', errorMessage);
                    showInviteResult('Error al enviar', errorMessage, true);
                }
            })
            .catch(function(error) {
                console.error('Network error:', error);
                showInviteResult('Error al enviar', 'Ocurrio un error de red. Intenta de nuevo.\\n\\nRevisa la consola del navegador y los logs del servidor para mas detalles.', true);
            })
            .finally(function() {
                // Restaurar boton
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalBtnText;
            });
            return false;
        }
    </script>
</body>
</html>
'''

# ==================== TEMPLATES FEEDBACK ====================

MIS_OPORTUNIDADES_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mis Oportunidades - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-info">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Mis Oportunidades Publicadas</span>
        </div>
    </nav>

    <div class="container py-4">
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

        <div class="card shadow">
            <div class="card-header bg-info text-white">
                <h4 class="mb-0"><i class="fas fa-list"></i> Oportunidades que He Publicado</h4>
                <p class="mb-0 mt-2 opacity-75">Todas las oportunidades que has creado</p>
            </div>
            <div class="card-body">
                {% if opportunities %}
                    <div class="row">
                        {% for opp in opportunities %}
                            <div class="col-md-6 mb-3">
                                <div class="card h-100">
                                    {% if opp.image_url %}
                                        <img src="{{ opp.image_url }}" class="card-img-top" alt="{{ opp.title }}"
                                             style="height: 200px; object-fit: cover;">
                                    {% endif %}
                                    <div class="card-body">
                                        <h5 class="card-title">{{ opp.title }}</h5>
                                        <p class="card-text">{{ opp.description[:150] }}...</p>
                                        <div class="mb-2">
                                            <span class="badge bg-primary">{{ opp.type }}</span>
                                            <span class="badge bg-secondary">{{ opp.industry }}</span>
                                        </div>
                                        {% if opp.expiration_date %}
                                            <p class="text-muted small">
                                                <i class="fas fa-calendar"></i>
                                                Vigencia: {{ opp.expiration_date }}
                                            </p>
                                        {% endif %}
                                        <p class="text-muted small">
                                            <i class="fas fa-clock"></i>
                                            Publicada: {{ opp.created_at }}
                                        </p>
                                        <a href="{{ url_for('opportunity_detail', opportunity_id=opp.id) }}" class="btn btn-outline-info btn-sm">
                                            <i class="fas fa-eye"></i> Abrir oportunidad
                                        </a>
                                    </div>
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                {% else %}
                    <div class="alert alert-info text-center">
                        <i class="fas fa-info-circle fa-3x mb-3"></i>
                        <h5>No has publicado oportunidades aún</h5>
                        <p>Comienza publicando tu primera oportunidad</p>
                        <a href="{{ url_for('subir_oportunidad') }}" class="btn btn-primary">
                            <i class="fas fa-plus"></i> Crear Oportunidad
                        </a>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

OPPORTUNITY_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ opportunity.title }} - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container">
            <a href="{{ back_url }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver
            </a>
            <span class="navbar-brand">Detalle de Oportunidad</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="row">
            <div class="col-lg-8 mb-4">
                <div class="card shadow-sm border-0 h-100">
                    {% if opportunity.image_url %}
                        <img src="{{ opportunity.image_url }}" class="card-img-top" alt="{{ opportunity.title }}" style="max-height: 380px; object-fit: cover;">
                    {% endif %}
                    <div class="card-body">
                        <div class="d-flex flex-wrap gap-2 mb-3">
                            <span class="badge bg-{{ 'primary' if opportunity.type == 'producto' else 'success' }}">{{ opportunity.type|title }}</span>
                            <span class="badge bg-secondary">{{ opportunity.industry or 'Sin industria' }}</span>
                            <span class="badge bg-{{ 'warning text-dark' if opportunity.tags else 'info text-dark' }}">
                                {{ 'Dirigida' if opportunity.tags else 'Pública' }}
                            </span>
                        </div>
                        <h1 class="h3 mb-3">{{ opportunity.title }}</h1>
                        <p class="text-muted mb-4">Revisión completa de la oportunidad y sus características.</p>

                        <h5>Descripción</h5>
                        <p class="mb-0" style="white-space: pre-line;">{{ opportunity.description }}</p>
                    </div>
                </div>
            </div>

            <div class="col-lg-4">
                <div class="card shadow-sm border-0 mb-4">
                    <div class="card-header bg-white">
                        <h5 class="mb-0"><i class="fas fa-list-check text-primary"></i> Características</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Publicado por:</strong><br>{{ opportunity.creator_name }}</p>
                        <p><strong>Email de contacto:</strong><br>{{ opportunity.creator_email or 'No disponible' }}</p>
                        <p><strong>Industria:</strong><br>{{ opportunity.industry or 'No especificada' }}</p>
                        <p><strong>Tipo:</strong><br>{{ opportunity.type|title }}</p>
                        <p><strong>Fecha de publicación:</strong><br>{{ opportunity.created_at[:10] if opportunity.created_at else 'No disponible' }}</p>
                        <p><strong>Vigencia:</strong><br>{{ opportunity.expiration_date or 'No especificada' }}</p>
                        <p class="mb-0"><strong>Estado de visibilidad:</strong><br>{{ 'Dirigida a contactos específicos' if opportunity.tags else 'Visible para tu red' }}</p>
                    </div>
                </div>

                {% if not is_owner %}
                <div class="card shadow-sm border-0">
                    <div class="card-body">
                        <h5 class="card-title">Siguiente acción</h5>
                        <p class="text-muted">Ya puedes revisar esta oportunidad a detalle. El siguiente paso es contactar al dueño si te interesa.</p>
                        <a href="{{ url_for('buscar_oportunidad') }}" class="btn btn-primary w-100">
                            <i class="fas fa-search"></i> Volver a oportunidades
                        </a>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

OPPORTUNITIES_STATUS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Estado de Oportunidades - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .status-expired { border-left: 4px solid #dc3545; }
        .status-expiring { border-left: 4px solid #ffc107; }
        .status-active { border-left: 4px solid #28a745; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-warning">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-dark">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand text-dark">Estado de Oportunidades</span>
        </div>
    </nav>

    <div class="container py-4">
        <div class="card shadow">
            <div class="card-header bg-warning">
                <h4 class="mb-0"><i class="fas fa-chart-line"></i> Estado de Todas las Oportunidades</h4>
                <p class="mb-0 mt-2 opacity-75">Monitoreo de vigencia de oportunidades en la plataforma</p>
                <small class="d-block mt-2">
                    Validación actual: en tiempo real cada vez que entras a esta vista. Recomendación operativa: revisar y depurar diariamente las oportunidades por vencer o vencidas.
                </small>
            </div>
            <div class="card-body">
                {% if opportunities %}
                    <div class="row mb-3">
                        <div class="col-md-4">
                            <div class="card bg-success text-white">
                                <div class="card-body text-center">
                                    <h3>{{ opportunities|selectattr('is_expired', 'equalto', False)|selectattr('is_expiring_soon', 'equalto', False)|list|length }}</h3>
                                    <p class="mb-0">Activas</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card bg-warning text-dark">
                                <div class="card-body text-center">
                                    <h3>{{ opportunities|selectattr('is_expiring_soon', 'equalto', True)|list|length }}</h3>
                                    <p class="mb-0">Por Vencer</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="card bg-danger text-white">
                                <div class="card-body text-center">
                                    <h3>{{ opportunities|selectattr('is_expired', 'equalto', True)|list|length }}</h3>
                                    <p class="mb-0">Vencidas</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Título</th>
                                    <th>Creador</th>
                                    <th>Tipo</th>
                                    <th>Vigencia</th>
                                    <th>Estado</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for opp in opportunities %}
                                    <tr class="{% if opp.is_expired %}status-expired{% elif opp.is_expiring_soon %}status-expiring{% else %}status-active{% endif %}">
                                        <td>
                                            <a href="{{ url_for('opportunity_detail', opportunity_id=opp.id) }}" class="text-decoration-none fw-bold">
                                                {{ opp.title }}
                                            </a>
                                        </td>
                                        <td>{{ opp.creator_name }}</td>
                                        <td><span class="badge bg-primary">{{ opp.type }}</span></td>
                                        <td>
                                            {% if opp.expiration_date %}
                                                {{ opp.expiration_date }}
                                            {% else %}
                                                <span class="text-muted">Sin especificar</span>
                                            {% endif %}
                                        </td>
                                        <td>
                                            {% if opp.is_expired %}
                                                <span class="badge bg-danger">
                                                    <i class="fas fa-times-circle"></i> Vencida
                                                </span>
                                            {% elif opp.is_expiring_soon %}
                                                <span class="badge bg-warning text-dark">
                                                    <i class="fas fa-exclamation-triangle"></i>
                                                    Vence en {{ opp.days_left }} día{{ 's' if opp.days_left != 1 else '' }}
                                                </span>
                                            {% else %}
                                                <span class="badge bg-success">
                                                    <i class="fas fa-check-circle"></i>
                                                    Activa ({{ opp.days_left }} días)
                                                </span>
                                            {% endif %}
                                        </td>
                                    </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                {% else %}
                    <div class="alert alert-info text-center">
                        <i class="fas fa-info-circle fa-3x mb-3"></i>
                        <h5>No hay oportunidades en la plataforma</h5>
                    </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# Template para vista de Mis Contactos
MIS_CONTACTOS_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mis Contactos - Golden Coyotes</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-primary">
        <div class="container">
            <a href="{{ url_for('dashboard') }}" class="btn btn-outline-light">
                <i class="fas fa-arrow-left"></i> Volver al Dashboard
            </a>
            <span class="navbar-brand">Mis Contactos</span>
        </div>
    </nav>

    <div class="container py-4">
        <!-- Métricas -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="fas fa-users text-primary"></i> Contactos Aceptados
                        </h5>
                        <h2 class="mb-0">{{ total_contacts }}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">
                            <i class="fas fa-clock text-warning"></i> Solicitudes Pendientes
                        </h5>
                        <h2 class="mb-0">{{ pending_count }}</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- Solicitudes Pendientes -->
        {% if pending_requests %}
        <div class="card mb-4">
            <div class="card-header bg-warning text-dark">
                <h5 class="mb-0">
                    <i class="fas fa-bell"></i> Solicitudes de Conexión Pendientes
                </h5>
            </div>
            <div class="card-body">
                <div class="row">
                    {% for request in pending_requests %}
                    <div class="col-md-6 mb-3">
                        <div class="card">
                            <div class="card-body">
                                <h6 class="card-title">
                                    <i class="fas fa-user"></i> {{ request.requester_name }}
                                </h6>
                                <p class="card-text small mb-2">
                                    <strong>Email:</strong> {{ request.requester_email }}<br>
                                    {% if request.company %}
                                    <strong>Empresa:</strong> {{ request.company }}<br>
                                    {% endif %}
                                    {% if request.position %}
                                    <strong>Cargo:</strong> {{ request.position }}<br>
                                    {% endif %}
                                    {% if request.industry %}
                                    <strong>Industria:</strong> {{ request.industry }}<br>
                                    {% endif %}
                                </p>
                                {% if request.message %}
                                <p class="card-text small text-muted fst-italic">
                                    "{{ request.message }}"
                                </p>
                                {% endif %}
                                <div class="d-grid gap-2 d-md-flex">
                                    <button class="btn btn-success btn-sm" onclick="acceptConnection('{{ request.id }}')">
                                        <i class="fas fa-check"></i> Aceptar
                                    </button>
                                    <button class="btn btn-danger btn-sm" onclick="rejectConnection('{{ request.id }}')">
                                        <i class="fas fa-times"></i> Rechazar
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Lista de Contactos -->
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">
                    <i class="fas fa-address-book"></i> Mis Contactos ({{ total_contacts }})
                </h5>
            </div>
            <div class="card-body">
                {% if contacts %}
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Nombre</th>
                                <th>Email</th>
                                <th>Empresa</th>
                                <th>Cargo</th>
                                <th>Industria</th>
                                <th>Desde</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for contact in contacts %}
                            <tr>
                                <td>
                                    <i class="fas fa-user-circle text-primary"></i>
                                    <strong>{{ contact.name }}</strong>
                                </td>
                                <td>{{ contact.email }}</td>
                                <td>{{ contact.company or '-' }}</td>
                                <td>{{ contact.position or '-' }}</td>
                                <td>{{ contact.industry or '-' }}</td>
                                <td>
                                    <small class="text-muted">
                                        {{ contact.created_at[:10] if contact.created_at else '-' }}
                                    </small>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% else %}
                <div class="alert alert-info text-center">
                    <i class="fas fa-info-circle fa-3x mb-3"></i>
                    <h5>Aún no tienes contactos en tu red</h5>
                    <p>Invita a tus amigos, familiares y colegas para empezar a compartir oportunidades.</p>
                    <a href="{{ url_for('invitar_contactos') }}" class="btn btn-primary">
                        <i class="fas fa-user-plus"></i> Invitar Contactos
                    </a>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function acceptConnection(connectionId) {
            try {
                const response = await fetch('/api/connections/accept', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ connection_id: connectionId })
                });

                const data = await response.json();

                if (data.success) {
                    alert('Conexión aceptada exitosamente');
                    location.reload();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error al aceptar conexión: ' + error);
            }
        }

        async function rejectConnection(connectionId) {
            if (!confirm('¿Estás seguro de rechazar esta solicitud?')) {
                return;
            }

            try {
                const response = await fetch('/api/connections/reject', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ connection_id: connectionId })
                });

                const data = await response.json();

                if (data.success) {
                    alert('Conexión rechazada');
                    location.reload();
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error al rechazar conexión: ' + error);
            }
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
