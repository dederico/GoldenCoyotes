#!/usr/bin/env python3
"""
Script para poblar la base de datos con datos de ejemplo realistas
"""
import sys
from datetime import datetime, timedelta
from database_setup import DatabaseManager

def seed_database():
    """Poblar base de datos con datos de ejemplo"""
    db = DatabaseManager()

    print("🌱 Poblando base de datos con ejemplos realistas...")

    # ==================== USUARIOS ====================
    print("\n👥 Creando usuarios...")

    users = [
        {
            'email': 'maria.lopez@techstart.mx',
            'password': 'demo123',
            'name': 'María López',
            'phone': '+52 55 1234 5678',
            'interests': 'Tecnología,Fintech,Startups',
            'company': 'TechStart México',
            'position': 'CEO & Founder'
        },
        {
            'email': 'carlos.rivera@inversiones.com',
            'password': 'demo123',
            'name': 'Carlos Rivera',
            'phone': '+52 81 9876 5432',
            'interests': 'Inversiones,Bienes Raíces,Finanzas',
            'company': 'Rivera Inversiones',
            'position': 'Director de Inversiones'
        },
        {
            'email': 'ana.martinez@ecommerce.mx',
            'password': 'demo123',
            'name': 'Ana Martínez',
            'phone': '+52 33 5555 1234',
            'interests': 'E-commerce,Marketing Digital,Retail',
            'company': 'MercadoLocal',
            'position': 'Directora de Operaciones'
        },
        {
            'email': 'roberto.garcia@consultoria.com',
            'password': 'demo123',
            'name': 'Roberto García',
            'phone': '+52 55 8765 4321',
            'interests': 'Consultoría,Transformación Digital,Negocios',
            'company': 'García & Asociados',
            'position': 'Socio Consultor'
        },
        {
            'email': 'lucia.fernandez@salud.mx',
            'password': 'demo123',
            'name': 'Lucía Fernández',
            'phone': '+52 55 2468 1357',
            'interests': 'Salud,Tecnología Médica,Innovación',
            'company': 'HealthTech Innovación',
            'position': 'Directora General'
        }
    ]

    user_ids = {}
    for user_data in users:
        user_id = db.create_user(
            email=user_data['email'],
            password=user_data['password'],
            name=user_data['name'],
            phone=user_data['phone'],
            interests=user_data['interests']
        )
        if user_id:
            user_ids[user_data['name']] = user_id
            print(f"   ✅ {user_data['name']} - {user_data['company']}")

    # ==================== OPORTUNIDADES ====================
    print("\n💼 Creando oportunidades...")

    today = datetime.now()

    opportunities = [
        {
            'user': 'María López',
            'title': 'Busco Socio Técnico para Plataforma Fintech',
            'description': 'Estamos desarrollando una plataforma de pagos digitales para PYMES en México. Necesitamos un CTO con experiencia en desarrollo backend (Python/Node.js), arquitectura de microservicios y sistemas de pago. Ofrecemos equity del 20% y participación en decisiones estratégicas.',
            'type': 'servicio',
            'industry': 'Fintech',
            'expiration_date': (today + timedelta(days=45)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Carlos Rivera',
            'title': 'Inversión en Proyectos Inmobiliarios Sustentables',
            'description': 'Fondo de inversión especializado en desarrollos inmobiliarios sustentables busca proyectos en etapa temprana. Ticket mínimo: $5M MXN. Sectores: vivienda vertical, co-living, espacios de trabajo híbrido. Ofrecemos mentoría estratégica y red de contactos.',
            'type': 'servicio',
            'industry': 'Inmobiliaria',
            'expiration_date': (today + timedelta(days=60)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Ana Martínez',
            'title': 'Proveedores de Logística Last-Mile para E-commerce',
            'description': 'MercadoLocal está escalando operaciones en CDMX, Guadalajara y Monterrey. Buscamos alianzas con operadores logísticos para entregas same-day. Volumen estimado: 5,000 pedidos/mes por ciudad. Contrato a 12 meses renovable.',
            'type': 'servicio',
            'industry': 'E-commerce',
            'expiration_date': (today + timedelta(days=30)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Roberto García',
            'title': 'Consultoría en Transformación Digital - Manufactura',
            'description': 'Ofrecemos servicios de consultoría especializada en transformación digital para empresas manufactureras. Experiencia implementando ERP, automatización de procesos, IoT industrial. Casos de éxito con reducción de costos del 25%. Primera sesión diagnóstico sin costo.',
            'type': 'servicio',
            'industry': 'Manufactura',
            'expiration_date': (today + timedelta(days=90)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Lucía Fernández',
            'title': 'Plataforma de Telemedicina Busca Clínicas Asociadas',
            'description': 'HealthTech Innovación conecta pacientes con especialistas mediante telemedicina. Buscamos clínicas y médicos independientes para red de prestadores. Sistema de referidos, agenda digital integrada, pago automático. Sin costos de entrada.',
            'type': 'servicio',
            'industry': 'Salud',
            'expiration_date': (today + timedelta(days=120)).strftime('%Y-%m-%d')
        },
        {
            'user': 'María López',
            'title': 'Desarrolladores Python/Django para MVP Fintech',
            'description': 'Proyecto urgente: necesitamos 2 desarrolladores Python con experiencia en Django y APIs REST para completar MVP de plataforma de préstamos P2P. Duración: 3 meses. Remoto. Posibilidad de incorporación permanente.',
            'type': 'servicio',
            'industry': 'Tecnología',
            'expiration_date': (today + timedelta(days=15)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Carlos Rivera',
            'title': 'Oportunidad: Franquicia de Cafetería Gourmet',
            'description': 'Modelo de negocio probado con 8 sucursales exitosas en zona premium. ROI: 18-24 meses. Inversión inicial: $2.5M MXN (incluye equipo, capacitación, marketing inicial). Soporte operativo continuo y marca consolidada.',
            'type': 'producto',
            'industry': 'Alimentaria',
            'expiration_date': (today + timedelta(days=20)).strftime('%Y-%m-%d')
        },
        {
            'user': 'Ana Martínez',
            'title': 'Espacios Publicitarios en Plataforma E-commerce',
            'description': 'MercadoLocal ofrece paquetes de publicidad digital con 50K impresiones mensuales garantizadas. Segmentación por ciudad, categoría de producto y perfil de comprador. Incluye reportes de performance y optimización de campañas.',
            'type': 'producto',
            'industry': 'Marketing Digital',
            'expiration_date': (today + timedelta(days=7)).strftime('%Y-%m-%d')
        }
    ]

    for opp in opportunities:
        user_id = user_ids.get(opp['user'])
        if user_id:
            opp_id = db.create_opportunity(
                user_id=user_id,
                title=opp['title'],
                description=opp['description'],
                opp_type=opp['type'],
                industry=opp['industry'],
                expiration_date=opp['expiration_date']
            )
            if opp_id:
                status_icon = "⏰" if (datetime.strptime(opp['expiration_date'], '%Y-%m-%d') - today).days < 14 else "✅"
                print(f"   {status_icon} {opp['title'][:60]}... ({opp['user']})")

    # ==================== CONEXIONES ====================
    print("\n🤝 Creando conexiones entre usuarios...")

    connections = [
        ('María López', 'Carlos Rivera', 'Nos conocimos en el evento de Startups MX 2024'),
        ('María López', 'Roberto García', 'Ex-compañeros en aceleradora de negocios'),
        ('Ana Martínez', 'Carlos Rivera', 'Contacto de red de inversionistas'),
        ('Lucía Fernández', 'María López', 'Alianza estratégica HealthTech-Fintech'),
        ('Roberto García', 'Ana Martínez', 'Cliente de consultoría exitosa')
    ]

    for user1_name, user2_name, message in connections:
        user1_id = user_ids.get(user1_name)
        user2_id = user_ids.get(user2_name)
        if user1_id and user2_id:
            db.create_connection(
                user_id=user1_id,
                connected_user_id=user2_id,
                message=message,
                status='accepted',
                accepted_at=datetime.now().isoformat()
            )
            print(f"   ✅ {user1_name} ↔ {user2_name}")

    print("\n" + "="*60)
    print("✨ Base de datos poblada exitosamente!")
    print("="*60)
    print("\n📊 Resumen:")
    print(f"   👥 Usuarios creados: {len(user_ids)}")
    print(f"   💼 Oportunidades creadas: {len(opportunities)}")
    print(f"   🤝 Conexiones creadas: {len(connections)}")
    print("\n🔐 Credenciales para login (todas usan password: demo123):")
    for user_data in users:
        print(f"   • {user_data['email']}")
    print("\n🌐 Accede a: https://goldencoyotes.onrender.com")
    print("\n")

if __name__ == "__main__":
    try:
        seed_database()
    except Exception as e:
        print(f"\n❌ Error al poblar base de datos: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
