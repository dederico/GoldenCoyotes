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
        },
        # Contactos adicionales para María (para demostrar scroll)
        {
            'email': 'pedro.sanchez@marketing.mx',
            'password': 'demo123',
            'name': 'Pedro Sánchez',
            'phone': '+52 55 1111 2222',
            'interests': 'Marketing,Publicidad,Redes Sociales',
            'company': 'Digital Marketing Pro',
            'position': 'Director Creativo'
        },
        {
            'email': 'sofia.torres@design.mx',
            'password': 'demo123',
            'name': 'Sofía Torres',
            'phone': '+52 55 3333 4444',
            'interests': 'Diseño,UX/UI,Producto',
            'company': 'Design Studio MX',
            'position': 'Lead Designer'
        },
        {
            'email': 'miguel.ramirez@legal.com',
            'password': 'demo123',
            'name': 'Miguel Ramírez',
            'phone': '+52 55 5555 6666',
            'interests': 'Legal,Corporativo,Startups',
            'company': 'Ramírez & Partners',
            'position': 'Abogado Corporativo'
        },
        {
            'email': 'isabel.morales@hr.mx',
            'password': 'demo123',
            'name': 'Isabel Morales',
            'phone': '+52 55 7777 8888',
            'interests': 'Recursos Humanos,Talento,Cultura Organizacional',
            'company': 'Talent Solutions',
            'position': 'HR Director'
        },
        {
            'email': 'jorge.castro@ventas.com',
            'password': 'demo123',
            'name': 'Jorge Castro',
            'phone': '+52 55 9999 0000',
            'interests': 'Ventas,B2B,Desarrollo de Negocios',
            'company': 'Sales Excellence',
            'position': 'VP de Ventas'
        },
        {
            'email': 'carmen.diaz@export.mx',
            'password': 'demo123',
            'name': 'Carmen Díaz',
            'phone': '+52 55 2222 3333',
            'interests': 'Comercio Internacional,Logística,Exportación',
            'company': 'Global Trade MX',
            'position': 'Gerente de Exportaciones'
        },
        {
            'email': 'ricardo.flores@tech.mx',
            'password': 'demo123',
            'name': 'Ricardo Flores',
            'phone': '+52 55 4444 5555',
            'interests': 'Desarrollo de Software,Cloud,DevOps',
            'company': 'CloudTech Solutions',
            'position': 'CTO'
        },
        {
            'email': 'elena.vargas@contenido.mx',
            'password': 'demo123',
            'name': 'Elena Vargas',
            'phone': '+52 55 6666 7777',
            'interests': 'Contenido,Social Media,Copywriting',
            'company': 'Content Creators MX',
            'position': 'Content Strategist'
        },
        {
            'email': 'fernando.ruiz@finanzas.com',
            'password': 'demo123',
            'name': 'Fernando Ruiz',
            'phone': '+52 55 8888 9999',
            'interests': 'Finanzas,Contabilidad,CFO Services',
            'company': 'Financial Advisory',
            'position': 'CFO'
        },
        {
            'email': 'laura.mendez@producto.mx',
            'password': 'demo123',
            'name': 'Laura Méndez',
            'phone': '+52 55 1234 5670',
            'interests': 'Product Management,Agile,Innovación',
            'company': 'Product Lab',
            'position': 'Product Manager'
        },
        {
            'email': 'daniel.ortiz@operaciones.com',
            'password': 'demo123',
            'name': 'Daniel Ortiz',
            'phone': '+52 55 2345 6781',
            'interests': 'Operaciones,Supply Chain,Procesos',
            'company': 'Operations Excellence',
            'position': 'Director de Operaciones'
        },
        {
            'email': 'patricia.gomez@datos.mx',
            'password': 'demo123',
            'name': 'Patricia Gómez',
            'phone': '+52 55 3456 7892',
            'interests': 'Data Science,Analytics,BI',
            'company': 'Data Insights MX',
            'position': 'Data Scientist'
        },
        {
            'email': 'alberto.herrera@innovacion.com',
            'password': 'demo123',
            'name': 'Alberto Herrera',
            'phone': '+52 55 4567 8903',
            'interests': 'Innovación,R&D,Tecnología',
            'company': 'Innovation Hub',
            'position': 'Chief Innovation Officer'
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
            'expiration_date': (today + timedelta(days=45)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1563986768609-322da13575f3?w=800'
        },
        {
            'user': 'Carlos Rivera',
            'title': 'Inversión en Proyectos Inmobiliarios Sustentables',
            'description': 'Fondo de inversión especializado en desarrollos inmobiliarios sustentables busca proyectos en etapa temprana. Ticket mínimo: $5M MXN. Sectores: vivienda vertical, co-living, espacios de trabajo híbrido. Ofrecemos mentoría estratégica y red de contactos.',
            'type': 'servicio',
            'industry': 'Inmobiliaria',
            'expiration_date': (today + timedelta(days=60)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=800'
        },
        {
            'user': 'Ana Martínez',
            'title': 'Proveedores de Logística Last-Mile para E-commerce',
            'description': 'MercadoLocal está escalando operaciones en CDMX, Guadalajara y Monterrey. Buscamos alianzas con operadores logísticos para entregas same-day. Volumen estimado: 5,000 pedidos/mes por ciudad. Contrato a 12 meses renovable.',
            'type': 'servicio',
            'industry': 'E-commerce',
            'expiration_date': (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1566576721346-d4a3b4eaeb55?w=800'
        },
        {
            'user': 'Roberto García',
            'title': 'Consultoría en Transformación Digital - Manufactura',
            'description': 'Ofrecemos servicios de consultoría especializada en transformación digital para empresas manufactureras. Experiencia implementando ERP, automatización de procesos, IoT industrial. Casos de éxito con reducción de costos del 25%. Primera sesión diagnóstico sin costo.',
            'type': 'servicio',
            'industry': 'Manufactura',
            'expiration_date': (today + timedelta(days=90)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=800'
        },
        {
            'user': 'Lucía Fernández',
            'title': 'Plataforma de Telemedicina Busca Clínicas Asociadas',
            'description': 'HealthTech Innovación conecta pacientes con especialistas mediante telemedicina. Buscamos clínicas y médicos independientes para red de prestadores. Sistema de referidos, agenda digital integrada, pago automático. Sin costos de entrada.',
            'type': 'servicio',
            'industry': 'Salud',
            'expiration_date': (today + timedelta(days=120)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=800'
        },
        {
            'user': 'María López',
            'title': 'Desarrolladores Python/Django para MVP Fintech',
            'description': 'Proyecto urgente: necesitamos 2 desarrolladores Python con experiencia en Django y APIs REST para completar MVP de plataforma de préstamos P2P. Duración: 3 meses. Remoto. Posibilidad de incorporación permanente.',
            'type': 'servicio',
            'industry': 'Tecnología',
            'expiration_date': (today + timedelta(days=15)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=800'
        },
        {
            'user': 'Carlos Rivera',
            'title': 'Oportunidad: Franquicia de Cafetería Gourmet',
            'description': 'Modelo de negocio probado con 8 sucursales exitosas en zona premium. ROI: 18-24 meses. Inversión inicial: $2.5M MXN (incluye equipo, capacitación, marketing inicial). Soporte operativo continuo y marca consolidada.',
            'type': 'producto',
            'industry': 'Alimentaria',
            'expiration_date': (today + timedelta(days=20)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1501339847302-ac426a4a7cbb?w=800'
        },
        {
            'user': 'Ana Martínez',
            'title': 'Espacios Publicitarios en Plataforma E-commerce',
            'description': 'MercadoLocal ofrece paquetes de publicidad digital con 50K impresiones mensuales garantizadas. Segmentación por ciudad, categoría de producto y perfil de comprador. Incluye reportes de performance y optimización de campañas.',
            'type': 'producto',
            'industry': 'Marketing Digital',
            'expiration_date': (today + timedelta(days=7)).strftime('%Y-%m-%d'),
            'image_url': 'https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=800'
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
                expiration_date=opp['expiration_date'],
                image_url=opp.get('image_url')
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
        ('Roberto García', 'Ana Martínez', 'Cliente de consultoría exitosa'),
        # Conexiones adicionales de María (para demostrar scroll)
        ('María López', 'Pedro Sánchez', 'Colaboramos en campaña de marketing digital'),
        ('María López', 'Sofía Torres', 'Diseñó la interfaz de nuestra app'),
        ('María López', 'Miguel Ramírez', 'Nos asesoró en temas legales de startup'),
        ('María López', 'Isabel Morales', 'Red de reclutamiento de talento tech'),
        ('María López', 'Jorge Castro', 'Contacto de desarrollo de negocios'),
        ('María López', 'Carmen Díaz', 'Nos ayudó con exportación de servicios'),
        ('María López', 'Ricardo Flores', 'CTO de empresa amiga'),
        ('María López', 'Elena Vargas', 'Maneja nuestras redes sociales'),
        ('María López', 'Fernando Ruiz', 'Nos apoyó en planeación financiera'),
        ('María López', 'Laura Méndez', 'Ex-compañera de producto'),
        ('María López', 'Daniel Ortiz', 'Consultoría en optimización de procesos'),
        ('María López', 'Patricia Gómez', 'Implementó analytics en nuestra plataforma'),
        ('María López', 'Alberto Herrera', 'Co-fundador de hub de innovación')
    ]

    for user1_name, user2_name, message in connections:
        user1_id = user_ids.get(user1_name)
        user2_id = user_ids.get(user2_name)
        if user1_id and user2_id:
            db.create_connection(
                user_id=user1_id,
                target_user_id=user2_id,
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
