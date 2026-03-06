#!/usr/bin/env python3
"""
Tests para Retroalimentación 4 - Filtro de Red y Gestión de Conexiones

Valida:
1. Filtro de oportunidades por red de contactos (network_only)
2. Endpoints REST para gestión de conexiones
3. Métodos de base de datos para conexiones transparentes
4. Vista de Mis Contactos
"""

import unittest
import sys
import os
import json
from datetime import datetime, timedelta

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_setup import DatabaseManager


class TestRetro4NetworkFiltering(unittest.TestCase):
    """Tests para filtro de oportunidades por red de contactos"""

    def setUp(self):
        """Setup test database"""
        # Usar base de datos de test
        self.db = DatabaseManager("test_retro4.db")

        # Crear usuarios de prueba
        self.user1_id = self.db.create_user(
            email="user1@test.com",
            password="test123",
            name="Usuario 1",
            industry="Tecnología",
            company="Tech Co"
        )

        self.user2_id = self.db.create_user(
            email="user2@test.com",
            password="test123",
            name="Usuario 2",
            industry="Fintech",
            company="Finance Inc"
        )

        self.user3_id = self.db.create_user(
            email="user3@test.com",
            password="test123",
            name="Usuario 3 (No conectado)",
            industry="E-commerce",
            company="Shop Co"
        )

        # Crear conexión entre user1 y user2 (aceptada)
        conn_id = self.db.create_connection(
            self.user1_id,
            self.user2_id,
            message="Conectemos",
            status="accepted",
            accepted_at=datetime.now().isoformat()
        )

        self.assertIsNotNone(conn_id, "La conexión debería haberse creado")

    def tearDown(self):
        """Cleanup test database"""
        if os.path.exists("test_retro4.db"):
            os.remove("test_retro4.db")

    def test_network_only_filter_shows_only_network_opportunities(self):
        """Test que network_only=True filtra solo oportunidades de mi red"""

        # Crear oportunidad de user2 (está conectado con user1)
        opp1_id = self.db.create_opportunity(
            user_id=self.user2_id,
            title="Oportunidad de mi red",
            description="Esta oportunidad debería aparecer",
            opp_type="producto",
            industry="Fintech"
        )

        # Crear oportunidad de user3 (NO está conectado con user1)
        opp2_id = self.db.create_opportunity(
            user_id=self.user3_id,
            title="Oportunidad fuera de mi red",
            description="Esta NO debería aparecer",
            opp_type="servicio",
            industry="E-commerce"
        )

        # Obtener oportunidades de la red de user1
        network_opps = self.db.get_opportunities(
            network_only=True,
            requesting_user_id=self.user1_id,
            limit=50
        )

        # Verificar que solo aparece la oportunidad de user2
        opp_ids = [opp['id'] for opp in network_opps]

        self.assertIn(opp1_id, opp_ids, "Debería incluir oportunidad de contacto conectado")
        self.assertNotIn(opp2_id, opp_ids, "NO debería incluir oportunidad de usuario no conectado")

    def test_get_opportunities_without_filter_shows_all(self):
        """Test que sin network_only muestra todas las oportunidades"""

        # Crear oportunidades de diferentes usuarios
        opp1 = self.db.create_opportunity(
            user_id=self.user2_id,
            title="Oportunidad 1",
            description="Test",
            opp_type="producto"
        )

        opp2 = self.db.create_opportunity(
            user_id=self.user3_id,
            title="Oportunidad 2",
            description="Test",
            opp_type="servicio"
        )

        # Sin filtro de red (admin view)
        all_opps = self.db.get_opportunities(limit=50)

        self.assertGreaterEqual(len(all_opps), 2, "Debería mostrar todas las oportunidades")

    def test_my_own_opportunities_excluded_from_network_filter(self):
        """Test que mis propias oportunidades no aparecen en el filtro de red"""

        # user1 crea una oportunidad
        my_opp = self.db.create_opportunity(
            user_id=self.user1_id,
            title="Mi propia oportunidad",
            description="No debería aparecer en network_only",
            opp_type="producto"
        )

        # Obtener oportunidades de red de user1
        network_opps = self.db.get_opportunities(
            network_only=True,
            requesting_user_id=self.user1_id,
            limit=50
        )

        opp_ids = [opp['id'] for opp in network_opps]

        self.assertNotIn(my_opp, opp_ids, "Mis propias oportunidades NO deberían aparecer en network_only")


class TestRetro4ConnectionManagement(unittest.TestCase):
    """Tests para gestión de conexiones transparentes"""

    def setUp(self):
        """Setup test database"""
        self.db = DatabaseManager("test_connections.db")

        # Crear usuarios
        self.user1_id = self.db.create_user(
            email="alice@test.com",
            password="test123",
            name="Alice"
        )

        self.user2_id = self.db.create_user(
            email="bob@test.com",
            password="test123",
            name="Bob"
        )

    def tearDown(self):
        """Cleanup"""
        if os.path.exists("test_connections.db"):
            os.remove("test_connections.db")

    def test_create_connection_request(self):
        """Test crear solicitud de conexión"""

        conn_id = self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            message="Hola Bob, conectemos",
            status="pending"
        )

        self.assertIsNotNone(conn_id, "Debería crear la conexión")

    def test_get_pending_connection_requests(self):
        """Test obtener solicitudes pendientes"""

        # user1 envía solicitud a user2
        self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            message="Conectemos",
            status="pending"
        )

        # user2 obtiene sus solicitudes pendientes
        pending = self.db.get_pending_connection_requests(self.user2_id)

        self.assertEqual(len(pending), 1, "Debería tener 1 solicitud pendiente")
        self.assertEqual(pending[0]['requester_name'], "Alice")

    def test_accept_connection(self):
        """Test aceptar solicitud de conexión"""

        # Crear solicitud
        conn_id = self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            message="Conectemos",
            status="pending"
        )

        # user2 acepta la solicitud
        success = self.db.accept_connection(conn_id, self.user2_id)

        self.assertTrue(success, "Debería aceptar la conexión")

        # Verificar que ahora están conectados
        is_connected = self.db.is_connected(self.user1_id, self.user2_id)
        self.assertTrue(is_connected, "Deberían estar conectados después de aceptar")

    def test_reject_connection(self):
        """Test rechazar solicitud de conexión"""

        # Crear solicitud
        conn_id = self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            message="Conectemos",
            status="pending"
        )

        # user2 rechaza la solicitud
        success = self.db.reject_connection(conn_id, self.user2_id)

        self.assertTrue(success, "Debería rechazar la conexión")

        # Verificar que NO están conectados
        is_connected = self.db.is_connected(self.user1_id, self.user2_id)
        self.assertFalse(is_connected, "NO deberían estar conectados después de rechazar")

    def test_is_connected_bidirectional(self):
        """Test que is_connected funciona en ambas direcciones"""

        # Crear y aceptar conexión
        conn_id = self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            status="accepted",
            accepted_at=datetime.now().isoformat()
        )

        # Verificar en ambas direcciones
        self.assertTrue(self.db.is_connected(self.user1_id, self.user2_id))
        self.assertTrue(self.db.is_connected(self.user2_id, self.user1_id))

    def test_get_user_connections_by_status(self):
        """Test obtener conexiones filtradas por estado"""

        user3_id = self.db.create_user(
            email="charlie@test.com",
            password="test123",
            name="Charlie"
        )

        # Crear conexión aceptada
        self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            status="accepted",
            accepted_at=datetime.now().isoformat()
        )

        # Crear conexión pendiente
        self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=user3_id,
            status="pending"
        )

        # Obtener solo aceptadas
        accepted = self.db.get_user_connections(self.user1_id, status='accepted')
        self.assertEqual(len(accepted), 1, "Debería tener 1 conexión aceptada")

        # Obtener solo pendientes
        pending = self.db.get_user_connections(self.user1_id, status='pending')
        self.assertEqual(len(pending), 1, "Debería tener 1 conexión pendiente")

    def test_cannot_accept_others_connection_requests(self):
        """Test que no puedo aceptar solicitudes que no son para mí"""

        user3_id = self.db.create_user(
            email="eve@test.com",
            password="test123",
            name="Eve (Atacante)"
        )

        # user1 envía solicitud a user2
        conn_id = self.db.create_connection(
            user_id=self.user1_id,
            target_user_id=self.user2_id,
            status="pending"
        )

        # user3 intenta aceptar una conexión que no le pertenece
        success = self.db.accept_connection(conn_id, user3_id)

        self.assertFalse(success, "NO debería poder aceptar conexión de otros")


class TestRetro4Integration(unittest.TestCase):
    """Tests de integración para flujo completo"""

    def setUp(self):
        """Setup"""
        self.db = DatabaseManager("test_integration.db")

        # Crear red de 3 usuarios
        self.alice_id = self.db.create_user(
            email="alice@test.com",
            password="test123",
            name="Alice",
            industry="Tecnología"
        )

        self.bob_id = self.db.create_user(
            email="bob@test.com",
            password="test123",
            name="Bob",
            industry="Fintech"
        )

        self.charlie_id = self.db.create_user(
            email="charlie@test.com",
            password="test123",
            name="Charlie",
            industry="E-commerce"
        )

        # Conectar Alice con Bob (aceptada)
        self.db.create_connection(
            self.alice_id,
            self.bob_id,
            status="accepted",
            accepted_at=datetime.now().isoformat()
        )

        # Charlie NO está conectado con nadie

    def tearDown(self):
        """Cleanup"""
        if os.path.exists("test_integration.db"):
            os.remove("test_integration.db")

    def test_complete_opportunity_visibility_workflow(self):
        """Test flujo completo: solo veo oportunidades de mi red"""

        # Bob crea una oportunidad (Alice debería verla)
        bob_opp = self.db.create_opportunity(
            user_id=self.bob_id,
            title="Oportunidad de Bob",
            description="Alice debería ver esto",
            opp_type="producto",
            industry="Fintech"
        )

        # Charlie crea una oportunidad (Alice NO debería verla)
        charlie_opp = self.db.create_opportunity(
            user_id=self.charlie_id,
            title="Oportunidad de Charlie",
            description="Alice NO debería ver esto",
            opp_type="servicio",
            industry="E-commerce"
        )

        # Alice busca oportunidades en su red
        alice_network_opps = self.db.get_opportunities(
            network_only=True,
            requesting_user_id=self.alice_id,
            limit=50
        )

        opp_titles = [opp['title'] for opp in alice_network_opps]

        self.assertIn("Oportunidad de Bob", opp_titles,
                      "Alice debería ver oportunidad de Bob (contacto conectado)")
        self.assertNotIn("Oportunidad de Charlie", opp_titles,
                         "Alice NO debería ver oportunidad de Charlie (no conectado)")

    def test_connection_request_workflow(self):
        """Test flujo completo de solicitud de conexión"""

        # 1. Charlie envía solicitud a Alice
        conn_id = self.db.create_connection(
            user_id=self.charlie_id,
            target_user_id=self.alice_id,
            message="Hola Alice, conectemos",
            status="pending"
        )

        # 2. Alice ve sus solicitudes pendientes
        alice_pending = self.db.get_pending_connection_requests(self.alice_id)
        self.assertEqual(len(alice_pending), 1)
        self.assertEqual(alice_pending[0]['requester_name'], "Charlie")

        # 3. Antes de aceptar, NO están conectados
        self.assertFalse(self.db.is_connected(self.alice_id, self.charlie_id))

        # 4. Alice acepta la solicitud
        success = self.db.accept_connection(conn_id, self.alice_id)
        self.assertTrue(success)

        # 5. Ahora SÍ están conectados
        self.assertTrue(self.db.is_connected(self.alice_id, self.charlie_id))

        # 6. Charlie crea una oportunidad
        charlie_new_opp = self.db.create_opportunity(
            user_id=self.charlie_id,
            title="Nueva oportunidad de Charlie",
            description="Ahora Alice SÍ debería verla",
            opp_type="producto"
        )

        # 7. Ahora Alice SÍ ve la oportunidad de Charlie
        alice_opps = self.db.get_opportunities(
            network_only=True,
            requesting_user_id=self.alice_id
        )

        opp_ids = [opp['id'] for opp in alice_opps]
        self.assertIn(charlie_new_opp, opp_ids,
                      "Alice debería ver oportunidad de Charlie después de conectarse")


def run_tests():
    """Ejecutar todos los tests"""
    print("=" * 70)
    print("SUITE DE TESTS - RETROALIMENTACIÓN 4")
    print("Filtro de Red y Gestión Transparente de Conexiones")
    print("=" * 70)
    print()

    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Agregar tests
    suite.addTests(loader.loadTestsFromTestCase(TestRetro4NetworkFiltering))
    suite.addTests(loader.loadTestsFromTestCase(TestRetro4ConnectionManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestRetro4Integration))

    # Ejecutar
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Fallidos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
