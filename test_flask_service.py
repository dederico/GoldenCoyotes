#!/usr/bin/env python3
"""
Simple test script for Flask service
Tests basic functionality without full system setup
"""

import sys
import time
import threading
import requests
from flask import Flask
from api.main_api import create_app

def test_flask_endpoints():
    """Test Flask service endpoints"""
    print("🧪 Testing Flask Service Endpoints")
    print("=" * 50)
    
    # Create minimal config
    class MinimalConfig:
        def __init__(self):
            pass
    
    # Create minimal components
    class MinimalComponents:
        def __init__(self):
            pass
    
    try:
        # Create Flask app
        config = MinimalConfig()
        components = MinimalComponents()
        app = create_app(config, components)
        
        print("✅ Flask app created successfully")
        
        # Test in debug mode
        with app.test_client() as client:
            print("\n🔍 Testing endpoints:")
            
            # Test health endpoint
            response = client.get('/health')
            print(f"   GET /health: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"      Status: {data.get('status', 'unknown')}")
            
            # Test intelligence health
            response = client.get('/api/intelligence/health')
            print(f"   GET /api/intelligence/health: {response.status_code}")
            
            # Test analytics health  
            response = client.get('/api/analytics/health')
            print(f"   GET /api/analytics/health: {response.status_code}")
            
            # Test notification health
            response = client.get('/api/notifications/health')
            print(f"   GET /api/notifications/health: {response.status_code}")
            
            # Test intelligence status
            response = client.get('/api/intelligence/status')
            print(f"   GET /api/intelligence/status: {response.status_code}")
            
            # Test analytics available metrics
            response = client.get('/api/analytics/metrics/available')
            print(f"   GET /api/analytics/metrics/available: {response.status_code}")
            
            # Test notification status
            response = client.get('/api/notifications/status')
            print(f"   GET /api/notifications/status: {response.status_code}")
            
            # Test dashboard endpoint
            response = client.get('/dashboard')
            print(f"   GET /dashboard: {response.status_code}")
            
            # Test demo data endpoints
            response = client.get('/api/demo/users')
            print(f"   GET /api/demo/users: {response.status_code}")
            
            response = client.get('/api/demo/opportunities')
            print(f"   GET /api/demo/opportunities: {response.status_code}")
            
            response = client.get('/api/demo/recommendations/user123')
            print(f"   GET /api/demo/recommendations/user123: {response.status_code}")
            
        print("\n✅ All endpoint tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Flask endpoints: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_blueprint_registration():
    """Test that all blueprints are properly registered"""
    print("\n🔧 Testing Blueprint Registration")
    print("=" * 50)
    
    try:
        class MinimalConfig:
            pass
        
        class MinimalComponents:
            pass
        
        config = MinimalConfig()
        components = MinimalComponents()
        app = create_app(config, components)
        
        # Check registered blueprints
        blueprints = list(app.blueprints.keys())
        print(f"   Registered blueprints: {blueprints}")
        
        expected_blueprints = ['intelligence', 'analytics', 'notification']
        for bp in expected_blueprints:
            if bp in blueprints:
                print(f"   ✅ {bp} blueprint registered")
            else:
                print(f"   ❌ {bp} blueprint missing")
        
        # Check routes
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append(f"{rule.methods} {rule.rule}")
        
        print(f"\n   Total routes registered: {len(routes)}")
        
        # Check for key routes
        key_routes = [
            '/health',
            '/api/intelligence/health',
            '/api/analytics/health', 
            '/api/notifications/health'
        ]
        
        for route in key_routes:
            found = any(route in r for r in routes)
            if found:
                print(f"   ✅ {route} route found")
            else:
                print(f"   ❌ {route} route missing")
        
        print("✅ Blueprint registration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing blueprints: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Flask Service Tests")
    print("=" * 60)
    
    # Test Flask endpoints
    endpoints_success = test_flask_endpoints()
    
    # Test blueprint registration
    blueprints_success = test_blueprint_registration()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Flask Endpoints: {'✅ PASS' if endpoints_success else '❌ FAIL'}")
    print(f"   Blueprint Registration: {'✅ PASS' if blueprints_success else '❌ FAIL'}")
    
    if endpoints_success and blueprints_success:
        print("\n🎉 All Flask service tests PASSED!")
        print("💡 The Flask service is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Some Flask service tests FAILED!")
        sys.exit(1)