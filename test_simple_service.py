#!/usr/bin/env python3
"""
Simple test to start just the Flask service without full initialization
"""

from flask import Flask
from api.main_api import create_app

def test_simple_flask():
    """Test Flask app creation with minimal setup"""
    print("🧪 Testing Simple Flask Service")
    print("=" * 50)
    
    try:
        # Minimal config and components
        class SimpleConfig:
            def __init__(self):
                self.debug = True
                
        class SimpleComponents:
            def __init__(self):
                pass
        
        config = SimpleConfig()
        components = SimpleComponents()
        
        # Create Flask app
        app = create_app(config, components)
        
        print("✅ Flask app created successfully")
        
        # Start development server
        print("🚀 Starting Flask development server on port 8080...")
        print("   Access at: http://localhost:8080")
        print("   Health check: http://localhost:8080/health")
        print("   Dashboard: http://localhost:8080/dashboard") 
        print("   API Status: http://localhost:8080/api/intelligence/status")
        print("\nPress Ctrl+C to stop the server")
        
        app.run(host='0.0.0.0', port=8080, debug=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_flask()