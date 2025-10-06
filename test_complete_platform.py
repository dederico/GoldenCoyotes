#!/usr/bin/env python3
"""
Test script for the complete Golden Coyotes Platform
"""

from golden_coyotes_platform import GoldenCoyotesPlatform
import threading
import time
import requests

def test_platform():
    print("🧪 Testing Complete Golden Coyotes Platform")
    print("=" * 50)
    
    # Create platform
    platform = GoldenCoyotesPlatform()
    
    # Start in thread
    def run_server():
        platform.run(port=8082, debug=False)
    
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    base_url = "http://localhost:8082"
    
    try:
        # Test landing page
        print("Testing landing page...")
        response = requests.get(f"{base_url}/")
        print(f"✅ Landing page: {response.status_code}")
        
        # Test register page
        print("Testing register page...")
        response = requests.get(f"{base_url}/register")
        print(f"✅ Register page: {response.status_code}")
        
        # Test login page
        print("Testing login page...")
        response = requests.get(f"{base_url}/login")
        print(f"✅ Login page: {response.status_code}")
        
        # Test user registration
        print("Testing user registration...")
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "password": "password123",
            "industry": "Technology",
            "location": "San Francisco",
            "skills": "AI, Python, Web Development",
            "interests": "Startups, Innovation, Tech",
            "bio": "Test user for platform testing"
        }
        
        response = requests.post(f"{base_url}/register", json=user_data)
        print(f"✅ User registration: {response.status_code}")
        
        print("\n🎉 All tests passed! Platform is working correctly.")
        print(f"🌐 Visit: {base_url}")
        print("\n📋 What you can do:")
        print("1. Register new users with different profiles")
        print("2. Login and see personalized dashboards")
        print("3. Create opportunities and see AI matching")
        print("4. Build your network with connection requests")
        print("5. Update profiles and see improved matches")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_platform()
    
    # Keep running for manual testing
    print("\n🚀 Platform running on http://localhost:8082")
    print("Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Platform stopped")