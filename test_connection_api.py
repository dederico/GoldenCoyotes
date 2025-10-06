#!/usr/bin/env python3
"""
Test Connection API - Simulate browser connection request
"""

import requests
import json

def test_connection_with_session():
    """Test connection API with proper session"""
    session = requests.Session()
    
    # Step 1: Login first
    print("🔑 Logging in...")
    login_response = session.post('http://localhost:8080/login', 
                                json={
                                    'email': 'testuser1@example.com',
                                    'password': 'password123'
                                })
    
    print(f"Login status: {login_response.status_code}")
    if login_response.status_code == 200:
        login_result = login_response.json()
        print(f"Login result: {login_result}")
        
        if login_result.get('success'):
            print("✅ Login successful")
            
            # Step 2: Try to connect
            print("\n🤝 Attempting connection...")
            connect_response = session.post('http://localhost:8080/api/connect',
                                          json={
                                              'target_user_id': 'd3d3b2e3-342c-4376-8192-52fe2101c565',
                                              'message': 'API test connection to new user'
                                          })
            
            print(f"Connect status: {connect_response.status_code}")
            print(f"Connect response: {connect_response.text}")
            
            if connect_response.status_code == 200:
                result = connect_response.json()
                print(f"✅ Connection result: {result}")
            else:
                print(f"❌ Connection failed: {connect_response.text}")
        else:
            print("❌ Login failed")
    else:
        print(f"❌ Login request failed: {login_response.text}")

if __name__ == "__main__":
    test_connection_with_session()