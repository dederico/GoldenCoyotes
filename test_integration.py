#!/usr/bin/env python3
"""
Integration tests for Business Dealer Intelligence Web Application
Tests all API endpoints and UI functionality
"""

import requests
import json
import time
import uuid
import subprocess
import signal
import os
from datetime import datetime

class WebAppIntegrationTest:
    """Integration test suite for the web application"""
    
    def __init__(self, base_url="http://localhost:8082"):
        self.base_url = base_url
        self.app_process = None
        
    def start_app(self):
        """Start the web application for testing"""
        print("🚀 Starting web application for integration testing...")
        
        # Start app in background
        self.app_process = subprocess.Popen([
            'python3', 'web_app.py', '--port', '8082'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for app to start
        time.sleep(3)
        
        # Check if app is running
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ Web application started successfully")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("❌ Failed to start web application")
        return False
    
    def stop_app(self):
        """Stop the web application"""
        if self.app_process:
            self.app_process.terminate()
            self.app_process.wait()
            print("🛑 Web application stopped")
    
    def test_homepage(self):
        """Test main dashboard page"""
        print("\n📊 Testing Homepage...")
        try:
            response = requests.get(f"{self.base_url}/")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert "Business Dealer Intelligence Dashboard" in response.text
            print("✅ Homepage loads correctly")
            return True
        except Exception as e:
            print(f"❌ Homepage test failed: {e}")
            return False
    
    def test_pages(self):
        """Test all main pages"""
        print("\n📄 Testing All Pages...")
        pages = [
            ("/users", "User Management"),
            ("/opportunities", "Business Opportunities"),
            ("/analytics", "Analytics Dashboard"),
            ("/notifications", "Notifications"),
            ("/settings", "Settings")
        ]
        
        success_count = 0
        for path, expected_content in pages:
            try:
                response = requests.get(f"{self.base_url}{path}")
                assert response.status_code == 200
                assert expected_content in response.text
                print(f"✅ {path} page loads correctly")
                success_count += 1
            except Exception as e:
                print(f"❌ {path} page failed: {e}")
        
        return success_count == len(pages)
    
    def test_user_creation(self):
        """Test user creation API"""
        print("\n👤 Testing User Creation API...")
        
        user_data = {
            "name": f"Test User {uuid.uuid4().hex[:8]}",
            "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
            "industry": "Technology",
            "location": "San Francisco",
            "interests": "AI, Testing, Automation"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/users",
                json=user_data,
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] == True
            assert 'user_id' in result
            
            print(f"✅ User created successfully: {result['user_id']}")
            return result['user_id']
        except Exception as e:
            print(f"❌ User creation failed: {e}")
            return None
    
    def test_opportunity_creation(self):
        """Test opportunity creation API"""
        print("\n💼 Testing Opportunity Creation API...")
        
        opp_data = {
            "title": f"Test Opportunity {uuid.uuid4().hex[:8]}",
            "description": "A test opportunity for integration testing",
            "type": "partnership",
            "industry": "Technology",
            "budget": "$50,000-100,000",
            "location": "San Francisco",
            "deadline": "2025-08-15"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/opportunities",
                json=opp_data,
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] == True
            assert 'opportunity_id' in result
            
            print(f"✅ Opportunity created successfully: {result['opportunity_id']}")
            return result['opportunity_id']
        except Exception as e:
            print(f"❌ Opportunity creation failed: {e}")
            return None
    
    def test_metrics_api(self):
        """Test metrics API"""
        print("\n📈 Testing Metrics API...")
        
        try:
            response = requests.get(f"{self.base_url}/api/metrics")
            assert response.status_code == 200
            
            metrics = response.json()
            required_fields = [
                'total_users', 'total_opportunities', 'total_interactions',
                'engagement_rate', 'recent_activity', 'active_opportunities'
            ]
            
            for field in required_fields:
                assert field in metrics, f"Missing field: {field}"
            
            print(f"✅ Metrics API working: {metrics}")
            return True
        except Exception as e:
            print(f"❌ Metrics API failed: {e}")
            return False
    
    def test_recommendations_api(self):
        """Test recommendations API"""
        print("\n🎯 Testing Recommendations API...")
        
        try:
            response = requests.get(f"{self.base_url}/api/recommendations/user1")
            assert response.status_code == 200
            
            result = response.json()
            assert 'recommendations' in result
            assert 'user_id' in result
            assert result['user_id'] == 'user1'
            
            print(f"✅ Recommendations API working: {len(result['recommendations'])} recommendations")
            return True
        except Exception as e:
            print(f"❌ Recommendations API failed: {e}")
            return False
    
    def test_interaction_recording(self):
        """Test interaction recording API"""
        print("\n🔄 Testing Interaction Recording API...")
        
        interaction_data = {
            "user_id": "user1",
            "opportunity_id": "opp1",
            "type": "view",
            "metadata": {"test": True}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/interactions",
                json=interaction_data,
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] == True
            assert 'interaction_id' in result
            
            print(f"✅ Interaction recorded successfully: {result['interaction_id']}")
            return True
        except Exception as e:
            print(f"❌ Interaction recording failed: {e}")
            return False
    
    def test_user_profile_api(self):
        """Test user profile API"""
        print("\n👤 Testing User Profile API...")
        
        try:
            response = requests.get(f"{self.base_url}/api/user-profile/user1")
            assert response.status_code == 200
            
            result = response.json()
            assert 'user' in result
            assert 'interactions' in result
            assert 'interaction_count' in result
            
            print(f"✅ User profile API working: {result['interaction_count']} interactions")
            return True
        except Exception as e:
            print(f"❌ User profile API failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete integration test suite"""
        print("🧪 Business Dealer Intelligence - Integration Test Suite")
        print("=" * 60)
        
        # Start app
        if not self.start_app():
            return False
        
        test_results = []
        
        try:
            # Run all tests
            test_results.append(("Homepage", self.test_homepage()))
            test_results.append(("All Pages", self.test_pages()))
            test_results.append(("User Creation", self.test_user_creation() is not None))
            test_results.append(("Opportunity Creation", self.test_opportunity_creation() is not None))
            test_results.append(("Metrics API", self.test_metrics_api()))
            test_results.append(("Recommendations API", self.test_recommendations_api()))
            test_results.append(("Interaction Recording", self.test_interaction_recording()))
            test_results.append(("User Profile API", self.test_user_profile_api()))
            
        finally:
            self.stop_app()
        
        # Results summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<25} {status}")
            if result:
                passed += 1
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed")
            return False

def main():
    """Run integration tests"""
    tester = WebAppIntegrationTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()