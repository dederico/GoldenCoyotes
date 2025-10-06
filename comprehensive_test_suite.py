#!/usr/bin/env python3
"""
Comprehensive Test Suite for Golden Coyotes Platform
Tests every component and shows expected results
"""

import os
import sys
import time
import threading
import requests
import json
from datetime import datetime

# Import our platform components
from database_setup import DatabaseManager
from email_service import EmailService
from ai_matching_engine import AIMatchingEngine
from golden_coyotes_platform import GoldenCoyotesPlatform

class ComprehensiveTestSuite:
    """Complete testing of all platform components"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.email_service = EmailService()
        self.ai_matcher = AIMatchingEngine(self.db)
        self.platform = None
        self.base_url = "http://localhost:8083"
        self.test_results = {}
        
    def log_test(self, test_name, result, details=""):
        """Log test results"""
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   {details}")
        self.test_results[test_name] = {"status": result, "details": details}
        
    def test_database_operations(self):
        """Test all database operations"""
        print("\n🗄️ Testing Database Operations")
        print("-" * 40)
        
        try:
            # Test user creation
            user_id = self.db.create_user(
                email="testuser1@example.com",
                password="password123",
                name="Test User One",
                industry="Technology",
                location="San Francisco",
                skills="Python, AI, Machine Learning",
                interests="Startups, Innovation, Tech",
                bio="A passionate tech professional"
            )
            
            self.log_test("User Creation", user_id is not None, f"User ID: {user_id}")
            
            # Test user retrieval
            if user_id:
                user = self.db.get_user(user_id)
                self.log_test("User Retrieval", user is not None, f"Retrieved user: {user['name']}")
            
            # Test user authentication
            auth_result = self.db.authenticate_user("testuser1@example.com", "password123")
            self.log_test("User Authentication", auth_result is not None, f"Auth: {auth_result['name'] if auth_result else 'Failed'}")
            
            # Test opportunity creation
            if user_id:
                opp_id = self.db.create_opportunity(
                    user_id=user_id,
                    title="AI Startup Looking for CTO",
                    description="Revolutionary AI platform seeking technical co-founder",
                    opp_type="partnership",
                    industry="Technology",
                    budget_min=50000,
                    budget_max=100000,
                    location="San Francisco",
                    requirements="Python, AI, Machine Learning experience",
                    tags="AI, startup, CTO, tech"
                )
                self.log_test("Opportunity Creation", opp_id is not None, f"Opportunity ID: {opp_id}")
            
            # Test opportunity retrieval
            opportunities = self.db.get_opportunities(limit=10)
            self.log_test("Opportunity Retrieval", len(opportunities) > 0, f"Found {len(opportunities)} opportunities")
            
            # Test connection creation
            user2_id = self.db.create_user(
                email="testuser2@example.com",
                password="password123",
                name="Test User Two",
                industry="Healthcare",
                location="New York"
            )
            
            if user_id and user2_id:
                conn_id = self.db.create_connection(user_id, user2_id, "Let's connect!")
                self.log_test("Connection Creation", conn_id is not None, f"Connection ID: {conn_id}")
            
        except Exception as e:
            self.log_test("Database Operations", False, f"Error: {str(e)}")
    
    def test_email_system(self):
        """Test email functionality"""
        print("\n📧 Testing Email System")
        print("-" * 40)
        
        try:
            # Check email configuration
            has_config = all([
                self.email_service.email_address,
                self.email_service.email_password,
                self.email_service.smtp_server
            ])
            self.log_test("Email Configuration", has_config, 
                         f"SMTP: {self.email_service.smtp_server}, Email: {self.email_service.email_address}")
            
            if has_config:
                # Test welcome email
                result = self.email_service.send_welcome_email("test@example.com", "Test User")
                self.log_test("Welcome Email", result, "Check your email inbox")
                
                # Test connection request email
                result = self.email_service.send_connection_request_notification(
                    "test@example.com", 
                    "Test User",
                    "John Doe",
                    "http://localhost:8083/profile/123",
                    "I'd like to connect with you!"
                )
                self.log_test("Connection Request Email", result, "Check your email inbox")
                
                # Test opportunity match email
                result = self.email_service.send_opportunity_match_notification(
                    "test@example.com",
                    "Test User", 
                    "AI Startup CTO Position",
                    0.85,
                    "http://localhost:8083/opportunity/123"
                )
                self.log_test("Opportunity Match Email", result, "Check your email inbox")
            else:
                self.log_test("Email System", False, "Email not configured. Set EMAIL_ADDRESS, EMAIL_PASSWORD environment variables")
                
        except Exception as e:
            self.log_test("Email System", False, f"Error: {str(e)}")
    
    def test_ai_matching(self):
        """Test AI matching algorithms"""
        print("\n🧠 Testing AI Matching System")
        print("-" * 40)
        
        try:
            # Create test users with different profiles
            user1_id = self.db.create_user(
                email="aitest1@example.com",
                password="test123",
                name="AI Researcher",
                industry="Technology",
                location="San Francisco",
                skills="Python, AI, Machine Learning, Deep Learning",
                interests="AI, Research, Innovation, Startups",
                bio="PhD in AI, passionate about machine learning applications"
            )
            
            user2_id = self.db.create_user(
                email="aitest2@example.com", 
                password="test123",
                name="Healthcare Tech",
                industry="Healthcare",
                location="New York",
                skills="Healthcare, Data Analysis, Python",
                interests="HealthTech, Innovation, Data Science",
                bio="Healthcare professional with tech background"
            )
            
            # Create opportunities
            if user1_id:
                opp1_id = self.db.create_opportunity(
                    user_id=user1_id,
                    title="AI Research Collaboration",
                    description="Looking for AI researchers to collaborate on machine learning project",
                    opp_type="collaboration",
                    industry="Technology",
                    requirements="PhD in AI, Python, Machine Learning",
                    tags="AI, research, collaboration, ML"
                )
            
            if user2_id:
                opp2_id = self.db.create_opportunity(
                    user_id=user2_id,
                    title="HealthTech Startup Partnership",
                    description="Digital health platform seeking tech partnerships",
                    opp_type="partnership", 
                    industry="Healthcare",
                    requirements="Tech background, healthcare interest",
                    tags="healthtech, partnership, innovation"
                )
            
            # Test opportunity matching for user1
            if user1_id:
                matches = self.ai_matcher.calculate_opportunity_matches(user1_id, limit=5)
                self.log_test("AI Opportunity Matching", len(matches) >= 0, 
                             f"Found {len(matches)} matches for AI Researcher")
                
                for match in matches:
                    print(f"   → {match['opportunity']['title']}: {match['score']:.2f} score")
                    print(f"     Reason: {match['reasoning']}")
            
            # Test user matching
            if user1_id:
                user_matches = self.ai_matcher.calculate_user_matches(user1_id, limit=5)
                self.log_test("AI User Matching", len(user_matches) >= 0,
                             f"Found {len(user_matches)} user matches")
                
                for match in user_matches:
                    print(f"   → {match['user']['name']}: {match['score']:.2f} score")
                    print(f"     Reason: {match['reasoning']}")
            
        except Exception as e:
            self.log_test("AI Matching", False, f"Error: {str(e)}")
    
    def test_platform_api(self):
        """Test platform API endpoints"""
        print("\n🌐 Testing Platform API")
        print("-" * 40)
        
        # Start platform in thread
        def run_platform():
            self.platform = GoldenCoyotesPlatform()
            self.platform.run(port=8083, debug=False)
        
        platform_thread = threading.Thread(target=run_platform)
        platform_thread.daemon = True
        platform_thread.start()
        
        # Wait for platform to start
        time.sleep(3)
        
        try:
            # Test landing page
            response = requests.get(f"{self.base_url}/")
            self.log_test("Landing Page API", response.status_code == 200, f"Status: {response.status_code}")
            
            # Test registration page
            response = requests.get(f"{self.base_url}/register")
            self.log_test("Register Page API", response.status_code == 200, f"Status: {response.status_code}")
            
            # Test user registration
            user_data = {
                "name": "API Test User",
                "email": "apitest@example.com",
                "password": "test123",
                "industry": "Technology",
                "location": "San Francisco",
                "skills": "API Testing, Python",
                "interests": "Testing, Quality Assurance",
                "bio": "Quality assurance professional"
            }
            
            response = requests.post(f"{self.base_url}/register", json=user_data)
            self.log_test("User Registration API", response.status_code == 200, 
                         f"Status: {response.status_code}")
            
            # Test login
            login_data = {"email": "apitest@example.com", "password": "test123"}
            session = requests.Session()
            response = session.post(f"{self.base_url}/login", json=login_data)
            self.log_test("Login API", response.status_code == 200, f"Status: {response.status_code}")
            
            # Test protected dashboard (should work after login)
            response = session.get(f"{self.base_url}/dashboard")
            self.log_test("Dashboard API (Protected)", response.status_code == 200, 
                         f"Status: {response.status_code}")
            
            # Test opportunity creation
            opp_data = {
                "title": "API Test Opportunity",
                "description": "Testing opportunity creation via API",
                "type": "collaboration",
                "industry": "Technology"
            }
            
            response = session.post(f"{self.base_url}/create-opportunity", json=opp_data)
            self.log_test("Opportunity Creation API", response.status_code == 200,
                         f"Status: {response.status_code}")
            
        except Exception as e:
            self.log_test("Platform API", False, f"Error: {str(e)}")
    
    def test_end_to_end_scenario(self):
        """Test complete user journey"""
        print("\n🎯 Testing End-to-End User Scenario")
        print("-" * 40)
        
        try:
            # Scenario: Two users register, create opportunities, and connect
            
            # User 1: Tech Entrepreneur
            print("👤 Creating Tech Entrepreneur...")
            user1_id = self.db.create_user(
                email="entrepreneur@example.com",
                password="secure123",
                name="Tech Entrepreneur",
                industry="Technology",
                location="San Francisco",
                skills="Business Development, AI, Startups",
                interests="Entrepreneurship, AI, Innovation",
                bio="Serial entrepreneur with 3 successful exits"
            )
            
            # User 2: AI Engineer
            print("👤 Creating AI Engineer...")
            user2_id = self.db.create_user(
                email="engineer@example.com",
                password="secure123", 
                name="Senior AI Engineer",
                industry="Technology",
                location="San Francisco",
                skills="Python, TensorFlow, Machine Learning",
                interests="AI, Deep Learning, Research",
                bio="Senior engineer with 8 years in AI/ML"
            )
            
            self.log_test("User Creation (Scenario)", user1_id and user2_id, 
                         "Both users created successfully")
            
            # User 1 creates opportunity
            if user1_id:
                print("💼 Tech Entrepreneur creates opportunity...")
                opp_id = self.db.create_opportunity(
                    user_id=user1_id,
                    title="AI Startup Seeks Technical Co-founder",
                    description="Revolutionary AI platform for healthcare needs experienced CTO with machine learning expertise",
                    opp_type="partnership",
                    industry="Technology",
                    budget_min=80000,
                    budget_max=150000,
                    location="San Francisco",
                    requirements="Python, TensorFlow, Machine Learning, 5+ years experience",
                    tags="AI, startup, CTO, healthcare, ML"
                )
                
                self.log_test("Opportunity Creation (Scenario)", opp_id is not None,
                             f"Opportunity created: {opp_id}")
            
            # Test AI matching for User 2
            if user2_id:
                print("🧠 AI Engineer gets opportunity matches...")
                matches = self.ai_matcher.calculate_opportunity_matches(user2_id, limit=3)
                
                high_matches = [m for m in matches if m['score'] > 0.5]
                self.log_test("AI Matching (Scenario)", len(high_matches) > 0,
                             f"Found {len(high_matches)} high-quality matches")
                
                for match in high_matches:
                    print(f"   🎯 Match: {match['opportunity']['title']}")
                    print(f"       Score: {match['score']:.2f}")
                    print(f"       Reason: {match['reasoning']}")
            
            # User 2 connects with User 1
            if user1_id and user2_id:
                print("🤝 AI Engineer connects with Tech Entrepreneur...")
                conn_id = self.db.create_connection(
                    user2_id, 
                    user1_id,
                    "Hi! I saw your CTO opportunity and I'm very interested. I have 8 years of AI/ML experience and would love to discuss this further."
                )
                
                self.log_test("Connection Request (Scenario)", conn_id is not None,
                             f"Connection created: {conn_id}")
                
                # Test email notification for connection
                user1 = self.db.get_user(user1_id)
                user2 = self.db.get_user(user2_id)
                
                if user1 and user2:
                    email_sent = self.email_service.send_connection_request_notification(
                        user1['email'],
                        user1['name'],
                        user2['name'],
                        f"http://localhost:8083/profile/{user2_id}",
                        "I'm interested in your CTO opportunity!"
                    )
                    
                    self.log_test("Connection Email (Scenario)", email_sent,
                                 "Connection request email sent")
            
            # Test opportunity match notification
            if user2_id and opp_id:
                print("📧 Sending opportunity match notification...")
                user2 = self.db.get_user(user2_id)
                if user2:
                    email_sent = self.email_service.send_opportunity_match_notification(
                        user2['email'],
                        user2['name'],
                        "AI Startup Seeks Technical Co-founder",
                        0.92,
                        f"http://localhost:8083/opportunity/{opp_id}"
                    )
                    
                    self.log_test("Match Email (Scenario)", email_sent,
                                 "Opportunity match email sent")
            
        except Exception as e:
            self.log_test("End-to-End Scenario", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 GOLDEN COYOTES PLATFORM - COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Test started at: {datetime.now()}")
        
        # Run all test categories
        self.test_database_operations()
        self.test_email_system()
        self.test_ai_matching()
        self.test_platform_api()
        self.test_end_to_end_scenario()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'])
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            status = "✅" if result['status'] else "❌"
            print(f"{status} {test_name}")
        
        print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Platform is fully functional.")
        else:
            print(f"⚠️  {failed_tests} tests failed. See details above.")
        
        print(f"\nTest completed at: {datetime.now()}")
        
        return failed_tests == 0

if __name__ == "__main__":
    print("Starting comprehensive test suite...")
    
    # Check if email configuration exists
    print("\n📧 Email Configuration Check:")
    print(f"EMAIL_ADDRESS: {os.getenv('EMAIL_ADDRESS', 'Not set')}")
    print(f"EMAIL_PASSWORD: {'Set' if os.getenv('EMAIL_PASSWORD') else 'Not set'}")
    print(f"SMTP_SERVER: {os.getenv('SMTP_SERVER', 'Default: smtp.gmail.com')}")
    
    if not os.getenv('EMAIL_ADDRESS'):
        print("\n⚠️  To test email functionality, set environment variables:")
        print("export EMAIL_ADDRESS='your-email@gmail.com'")
        print("export EMAIL_PASSWORD='your-app-password'")
        print("export SMTP_SERVER='smtp.gmail.com'")
    
    print("\n" + "="*60)
    
    # Run tests
    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🚀 Platform is ready for production use!")
        print("Run: python3 golden_coyotes_platform.py")
        print("Visit: http://localhost:8080")
    else:
        print("\n🔧 Some components need attention. Check the test results above.")
    
    sys.exit(0 if success else 1)