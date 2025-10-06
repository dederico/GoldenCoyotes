#!/usr/bin/env python3
"""
Test AI Matching Fix
"""

from database_setup import DatabaseManager
from ai_matching_engine import AIMatchingEngine

def test_ai_matching_fix():
    print("🧪 Testing AI Matching Fix")
    print("=" * 40)
    
    # Initialize components
    db = DatabaseManager()
    ai_matcher = AIMatchingEngine(db)
    
    try:
        # Create test user with some None values (simulating database state)
        user_id = db.create_user(
            email="test_ai@example.com",
            password="test123",
            name="AI Test User",
            industry="Technology",
            location="San Francisco",
            skills="Python, AI, Machine Learning",  # This will be stored
            interests="AI, Innovation",              # This will be stored  
            bio=None,                               # This might be None
            company=None,                           # This might be None
            position=None                           # This might be None
        )
        
        print(f"✅ Created test user: {user_id}")
        
        # Create test opportunity
        opp_id = db.create_opportunity(
            user_id=user_id,
            title="AI Research Position",
            description="Looking for AI researcher with Python and machine learning experience",
            opp_type="collaboration",
            industry="Technology",
            requirements="Python, AI, Machine Learning",
            tags="AI, research, python"
        )
        
        print(f"✅ Created test opportunity: {opp_id}")
        
        # Test AI matching (this should not crash now)
        print("🧠 Testing AI matching...")
        matches = ai_matcher.calculate_opportunity_matches(user_id, limit=3)
        print(f"✅ AI matching successful: {len(matches)} matches found")
        
        # Show match details
        for match in matches:
            print(f"   📋 {match['opportunity']['title']}")
            print(f"      Score: {match['score']:.2f}")
            print(f"      Reason: {match['reasoning']}")
        
        # Test user matching
        print("👥 Testing user matching...")
        user_matches = ai_matcher.calculate_user_matches(user_id, limit=3)
        print(f"✅ User matching successful: {len(user_matches)} matches found")
        
        for match in user_matches:
            print(f"   👤 {match['user']['name']}")
            print(f"      Score: {match['score']:.2f}")
            print(f"      Reason: {match['reasoning']}")
        
        print("\n🎉 AI Matching Fix Successful!")
        return True
        
    except Exception as e:
        print(f"❌ AI Matching still has issues: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ai_matching_fix()