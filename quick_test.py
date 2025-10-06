#!/usr/bin/env python3

from golden_coyotes_platform import GoldenCoyotesPlatform

print("Testing platform creation...")
platform = GoldenCoyotesPlatform()
print("✅ Platform created successfully!")

print("Testing database...")
user_id = platform.db.create_user(
    email="test@example.com",
    password="test123",
    name="Test User",
    industry="Technology",
    location="San Francisco"
)
print(f"✅ User created: {user_id}")

print("Testing AI matcher...")
matches = platform.ai_matcher.calculate_opportunity_matches("test_user", limit=5)
print(f"✅ AI matcher working: {len(matches)} matches")

print("✅ All components working!")
print("\nTo run the platform:")
print("python3 golden_coyotes_platform.py")
print("Then visit: http://localhost:8080")