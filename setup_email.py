#!/usr/bin/env python3
"""
Email Configuration Setup for Golden Coyotes Platform
Sets up email environment variables for testing
"""

import os
import getpass

def setup_email_config():
    """Setup email configuration interactively"""
    print("📧 Golden Coyotes Email Configuration Setup")
    print("=" * 50)
    
    print("\n🔧 To enable email notifications, you need:")
    print("1. Gmail account with 2-factor authentication enabled")
    print("2. Gmail App Password (not your regular password)")
    print("3. SMTP settings")
    
    choice = input("\nDo you want to configure email now? (y/n): ").lower()
    
    if choice != 'y':
        print("\n⚠️  Email notifications will be disabled.")
        print("You can configure them later by setting environment variables:")
        print("export EMAIL_ADDRESS='your-email@gmail.com'")
        print("export EMAIL_PASSWORD='your-app-password'")
        print("export SMTP_SERVER='smtp.gmail.com'")
        print("export SMTP_PORT='587'")
        return False
    
    print("\n📋 Email Configuration:")
    
    # Get email address
    email = input("Enter your Gmail address: ").strip()
    if not email or '@' not in email:
        print("❌ Invalid email address")
        return False
    
    # Get app password
    print("\n🔑 Gmail App Password Setup:")
    print("1. Go to https://myaccount.google.com/security")
    print("2. Enable 2-factor authentication")
    print("3. Go to 'App passwords' and generate a new password")
    print("4. Enter the 16-character app password below")
    
    app_password = getpass.getpass("Enter Gmail App Password: ").strip()
    if not app_password:
        print("❌ App password required")
        return False
    
    # SMTP settings
    smtp_server = input("SMTP Server (default: smtp.gmail.com): ").strip() or "smtp.gmail.com"
    smtp_port = input("SMTP Port (default: 587): ").strip() or "587"
    
    # Test email configuration
    print("\n🧪 Testing email configuration...")
    
    # Set environment variables
    os.environ['EMAIL_ADDRESS'] = email
    os.environ['EMAIL_PASSWORD'] = app_password
    os.environ['SMTP_SERVER'] = smtp_server
    os.environ['SMTP_PORT'] = smtp_port
    
    # Test email sending
    try:
        from email_service import EmailService
        email_service = EmailService()
        
        test_email = input(f"Send test email to (default: {email}): ").strip() or email
        
        print("📤 Sending test email...")
        result = email_service.send_welcome_email(test_email, "Test User")
        
        if result:
            print("✅ Email configuration successful!")
            print("📧 Test email sent successfully!")
            
            # Save to file for future use
            with open('.env', 'w') as f:
                f.write(f"EMAIL_ADDRESS={email}\n")
                f.write(f"EMAIL_PASSWORD={app_password}\n")
                f.write(f"SMTP_SERVER={smtp_server}\n")
                f.write(f"SMTP_PORT={smtp_port}\n")
            
            print("💾 Configuration saved to .env file")
            return True
        else:
            print("❌ Email sending failed. Please check your credentials.")
            return False
            
    except Exception as e:
        print(f"❌ Email configuration error: {e}")
        return False

def load_email_config():
    """Load email configuration from .env file"""
    try:
        if os.path.exists('.env'):
            print("📁 Loading email configuration from .env file...")
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            print("✅ Email configuration loaded")
            return True
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Golden Coyotes Email Setup")
    
    # Try to load existing configuration
    if load_email_config():
        choice = input("Configuration found. Reconfigure? (y/n): ").lower()
        if choice != 'y':
            print("✅ Using existing email configuration")
            exit(0)
    
    # Setup new configuration
    if setup_email_config():
        print("\n🎉 Email setup complete!")
        print("📱 You can now run the platform with email notifications:")
        print("python3 golden_coyotes_platform.py")
    else:
        print("\n⚠️  Email setup skipped or failed.")
        print("Platform will run without email notifications.")