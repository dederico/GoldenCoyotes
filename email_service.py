#!/usr/bin/env python3
"""
Email notification service for Golden Coyotes Platform
SMTP integration for real email notifications
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """Handles email notifications and communications"""
    
    def __init__(self):
        # Email configuration - use environment variables for security
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_address = os.getenv('EMAIL_ADDRESS', 'your-email@gmail.com')
        self.email_password = os.getenv('EMAIL_PASSWORD', 'your-app-password')
        self.sender_name = os.getenv('SENDER_NAME', 'Golden Coyotes Platform')
        
        # Email templates
        self.templates = {
            'welcome': self.get_welcome_template(),
            'opportunity_match': self.get_opportunity_match_template(),
            'connection_request': self.get_connection_request_template(),
            'message_notification': self.get_message_notification_template(),
            'weekly_digest': self.get_weekly_digest_template()
        }
    
    def send_email(self, to_email, subject, html_content, text_content=None):
        """Send email with HTML content"""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.sender_name} <{self.email_address}>"
            message["To"] = to_email
            
            # Add text version if provided
            if text_content:
                text_part = MIMEText(text_content, "plain")
                message.attach(text_part)
            
            # Add HTML version
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email_address, self.email_password)
                server.sendmail(self.email_address, to_email, message.as_string())
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_welcome_email(self, user_email, user_name):
        """Send welcome email to new user"""
        subject = "Welcome to Golden Coyotes Platform! 🎉"
        html_content = self.templates['welcome'].format(
            user_name=user_name,
            platform_url="http://localhost:8080",
            year=datetime.now().year
        )
        
        return self.send_email(user_email, subject, html_content)
    
    def send_opportunity_match_notification(self, user_email, user_name, opportunity_title, match_score, opportunity_url):
        """Send opportunity match notification"""
        subject = f"🎯 New Opportunity Match: {opportunity_title}"
        html_content = self.templates['opportunity_match'].format(
            user_name=user_name,
            opportunity_title=opportunity_title,
            match_score=int(match_score * 100),
            opportunity_url=opportunity_url,
            platform_url="http://localhost:8080",
            year=datetime.now().year
        )
        
        return self.send_email(user_email, subject, html_content)
    
    def send_connection_request_notification(self, user_email, user_name, requester_name, requester_profile_url, message=""):
        """Send connection request notification"""
        subject = f"🤝 New Connection Request from {requester_name}"
        html_content = self.templates['connection_request'].format(
            user_name=user_name,
            requester_name=requester_name,
            requester_profile_url=requester_profile_url,
            personal_message=message if message else "No personal message included.",
            connections_url="http://localhost:8080/connections",
            platform_url="http://localhost:8080",
            year=datetime.now().year
        )
        
        return self.send_email(user_email, subject, html_content)
    
    def send_message_notification(self, user_email, user_name, sender_name, message_subject, message_url):
        """Send new message notification"""
        subject = f"💬 New Message from {sender_name}"
        html_content = self.templates['message_notification'].format(
            user_name=user_name,
            sender_name=sender_name,
            message_subject=message_subject,
            message_url=message_url,
            platform_url="http://localhost:8080",
            year=datetime.now().year
        )
        
        return self.send_email(user_email, subject, html_content)
    
    def send_weekly_digest(self, user_email, user_name, stats):
        """Send weekly activity digest"""
        subject = "📊 Your Weekly Golden Coyotes Digest"
        html_content = self.templates['weekly_digest'].format(
            user_name=user_name,
            new_opportunities=stats.get('new_opportunities', 0),
            new_connections=stats.get('new_connections', 0),
            profile_views=stats.get('profile_views', 0),
            messages_received=stats.get('messages_received', 0),
            platform_url="http://localhost:8080",
            year=datetime.now().year
        )
        
        return self.send_email(user_email, subject, html_content)
    
    def get_welcome_template(self):
        """Welcome email template"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to Golden Coyotes</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .button {{ display: inline-block; background: #667eea; color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; margin: 20px 0; }}
                .footer {{ background: #333; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎉 Welcome to Golden Coyotes!</h1>
                    <p>Your journey to meaningful business connections starts here</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p>Thank you for joining the Golden Coyotes platform! We're excited to help you discover new business opportunities and build valuable connections.</p>
                    
                    <h3>🚀 Get Started:</h3>
                    <ul>
                        <li>Complete your profile to get better matches</li>
                        <li>Browse opportunities in your industry</li>
                        <li>Connect with like-minded professionals</li>
                        <li>Create your own business opportunities</li>
                    </ul>
                    
                    <center>
                        <a href="{platform_url}" class="button">Explore Platform</a>
                    </center>
                    
                    <p>Our AI-powered matching system will help you find the most relevant opportunities based on your interests, industry, and professional background.</p>
                    
                    <p>If you have any questions, feel free to reach out to our support team.</p>
                    
                    <p>Welcome aboard!</p>
                    <p><strong>The Golden Coyotes Team</strong></p>
                </div>
                <div class="footer">
                    <p>&copy; {year} Golden Coyotes Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def get_opportunity_match_template(self):
        """Opportunity match notification template"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Opportunity Match</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .match-score {{ background: #28a745; color: white; padding: 10px 20px; border-radius: 25px; display: inline-block; font-weight: bold; }}
                .button {{ display: inline-block; background: #28a745; color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; margin: 20px 0; }}
                .footer {{ background: #333; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 New Opportunity Match!</h1>
                    <p>Our AI found a great match for you</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p>Great news! Our AI matching system has identified a high-potential opportunity that aligns with your profile and interests.</p>
                    
                    <h3>📋 Opportunity Details:</h3>
                    <p><strong>Title:</strong> {opportunity_title}</p>
                    <p><strong>Match Score:</strong> <span class="match-score">{match_score}% Match</span></p>
                    
                    <p>This opportunity matches your professional background, interests, and industry expertise. We recommend reviewing it as soon as possible.</p>
                    
                    <center>
                        <a href="{opportunity_url}" class="button">View Opportunity</a>
                    </center>
                    
                    <p>Don't miss out on this potential connection. The best opportunities often go to those who act quickly!</p>
                    
                    <p>Happy networking!</p>
                    <p><strong>The Golden Coyotes Team</strong></p>
                </div>
                <div class="footer">
                    <p>&copy; {year} Golden Coyotes Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def get_connection_request_template(self):
        """Connection request notification template"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Connection Request</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .message-box {{ background: #e9ecef; padding: 15px; border-left: 4px solid #6f42c1; margin: 20px 0; }}
                .button {{ display: inline-block; background: #6f42c1; color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; margin: 20px 0; }}
                .footer {{ background: #333; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤝 New Connection Request</h1>
                    <p>Someone wants to connect with you</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p><strong>{requester_name}</strong> has sent you a connection request on Golden Coyotes!</p>
                    
                    <div class="message-box">
                        <p><strong>Personal Message:</strong></p>
                        <p>"{personal_message}"</p>
                    </div>
                    
                    <p>Building your professional network is key to discovering new opportunities. We encourage you to review their profile and consider connecting.</p>
                    
                    <center>
                        <a href="{requester_profile_url}" class="button">View Profile</a>
                        <a href="{connections_url}" class="button">Manage Connections</a>
                    </center>
                    
                    <p>Growing your network opens doors to new collaborations, partnerships, and business opportunities.</p>
                    
                    <p>Best regards,</p>
                    <p><strong>The Golden Coyotes Team</strong></p>
                </div>
                <div class="footer">
                    <p>&copy; {year} Golden Coyotes Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def get_message_notification_template(self):
        """Message notification template"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>New Message</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #17a2b8 0%, #6610f2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .button {{ display: inline-block; background: #17a2b8; color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; margin: 20px 0; }}
                .footer {{ background: #333; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>💬 New Message</h1>
                    <p>You have a new message waiting</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p><strong>{sender_name}</strong> has sent you a new message on Golden Coyotes.</p>
                    
                    <p><strong>Subject:</strong> {message_subject}</p>
                    
                    <p>Log in to your account to read the full message and respond.</p>
                    
                    <center>
                        <a href="{message_url}" class="button">Read Message</a>
                    </center>
                    
                    <p>Stay connected and keep the conversation going!</p>
                    
                    <p>Best regards,</p>
                    <p><strong>The Golden Coyotes Team</strong></p>
                </div>
                <div class="footer">
                    <p>&copy; {year} Golden Coyotes Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    def get_weekly_digest_template(self):
        """Weekly digest template"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weekly Digest</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #fd7e14 0%, #dc3545 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f8f9fa; padding: 30px; }}
                .stats {{ background: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .stat-item {{ display: inline-block; text-align: center; margin: 10px 20px; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #fd7e14; }}
                .button {{ display: inline-block; background: #fd7e14; color: white; padding: 12px 30px; text-decoration: none; border-radius: 25px; margin: 20px 0; }}
                .footer {{ background: #333; color: white; padding: 20px; text-align: center; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Your Weekly Digest</h1>
                    <p>Here's what happened in your network this week</p>
                </div>
                <div class="content">
                    <h2>Hello {user_name},</h2>
                    <p>Here's a summary of your activity and opportunities from the past week on Golden Coyotes.</p>
                    
                    <div class="stats">
                        <h3>📈 Your Week in Numbers</h3>
                        <div class="stat-item">
                            <div class="stat-number">{new_opportunities}</div>
                            <div>New Opportunities</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{new_connections}</div>
                            <div>New Connections</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{profile_views}</div>
                            <div>Profile Views</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number">{messages_received}</div>
                            <div>Messages Received</div>
                        </div>
                    </div>
                    
                    <p>Keep engaging with the platform to maximize your networking potential and discover new business opportunities!</p>
                    
                    <center>
                        <a href="{platform_url}" class="button">Visit Platform</a>
                    </center>
                    
                    <p>Have a great week ahead!</p>
                    <p><strong>The Golden Coyotes Team</strong></p>
                </div>
                <div class="footer">
                    <p>&copy; {year} Golden Coyotes Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''