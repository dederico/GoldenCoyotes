#!/usr/bin/env python3
"""
Database setup and models for Golden Coyotes Platform
SQLite database with proper persistence and relationships
"""

import sqlite3
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """Manages SQLite database operations for the platform"""
    
    def __init__(self, db_path="golden_coyotes.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    industry TEXT,
                    location TEXT,
                    bio TEXT,
                    skills TEXT,
                    interests TEXT,
                    company TEXT,
                    position TEXT,
                    phone TEXT,
                    website TEXT,
                    linkedin TEXT,
                    profile_image TEXT,
                    is_verified INTEGER DEFAULT 0,
                    user_role TEXT DEFAULT 'user',
                    admin_level INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Opportunities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS opportunities (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    industry TEXT,
                    budget_min INTEGER,
                    budget_max INTEGER,
                    location TEXT,
                    deadline DATE,
                    expiration_date DATE,
                    requirements TEXT,
                    tags TEXT,
                    is_active INTEGER DEFAULT 1,
                    views INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # User connections/network table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS connections (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    connected_user_id TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accepted_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (connected_user_id) REFERENCES users (id),
                    UNIQUE(user_id, connected_user_id)
                )
            ''')
            
            # User interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    opportunity_id TEXT,
                    target_user_id TEXT,
                    type TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id),
                    FOREIGN KEY (target_user_id) REFERENCES users (id)
                )
            ''')
            
            # Notifications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    type TEXT NOT NULL,
                    channel TEXT DEFAULT 'in_app',
                    is_read INTEGER DEFAULT 0,
                    is_sent INTEGER DEFAULT 0,
                    opportunity_id TEXT,
                    related_user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id),
                    FOREIGN KEY (related_user_id) REFERENCES users (id)
                )
            ''')
            
            # Messages table for user communication
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    opportunity_id TEXT,
                    subject TEXT,
                    content TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_id) REFERENCES users (id),
                    FOREIGN KEY (recipient_id) REFERENCES users (id),
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id)
                )
            ''')
            
            # AI matching scores table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_matches (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    opportunity_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    reasoning TEXT,
                    factors TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id),
                    UNIQUE(user_id, opportunity_id)
                )
            ''')
            
            # Admin activity logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS admin_logs (
                    id TEXT PRIMARY KEY,
                    admin_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT,
                    target_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (admin_id) REFERENCES users (id)
                )
            ''')
            
            conn.commit()
            print("✅ Database initialized successfully")
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users(user_role)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_opportunities_user ON opportunities(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_connections_users ON connections(user_id, connected_user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_admin_logs_admin ON admin_logs(admin_id)')
            conn.commit()
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def hash_password(self, password):
        """Hash password with salt"""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt + pwdhash
    
    def verify_password(self, stored_password, provided_password):
        """Verify password against hash"""
        salt = stored_password[:32]
        stored_hash = stored_password[32:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwdhash == stored_hash
    
    def create_user(self, email, password, name, **kwargs):
        """Create new user account"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            user_id = str(uuid.uuid4())
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (id, email, password_hash, name, industry, location, bio, skills, interests, company, position, phone, user_role, admin_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, email, password_hash, name,
                kwargs.get('industry'), kwargs.get('location'), kwargs.get('bio'),
                kwargs.get('skills'), kwargs.get('interests'), kwargs.get('company'),
                kwargs.get('position'), kwargs.get('phone'),
                kwargs.get('user_role', 'user'), kwargs.get('admin_level', 0)
            ))
            
            conn.commit()
            return user_id
            
        except sqlite3.IntegrityError:
            return None  # Email already exists
        except Exception as e:
            print(f"Error creating user: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def authenticate_user(self, email, password):
        """Authenticate user login"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT id, password_hash, name, user_role, admin_level FROM users WHERE email = ? AND status = "active"', (email,))
            result = cursor.fetchone()
            
            if result and self.verify_password(result[1], password):
                user_id, _, name, user_role, admin_level = result
                # Update last login
                cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
                conn.commit()
                return {'id': user_id, 'name': name, 'email': email, 'user_role': user_role, 'admin_level': admin_level}
            
            return None
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return None
        finally:
            conn.close()
    
    def get_user(self, user_id):
        """Get user by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
        finally:
            conn.close()
    
    def create_opportunity(self, user_id, title, description, opp_type, **kwargs):
        """Create new opportunity"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            opp_id = str(uuid.uuid4())

            cursor.execute('''
                INSERT INTO opportunities (id, user_id, title, description, type, industry, budget_min, budget_max, location, deadline, expiration_date, requirements, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opp_id, user_id, title, description, opp_type,
                kwargs.get('industry'), kwargs.get('budget_min'), kwargs.get('budget_max'),
                kwargs.get('location'), kwargs.get('deadline'), kwargs.get('expiration_date'), kwargs.get('requirements'), kwargs.get('tags')
            ))

            conn.commit()
            return opp_id

        except Exception as e:
            print(f"Error creating opportunity: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_opportunities(self, user_id=None, limit=50):
        """Get opportunities, optionally filtered by user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            if user_id:
                cursor.execute('''
                    SELECT o.*, u.name as creator_name 
                    FROM opportunities o 
                    JOIN users u ON o.user_id = u.id 
                    WHERE o.user_id = ? AND o.is_active = 1 
                    ORDER BY o.created_at DESC LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT o.*, u.name as creator_name 
                    FROM opportunities o 
                    JOIN users u ON o.user_id = u.id 
                    WHERE o.is_active = 1 
                    ORDER BY o.created_at DESC LIMIT ?
                ''', (limit,))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting opportunities: {e}")
            return []
        finally:
            conn.close()
    
    def create_connection(self, user_id, target_user_id, message="", status="pending", accepted_at=None):
        """Create connection request between users"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            conn_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO connections (id, user_id, connected_user_id, status, message, accepted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (conn_id, user_id, target_user_id, status, message, accepted_at))
            
            conn.commit()
            return conn_id
            
        except sqlite3.IntegrityError:
            return None  # Connection already exists
        except Exception as e:
            print(f"Error creating connection: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_user_connections(self, user_id):
        """Get user's connections"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT c.*, u.name, u.email, u.industry, u.location
                FROM connections c
                JOIN users u ON (c.connected_user_id = u.id OR c.user_id = u.id)
                WHERE (c.user_id = ? OR c.connected_user_id = ?) 
                AND c.status = 'accepted' 
                AND u.id != ?
            ''', (user_id, user_id, user_id))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting connections: {e}")
            return []
        finally:
            conn.close()
    
    def send_message(self, sender_id, recipient_id, subject, content, opportunity_id=None):
        """Send message between users"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            message_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO messages (id, sender_id, recipient_id, opportunity_id, subject, content, is_read)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            ''', (message_id, sender_id, recipient_id, opportunity_id, subject, content))
            
            conn.commit()
            return message_id
            
        except Exception as e:
            print(f"Error sending message: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_user_messages(self, user_id, limit=50):
        """Get user's messages (received and sent)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT m.*, 
                       sender.name as sender_name, sender.email as sender_email,
                       recipient.name as recipient_name, recipient.email as recipient_email
                FROM messages m
                JOIN users sender ON m.sender_id = sender.id
                JOIN users recipient ON m.recipient_id = recipient.id
                WHERE m.sender_id = ? OR m.recipient_id = ?
                ORDER BY m.created_at DESC LIMIT ?
            ''', (user_id, user_id, limit))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []
        finally:
            conn.close()
    
    def get_conversation(self, user1_id, user2_id, limit=50):
        """Get conversation between two users"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT m.*, 
                       sender.name as sender_name, sender.email as sender_email,
                       recipient.name as recipient_name, recipient.email as recipient_email
                FROM messages m
                JOIN users sender ON m.sender_id = sender.id
                JOIN users recipient ON m.recipient_id = recipient.id
                WHERE (m.sender_id = ? AND m.recipient_id = ?) 
                   OR (m.sender_id = ? AND m.recipient_id = ?)
                ORDER BY m.created_at ASC LIMIT ?
            ''', (user1_id, user2_id, user2_id, user1_id, limit))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return []
        finally:
            conn.close()
    
    def mark_message_read(self, message_id, user_id):
        """Mark message as read"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE messages SET is_read = 1 
                WHERE id = ? AND recipient_id = ?
            ''', (message_id, user_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error marking message as read: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def create_admin_user(self, email, password, name, admin_level=1):
        """Create admin user account"""
        return self.create_user(
            email=email, 
            password=password, 
            name=name,
            user_role='admin',
            admin_level=admin_level,
            industry='Technology',
            location='Golden Coyotes HQ'
        )
    
    def log_admin_action(self, admin_id, action, target_type=None, target_id=None, details=None, ip_address=None, user_agent=None):
        """Log admin action for audit trail"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            log_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO admin_logs (id, admin_id, action, target_type, target_id, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (log_id, admin_id, action, target_type, target_id, details, ip_address, user_agent))
            
            conn.commit()
            return log_id
            
        except Exception as e:
            print(f"Error logging admin action: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_platform_statistics(self):
        """Get platform-wide statistics for admin dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            stats = {}
            
            # User statistics
            cursor.execute('SELECT COUNT(*) FROM users WHERE status = "active"')
            stats['total_users'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM users WHERE created_at >= date("now", "-30 days")')
            stats['new_users_30d'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM users WHERE last_login >= date("now", "-7 days")')
            stats['active_users_7d'] = cursor.fetchone()[0]
            
            # Opportunity statistics
            cursor.execute('SELECT COUNT(*) FROM opportunities WHERE is_active = 1')
            stats['total_opportunities'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM opportunities WHERE created_at >= date("now", "-30 days")')
            stats['new_opportunities_30d'] = cursor.fetchone()[0]
            
            # Connection statistics
            cursor.execute('SELECT COUNT(*) FROM connections WHERE status = "accepted"')
            stats['total_connections'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM connections WHERE created_at >= date("now", "-30 days")')
            stats['new_connections_30d'] = cursor.fetchone()[0]
            
            # Message statistics
            cursor.execute('SELECT COUNT(*) FROM messages')
            stats['total_messages'] = cursor.fetchone()[0]
            
            # Industry breakdown
            cursor.execute('SELECT industry, COUNT(*) FROM users WHERE industry IS NOT NULL GROUP BY industry ORDER BY COUNT(*) DESC LIMIT 10')
            stats['top_industries'] = cursor.fetchall()
            
            # Location breakdown
            cursor.execute('SELECT location, COUNT(*) FROM users WHERE location IS NOT NULL GROUP BY location ORDER BY COUNT(*) DESC LIMIT 10')
            stats['top_locations'] = cursor.fetchall()
            
            return stats
            
        except Exception as e:
            print(f"Error getting platform statistics: {e}")
            return {}
        finally:
            conn.close()
    
    def get_all_users(self, limit=100, offset=0):
        """Get all users for admin management"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT id, email, name, industry, location, user_role, admin_level, 
                       is_verified, status, created_at, last_login
                FROM users 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
        finally:
            conn.close()
    
    def update_user_status(self, user_id, status):
        """Update user status (admin action)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('UPDATE users SET status = ? WHERE id = ?', (status, user_id))
            conn.commit()
            return cursor.rowcount > 0
            
        except Exception as e:
            print(f"Error updating user status: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_admin_logs(self, limit=100, offset=0):
        """Get admin activity logs"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT al.*, u.name as admin_name, u.email as admin_email
                FROM admin_logs al
                JOIN users u ON al.admin_id = u.id
                ORDER BY al.created_at DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"Error getting admin logs: {e}")
            return []
        finally:
            conn.close()

if __name__ == "__main__":
    # Initialize database
    db = DatabaseManager()
    print("Database setup complete!")
    
    # Create default admin user
    admin_id = db.create_admin_user(
        email="admin@goldencoyotes.com",
        password="GC_Admin_2024!",
        name="Golden Coyotes Admin",
        admin_level=3
    )
    
    if admin_id:
        print(f"✅ Default admin user created with ID: {admin_id}")
        print("📧 Email: admin@goldencoyotes.com")
        print("🔐 Password: GC_Admin_2024!")
    else:
        print("ℹ️  Default admin user already exists")
