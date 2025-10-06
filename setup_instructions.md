# Golden Coyotes Platform - Setup Instructions

## 🚀 Complete Multi-User Business Networking Platform

This is a fully functional business networking platform with user authentication, database persistence, AI matching, and email notifications.

## ✨ Key Features

### 🔐 User Authentication
- **User Registration** with comprehensive profile creation
- **Secure Login/Logout** with session management
- **Password Hashing** with salt for security
- **Profile Management** with completion tracking

### 🗄️ Database Persistence
- **SQLite Database** with proper schema and relationships
- **Data Persistence** - users and data survive restarts
- **User Connections** and networking functionality
- **Opportunity Management** with user ownership

### 🧠 AI-Powered Matching
- **Real AI Matching** using TF-IDF and semantic analysis
- **OpenAI Integration** for advanced text embeddings (optional)
- **Multi-factor Scoring** considering industry, skills, interests, location
- **Smart Recommendations** for both opportunities and user connections

### 📧 Email Notifications
- **SMTP Integration** for real email sending
- **Welcome Emails** for new users
- **Opportunity Match Notifications** when AI finds good matches
- **Connection Request Notifications** for networking
- **Professional Email Templates** with HTML styling

### 🌐 Complete User Interface
- **Landing Page** with modern design
- **User Dashboard** with personalized content
- **Opportunity Management** (browse, create, manage)
- **Network Building** with connection requests
- **Profile Management** with completion tracking

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
pip install flask flask-cors sqlite3 scikit-learn numpy openai
```

### 2. Configure Email (Optional)

For email notifications to work, set these environment variables:

```bash
export EMAIL_ADDRESS="dederico@gmail.com"
export EMAIL_PASSWORD="miguelpendejo2."  # Use Gmail App Password
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
```

**Note:** To use Gmail:
1. Enable 2-factor authentication
2. Generate an App Password in Google Account settings
3. Use the App Password, not your regular password

### 3. Configure OpenAI (Optional)

For advanced AI matching with embeddings:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 4. Initialize Database

```bash
python3 database_setup.py
```

### 5. Run the Platform

```bash
python3 golden_coyotes_platform.py
```

Access the platform at: **http://localhost:8080**

## 🎯 How to Use

### For New Users

1. **Visit the Landing Page** at http://localhost:8080
2. **Click "Join Now"** to create an account
3. **Fill out the registration form** with your professional details
4. **Complete your profile** to get better AI matches
5. **Start networking** and creating opportunities!

### Key User Journey

1. **Register** with professional information
2. **Complete Profile** (industry, skills, interests, bio)
3. **Browse Opportunities** or create your own
4. **Get AI Matches** based on your profile
5. **Connect with Users** who match your interests
6. **Receive Email Notifications** for matches and connections

## 🧪 Testing the Platform

### Test Different User Types

Create multiple test accounts with different:
- **Industries** (Technology, Healthcare, Finance, etc.)
- **Locations** (different cities)
- **Skills** (AI, Marketing, Sales, etc.)
- **Interests** (Startups, Innovation, etc.)

### Test AI Matching

1. Create users with **similar interests** → Should get high match scores
2. Create opportunities in **different industries** → Should see industry-based matching
3. Use **specific keywords** in skills/interests → Should see semantic matching

### Test Email Notifications

1. **Register new users** → Should receive welcome emails
2. **Create opportunities** → Other users should get match notifications
3. **Send connection requests** → Should receive connection emails

## 🔧 System Architecture

### Database Schema

- **users** - User accounts and profiles
- **opportunities** - Business opportunities
- **connections** - User network relationships
- **interactions** - User activity tracking
- **notifications** - System notifications
- **messages** - User-to-user messaging
- **ai_matches** - AI-calculated match scores

### AI Matching Algorithm

The system uses multiple factors for intelligent matching:

1. **Industry Match** (25%) - Exact or related industry scoring
2. **Skills Match** (20%) - Skills vs. opportunity requirements
3. **Interests Match** (20%) - Interest keywords in descriptions
4. **Location Proximity** (10%) - Geographic proximity scoring
5. **Experience Level** (10%) - Professional level compatibility
6. **Semantic Similarity** (15%) - Text similarity using TF-IDF/embeddings

### Security Features

- **Password Hashing** with PBKDF2 and salt
- **Session Management** with secure session keys
- **Input Validation** and SQL injection prevention
- **User Authorization** for data access

## 🎉 What This Solves

### Previous Issues Fixed

1. **❌ No Database Persistence** → **✅ SQLite with proper persistence**
2. **❌ Hardcoded Data** → **✅ Real user-generated content**
3. **❌ No Email Notifications** → **✅ Real SMTP email system**
4. **❌ Hardcoded Matching** → **✅ AI-powered matching algorithm**
5. **❌ Single User Interface** → **✅ Multi-user platform with authentication**
6. **❌ No User Networks** → **✅ Connection requests and networking**

### New Capabilities

- **Multi-tenant Platform** - Each user has their own account and data
- **Real AI Intelligence** - Smart matching based on profile analysis
- **Professional Networking** - Users can connect and build networks
- **Email Integration** - Real notifications and communications
- **Scalable Architecture** - Database-driven, can handle many users

## 🚀 Next Steps

The platform is production-ready with:
- ✅ Complete user authentication system
- ✅ Database persistence and relationships
- ✅ AI-powered matching and recommendations
- ✅ Email notification system
- ✅ Multi-user networking capabilities
- ✅ Professional user interface

**You can now register users, create opportunities, build networks, and see real AI-powered matching in action!**