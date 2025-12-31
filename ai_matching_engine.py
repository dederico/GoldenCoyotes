#!/usr/bin/env python3
"""
AI-Powered Matching Engine for Golden Coyotes Platform
Real intelligent matching using OpenAI embeddings and ML algorithms
"""

import sys
print("⏳ Cargando módulos de IA... (esto puede tardar 20-30 segundos)", flush=True)

import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import logging
from datetime import datetime, timedelta
import os

print("✅ Módulos de IA cargados correctamente", flush=True)
logger = logging.getLogger(__name__)

class AIMatchingEngine:
    """Advanced AI matching system for users and opportunities"""
    
    def __init__(self, db_manager, openai_api_key=None):
        self.db = db_manager
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Weights for different matching factors
        self.weights = {
            'industry_match': 0.25,
            'skills_match': 0.20,
            'interests_match': 0.20,
            'location_proximity': 0.10,
            'experience_level': 0.10,
            'semantic_similarity': 0.15
        }
    
    def get_embeddings(self, texts, model="text-embedding-3-small"):
        """Get OpenAI embeddings for texts"""
        try:
            if not self.openai_api_key:
                logger.warning("OpenAI API key not configured, using fallback matching")
                return None
            
            response = openai.embeddings.create(
                input=texts,
                model=model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return None
    
    def calculate_opportunity_matches(self, user_id, limit=10):
        """Calculate AI-powered matches for user with opportunities"""
        user = self.db.get_user(user_id)
        if not user:
            return []
        
        # Get all active opportunities (excluding user's own)
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT o.*, u.name as creator_name, u.industry as creator_industry
                FROM opportunities o
                JOIN users u ON o.user_id = u.id
                WHERE o.is_active = 1 AND o.user_id != ?
                ORDER BY o.created_at DESC
            ''', (user_id,))
            
            opportunities = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            opportunities = [dict(zip(columns, row)) for row in opportunities]
            
        except Exception as e:
            logger.error(f"Error fetching opportunities: {e}")
            return []
        finally:
            conn.close()
        
        if not opportunities:
            return []
        
        matches = []
        user_profile = self._build_user_profile(user)
        
        for opp in opportunities:
            try:
                score = self._calculate_match_score(user_profile, opp)
                reasoning = self._generate_match_reasoning(user_profile, opp, score)
                
                if score > 0.3:  # Minimum threshold
                    matches.append({
                        'opportunity': opp,
                        'score': round(score, 3),
                        'reasoning': reasoning,
                        'factors': self._get_match_factors(user_profile, opp)
                    })
                    
                    # Store in database for future reference
                    self._store_match(user_id, opp['id'], score, reasoning)
                    
            except Exception as e:
                logger.error(f"Error calculating match for opportunity {opp.get('id')}: {e}")
                continue
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]
    
    def calculate_user_matches(self, user_id, limit=10):
        """Find matching users for networking"""
        user = self.db.get_user(user_id)
        if not user:
            return []
        
        # Get all other users
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM users 
                WHERE id != ? AND status = 'active'
                ORDER BY created_at DESC
            ''', (user_id,))
            
            users = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            users = [dict(zip(columns, row)) for row in users]
            
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return []
        finally:
            conn.close()
        
        if not users:
            return []
        
        matches = []
        user_profile = self._build_user_profile(user)
        
        for other_user in users:
            try:
                other_profile = self._build_user_profile(other_user)
                score = self._calculate_user_match_score(user_profile, other_profile)
                
                if score > 0.4:  # Higher threshold for user matches
                    matches.append({
                        'user': other_user,
                        'score': round(score, 3),
                        'reasoning': self._generate_user_match_reasoning(user_profile, other_profile),
                        'common_interests': self._find_common_interests(user_profile, other_profile)
                    })
                    
            except Exception as e:
                logger.error(f"Error calculating user match for {other_user.get('id')}: {e}")
                continue
        
        # Sort by score and return top matches
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:limit]
    
    def _build_user_profile(self, user):
        """Build comprehensive user profile for matching"""
        # Safely get string values, converting None to empty string
        def safe_str(value):
            return str(value) if value is not None else ''
        
        profile = {
            'id': user['id'],
            'name': user['name'],
            'industry': safe_str(user.get('industry', '')),
            'location': safe_str(user.get('location', '')),
            'skills': safe_str(user.get('skills', '')),
            'interests': safe_str(user.get('interests', '')),
            'bio': safe_str(user.get('bio', '')),
            'company': safe_str(user.get('company', '')),
            'position': safe_str(user.get('position', ''))
        }
        
        # Create combined text for semantic analysis
        profile['combined_text'] = ' '.join([
            profile['industry'],
            profile['skills'],
            profile['interests'],
            profile['bio'],
            profile['position']
        ]).strip()
        
        # Parse lists from comma-separated strings
        profile['skills_list'] = [s.strip().lower() for s in profile['skills'].split(',') if s.strip()]
        profile['interests_list'] = [i.strip().lower() for i in profile['interests'].split(',') if i.strip()]
        
        return profile
    
    def _calculate_match_score(self, user_profile, opportunity):
        """Calculate comprehensive match score between user and opportunity"""
        scores = {}
        
        # Industry match
        scores['industry_match'] = self._calculate_industry_match(
            user_profile['industry'], 
            opportunity.get('industry', '')
        )
        
        # Skills match
        scores['skills_match'] = self._calculate_skills_match(
            user_profile['skills_list'],
            opportunity.get('requirements', ''),
            opportunity.get('tags', '')
        )
        
        # Interests match
        scores['interests_match'] = self._calculate_interests_match(
            user_profile['interests_list'],
            opportunity.get('description', ''),
            opportunity.get('title', '')
        )
        
        # Location proximity
        scores['location_proximity'] = self._calculate_location_match(
            user_profile['location'],
            opportunity.get('location', '')
        )
        
        # Semantic similarity using text analysis
        scores['semantic_similarity'] = self._calculate_semantic_similarity(
            user_profile['combined_text'],
            f"{opportunity.get('title', '')} {opportunity.get('description', '')} {opportunity.get('requirements', '')}"
        )
        
        # Calculate weighted final score
        final_score = sum(
            scores[factor] * self.weights[factor] 
            for factor in scores if factor in self.weights
        )
        
        return min(final_score, 1.0)
    
    def _calculate_user_match_score(self, user1_profile, user2_profile):
        """Calculate match score between two users"""
        scores = {}
        
        # Industry compatibility
        scores['industry_match'] = self._calculate_industry_match(
            user1_profile['industry'], 
            user2_profile['industry']
        )
        
        # Skills complementarity
        scores['skills_match'] = self._calculate_skills_complementarity(
            user1_profile['skills_list'],
            user2_profile['skills_list']
        )
        
        # Interests overlap
        scores['interests_match'] = self._calculate_interests_overlap(
            user1_profile['interests_list'],
            user2_profile['interests_list']
        )
        
        # Location proximity
        scores['location_proximity'] = self._calculate_location_match(
            user1_profile['location'],
            user2_profile['location']
        )
        
        # Semantic similarity
        scores['semantic_similarity'] = self._calculate_semantic_similarity(
            user1_profile['combined_text'],
            user2_profile['combined_text']
        )
        
        # Calculate weighted final score
        final_score = sum(
            scores[factor] * self.weights[factor] 
            for factor in scores if factor in self.weights
        )
        
        return min(final_score, 1.0)
    
    def _calculate_industry_match(self, industry1, industry2):
        """Calculate industry match score"""
        if not industry1 or not industry2:
            return 0.0
        
        industry1 = industry1.lower().strip()
        industry2 = industry2.lower().strip()
        
        if industry1 == industry2:
            return 1.0
        
        # Related industries mapping
        related_industries = {
            'technology': ['software', 'tech', 'it', 'ai', 'data'],
            'healthcare': ['medical', 'health', 'pharma', 'biotech'],
            'finance': ['banking', 'fintech', 'investment', 'insurance'],
            'education': ['training', 'learning', 'academic', 'university']
        }
        
        for main_industry, related in related_industries.items():
            if industry1 in related and industry2 in related:
                return 0.7
            if (industry1 == main_industry and industry2 in related) or \
               (industry2 == main_industry and industry1 in related):
                return 0.8
        
        return 0.0
    
    def _calculate_skills_match(self, user_skills, requirements, tags):
        """Calculate skills match with opportunity requirements"""
        if not user_skills:
            return 0.0
        
        req_text = f"{requirements} {tags}".lower()
        if not req_text.strip():
            return 0.5  # Neutral if no requirements specified
        
        matches = 0
        for skill in user_skills:
            if skill in req_text:
                matches += 1
        
        return min(matches / max(len(user_skills), 1), 1.0)
    
    def _calculate_skills_complementarity(self, skills1, skills2):
        """Calculate how well skills complement each other"""
        if not skills1 or not skills2:
            return 0.0
        
        # Calculate overlap and complementarity
        overlap = len(set(skills1) & set(skills2))
        total_unique = len(set(skills1) | set(skills2))
        
        # Balance between some overlap and complementarity
        if total_unique == 0:
            return 0.0
        
        overlap_ratio = overlap / total_unique
        
        # Optimal overlap is around 30-50%
        if 0.3 <= overlap_ratio <= 0.5:
            return 1.0
        elif overlap_ratio < 0.3:
            return overlap_ratio / 0.3 * 0.8
        else:
            return max(0.0, 1.0 - (overlap_ratio - 0.5) * 2)
    
    def _calculate_interests_match(self, user_interests, description, title):
        """Calculate interests match with opportunity"""
        if not user_interests:
            return 0.0
        
        text = f"{title} {description}".lower()
        if not text.strip():
            return 0.0
        
        matches = 0
        for interest in user_interests:
            if interest in text:
                matches += 1
        
        return min(matches / max(len(user_interests), 1), 1.0)
    
    def _calculate_interests_overlap(self, interests1, interests2):
        """Calculate interests overlap between users"""
        if not interests1 or not interests2:
            return 0.0
        
        overlap = len(set(interests1) & set(interests2))
        total = len(set(interests1) | set(interests2))
        
        return overlap / max(total, 1)
    
    def _calculate_location_match(self, location1, location2):
        """Calculate location proximity score"""
        if not location1 or not location2:
            return 0.5  # Neutral if location not specified
        
        location1 = location1.lower().strip()
        location2 = location2.lower().strip()
        
        if location1 == location2:
            return 1.0
        
        # Simple city/state matching
        if location1 in location2 or location2 in location1:
            return 0.8
        
        # Same country/region heuristics (simplified)
        us_states = ['california', 'new york', 'texas', 'florida', 'illinois']
        for state in us_states:
            if state in location1 and state in location2:
                return 0.6
        
        return 0.2  # Different locations
    
    def _calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using TF-IDF"""
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            # Use TF-IDF for semantic similarity
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _generate_match_reasoning(self, user_profile, opportunity, score):
        """Generate human-readable reasoning for the match"""
        reasons = []
        
        # Industry match
        if user_profile['industry'] and opportunity.get('industry'):
            if user_profile['industry'].lower() == opportunity['industry'].lower():
                reasons.append(f"Perfect industry match ({user_profile['industry']})")
            elif self._calculate_industry_match(user_profile['industry'], opportunity['industry']) > 0.7:
                reasons.append(f"Related industry experience ({user_profile['industry']} → {opportunity['industry']})")
        
        # Skills match
        matching_skills = []
        req_text = f"{opportunity.get('requirements', '')} {opportunity.get('tags', '')}".lower()
        for skill in user_profile['skills_list']:
            if skill in req_text:
                matching_skills.append(skill)
        
        if matching_skills:
            reasons.append(f"Relevant skills: {', '.join(matching_skills[:3])}")
        
        # Location
        if user_profile['location'] and opportunity.get('location'):
            if self._calculate_location_match(user_profile['location'], opportunity['location']) > 0.8:
                reasons.append(f"Local opportunity ({opportunity['location']})")
        
        # Experience level
        if user_profile['position'] and 'senior' in user_profile['position'].lower():
            reasons.append("Senior-level experience")
        
        if not reasons:
            reasons.append("Profile compatibility based on AI analysis")
        
        return "; ".join(reasons)
    
    def _generate_user_match_reasoning(self, user1_profile, user2_profile):
        """Generate reasoning for user-to-user matches"""
        reasons = []
        
        # Common interests
        common_interests = list(set(user1_profile['interests_list']) & set(user2_profile['interests_list']))
        if common_interests:
            reasons.append(f"Shared interests: {', '.join(common_interests[:3])}")
        
        # Industry connection
        if user1_profile['industry'] and user2_profile['industry']:
            if self._calculate_industry_match(user1_profile['industry'], user2_profile['industry']) > 0.7:
                reasons.append(f"Industry connection ({user1_profile['industry']} ↔ {user2_profile['industry']})")
        
        # Complementary skills
        if len(set(user1_profile['skills_list']) - set(user2_profile['skills_list'])) > 0:
            reasons.append("Complementary skill sets")
        
        if not reasons:
            reasons.append("Professional compatibility")
        
        return "; ".join(reasons)
    
    def _get_match_factors(self, user_profile, opportunity):
        """Get detailed match factors as JSON"""
        return json.dumps({
            'industry_score': self._calculate_industry_match(user_profile['industry'], opportunity.get('industry', '')),
            'skills_score': self._calculate_skills_match(user_profile['skills_list'], opportunity.get('requirements', ''), opportunity.get('tags', '')),
            'location_score': self._calculate_location_match(user_profile['location'], opportunity.get('location', '')),
            'interests_score': self._calculate_interests_match(user_profile['interests_list'], opportunity.get('description', ''), opportunity.get('title', ''))
        })
    
    def _find_common_interests(self, user1_profile, user2_profile):
        """Find common interests between users"""
        return list(set(user1_profile['interests_list']) & set(user2_profile['interests_list']))
    
    def _store_match(self, user_id, opportunity_id, score, reasoning):
        """Store match in database for future reference"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        try:
            match_id = f"{user_id}_{opportunity_id}"
            cursor.execute('''
                INSERT OR REPLACE INTO ai_matches (id, user_id, opportunity_id, score, reasoning, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (match_id, user_id, opportunity_id, score, reasoning))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing match: {e}")
            conn.rollback()
        finally:
            conn.close()

if __name__ == "__main__":
    # Test the matching engine
    from database_setup import DatabaseManager
    
    db = DatabaseManager()
    matcher = AIMatchingEngine(db)
    print("AI Matching Engine initialized successfully!")