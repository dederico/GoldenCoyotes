#!/usr/bin/env python3
"""
Opportunity Matcher
Vector similarity matching with scikit-learn for the Business Dealer Intelligence System

This matcher:
- Matches opportunities to users based on multi-dimensional similarity
- Uses vector similarity (cosine, euclidean) for matching
- Implements collaborative filtering for enhanced recommendations
- Scores opportunities based on user preferences and behavior patterns
- Provides real-time matching with caching for performance

Following Task 5 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from .data_models import Opportunity, UserProfile, OpportunityScore, OpportunityType
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


@dataclass
class MatchingCriteria:
    """Criteria for opportunity matching"""
    
    user_id: str
    industry_preferences: List[str]
    location_preferences: List[str]
    budget_range: Tuple[float, float]
    opportunity_types: List[str]
    keyword_preferences: List[str]
    minimum_score: float = 0.5
    max_results: int = 10


@dataclass
class OpportunityMatch:
    """Result of opportunity matching"""
    
    opportunity_id: str
    user_id: str
    overall_score: float
    component_scores: Dict[str, float]
    matching_factors: List[str]
    recommendation_reasons: List[str]
    matched_at: datetime


class OpportunityMatcher:
    """
    Opportunity matching engine using vector similarity and collaborative filtering
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Opportunity Matcher
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Initializing Opportunity Matcher")
        
        # Initialize content processor for embeddings
        self.content_processor = ContentProcessor(config, ml_config)
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Matching metrics
        self.matches_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info("✅ Opportunity Matcher initialized successfully")
    
    def _setup_redis_client(self):
        """Setup Redis client for caching matches"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for matching cache initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for matching cache: {e}")
            self.redis_enabled = False
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for opportunity matching")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_ml_components(self):
        """Setup ML components for matching"""
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Matching weights
        self.matching_weights = {
            'industry_match': 0.25,
            'location_match': 0.15,
            'content_similarity': 0.25,
            'behavior_match': 0.20,
            'network_strength': 0.10,
            'timing_score': 0.05
        }
        
        self.logger.info("✅ ML components initialized for matching")
    
    async def find_matching_opportunities(
        self,
        user_id: str,
        criteria: MatchingCriteria = None,
        use_cache: bool = True
    ) -> List[OpportunityMatch]:
        """
        Find matching opportunities for a user
        
        Args:
            user_id: ID of the user
            criteria: Optional matching criteria
            use_cache: Whether to use caching
            
        Returns:
            List of opportunity matches
        """
        try:
            self.logger.info(f"🔍 Finding matching opportunities for user {user_id}")
            
            # Check cache first
            if use_cache:
                cached_matches = await self._get_cached_matches(user_id)
                if cached_matches:
                    self.cache_hits += 1
                    self.logger.info(f"🎯 Using cached matches for user {user_id}")
                    return cached_matches
            
            self.cache_misses += 1
            
            # Get user profile and preferences
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                self.logger.warning(f"⚠️ No profile found for user {user_id}")
                return []
            
            # Use provided criteria or generate from user profile
            if criteria is None:
                criteria = await self._generate_matching_criteria(user_id, user_profile)
            
            # Get candidate opportunities
            candidate_opportunities = await self._get_candidate_opportunities(criteria)
            
            if not candidate_opportunities:
                self.logger.info(f"ℹ️ No candidate opportunities found for user {user_id}")
                return []
            
            # Calculate similarity scores
            matches = []
            for opportunity in candidate_opportunities:
                match_score = await self._calculate_match_score(
                    user_id, user_profile, opportunity, criteria
                )
                
                if match_score.overall_score >= criteria.minimum_score:
                    matches.append(match_score)
            
            # Sort by score and limit results
            matches.sort(key=lambda x: x.overall_score, reverse=True)
            matches = matches[:criteria.max_results]
            
            # Cache results
            if use_cache:
                await self._cache_matches(user_id, matches)
            
            # Store matches in database
            await self._store_matches_in_db(matches)
            
            self.matches_generated += len(matches)
            self.logger.info(f"✅ Found {len(matches)} matching opportunities for user {user_id}")
            
            return matches
            
        except Exception as e:
            self.logger.error(f"❌ Error finding matching opportunities: {e}")
            return []
    
    async def score_opportunity_for_user(
        self,
        user_id: str,
        opportunity_id: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """
        Score a specific opportunity for a user
        
        Args:
            user_id: ID of the user
            opportunity_id: ID of the opportunity
            detailed: Whether to return detailed scoring breakdown
            
        Returns:
            Dictionary containing score and details
        """
        try:
            self.logger.info(f"📊 Scoring opportunity {opportunity_id} for user {user_id}")
            
            # Get user profile and opportunity
            user_profile = await self._get_user_profile(user_id)
            opportunity = await self._get_opportunity(opportunity_id)
            
            if not user_profile or not opportunity:
                return {"error": "User profile or opportunity not found"}
            
            # Generate basic criteria
            criteria = await self._generate_matching_criteria(user_id, user_profile)
            
            # Calculate detailed match score
            match_score = await self._calculate_match_score(
                user_id, user_profile, opportunity, criteria
            )
            
            # Format result
            result = {
                "user_id": user_id,
                "opportunity_id": opportunity_id,
                "overall_score": match_score.overall_score,
                "scored_at": datetime.now().isoformat()
            }
            
            if detailed:
                result.update({
                    "component_scores": match_score.component_scores,
                    "matching_factors": match_score.matching_factors,
                    "recommendation_reasons": match_score.recommendation_reasons
                })
            
            self.logger.info(f"✅ Opportunity scored: {match_score.overall_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring opportunity: {e}")
            return {"error": str(e)}
    
    async def get_collaborative_matches(
        self,
        user_id: str,
        max_results: int = 10
    ) -> List[OpportunityMatch]:
        """
        Get matches using collaborative filtering
        
        Args:
            user_id: ID of the user
            max_results: Maximum number of results
            
        Returns:
            List of collaborative matches
        """
        try:
            self.logger.info(f"👥 Getting collaborative matches for user {user_id}")
            
            # Find similar users
            similar_users = await self._find_similar_users(user_id)
            
            if not similar_users:
                self.logger.info(f"ℹ️ No similar users found for {user_id}")
                return []
            
            # Get opportunities liked by similar users
            collaborative_opportunities = await self._get_collaborative_opportunities(
                user_id, similar_users
            )
            
            # Score and rank opportunities
            matches = []
            for opportunity in collaborative_opportunities:
                user_profile = await self._get_user_profile(user_id)
                criteria = await self._generate_matching_criteria(user_id, user_profile)
                
                match_score = await self._calculate_match_score(
                    user_id, user_profile, opportunity, criteria
                )
                
                # Boost score for collaborative filtering
                match_score.overall_score *= 1.1
                match_score.matching_factors.append("collaborative_filtering")
                match_score.recommendation_reasons.append("Liked by similar users")
                
                matches.append(match_score)
            
            # Sort and limit results
            matches.sort(key=lambda x: x.overall_score, reverse=True)
            matches = matches[:max_results]
            
            self.logger.info(f"✅ Found {len(matches)} collaborative matches")
            return matches
            
        except Exception as e:
            self.logger.error(f"❌ Error getting collaborative matches: {e}")
            return []
    
    async def _calculate_match_score(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        opportunity: Dict[str, Any],
        criteria: MatchingCriteria
    ) -> OpportunityMatch:
        """Calculate comprehensive match score"""
        try:
            component_scores = {}
            matching_factors = []
            recommendation_reasons = []
            
            # 1. Industry match
            industry_score = self._calculate_industry_match(
                user_profile.get('industry', ''),
                opportunity.get('industry', ''),
                criteria.industry_preferences
            )
            component_scores['industry_match'] = industry_score
            
            if industry_score > 0.7:
                matching_factors.append('industry')
                recommendation_reasons.append(f"Strong industry match: {opportunity.get('industry', '')}")
            
            # 2. Location match
            location_score = self._calculate_location_match(
                user_profile.get('location', ''),
                opportunity.get('location', ''),
                criteria.location_preferences
            )
            component_scores['location_match'] = location_score
            
            if location_score > 0.6:
                matching_factors.append('location')
                recommendation_reasons.append(f"Location match: {opportunity.get('location', '')}")
            
            # 3. Content similarity
            content_score = await self._calculate_content_similarity(
                user_profile, opportunity
            )
            component_scores['content_similarity'] = content_score
            
            if content_score > 0.6:
                matching_factors.append('content')
                recommendation_reasons.append("High content similarity to your interests")
            
            # 4. Behavior match
            behavior_score = await self._calculate_behavior_match(user_id, opportunity)
            component_scores['behavior_match'] = behavior_score
            
            if behavior_score > 0.5:
                matching_factors.append('behavior')
                recommendation_reasons.append("Matches your interaction patterns")
            
            # 5. Network strength
            network_score = await self._calculate_network_strength(user_id, opportunity)
            component_scores['network_strength'] = network_score
            
            if network_score > 0.4:
                matching_factors.append('network')
                recommendation_reasons.append("Strong network connections")
            
            # 6. Timing score
            timing_score = self._calculate_timing_score(opportunity)
            component_scores['timing_score'] = timing_score
            
            if timing_score > 0.6:
                matching_factors.append('timing')
                recommendation_reasons.append("Great timing for this opportunity")
            
            # Calculate weighted overall score
            overall_score = sum(
                component_scores.get(component, 0) * weight
                for component, weight in self.matching_weights.items()
            )
            
            return OpportunityMatch(
                opportunity_id=opportunity['id'],
                user_id=user_id,
                overall_score=overall_score,
                component_scores=component_scores,
                matching_factors=matching_factors,
                recommendation_reasons=recommendation_reasons,
                matched_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating match score: {e}")
            return OpportunityMatch(
                opportunity_id=opportunity.get('id', ''),
                user_id=user_id,
                overall_score=0.0,
                component_scores={},
                matching_factors=[],
                recommendation_reasons=[f"Error in scoring: {str(e)}"],
                matched_at=datetime.now()
            )
    
    def _calculate_industry_match(
        self,
        user_industry: str,
        opportunity_industry: str,
        preferred_industries: List[str]
    ) -> float:
        """Calculate industry match score"""
        if not user_industry or not opportunity_industry:
            return 0.0
        
        # Exact match
        if user_industry.lower() == opportunity_industry.lower():
            return 1.0
        
        # Preference match
        if opportunity_industry.lower() in [ind.lower() for ind in preferred_industries]:
            return 0.8
        
        # Related industry match (simplified)
        industry_relations = {
            'technology': ['software', 'tech', 'it', 'digital'],
            'finance': ['banking', 'fintech', 'financial'],
            'healthcare': ['medical', 'pharma', 'health'],
            'retail': ['ecommerce', 'commerce', 'sales']
        }
        
        for base_industry, related in industry_relations.items():
            if (user_industry.lower() in related and 
                opportunity_industry.lower() in related):
                return 0.6
        
        return 0.2  # Default low score for different industries
    
    def _calculate_location_match(
        self,
        user_location: str,
        opportunity_location: str,
        preferred_locations: List[str]
    ) -> float:
        """Calculate location match score"""
        if not user_location or not opportunity_location:
            return 0.5  # Neutral score for missing location
        
        # Exact match
        if user_location.lower() == opportunity_location.lower():
            return 1.0
        
        # Preference match
        if opportunity_location.lower() in [loc.lower() for loc in preferred_locations]:
            return 0.8
        
        # Same city/region match (simplified)
        user_parts = user_location.lower().split(',')
        opp_parts = opportunity_location.lower().split(',')
        
        if len(user_parts) > 1 and len(opp_parts) > 1:
            if user_parts[0].strip() == opp_parts[0].strip():  # Same city
                return 0.7
            if user_parts[-1].strip() == opp_parts[-1].strip():  # Same country/state
                return 0.4
        
        return 0.1  # Different locations
    
    async def _calculate_content_similarity(
        self,
        user_profile: Dict[str, Any],
        opportunity: Dict[str, Any]
    ) -> float:
        """Calculate content similarity using embeddings"""
        try:
            # Get user interests and preferences
            user_interests = user_profile.get('interests', [])
            user_preferences = user_profile.get('preferences', {})
            
            # Create user content representation
            user_content = " ".join(user_interests) + " " + str(user_preferences)
            
            # Get opportunity content
            opp_content = f"{opportunity.get('title', '')} {opportunity.get('description', '')}"
            
            # Generate embeddings
            user_embedding = await self.content_processor.generate_embedding(
                user_content, "user_profile"
            )
            
            opp_embedding = await self.content_processor.generate_embedding(
                opp_content, "opportunity"
            )
            
            if "error" in user_embedding or "error" in opp_embedding:
                return 0.5  # Default score on error
            
            # Calculate cosine similarity
            user_vector = np.array(user_embedding["embedding_vector"]).reshape(1, -1)
            opp_vector = np.array(opp_embedding["embedding_vector"]).reshape(1, -1)
            
            similarity = cosine_similarity(user_vector, opp_vector)[0][0]
            
            return max(0.0, min(1.0, similarity))  # Ensure 0-1 range
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating content similarity: {e}")
            return 0.5
    
    async def _calculate_behavior_match(
        self,
        user_id: str,
        opportunity: Dict[str, Any]
    ) -> float:
        """Calculate behavior match score"""
        try:
            # Get user behavior patterns
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user interactions
            cursor.execute("""
                SELECT interaction_type, COUNT(*) as count
                FROM user_interactions
                WHERE user_id = ? AND timestamp > ?
                GROUP BY interaction_type
            """, (user_id, (datetime.now() - timedelta(days=30)).isoformat()))
            
            interactions = cursor.fetchall()
            conn.close()
            
            if not interactions:
                return 0.3  # Default score for new users
            
            # Calculate engagement patterns
            total_interactions = sum(count for _, count in interactions)
            engagement_score = min(total_interactions / 100, 1.0)  # Normalize
            
            # Check for relevant interaction types
            relevant_types = ['contact', 'save', 'share']
            relevant_interactions = sum(
                count for interaction_type, count in interactions
                if interaction_type in relevant_types
            )
            
            conversion_tendency = relevant_interactions / max(total_interactions, 1)
            
            # Combine scores
            behavior_score = (engagement_score * 0.6) + (conversion_tendency * 0.4)
            
            return behavior_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating behavior match: {e}")
            return 0.3
    
    async def _calculate_network_strength(
        self,
        user_id: str,
        opportunity: Dict[str, Any]
    ) -> float:
        """Calculate network strength score"""
        try:
            # Get user network connections
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get network analytics
            cursor.execute("""
                SELECT AVG(connection_strength), COUNT(*)
                FROM network_analytics
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result or result[0] is None:
                return 0.2  # Default score for users without network data
            
            avg_strength, connection_count = result
            
            # Calculate network score
            strength_score = min(avg_strength, 1.0)
            activity_score = min(connection_count / 50, 1.0)  # Normalize
            
            network_score = (strength_score * 0.7) + (activity_score * 0.3)
            
            return network_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating network strength: {e}")
            return 0.2
    
    def _calculate_timing_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate timing score for opportunity"""
        try:
            # Get opportunity timing information
            created_at = opportunity.get('created_at')
            deadline = opportunity.get('deadline')
            
            if not created_at:
                return 0.5
            
            # Parse dates
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            current_date = datetime.now()
            
            # Calculate freshness score
            days_since_creation = (current_date - created_date).days
            freshness_score = max(0, 1 - (days_since_creation / 30))  # Decay over 30 days
            
            # Calculate urgency score
            urgency_score = 0.5
            if deadline:
                deadline_date = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
                days_until_deadline = (deadline_date - current_date).days
                
                if days_until_deadline > 0:
                    urgency_score = min(1.0, 30 / days_until_deadline)  # Higher score for closer deadlines
            
            # Combine scores
            timing_score = (freshness_score * 0.6) + (urgency_score * 0.4)
            
            return timing_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating timing score: {e}")
            return 0.5
    
    # Helper methods for data retrieval
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user profile
            cursor.execute("""
                SELECT user_id, industry, location, company_size, job_role, 
                       interests, preferences, engagement_score, last_active
                FROM user_profiles
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'user_id': result[0],
                    'industry': result[1],
                    'location': result[2],
                    'company_size': result[3],
                    'job_role': result[4],
                    'interests': json.loads(result[5] or '[]'),
                    'preferences': json.loads(result[6] or '{}'),
                    'engagement_score': result[7],
                    'last_active': result[8]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user profile: {e}")
            return None
    
    async def _get_opportunity(self, opportunity_id: str) -> Optional[Dict[str, Any]]:
        """Get opportunity from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get opportunity
            cursor.execute("""
                SELECT id, title, description, opportunity_type, industry, 
                       location, budget_range, deadline, contact_info, tags, 
                       status, created_at, updated_at
                FROM opportunities
                WHERE id = ?
            """, (opportunity_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'title': result[1],
                    'description': result[2],
                    'opportunity_type': result[3],
                    'industry': result[4],
                    'location': result[5],
                    'budget_range': result[6],
                    'deadline': result[7],
                    'contact_info': json.loads(result[8] or '{}'),
                    'tags': json.loads(result[9] or '[]'),
                    'status': result[10],
                    'created_at': result[11],
                    'updated_at': result[12]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting opportunity: {e}")
            return None
    
    async def _generate_matching_criteria(
        self,
        user_id: str,
        user_profile: Dict[str, Any]
    ) -> MatchingCriteria:
        """Generate matching criteria from user profile"""
        return MatchingCriteria(
            user_id=user_id,
            industry_preferences=user_profile.get('interests', []),
            location_preferences=[user_profile.get('location', '')],
            budget_range=(0.0, float('inf')),
            opportunity_types=['buyer', 'seller', 'service', 'partnership'],
            keyword_preferences=user_profile.get('interests', []),
            minimum_score=0.3,
            max_results=20
        )
    
    async def _get_candidate_opportunities(
        self,
        criteria: MatchingCriteria
    ) -> List[Dict[str, Any]]:
        """Get candidate opportunities based on criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic query for active opportunities
            cursor.execute("""
                SELECT id, title, description, opportunity_type, industry, 
                       location, budget_range, deadline, contact_info, tags, 
                       status, created_at, updated_at
                FROM opportunities
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT 100
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            opportunities = []
            for result in results:
                opportunities.append({
                    'id': result[0],
                    'title': result[1],
                    'description': result[2],
                    'opportunity_type': result[3],
                    'industry': result[4],
                    'location': result[5],
                    'budget_range': result[6],
                    'deadline': result[7],
                    'contact_info': json.loads(result[8] or '{}'),
                    'tags': json.loads(result[9] or '[]'),
                    'status': result[10],
                    'created_at': result[11],
                    'updated_at': result[12]
                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error getting candidate opportunities: {e}")
            return []
    
    async def _find_similar_users(self, user_id: str) -> List[str]:
        """Find similar users for collaborative filtering"""
        try:
            # Get user behavior patterns
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get users with similar interaction patterns
            cursor.execute("""
                SELECT DISTINCT u2.user_id
                FROM user_interactions u1
                JOIN user_interactions u2 ON u1.opportunity_id = u2.opportunity_id
                WHERE u1.user_id = ? AND u2.user_id != ?
                GROUP BY u2.user_id
                HAVING COUNT(*) > 2
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """, (user_id, user_id))
            
            similar_users = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return similar_users
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar users: {e}")
            return []
    
    async def _get_collaborative_opportunities(
        self,
        user_id: str,
        similar_users: List[str]
    ) -> List[Dict[str, Any]]:
        """Get opportunities liked by similar users"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get opportunities with positive interactions from similar users
            placeholders = ','.join(['?'] * len(similar_users))
            cursor.execute(f"""
                SELECT DISTINCT o.id, o.title, o.description, o.opportunity_type, 
                       o.industry, o.location, o.budget_range, o.deadline, 
                       o.contact_info, o.tags, o.status, o.created_at, o.updated_at
                FROM opportunities o
                JOIN user_interactions ui ON o.id = ui.opportunity_id
                WHERE ui.user_id IN ({placeholders})
                AND ui.interaction_type IN ('contact', 'save', 'share')
                AND o.id NOT IN (
                    SELECT opportunity_id FROM user_interactions 
                    WHERE user_id = ?
                )
                AND o.status = 'active'
                ORDER BY o.created_at DESC
                LIMIT 50
            """, similar_users + [user_id])
            
            results = cursor.fetchall()
            conn.close()
            
            opportunities = []
            for result in results:
                opportunities.append({
                    'id': result[0],
                    'title': result[1],
                    'description': result[2],
                    'opportunity_type': result[3],
                    'industry': result[4],
                    'location': result[5],
                    'budget_range': result[6],
                    'deadline': result[7],
                    'contact_info': json.loads(result[8] or '{}'),
                    'tags': json.loads(result[9] or '[]'),
                    'status': result[10],
                    'created_at': result[11],
                    'updated_at': result[12]
                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error getting collaborative opportunities: {e}")
            return []
    
    # Cache management
    async def _get_cached_matches(self, user_id: str) -> Optional[List[OpportunityMatch]]:
        """Get cached matches from Redis"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"matches:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                matches = []
                
                for match_data in data:
                    match = OpportunityMatch(
                        opportunity_id=match_data['opportunity_id'],
                        user_id=match_data['user_id'],
                        overall_score=match_data['overall_score'],
                        component_scores=match_data['component_scores'],
                        matching_factors=match_data['matching_factors'],
                        recommendation_reasons=match_data['recommendation_reasons'],
                        matched_at=datetime.fromisoformat(match_data['matched_at'])
                    )
                    matches.append(match)
                
                return matches
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_matches(self, user_id: str, matches: List[OpportunityMatch]):
        """Cache matches in Redis"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"matches:{user_id}"
            ttl = self.config.redis.recommendation_cache_ttl
            
            # Serialize matches
            serialized_matches = []
            for match in matches:
                serialized_matches.append({
                    'opportunity_id': match.opportunity_id,
                    'user_id': match.user_id,
                    'overall_score': match.overall_score,
                    'component_scores': match.component_scores,
                    'matching_factors': match.matching_factors,
                    'recommendation_reasons': match.recommendation_reasons,
                    'matched_at': match.matched_at.isoformat()
                })
            
            # Store in cache
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(serialized_matches)
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    async def _store_matches_in_db(self, matches: List[OpportunityMatch]):
        """Store matches in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for match in matches:
                # Create OpportunityScore record
                cursor.execute("""
                    INSERT OR REPLACE INTO opportunity_scores 
                    (id, opportunity_id, user_id, relevance_score, success_probability, 
                     factors, calculated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"score_{match.user_id}_{match.opportunity_id}",
                    match.opportunity_id,
                    match.user_id,
                    match.overall_score,
                    match.overall_score * 0.8,  # Approximation
                    json.dumps(match.component_scores),
                    match.matched_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing matches in database: {e}")
    
    # Status and management methods
    def get_matcher_status(self) -> Dict[str, Any]:
        """Get opportunity matcher status"""
        return {
            "status": "operational",
            "matches_generated": self.matches_generated,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "matching_weights": self.matching_weights,
            "configuration": {
                "redis_enabled": self.redis_enabled,
                "ml_components": list(self.matching_weights.keys())
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_matching_cache(self, user_id: str = None) -> int:
        """Clear matching cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if user_id:
                pattern = f"matches:{user_id}"
            else:
                pattern = "matches:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing matching cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the opportunity matcher
    async def test_opportunity_matcher():
        print("🎯 Testing Opportunity Matcher")
        print("=" * 50)
        
        try:
            matcher = OpportunityMatcher()
            
            # Test finding matches
            matches = await matcher.find_matching_opportunities("test_user_123")
            print(f"Matches Found: {len(matches)}")
            
            # Test scoring specific opportunity
            score_result = await matcher.score_opportunity_for_user(
                "test_user_123", 
                "test_opp_456", 
                detailed=True
            )
            print(f"Score Result: {score_result}")
            
            # Test collaborative matches
            collaborative_matches = await matcher.get_collaborative_matches("test_user_123")
            print(f"Collaborative Matches: {len(collaborative_matches)}")
            
            # Test status
            status = matcher.get_matcher_status()
            print(f"Matcher Status: {status}")
            
            print("✅ Opportunity Matcher test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_opportunity_matcher())