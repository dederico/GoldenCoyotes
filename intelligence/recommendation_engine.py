#!/usr/bin/env python3
"""
Recommendation Engine
Personalized recommendations with collaborative filtering for the Business Dealer Intelligence System

This engine:
- Generates personalized recommendations using multiple algorithms
- Implements collaborative filtering and content-based filtering
- Provides real-time recommendation generation and ranking
- Tracks recommendation performance and user feedback
- Handles cold start problems for new users

Following Task 6 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from .data_models import Recommendation, RecommendationType, UserProfile, Opportunity
from .behavior_analyzer import BehaviorAnalyzer
from .opportunity_matcher import OpportunityMatcher
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    """Request for recommendations"""
    
    user_id: str
    max_recommendations: int = 10
    recommendation_types: List[str] = None
    exclude_seen: bool = True
    include_reasons: bool = True
    context: Dict[str, Any] = None


@dataclass
class RecommendationResult:
    """Result of recommendation generation"""
    
    user_id: str
    recommendations: List[Recommendation]
    total_generated: int
    generation_time_ms: int
    algorithms_used: List[str]
    metadata: Dict[str, Any]
    generated_at: datetime


class RecommendationEngine:
    """
    Recommendation engine that generates personalized recommendations using multiple algorithms
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Recommendation Engine
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Initializing Recommendation Engine")
        
        # Initialize component engines
        self.behavior_analyzer = BehaviorAnalyzer(config, ml_config)
        self.opportunity_matcher = OpportunityMatcher(config, ml_config)
        self.content_processor = ContentProcessor(config, ml_config)
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Recommendation algorithms and their weights
        self.algorithm_weights = {
            'collaborative_filtering': 0.30,
            'content_based': 0.25,
            'behavior_based': 0.20,
            'trending': 0.15,
            'location_based': 0.10
        }
        
        # Performance metrics
        self.recommendations_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.feedback_received = 0
        
        self.logger.info("✅ Recommendation Engine initialized successfully")
    
    def _setup_redis_client(self):
        """Setup Redis client for caching recommendations"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for recommendation caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for recommendation caching: {e}")
            self.redis_enabled = False
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for recommendations")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_ml_components(self):
        """Setup ML components for recommendations"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.svd_model = TruncatedSVD(
            n_components=50,
            random_state=self.ml_config.random_seed
        )
        
        # User-item interaction matrix (will be populated dynamically)
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        
        self.logger.info("✅ ML components initialized for recommendations")
    
    async def generate_recommendations(
        self,
        request: RecommendationRequest,
        use_cache: bool = True
    ) -> RecommendationResult:
        """
        Generate personalized recommendations
        
        Args:
            request: Recommendation request
            use_cache: Whether to use caching
            
        Returns:
            RecommendationResult with generated recommendations
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"🎯 Generating recommendations for user {request.user_id}")
            
            # Check cache first
            if use_cache:
                cached_recommendations = await self._get_cached_recommendations(request.user_id)
                if cached_recommendations:
                    self.cache_hits += 1
                    self.logger.info(f"🎯 Using cached recommendations for user {request.user_id}")
                    return cached_recommendations
            
            self.cache_misses += 1
            
            # Get user profile and behavior
            user_profile = await self._get_user_profile(request.user_id)
            user_behavior = await self._get_user_behavior(request.user_id)
            
            # Initialize recommendation lists
            all_recommendations = []
            algorithms_used = []
            
            # 1. Collaborative Filtering Recommendations
            if user_behavior and len(user_behavior) > 3:
                collab_recs = await self._generate_collaborative_recommendations(
                    request.user_id, user_behavior, request.max_recommendations
                )
                all_recommendations.extend(collab_recs)
                algorithms_used.append('collaborative_filtering')
            
            # 2. Content-Based Recommendations
            if user_profile:
                content_recs = await self._generate_content_based_recommendations(
                    request.user_id, user_profile, request.max_recommendations
                )
                all_recommendations.extend(content_recs)
                algorithms_used.append('content_based')
            
            # 3. Behavior-Based Recommendations
            if user_behavior:
                behavior_recs = await self._generate_behavior_based_recommendations(
                    request.user_id, user_behavior, request.max_recommendations
                )
                all_recommendations.extend(behavior_recs)
                algorithms_used.append('behavior_based')
            
            # 4. Trending Recommendations
            trending_recs = await self._generate_trending_recommendations(
                request.user_id, request.max_recommendations
            )
            all_recommendations.extend(trending_recs)
            algorithms_used.append('trending')
            
            # 5. Location-Based Recommendations
            if user_profile and user_profile.get('location'):
                location_recs = await self._generate_location_based_recommendations(
                    request.user_id, user_profile.get('location'), request.max_recommendations
                )
                all_recommendations.extend(location_recs)
                algorithms_used.append('location_based')
            
            # 6. Handle cold start (new users)
            if not all_recommendations:
                cold_start_recs = await self._generate_cold_start_recommendations(
                    request.user_id, request.max_recommendations
                )
                all_recommendations.extend(cold_start_recs)
                algorithms_used.append('cold_start')
            
            # Combine and rank recommendations
            final_recommendations = await self._combine_and_rank_recommendations(
                all_recommendations, request
            )
            
            # Filter seen recommendations if requested
            if request.exclude_seen:
                final_recommendations = await self._filter_seen_recommendations(
                    request.user_id, final_recommendations
                )
            
            # Limit to requested number
            final_recommendations = final_recommendations[:request.max_recommendations]
            
            # Store recommendations in database
            await self._store_recommendations(final_recommendations)
            
            # Create result
            end_time = datetime.now()
            generation_time = int((end_time - start_time).total_seconds() * 1000)
            
            result = RecommendationResult(
                user_id=request.user_id,
                recommendations=final_recommendations,
                total_generated=len(final_recommendations),
                generation_time_ms=generation_time,
                algorithms_used=algorithms_used,
                metadata={
                    'user_profile_available': user_profile is not None,
                    'user_behavior_available': user_behavior is not None,
                    'cache_used': False,
                    'total_candidates': len(all_recommendations)
                },
                generated_at=datetime.now()
            )
            
            # Cache the result
            if use_cache:
                await self._cache_recommendations(request.user_id, result)
            
            self.recommendations_generated += len(final_recommendations)
            self.logger.info(f"✅ Generated {len(final_recommendations)} recommendations for user {request.user_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recommendations: {e}")
            return RecommendationResult(
                user_id=request.user_id,
                recommendations=[],
                total_generated=0,
                generation_time_ms=0,
                algorithms_used=[],
                metadata={'error': str(e)},
                generated_at=datetime.now()
            )
    
    async def _generate_collaborative_recommendations(
        self,
        user_id: str,
        user_behavior: List[Dict[str, Any]],
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate collaborative filtering recommendations"""
        try:
            self.logger.info(f"👥 Generating collaborative recommendations for user {user_id}")
            
            # Get user-item interaction matrix
            user_item_matrix = await self._build_user_item_matrix()
            
            if user_item_matrix is None or user_item_matrix.empty:
                return []
            
            # Find similar users
            similar_users = await self._find_similar_users_collaborative(
                user_id, user_item_matrix
            )
            
            if not similar_users:
                return []
            
            # Get items liked by similar users
            recommendations = []
            for similar_user_id, similarity_score in similar_users:
                similar_user_items = await self._get_user_positive_interactions(similar_user_id)
                
                for item_id, interaction_score in similar_user_items:
                    # Skip if user already interacted with this item
                    if await self._user_has_interacted(user_id, item_id):
                        continue
                    
                    # Calculate recommendation score
                    score = similarity_score * interaction_score * 0.8  # Collaborative weight
                    
                    recommendation = Recommendation(
                        user_id=user_id,
                        opportunity_id=item_id,
                        recommendation_type=RecommendationType.SIMILAR_USERS,
                        score=score,
                        reasoning=f"Recommended based on similar user preferences (similarity: {similarity_score:.2f})"
                    )
                    
                    recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating collaborative recommendations: {e}")
            return []
    
    async def _generate_content_based_recommendations(
        self,
        user_id: str,
        user_profile: Dict[str, Any],
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate content-based recommendations"""
        try:
            self.logger.info(f"📄 Generating content-based recommendations for user {user_id}")
            
            # Create user content profile
            user_interests = user_profile.get('interests', [])
            user_preferences = user_profile.get('preferences', {})
            user_industry = user_profile.get('industry', '')
            
            user_content = f"{' '.join(user_interests)} {user_industry} {str(user_preferences)}"
            
            # Get user content embedding
            user_embedding = await self.content_processor.generate_embedding(
                user_content, "user_profile"
            )
            
            if "error" in user_embedding:
                return []
            
            # Find similar content
            similar_content = await self.content_processor.find_similar_content(
                user_content, "opportunity", max_results=max_recommendations * 2
            )
            
            recommendations = []
            for content_match in similar_content:
                # Get opportunity details
                opportunity = await self._get_opportunity(content_match['content_id'])
                
                if opportunity:
                    recommendation = Recommendation(
                        user_id=user_id,
                        opportunity_id=content_match['content_id'],
                        recommendation_type=RecommendationType.PERSONALIZED,
                        score=content_match['similarity_score'] * 0.7,  # Content-based weight
                        reasoning=f"Recommended based on your interests and preferences (similarity: {content_match['similarity_score']:.2f})"
                    )
                    
                    recommendations.append(recommendation)
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating content-based recommendations: {e}")
            return []
    
    async def _generate_behavior_based_recommendations(
        self,
        user_id: str,
        user_behavior: List[Dict[str, Any]],
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate behavior-based recommendations"""
        try:
            self.logger.info(f"🎭 Generating behavior-based recommendations for user {user_id}")
            
            # Analyze user behavior patterns
            behavior_analysis = await self.behavior_analyzer.analyze_user_patterns(user_id)
            
            if "error" in behavior_analysis:
                return []
            
            # Get behavior insights
            preferred_categories = behavior_analysis.get('behavior_pattern', {}).get('preferred_categories', [])
            peak_hours = behavior_analysis.get('behavior_pattern', {}).get('peak_activity_hours', [])
            
            # Find opportunities matching behavior patterns
            opportunities = await self._find_opportunities_by_behavior(
                preferred_categories, peak_hours
            )
            
            recommendations = []
            for opportunity in opportunities:
                # Calculate behavior match score
                score = self._calculate_behavior_match_score(
                    behavior_analysis, opportunity
                )
                
                recommendation = Recommendation(
                    user_id=user_id,
                    opportunity_id=opportunity['id'],
                    recommendation_type=RecommendationType.BEHAVIORAL_PATTERN,
                    score=score,
                    reasoning=f"Recommended based on your behavior patterns and activity times"
                )
                
                recommendations.append(recommendation)
            
            # Sort by score and return top recommendations
            recommendations.sort(key=lambda x: x.score, reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating behavior-based recommendations: {e}")
            return []
    
    async def _generate_trending_recommendations(
        self,
        user_id: str,
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate trending recommendations"""
        try:
            self.logger.info(f"📈 Generating trending recommendations for user {user_id}")
            
            # Get trending opportunities based on recent interactions
            trending_opportunities = await self._get_trending_opportunities()
            
            recommendations = []
            for opportunity in trending_opportunities:
                # Calculate trending score
                score = self._calculate_trending_score(opportunity)
                
                recommendation = Recommendation(
                    user_id=user_id,
                    opportunity_id=opportunity['id'],
                    recommendation_type=RecommendationType.TRENDING,
                    score=score,
                    reasoning=f"Trending opportunity with high engagement from other users"
                )
                
                recommendations.append(recommendation)
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trending recommendations: {e}")
            return []
    
    async def _generate_location_based_recommendations(
        self,
        user_id: str,
        user_location: str,
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate location-based recommendations"""
        try:
            self.logger.info(f"📍 Generating location-based recommendations for user {user_id}")
            
            # Get opportunities in user's location
            location_opportunities = await self._get_opportunities_by_location(user_location)
            
            recommendations = []
            for opportunity in location_opportunities:
                # Calculate location match score
                score = self._calculate_location_match_score(user_location, opportunity)
                
                recommendation = Recommendation(
                    user_id=user_id,
                    opportunity_id=opportunity['id'],
                    recommendation_type=RecommendationType.LOCATION_BASED,
                    score=score,
                    reasoning=f"Recommended based on your location: {user_location}"
                )
                
                recommendations.append(recommendation)
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating location-based recommendations: {e}")
            return []
    
    async def _generate_cold_start_recommendations(
        self,
        user_id: str,
        max_recommendations: int
    ) -> List[Recommendation]:
        """Generate recommendations for new users (cold start)"""
        try:
            self.logger.info(f"🆕 Generating cold start recommendations for user {user_id}")
            
            # Get popular opportunities
            popular_opportunities = await self._get_popular_opportunities()
            
            recommendations = []
            for opportunity in popular_opportunities:
                # Use popularity score
                score = 0.5  # Default score for cold start
                
                recommendation = Recommendation(
                    user_id=user_id,
                    opportunity_id=opportunity['id'],
                    recommendation_type=RecommendationType.TRENDING,
                    score=score,
                    reasoning="Popular opportunity recommended for new users"
                )
                
                recommendations.append(recommendation)
            
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error(f"❌ Error generating cold start recommendations: {e}")
            return []
    
    async def _combine_and_rank_recommendations(
        self,
        all_recommendations: List[Recommendation],
        request: RecommendationRequest
    ) -> List[Recommendation]:
        """Combine and rank recommendations from different algorithms"""
        try:
            # Group recommendations by opportunity_id
            opportunity_recommendations = {}
            
            for rec in all_recommendations:
                if rec.opportunity_id not in opportunity_recommendations:
                    opportunity_recommendations[rec.opportunity_id] = []
                opportunity_recommendations[rec.opportunity_id].append(rec)
            
            # Combine scores for each opportunity
            combined_recommendations = []
            
            for opportunity_id, recs in opportunity_recommendations.items():
                # Calculate combined score
                combined_score = 0
                algorithm_scores = {}
                reasoning_parts = []
                
                for rec in recs:
                    algorithm_type = rec.recommendation_type.value
                    weight = self.algorithm_weights.get(algorithm_type, 0.1)
                    
                    combined_score += rec.score * weight
                    algorithm_scores[algorithm_type] = rec.score
                    reasoning_parts.append(rec.reasoning)
                
                # Create combined recommendation
                combined_rec = Recommendation(
                    user_id=request.user_id,
                    opportunity_id=opportunity_id,
                    recommendation_type=RecommendationType.PERSONALIZED,
                    score=combined_score,
                    reasoning=" | ".join(reasoning_parts)
                )
                
                combined_recommendations.append(combined_rec)
            
            # Sort by combined score
            combined_recommendations.sort(key=lambda x: x.score, reverse=True)
            
            return combined_recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error combining recommendations: {e}")
            return all_recommendations
    
    async def record_recommendation_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        feedback_type: str,
        feedback_value: Any = None
    ) -> bool:
        """
        Record feedback on a recommendation
        
        Args:
            user_id: ID of the user
            recommendation_id: ID of the recommendation
            feedback_type: Type of feedback (click, dismiss, rate, etc.)
            feedback_value: Optional feedback value
            
        Returns:
            True if feedback was recorded successfully
        """
        try:
            self.logger.info(f"📝 Recording feedback for recommendation {recommendation_id}")
            
            # Update recommendation in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if feedback_type == 'click':
                cursor.execute("""
                    UPDATE recommendations 
                    SET clicked_at = ? 
                    WHERE id = ? AND user_id = ?
                """, (datetime.now().isoformat(), recommendation_id, user_id))
            
            elif feedback_type == 'dismiss':
                cursor.execute("""
                    UPDATE recommendations 
                    SET dismissed_at = ? 
                    WHERE id = ? AND user_id = ?
                """, (datetime.now().isoformat(), recommendation_id, user_id))
            
            conn.commit()
            conn.close()
            
            self.feedback_received += 1
            self.logger.info(f"✅ Feedback recorded for recommendation {recommendation_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording recommendation feedback: {e}")
            return False
    
    # Helper methods
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
    
    async def _get_user_behavior(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get user behavior from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT opportunity_id, interaction_type, timestamp, duration, metadata
                FROM user_interactions
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (user_id, (datetime.now() - timedelta(days=30)).isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            behavior = []
            for result in results:
                behavior.append({
                    'opportunity_id': result[0],
                    'interaction_type': result[1],
                    'timestamp': result[2],
                    'duration': result[3],
                    'metadata': json.loads(result[4] or '{}')
                })
            
            return behavior
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user behavior: {e}")
            return None
    
    async def _get_opportunity(self, opportunity_id: str) -> Optional[Dict[str, Any]]:
        """Get opportunity from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
    
    async def _build_user_item_matrix(self) -> Optional[pd.DataFrame]:
        """Build user-item interaction matrix"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get user interactions
            query = """
                SELECT user_id, opportunity_id, interaction_type, COUNT(*) as interaction_count
                FROM user_interactions
                WHERE timestamp > ?
                GROUP BY user_id, opportunity_id, interaction_type
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=[(datetime.now() - timedelta(days=30)).isoformat()]
            )
            
            conn.close()
            
            if df.empty:
                return None
            
            # Create interaction scores
            interaction_weights = {'view': 1, 'click': 2, 'share': 3, 'contact': 5, 'save': 4}
            df['score'] = df.apply(
                lambda row: row['interaction_count'] * interaction_weights.get(row['interaction_type'], 1),
                axis=1
            )
            
            # Group by user and item
            user_item_scores = df.groupby(['user_id', 'opportunity_id'])['score'].sum().reset_index()
            
            # Create pivot table
            user_item_matrix = user_item_scores.pivot(
                index='user_id', 
                columns='opportunity_id', 
                values='score'
            ).fillna(0)
            
            return user_item_matrix
            
        except Exception as e:
            self.logger.error(f"❌ Error building user-item matrix: {e}")
            return None
    
    async def _find_similar_users_collaborative(
        self,
        user_id: str,
        user_item_matrix: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """Find similar users using collaborative filtering"""
        try:
            if user_id not in user_item_matrix.index:
                return []
            
            # Calculate user similarity matrix
            user_similarity = cosine_similarity(user_item_matrix)
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=user_item_matrix.index,
                columns=user_item_matrix.index
            )
            
            # Get similar users
            similar_users = user_similarity_df[user_id].sort_values(ascending=False)
            
            # Return top similar users (excluding self)
            similar_users = similar_users.drop(user_id)
            
            return [(user, score) for user, score in similar_users.head(10).items() if score > 0.1]
            
        except Exception as e:
            self.logger.error(f"❌ Error finding similar users: {e}")
            return []
    
    async def _get_user_positive_interactions(self, user_id: str) -> List[Tuple[str, float]]:
        """Get positive interactions for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT opportunity_id, interaction_type, COUNT(*) as count
                FROM user_interactions
                WHERE user_id = ? AND interaction_type IN ('contact', 'save', 'share')
                GROUP BY opportunity_id, interaction_type
                ORDER BY count DESC
                LIMIT 20
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            # Calculate interaction scores
            interaction_weights = {'contact': 5, 'save': 4, 'share': 3}
            
            item_scores = {}
            for opportunity_id, interaction_type, count in results:
                score = count * interaction_weights.get(interaction_type, 1)
                
                if opportunity_id not in item_scores:
                    item_scores[opportunity_id] = 0
                item_scores[opportunity_id] += score
            
            # Normalize scores
            max_score = max(item_scores.values()) if item_scores else 1
            normalized_scores = [(item_id, score/max_score) for item_id, score in item_scores.items()]
            
            return normalized_scores
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user positive interactions: {e}")
            return []
    
    async def _user_has_interacted(self, user_id: str, opportunity_id: str) -> bool:
        """Check if user has interacted with opportunity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM user_interactions
                WHERE user_id = ? AND opportunity_id = ?
            """, (user_id, opportunity_id))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error checking user interaction: {e}")
            return False
    
    async def _find_opportunities_by_behavior(
        self,
        preferred_categories: List[str],
        peak_hours: List[int]
    ) -> List[Dict[str, Any]]:
        """Find opportunities matching behavior patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get opportunities with matching tags or industry
            if preferred_categories:
                placeholders = ','.join(['?' for _ in preferred_categories])
                query = f"""
                    SELECT id, title, description, opportunity_type, industry, 
                           location, budget_range, deadline, contact_info, tags, 
                           status, created_at, updated_at
                    FROM opportunities
                    WHERE status = 'active' AND (
                        industry IN ({placeholders}) OR
                        tags LIKE '%' || ? || '%'
                    )
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                
                params = preferred_categories + [preferred_categories[0]]
                cursor.execute(query, params)
            else:
                cursor.execute("""
                    SELECT id, title, description, opportunity_type, industry, 
                           location, budget_range, deadline, contact_info, tags, 
                           status, created_at, updated_at
                    FROM opportunities
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 20
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
            self.logger.error(f"❌ Error finding opportunities by behavior: {e}")
            return []
    
    def _calculate_behavior_match_score(
        self,
        behavior_analysis: Dict[str, Any],
        opportunity: Dict[str, Any]
    ) -> float:
        """Calculate behavior match score"""
        try:
            behavior_pattern = behavior_analysis.get('behavior_pattern', {})
            
            # Base score from engagement
            engagement_score = behavior_pattern.get('engagement_score', 0.5)
            
            # Category match bonus
            preferred_categories = behavior_pattern.get('preferred_categories', [])
            opportunity_tags = opportunity.get('tags', [])
            
            category_match = 0
            for category in preferred_categories:
                if category.lower() in [tag.lower() for tag in opportunity_tags]:
                    category_match += 0.2
            
            # Industry match bonus
            if opportunity.get('industry', '').lower() in [cat.lower() for cat in preferred_categories]:
                category_match += 0.3
            
            # Combine scores
            final_score = (engagement_score * 0.6) + (min(category_match, 1.0) * 0.4)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating behavior match score: {e}")
            return 0.5
    
    async def _get_trending_opportunities(self) -> List[Dict[str, Any]]:
        """Get trending opportunities"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get opportunities with most interactions in last 7 days
            cursor.execute("""
                SELECT o.id, o.title, o.description, o.opportunity_type, o.industry, 
                       o.location, o.budget_range, o.deadline, o.contact_info, o.tags, 
                       o.status, o.created_at, o.updated_at, COUNT(ui.id) as interaction_count
                FROM opportunities o
                LEFT JOIN user_interactions ui ON o.id = ui.opportunity_id
                WHERE o.status = 'active' AND ui.timestamp > ?
                GROUP BY o.id
                ORDER BY interaction_count DESC, o.created_at DESC
                LIMIT 20
            """, ((datetime.now() - timedelta(days=7)).isoformat(),))
            
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
                    'updated_at': result[12],
                    'interaction_count': result[13]
                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error getting trending opportunities: {e}")
            return []
    
    def _calculate_trending_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate trending score"""
        try:
            interaction_count = opportunity.get('interaction_count', 0)
            
            # Calculate recency score
            created_at = datetime.fromisoformat(opportunity.get('created_at', ''))
            days_since_creation = (datetime.now() - created_at).days
            recency_score = max(0, 1 - (days_since_creation / 7))  # Decay over 7 days
            
            # Calculate interaction score
            interaction_score = min(interaction_count / 10, 1.0)  # Normalize to 0-1
            
            # Combine scores
            trending_score = (interaction_score * 0.7) + (recency_score * 0.3)
            
            return trending_score
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating trending score: {e}")
            return 0.5
    
    async def _get_opportunities_by_location(self, location: str) -> List[Dict[str, Any]]:
        """Get opportunities by location"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, description, opportunity_type, industry, 
                       location, budget_range, deadline, contact_info, tags, 
                       status, created_at, updated_at
                FROM opportunities
                WHERE status = 'active' AND location LIKE ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (f'%{location}%',))
            
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
            self.logger.error(f"❌ Error getting opportunities by location: {e}")
            return []
    
    def _calculate_location_match_score(self, user_location: str, opportunity: Dict[str, Any]) -> float:
        """Calculate location match score"""
        try:
            opp_location = opportunity.get('location', '')
            
            if not opp_location:
                return 0.3  # Neutral score for missing location
            
            # Exact match
            if user_location.lower() == opp_location.lower():
                return 1.0
            
            # Partial match
            user_parts = user_location.lower().split(',')
            opp_parts = opp_location.lower().split(',')
            
            if len(user_parts) > 1 and len(opp_parts) > 1:
                if user_parts[0].strip() == opp_parts[0].strip():  # Same city
                    return 0.8
                if user_parts[-1].strip() == opp_parts[-1].strip():  # Same country/state
                    return 0.5
            
            return 0.2  # Different locations
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating location match score: {e}")
            return 0.3
    
    async def _get_popular_opportunities(self) -> List[Dict[str, Any]]:
        """Get popular opportunities for cold start"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT o.id, o.title, o.description, o.opportunity_type, o.industry, 
                       o.location, o.budget_range, o.deadline, o.contact_info, o.tags, 
                       o.status, o.created_at, o.updated_at, COUNT(ui.id) as interaction_count
                FROM opportunities o
                LEFT JOIN user_interactions ui ON o.id = ui.opportunity_id
                WHERE o.status = 'active'
                GROUP BY o.id
                ORDER BY interaction_count DESC, o.created_at DESC
                LIMIT 10
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
                    'updated_at': result[12],
                    'interaction_count': result[13]
                })
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error getting popular opportunities: {e}")
            return []
    
    async def _filter_seen_recommendations(
        self,
        user_id: str,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Filter out already seen recommendations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get seen opportunity IDs
            cursor.execute("""
                SELECT DISTINCT opportunity_id FROM recommendations
                WHERE user_id = ? AND generated_at > ?
            """, (user_id, (datetime.now() - timedelta(days=7)).isoformat()))
            
            seen_opportunities = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            # Filter recommendations
            filtered = [rec for rec in recommendations if rec.opportunity_id not in seen_opportunities]
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering seen recommendations: {e}")
            return recommendations
    
    async def _store_recommendations(self, recommendations: List[Recommendation]):
        """Store recommendations in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rec in recommendations:
                cursor.execute("""
                    INSERT INTO recommendations 
                    (id, user_id, opportunity_id, recommendation_type, score, 
                     reasoning, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec.id,
                    rec.user_id,
                    rec.opportunity_id,
                    rec.recommendation_type.value,
                    rec.score,
                    rec.reasoning,
                    rec.generated_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing recommendations: {e}")
    
    # Cache management
    async def _get_cached_recommendations(self, user_id: str) -> Optional[RecommendationResult]:
        """Get cached recommendations"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"recommendations:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                
                # Deserialize recommendations
                recommendations = []
                for rec_data in data['recommendations']:
                    rec = Recommendation(
                        id=rec_data['id'],
                        user_id=rec_data['user_id'],
                        opportunity_id=rec_data['opportunity_id'],
                        recommendation_type=RecommendationType(rec_data['recommendation_type']),
                        score=rec_data['score'],
                        reasoning=rec_data['reasoning'],
                        generated_at=datetime.fromisoformat(rec_data['generated_at'])
                    )
                    recommendations.append(rec)
                
                return RecommendationResult(
                    user_id=data['user_id'],
                    recommendations=recommendations,
                    total_generated=data['total_generated'],
                    generation_time_ms=data['generation_time_ms'],
                    algorithms_used=data['algorithms_used'],
                    metadata=data['metadata'],
                    generated_at=datetime.fromisoformat(data['generated_at'])
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_recommendations(self, user_id: str, result: RecommendationResult):
        """Cache recommendations"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"recommendations:{user_id}"
            ttl = self.config.redis.recommendation_cache_ttl
            
            # Serialize recommendations
            serialized_recs = []
            for rec in result.recommendations:
                serialized_recs.append({
                    'id': rec.id,
                    'user_id': rec.user_id,
                    'opportunity_id': rec.opportunity_id,
                    'recommendation_type': rec.recommendation_type.value,
                    'score': rec.score,
                    'reasoning': rec.reasoning,
                    'generated_at': rec.generated_at.isoformat()
                })
            
            # Serialize result
            serialized_result = {
                'user_id': result.user_id,
                'recommendations': serialized_recs,
                'total_generated': result.total_generated,
                'generation_time_ms': result.generation_time_ms,
                'algorithms_used': result.algorithms_used,
                'metadata': result.metadata,
                'generated_at': result.generated_at.isoformat()
            }
            
            # Store in cache
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(serialized_result)
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Status and management methods
    def get_engine_status(self) -> Dict[str, Any]:
        """Get recommendation engine status"""
        return {
            "status": "operational",
            "recommendations_generated": self.recommendations_generated,
            "feedback_received": self.feedback_received,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "algorithm_weights": self.algorithm_weights,
            "configuration": {
                "redis_enabled": self.redis_enabled,
                "algorithms_available": list(self.algorithm_weights.keys())
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_recommendation_cache(self, user_id: str = None) -> int:
        """Clear recommendation cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if user_id:
                pattern = f"recommendations:{user_id}"
            else:
                pattern = "recommendations:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing recommendation cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the recommendation engine
    async def test_recommendation_engine():
        print("🎯 Testing Recommendation Engine")
        print("=" * 50)
        
        try:
            engine = RecommendationEngine()
            
            # Test recommendation generation
            request = RecommendationRequest(
                user_id="test_user_123",
                max_recommendations=10,
                exclude_seen=True,
                include_reasons=True
            )
            
            result = await engine.generate_recommendations(request)
            print(f"Generated {result.total_generated} recommendations")
            print(f"Algorithms used: {result.algorithms_used}")
            print(f"Generation time: {result.generation_time_ms}ms")
            
            # Test feedback recording
            if result.recommendations:
                feedback_success = await engine.record_recommendation_feedback(
                    "test_user_123",
                    result.recommendations[0].id,
                    "click"
                )
                print(f"Feedback recorded: {feedback_success}")
            
            # Test status
            status = engine.get_engine_status()
            print(f"Engine Status: {status}")
            
            print("✅ Recommendation Engine test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_recommendation_engine())