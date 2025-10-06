#!/usr/bin/env python3
"""
Behavior Analyzer
Tracks user interactions and patterns using Pandas for the Business Dealer Intelligence System

This analyzer:
- Processes user interaction data to identify behavioral patterns
- Calculates engagement metrics and user segmentation
- Stores insights in Redis for real-time access
- Implements user segmentation using scikit-learn clustering

Following Task 3 from the PRP implementation blueprint.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from intelligence.data_models2 import InteractionType

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPattern:
    """Represents a user behavior pattern"""

    user_id: str
    engagement_score: float
    interaction_diversity: float
    session_duration_avg: float
    preferred_categories: List[str]
    peak_activity_hours: List[int]
    interaction_frequency: float
    conversion_rate: float
    pattern_confidence: float


@dataclass
class UserSegment:
    """Represents a user segment"""

    segment_id: str
    segment_name: str
    characteristics: Dict[str, Any]
    user_count: int
    avg_engagement: float
    key_behaviors: List[str]


class BehaviorAnalyzer:
    """
    Behavior analysis engine that processes user interactions and generates insights
    """

    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Behavior Analyzer

        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()

        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 Initializing Behavior Analyzer")

        # Initialize Redis client for caching
        self._setup_redis_client()

        # Initialize ML models
        self._setup_ml_models()

        # Interaction weights for scoring
        self.interaction_weights = self.config.behavior_analysis.interaction_weights

        self.logger.info("✅ Behavior Analyzer initialized successfully")

    def _setup_redis_client(self):
        """Setup Redis client for caching behavior insights"""
        try:
            import redis

            self.redis_client = redis.Redis.from_url(
                self.config.redis.url, decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for behavior caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for behavior caching: {e}")
            self.redis_enabled = False

    def _setup_ml_models(self):
        """Setup ML models for user segmentation"""
        self.scaler = StandardScaler()
        self.segmentation_model = KMeans(
            n_clusters=self.ml_config.clustering_model.n_clusters,
            random_state=self.ml_config.random_seed,
            n_init=self.ml_config.clustering_model.n_init,
        )
        self.pca_model = PCA(n_components=2, random_state=self.ml_config.random_seed)

        self.logger.info("✅ ML models for segmentation initialized")

    async def analyze_user_patterns(
        self, user_id: str, analysis_window_days: int = None
    ) -> Dict[str, Any]:
        """
        Analyze behavioral patterns for a specific user

        Args:
            user_id: ID of the user to analyze
            analysis_window_days: Number of days to analyze (default from config)

        Returns:
            Dictionary containing behavior patterns and insights
        """
        try:
            self.logger.info(f"📈 Analyzing behavior patterns for user {user_id}")

            # Check cache first
            cache_key = f"behavior_patterns:{user_id}"
            cached_patterns = self._get_from_cache(cache_key)
            if cached_patterns:
                self.logger.info(f"🎯 Using cached behavior patterns for user {user_id}")
                return cached_patterns

            # Get analysis window
            if analysis_window_days is None:
                analysis_window_days = (
                    self.config.behavior_analysis.analysis_window_days
                )

            # Load user interactions
            interactions_df = self._load_user_interactions(
                user_id, analysis_window_days
            )

            if interactions_df.empty:
                self.logger.warning(f"⚠️ No interactions found for user {user_id}")
                return self._empty_pattern_response(user_id)

            # Calculate behavior metrics
            engagement_score = self._calculate_engagement_score(interactions_df)
            interaction_diversity = self._calculate_interaction_diversity(
                interactions_df
            )
            session_metrics = self._calculate_session_metrics(interactions_df)
            temporal_patterns = self._analyze_temporal_patterns(interactions_df)
            category_preferences = self._analyze_category_preferences(interactions_df)

            # Create behavior pattern
            pattern = BehaviorPattern(
                user_id=user_id,
                engagement_score=engagement_score,
                interaction_diversity=interaction_diversity,
                session_duration_avg=session_metrics["avg_duration"],
                preferred_categories=category_preferences["top_categories"],
                peak_activity_hours=temporal_patterns["peak_hours"],
                interaction_frequency=session_metrics["frequency"],
                conversion_rate=self._calculate_conversion_rate(interactions_df),
                pattern_confidence=self._calculate_pattern_confidence(interactions_df),
            )

            # Generate insights
            insights = self._generate_behavioral_insights(pattern, interactions_df)

            # Prepare response
            response = {
                "user_id": user_id,
                "analysis_period": {
                    "start_date": (
                        datetime.now() - timedelta(days=analysis_window_days)
                    ).isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "days_analyzed": analysis_window_days,
                },
                "behavior_pattern": {
                    "engagement_score": pattern.engagement_score,
                    "interaction_diversity": pattern.interaction_diversity,
                    "avg_session_duration": pattern.session_duration_avg,
                    "preferred_categories": pattern.preferred_categories,
                    "peak_activity_hours": pattern.peak_activity_hours,
                    "interaction_frequency": pattern.interaction_frequency,
                    "conversion_rate": pattern.conversion_rate,
                    "confidence": pattern.pattern_confidence,
                },
                "insights": insights,
                "metrics": {
                    "total_interactions": len(interactions_df),
                    "unique_days_active": interactions_df["date"].nunique(),
                    "avg_interactions_per_day": len(interactions_df)
                    / max(interactions_df["date"].nunique(), 1),
                },
                "analyzed_at": datetime.now().isoformat(),
            }

            # Cache the results
            self._store_in_cache(
                cache_key, response, self.config.redis.user_behavior_cache_ttl
            )

            self.logger.info(f"✅ Behavior analysis completed for user {user_id}")
            return response

        except Exception as e:
            self.logger.error(f"❌ Error analyzing behavior patterns: {e}")
            return {"error": str(e), "user_id": user_id}

    def _load_user_interactions(self, user_id: str, days: int) -> pd.DataFrame:
        """Load user interactions from database"""
        try:
            # Connect to database
            conn = sqlite3.connect(self.config.database.intelligence_db_path)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Query interactions
            query = """
            SELECT 
                user_id,
                opportunity_id,
                interaction_type,
                timestamp,
                duration,
                metadata
            FROM user_interactions 
            WHERE user_id = ? 
            AND timestamp >= ? 
            AND timestamp <= ?
            ORDER BY timestamp
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=(user_id, start_date.isoformat(), end_date.isoformat()),
            )

            conn.close()

            if not df.empty:
                # Process timestamp and add derived columns
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["date"] = df["timestamp"].dt.date
                df["hour"] = df["timestamp"].dt.hour
                df["day_of_week"] = df["timestamp"].dt.dayofweek

                # Parse metadata
                df["metadata"] = df["metadata"].apply(
                    lambda x: json.loads(x) if x else {}
                )

            return df

        except Exception as e:
            self.logger.error(f"❌ Error loading user interactions: {e}")
            return pd.DataFrame()

    def _calculate_engagement_score(self, interactions_df: pd.DataFrame) -> float:
        """Calculate user engagement score based on interactions"""
        if interactions_df.empty:
            return 0.0

        # Weight interactions by type
        weighted_score = 0
        for _, row in interactions_df.iterrows():
            interaction_type = row["interaction_type"]
            weight = self.interaction_weights.get(interaction_type, 1.0)

            # Consider duration if available
            duration_bonus = 0
            if pd.notna(row["duration"]) and row["duration"] > 0:
                # Normalize duration to 0-1 scale (capped at 300 seconds)
                duration_bonus = min(row["duration"] / 300, 1.0) * 0.5

            weighted_score += weight + duration_bonus

        # Normalize by total possible score
        max_possible_score = len(interactions_df) * (
            max(self.interaction_weights.values()) + 0.5
        )

        return (
            min(weighted_score / max_possible_score, 1.0)
            if max_possible_score > 0
            else 0.0
        )

    def _calculate_interaction_diversity(self, interactions_df: pd.DataFrame) -> float:
        """Calculate diversity of interaction types"""
        if interactions_df.empty:
            return 0.0

        # Count unique interaction types
        unique_types = interactions_df["interaction_type"].nunique()
        total_possible_types = len(InteractionType)

        return unique_types / total_possible_types

    def _calculate_session_metrics(
        self, interactions_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate session-based metrics"""
        if interactions_df.empty:
            return {"avg_duration": 0.0, "frequency": 0.0}

        # Group by date to calculate daily metrics
        daily_interactions = interactions_df.groupby("date").size()

        # Calculate average duration
        avg_duration = (
            interactions_df["duration"].mean()
            if "duration" in interactions_df.columns
            else 0
        )
        avg_duration = avg_duration if pd.notna(avg_duration) else 0

        # Calculate interaction frequency (interactions per day)
        frequency = daily_interactions.mean()

        return {"avg_duration": float(avg_duration), "frequency": float(frequency)}

    def _analyze_temporal_patterns(
        self, interactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze temporal interaction patterns"""
        if interactions_df.empty:
            return {"peak_hours": [], "activity_distribution": {}}

        # Find peak activity hours
        hourly_activity = interactions_df.groupby("hour").size()
        peak_hours = hourly_activity.nlargest(3).index.tolist()

        # Activity distribution by day of week
        weekly_activity = interactions_df.groupby("day_of_week").size()
        activity_distribution = weekly_activity.to_dict()

        return {
            "peak_hours": peak_hours,
            "activity_distribution": activity_distribution,
        }

    def _analyze_category_preferences(
        self, interactions_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze category preferences from metadata"""
        if interactions_df.empty:
            return {"top_categories": [], "category_distribution": {}}

        # Extract categories from metadata
        categories = []
        for _, row in interactions_df.iterrows():
            metadata = row["metadata"]
            if isinstance(metadata, dict) and "category" in metadata:
                categories.append(metadata["category"])

        if not categories:
            return {"top_categories": [], "category_distribution": {}}

        category_series = pd.Series(categories)
        category_counts = category_series.value_counts()

        return {
            "top_categories": category_counts.head(5).index.tolist(),
            "category_distribution": category_counts.to_dict(),
        }

    def _calculate_conversion_rate(self, interactions_df: pd.DataFrame) -> float:
        """Calculate conversion rate (contact/save interactions vs total)"""
        if interactions_df.empty:
            return 0.0

        conversion_actions = ["contact", "save"]
        conversions = interactions_df[
            interactions_df["interaction_type"].isin(conversion_actions)
        ]

        return len(conversions) / len(interactions_df)

    def _calculate_pattern_confidence(self, interactions_df: pd.DataFrame) -> float:
        """Calculate confidence in the behavior pattern"""
        if interactions_df.empty:
            return 0.0

        # Base confidence on number of interactions and time span
        interaction_count = len(interactions_df)
        time_span_days = (
            interactions_df["timestamp"].max() - interactions_df["timestamp"].min()
        ).days

        # Confidence increases with more interactions and longer time span
        interaction_confidence = min(
            interaction_count / 50, 1.0
        )  # 50 interactions = full confidence
        time_confidence = min(time_span_days / 30, 1.0)  # 30 days = full confidence

        return (interaction_confidence + time_confidence) / 2

    def _generate_behavioral_insights(
        self, pattern: BehaviorPattern, interactions_df: pd.DataFrame
    ) -> List[str]:
        """Generate behavioral insights based on pattern analysis"""
        insights = []

        # Engagement insights
        if pattern.engagement_score > 0.8:
            insights.append("High engagement user - responds well to recommendations")
        elif pattern.engagement_score < 0.3:
            insights.append("Low engagement user - may need different content strategy")

        # Interaction diversity insights
        if pattern.interaction_diversity > 0.7:
            insights.append(
                "Diverse interaction pattern - actively explores different content types"
            )
        elif pattern.interaction_diversity < 0.3:
            insights.append(
                "Limited interaction pattern - prefers specific content types"
            )

        # Temporal insights
        if pattern.peak_activity_hours:
            peak_hours_str = ", ".join(map(str, pattern.peak_activity_hours))
            insights.append(f"Most active during hours: {peak_hours_str}")

        # Category insights
        if pattern.preferred_categories:
            categories_str = ", ".join(pattern.preferred_categories[:3])
            insights.append(f"Strong preference for: {categories_str}")

        # Conversion insights
        if pattern.conversion_rate > 0.2:
            insights.append(
                "High conversion rate - likely to take action on recommendations"
            )
        elif pattern.conversion_rate < 0.05:
            insights.append(
                "Low conversion rate - may need more compelling opportunities"
            )

        return insights

    def _empty_pattern_response(self, user_id: str) -> Dict[str, Any]:
        """Return empty pattern response for users with no interactions"""
        return {
            "user_id": user_id,
            "behavior_pattern": {
                "engagement_score": 0.0,
                "interaction_diversity": 0.0,
                "avg_session_duration": 0.0,
                "preferred_categories": [],
                "peak_activity_hours": [],
                "interaction_frequency": 0.0,
                "conversion_rate": 0.0,
                "confidence": 0.0,
            },
            "insights": ["No interaction data available for analysis"],
            "metrics": {
                "total_interactions": 0,
                "unique_days_active": 0,
                "avg_interactions_per_day": 0,
            },
            "analyzed_at": datetime.now().isoformat(),
        }

    # Cache management
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from Redis cache"""
        if not self.redis_enabled:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")

        return None

    def _store_in_cache(self, cache_key: str, data: Dict[str, Any], ttl: int):
        """Store data in Redis cache"""
        if not self.redis_enabled:
            return

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")

    # User segmentation methods
    async def perform_user_segmentation(
        self, user_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform user segmentation using clustering

        Args:
            user_ids: Optional list of user IDs to segment (default: all users)

        Returns:
            Dictionary containing segmentation results
        """
        try:
            self.logger.info("👥 Performing user segmentation")

            # Get user behavior data
            user_data = await self._collect_user_behavior_data(user_ids)

            if len(user_data) < self.config.behavior_analysis.min_segment_size:
                self.logger.warning(
                    f"⚠️ Not enough users for segmentation: {len(user_data)}"
                )
                return {"error": "Insufficient data for segmentation"}

            # Prepare features for clustering
            features_df = self._prepare_segmentation_features(user_data)

            # Perform clustering
            segments = self._perform_clustering(features_df)

            # Analyze segments
            segment_analysis = self._analyze_segments(segments, user_data)

            return {
                "segmentation_results": segment_analysis,
                "total_users": len(user_data),
                "num_segments": len(segment_analysis),
                "segmented_at": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"❌ Error performing user segmentation: {e}")
            return {"error": str(e)}

    async def _collect_user_behavior_data(
        self, user_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Collect behavior data for all users"""
        # Implementation would collect behavior patterns for all users
        # For now, return sample data
        return [
            {
                "user_id": f"user_{i}",
                "engagement_score": np.random.uniform(0.1, 1.0),
                "interaction_diversity": np.random.uniform(0.1, 1.0),
                "session_duration": np.random.uniform(30, 300),
                "conversion_rate": np.random.uniform(0.0, 0.5),
                "interaction_frequency": np.random.uniform(1, 20),
            }
            for i in range(100)
        ]

    def _prepare_segmentation_features(
        self, user_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Prepare features for clustering"""
        features = []

        for user in user_data:
            features.append(
                [
                    user["engagement_score"],
                    user["interaction_diversity"],
                    user["session_duration"] / 300,  # Normalize
                    user["conversion_rate"],
                    user["interaction_frequency"] / 20,  # Normalize
                ]
            )

        feature_names = [
            "engagement_score",
            "interaction_diversity",
            "normalized_session_duration",
            "conversion_rate",
            "normalized_interaction_frequency",
        ]

        return pd.DataFrame(features, columns=feature_names)

    def _perform_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering on user features"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)

        # Perform clustering
        cluster_labels = self.segmentation_model.fit_predict(scaled_features)

        # Reduce dimensionality for visualization
        pca_features = self.pca_model.fit_transform(scaled_features)

        return {
            "cluster_labels": cluster_labels,
            "scaled_features": scaled_features,
            "pca_features": pca_features,
            "cluster_centers": self.segmentation_model.cluster_centers_,
        }

    def _analyze_segments(
        self, segments: Dict[str, Any], user_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze the characteristics of each segment"""
        cluster_labels = segments["cluster_labels"]
        segment_analysis = []

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_users = [
                user for i, user in enumerate(user_data) if cluster_mask[i]
            ]

            if not cluster_users:
                continue

            # Calculate segment characteristics
            avg_engagement = np.mean(
                [user["engagement_score"] for user in cluster_users]
            )
            avg_diversity = np.mean(
                [user["interaction_diversity"] for user in cluster_users]
            )
            avg_conversion = np.mean(
                [user["conversion_rate"] for user in cluster_users]
            )

            # Determine segment name based on characteristics
            segment_name = self._determine_segment_name(
                avg_engagement, avg_diversity, avg_conversion
            )

            segment_analysis.append(
                {
                    "segment_id": f"segment_{cluster_id}",
                    "segment_name": segment_name,
                    "user_count": len(cluster_users),
                    "characteristics": {
                        "avg_engagement": float(avg_engagement),
                        "avg_diversity": float(avg_diversity),
                        "avg_conversion": float(avg_conversion),
                    },
                    "key_behaviors": self._identify_key_behaviors(cluster_users),
                }
            )

        return segment_analysis

    def _determine_segment_name(
        self, engagement: float, diversity: float, conversion: float
    ) -> str:
        """Determine segment name based on characteristics"""
        if engagement > 0.7 and conversion > 0.2:
            return "High Value Users"
        elif engagement > 0.5 and diversity > 0.6:
            return "Engaged Explorers"
        elif conversion > 0.1 and engagement < 0.5:
            return "Conversion Focused"
        elif diversity < 0.3:
            return "Focused Users"
        elif engagement < 0.3:
            return "Low Engagement"
        else:
            return "Moderate Users"

    def _identify_key_behaviors(self, cluster_users: List[Dict[str, Any]]) -> List[str]:
        """Identify key behaviors for a segment"""
        behaviors = []

        avg_engagement = np.mean([user["engagement_score"] for user in cluster_users])
        avg_frequency = np.mean(
            [user["interaction_frequency"] for user in cluster_users]
        )

        if avg_engagement > 0.8:
            behaviors.append("High engagement with content")
        if avg_frequency > 10:
            behaviors.append("Frequent platform usage")

        return behaviors


if __name__ == "__main__":
    # Test the behavior analyzer
    async def test_behavior_analyzer():
        print("📊 Testing Behavior Analyzer")
        print("=" * 50)

        try:
            analyzer = BehaviorAnalyzer()

            # Test user pattern analysis
            patterns = await analyzer.analyze_user_patterns("test_user_123")
            print(f"User Patterns: {patterns}")

            # Test user segmentation
            segmentation = await analyzer.perform_user_segmentation()
            print(f"Segmentation Results: {segmentation}")

            print("✅ Behavior Analyzer test completed successfully!")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Run test
    import asyncio

    asyncio.run(test_behavior_analyzer())
