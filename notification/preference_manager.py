#!/usr/bin/env python3
"""
Preference Manager
User notification preferences management system

This manager:
- Manages user notification preferences and settings
- Tracks user feedback and behavior patterns
- Learns from user interactions to improve preferences
- Provides preference-based filtering and routing
- Supports dynamic preference updates

Following Task 9 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from .smart_prioritizer import NotificationCategory, NotificationPriority
from .delivery_optimizer import DeliveryChannel

logger = logging.getLogger(__name__)


class PreferenceType(Enum):
    """Preference types"""
    CHANNEL = "channel"
    CATEGORY = "category"
    TIMING = "timing"
    FREQUENCY = "frequency"
    CONTENT = "content"


class FeedbackType(Enum):
    """Feedback types"""
    VIEWED = "viewed"
    CLICKED = "clicked"
    DISMISSED = "dismissed"
    BLOCKED = "blocked"
    REPORTED = "reported"


@dataclass
class UserPreference:
    """User notification preference"""
    
    user_id: str
    preference_type: PreferenceType
    preference_key: str
    preference_value: Any
    confidence_score: float
    learned_from_behavior: bool
    last_updated: datetime
    expires_at: Optional[datetime] = None


@dataclass
class NotificationFeedback:
    """Notification feedback"""
    
    notification_id: str
    user_id: str
    feedback_type: FeedbackType
    feedback_value: Any
    channel: DeliveryChannel
    category: NotificationCategory
    priority: NotificationPriority
    feedback_time: datetime
    metadata: Dict[str, Any]


@dataclass
class PreferenceProfile:
    """User preference profile"""
    
    user_id: str
    channel_preferences: Dict[str, float]
    category_preferences: Dict[str, float]
    timing_preferences: Dict[str, Any]
    frequency_preferences: Dict[str, int]
    content_preferences: Dict[str, Any]
    last_updated: datetime
    confidence_score: float


class PreferenceManager:
    """
    User notification preferences management system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Preference Manager
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("⚙️ Initializing Preference Manager")
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client
        self._setup_redis_client()
        
        # Initialize preference models
        self._setup_preference_models()
        
        # Performance tracking
        self.preferences_updated = 0
        self.feedback_processed = 0
        self.profiles_generated = 0
        
        # Default preferences
        self.default_preferences = {
            'channels': {
                DeliveryChannel.PUSH.value: 0.8,
                DeliveryChannel.EMAIL.value: 0.6,
                DeliveryChannel.SMS.value: 0.4,
                DeliveryChannel.IN_APP.value: 0.9,
                DeliveryChannel.SLACK.value: 0.3,
                DeliveryChannel.WEBHOOK.value: 0.1
            },
            'categories': {
                NotificationCategory.OPPORTUNITY.value: 0.9,
                NotificationCategory.INSIGHT.value: 0.8,
                NotificationCategory.ALERT.value: 0.7,
                NotificationCategory.RECOMMENDATION.value: 0.6,
                NotificationCategory.SYSTEM.value: 0.3
            },
            'timing': {
                'business_hours_only': False,
                'quiet_hours_start': 22,
                'quiet_hours_end': 7,
                'weekend_delivery': True
            },
            'frequency': {
                'daily_limit': 20,
                'hourly_limit': 5,
                'batch_notifications': True
            }
        }
        
        self.logger.info("✅ Preference Manager initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for preference management")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching preferences"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for preference caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for preference caching: {e}")
            self.redis_enabled = False
    
    def _setup_preference_models(self):
        """Setup preference learning models"""
        try:
            # Feedback weights for learning
            self.feedback_weights = {
                FeedbackType.VIEWED: 0.1,
                FeedbackType.CLICKED: 0.8,
                FeedbackType.DISMISSED: -0.3,
                FeedbackType.BLOCKED: -0.8,
                FeedbackType.REPORTED: -1.0
            }
            
            # Learning rates
            self.learning_rates = {
                'channel': 0.1,
                'category': 0.15,
                'timing': 0.05,
                'frequency': 0.2
            }
            
            self.logger.info("✅ Preference learning models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up preference models: {e}")
            raise
    
    async def get_user_preferences(
        self,
        user_id: str,
        use_cache: bool = True
    ) -> PreferenceProfile:
        """
        Get user preference profile
        
        Args:
            user_id: User identifier
            use_cache: Whether to use cached preferences
            
        Returns:
            PreferenceProfile with user preferences
        """
        try:
            self.logger.info(f"⚙️ Getting preferences for user {user_id}")
            
            # Check cache first
            if use_cache:
                cached_profile = await self._get_cached_profile(user_id)
                if cached_profile:
                    return cached_profile
            
            # Get stored preferences
            preferences = await self._get_stored_preferences(user_id)
            
            # Generate profile
            profile = await self._generate_preference_profile(user_id, preferences)
            
            # Cache the profile
            if use_cache:
                await self._cache_profile(profile)
            
            self.logger.info(f"✅ Retrieved preferences for user {user_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user preferences: {e}")
            return await self._create_default_profile(user_id)
    
    async def update_user_preference(
        self,
        user_id: str,
        preference_type: PreferenceType,
        preference_key: str,
        preference_value: Any,
        confidence_score: float = 1.0
    ) -> bool:
        """
        Update user preference
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_key: Preference key
            preference_value: Preference value
            confidence_score: Confidence in the preference
            
        Returns:
            True if updated successfully
        """
        try:
            self.logger.info(f"⚙️ Updating preference for user {user_id}: {preference_key}")
            
            # Create preference object
            preference = UserPreference(
                user_id=user_id,
                preference_type=preference_type,
                preference_key=preference_key,
                preference_value=preference_value,
                confidence_score=confidence_score,
                learned_from_behavior=False,
                last_updated=datetime.now()
            )
            
            # Store preference
            await self._store_preference(preference)
            
            # Clear cache
            await self._clear_user_cache(user_id)
            
            self.preferences_updated += 1
            self.logger.info(f"✅ Updated preference for user {user_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating user preference: {e}")
            return False
    
    async def record_feedback(
        self,
        notification_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        feedback_value: Any = None,
        channel: DeliveryChannel = None,
        category: NotificationCategory = None,
        priority: NotificationPriority = None
    ) -> bool:
        """
        Record user feedback on notification
        
        Args:
            notification_id: Notification identifier
            user_id: User identifier
            feedback_type: Type of feedback
            feedback_value: Feedback value
            channel: Delivery channel
            category: Notification category
            priority: Notification priority
            
        Returns:
            True if recorded successfully
        """
        try:
            self.logger.info(f"⚙️ Recording feedback for notification {notification_id}")
            
            # Create feedback object
            feedback = NotificationFeedback(
                notification_id=notification_id,
                user_id=user_id,
                feedback_type=feedback_type,
                feedback_value=feedback_value,
                channel=channel or DeliveryChannel.PUSH,
                category=category or NotificationCategory.SYSTEM,
                priority=priority or NotificationPriority.MEDIUM,
                feedback_time=datetime.now(),
                metadata={}
            )
            
            # Store feedback
            await self._store_feedback(feedback)
            
            # Learn from feedback
            await self._learn_from_feedback(feedback)
            
            self.feedback_processed += 1
            self.logger.info(f"✅ Recorded feedback for notification {notification_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error recording feedback: {e}")
            return False
    
    async def should_deliver_notification(
        self,
        user_id: str,
        channel: DeliveryChannel,
        category: NotificationCategory,
        priority: NotificationPriority,
        current_time: datetime = None
    ) -> Tuple[bool, float]:
        """
        Check if notification should be delivered based on preferences
        
        Args:
            user_id: User identifier
            channel: Delivery channel
            category: Notification category
            priority: Notification priority
            current_time: Current time for timing checks
            
        Returns:
            Tuple of (should_deliver, confidence_score)
        """
        try:
            self.logger.info(f"⚙️ Checking delivery preferences for user {user_id}")
            
            # Get user preferences
            profile = await self.get_user_preferences(user_id)
            
            # Check channel preference
            channel_score = profile.channel_preferences.get(channel.value, 0.5)
            
            # Check category preference
            category_score = profile.category_preferences.get(category.value, 0.5)
            
            # Check timing preference
            timing_score = await self._check_timing_preference(
                profile, current_time or datetime.now()
            )
            
            # Check frequency limits
            frequency_score = await self._check_frequency_limits(profile, user_id)
            
            # Calculate overall score
            overall_score = (
                channel_score * 0.3 +
                category_score * 0.25 +
                timing_score * 0.25 +
                frequency_score * 0.2
            )
            
            # Priority boost
            priority_boost = {
                NotificationPriority.CRITICAL: 0.3,
                NotificationPriority.HIGH: 0.2,
                NotificationPriority.MEDIUM: 0.1,
                NotificationPriority.LOW: 0.0
            }.get(priority, 0.0)
            
            final_score = min(1.0, overall_score + priority_boost)
            
            # Determine if should deliver
            should_deliver = final_score > 0.5
            
            self.logger.info(f"✅ Delivery decision: {should_deliver} (score: {final_score:.3f})")
            
            return should_deliver, final_score
            
        except Exception as e:
            self.logger.error(f"❌ Error checking delivery preferences: {e}")
            return True, 0.5  # Default to deliver
    
    async def _generate_preference_profile(
        self,
        user_id: str,
        preferences: List[UserPreference]
    ) -> PreferenceProfile:
        """Generate preference profile from stored preferences"""
        try:
            # Initialize with defaults
            channel_preferences = self.default_preferences['channels'].copy()
            category_preferences = self.default_preferences['categories'].copy()
            timing_preferences = self.default_preferences['timing'].copy()
            frequency_preferences = self.default_preferences['frequency'].copy()
            content_preferences = {}
            
            # Apply stored preferences
            for pref in preferences:
                if pref.preference_type == PreferenceType.CHANNEL:
                    channel_preferences[pref.preference_key] = pref.preference_value
                elif pref.preference_type == PreferenceType.CATEGORY:
                    category_preferences[pref.preference_key] = pref.preference_value
                elif pref.preference_type == PreferenceType.TIMING:
                    timing_preferences[pref.preference_key] = pref.preference_value
                elif pref.preference_type == PreferenceType.FREQUENCY:
                    frequency_preferences[pref.preference_key] = pref.preference_value
                elif pref.preference_type == PreferenceType.CONTENT:
                    content_preferences[pref.preference_key] = pref.preference_value
            
            # Calculate confidence score
            confidence_score = np.mean([p.confidence_score for p in preferences]) if preferences else 0.5
            
            return PreferenceProfile(
                user_id=user_id,
                channel_preferences=channel_preferences,
                category_preferences=category_preferences,
                timing_preferences=timing_preferences,
                frequency_preferences=frequency_preferences,
                content_preferences=content_preferences,
                last_updated=datetime.now(),
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating preference profile: {e}")
            return await self._create_default_profile(user_id)
    
    async def _learn_from_feedback(self, feedback: NotificationFeedback):
        """Learn preferences from user feedback"""
        try:
            # Get feedback weight
            weight = self.feedback_weights.get(feedback.feedback_type, 0.0)
            
            if weight == 0.0:
                return
            
            # Update channel preference
            await self._update_learned_preference(
                feedback.user_id,
                PreferenceType.CHANNEL,
                feedback.channel.value,
                weight,
                self.learning_rates['channel']
            )
            
            # Update category preference
            await self._update_learned_preference(
                feedback.user_id,
                PreferenceType.CATEGORY,
                feedback.category.value,
                weight,
                self.learning_rates['category']
            )
            
            # Update timing preference if applicable
            if feedback.feedback_type in [FeedbackType.VIEWED, FeedbackType.CLICKED]:
                hour = feedback.feedback_time.hour
                await self._update_timing_preference(feedback.user_id, hour, weight)
            
            self.logger.info(f"✅ Learned from feedback: {feedback.feedback_type.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error learning from feedback: {e}")
    
    async def _update_learned_preference(
        self,
        user_id: str,
        preference_type: PreferenceType,
        preference_key: str,
        weight: float,
        learning_rate: float
    ):
        """Update learned preference"""
        try:
            # Get current preference
            current_preferences = await self._get_stored_preferences(user_id)
            
            current_value = None
            for pref in current_preferences:
                if (pref.preference_type == preference_type and 
                    pref.preference_key == preference_key):
                    current_value = pref.preference_value
                    break
            
            # Use default if no current value
            if current_value is None:
                if preference_type == PreferenceType.CHANNEL:
                    current_value = self.default_preferences['channels'].get(preference_key, 0.5)
                elif preference_type == PreferenceType.CATEGORY:
                    current_value = self.default_preferences['categories'].get(preference_key, 0.5)
                else:
                    current_value = 0.5
            
            # Update with learning rate
            new_value = current_value + (learning_rate * weight)
            new_value = max(0.0, min(1.0, new_value))  # Clamp to [0, 1]
            
            # Store updated preference
            preference = UserPreference(
                user_id=user_id,
                preference_type=preference_type,
                preference_key=preference_key,
                preference_value=new_value,
                confidence_score=0.8,
                learned_from_behavior=True,
                last_updated=datetime.now()
            )
            
            await self._store_preference(preference)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating learned preference: {e}")
    
    async def _update_timing_preference(self, user_id: str, hour: int, weight: float):
        """Update timing preference based on feedback"""
        try:
            # This is a simplified version - would be more complex in reality
            if weight > 0:
                # Positive feedback - user likes notifications at this time
                preference_key = f"preferred_hour_{hour}"
                await self._update_learned_preference(
                    user_id,
                    PreferenceType.TIMING,
                    preference_key,
                    weight,
                    self.learning_rates['timing']
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error updating timing preference: {e}")
    
    async def _check_timing_preference(
        self,
        profile: PreferenceProfile,
        current_time: datetime
    ) -> float:
        """Check timing preference"""
        try:
            timing_prefs = profile.timing_preferences
            
            # Check quiet hours
            quiet_start = timing_prefs.get('quiet_hours_start', 22)
            quiet_end = timing_prefs.get('quiet_hours_end', 7)
            
            current_hour = current_time.hour
            
            # Check if in quiet hours
            if quiet_start > quiet_end:  # Overnight quiet hours
                in_quiet_hours = current_hour >= quiet_start or current_hour <= quiet_end
            else:  # Same day quiet hours
                in_quiet_hours = quiet_start <= current_hour <= quiet_end
            
            if in_quiet_hours:
                return 0.2  # Low score during quiet hours
            
            # Check business hours preference
            business_only = timing_prefs.get('business_hours_only', False)
            if business_only and (current_hour < 9 or current_hour > 17):
                return 0.3  # Low score outside business hours
            
            # Check weekend preference
            weekend_delivery = timing_prefs.get('weekend_delivery', True)
            if not weekend_delivery and current_time.weekday() >= 5:
                return 0.3  # Low score on weekends
            
            # Check preferred hours
            preferred_key = f"preferred_hour_{current_hour}"
            if preferred_key in timing_prefs:
                return timing_prefs[preferred_key]
            
            return 0.8  # Default good score
            
        except Exception as e:
            self.logger.error(f"❌ Error checking timing preference: {e}")
            return 0.5
    
    async def _check_frequency_limits(
        self,
        profile: PreferenceProfile,
        user_id: str
    ) -> float:
        """Check frequency limits"""
        try:
            freq_prefs = profile.frequency_preferences
            
            # Get recent notification count
            recent_count = await self._get_recent_notification_count(user_id, hours=1)
            hourly_limit = freq_prefs.get('hourly_limit', 5)
            
            if recent_count >= hourly_limit:
                return 0.1  # Very low score if over limit
            
            # Calculate score based on current usage
            usage_ratio = recent_count / hourly_limit
            return 1.0 - (usage_ratio * 0.5)  # Reduce score as usage increases
            
        except Exception as e:
            self.logger.error(f"❌ Error checking frequency limits: {e}")
            return 0.5
    
    async def _create_default_profile(self, user_id: str) -> PreferenceProfile:
        """Create default preference profile"""
        return PreferenceProfile(
            user_id=user_id,
            channel_preferences=self.default_preferences['channels'].copy(),
            category_preferences=self.default_preferences['categories'].copy(),
            timing_preferences=self.default_preferences['timing'].copy(),
            frequency_preferences=self.default_preferences['frequency'].copy(),
            content_preferences={},
            last_updated=datetime.now(),
            confidence_score=0.5
        )
    
    # Database methods
    async def _get_stored_preferences(self, user_id: str) -> List[UserPreference]:
        """Get stored preferences from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM user_preferences 
                WHERE user_id = ?
                ORDER BY last_updated DESC
            """, (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            preferences = []
            for row in results:
                preferences.append(UserPreference(
                    user_id=row[0],
                    preference_type=PreferenceType(row[1]),
                    preference_key=row[2],
                    preference_value=json.loads(row[3]) if row[3] else None,
                    confidence_score=row[4],
                    learned_from_behavior=bool(row[5]),
                    last_updated=datetime.fromisoformat(row[6])
                ))
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"❌ Error getting stored preferences: {e}")
            return []
    
    async def _store_preference(self, preference: UserPreference):
        """Store preference in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT,
                    preference_type TEXT,
                    preference_key TEXT,
                    preference_value TEXT,
                    confidence_score REAL,
                    learned_from_behavior INTEGER,
                    last_updated TEXT,
                    PRIMARY KEY (user_id, preference_type, preference_key)
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO user_preferences
                (user_id, preference_type, preference_key, preference_value,
                 confidence_score, learned_from_behavior, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                preference.user_id,
                preference.preference_type.value,
                preference.preference_key,
                json.dumps(preference.preference_value),
                preference.confidence_score,
                int(preference.learned_from_behavior),
                preference.last_updated.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing preference: {e}")
    
    async def _store_feedback(self, feedback: NotificationFeedback):
        """Store feedback in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notification_feedback (
                    notification_id TEXT,
                    user_id TEXT,
                    feedback_type TEXT,
                    feedback_value TEXT,
                    channel TEXT,
                    category TEXT,
                    priority TEXT,
                    feedback_time TEXT,
                    metadata TEXT,
                    PRIMARY KEY (notification_id, user_id, feedback_type)
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO notification_feedback
                (notification_id, user_id, feedback_type, feedback_value,
                 channel, category, priority, feedback_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.notification_id,
                feedback.user_id,
                feedback.feedback_type.value,
                json.dumps(feedback.feedback_value),
                feedback.channel.value,
                feedback.category.value,
                feedback.priority.value,
                feedback.feedback_time.isoformat(),
                json.dumps(feedback.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing feedback: {e}")
    
    async def _get_recent_notification_count(self, user_id: str, hours: int = 1) -> int:
        """Get recent notification count"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT COUNT(*) FROM delivery_results
                WHERE user_id = ? AND delivered_at > ? AND status = 'delivered'
            """, (user_id, since_time.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recent notification count: {e}")
            return 0
    
    # Cache management methods
    async def _get_cached_profile(self, user_id: str) -> Optional[PreferenceProfile]:
        """Get cached preference profile"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"profile:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return PreferenceProfile(
                    user_id=data['user_id'],
                    channel_preferences=data['channel_preferences'],
                    category_preferences=data['category_preferences'],
                    timing_preferences=data['timing_preferences'],
                    frequency_preferences=data['frequency_preferences'],
                    content_preferences=data['content_preferences'],
                    last_updated=datetime.fromisoformat(data['last_updated']),
                    confidence_score=data['confidence_score']
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_profile(self, profile: PreferenceProfile):
        """Cache preference profile"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"profile:{profile.user_id}"
            ttl = 3600  # 1 hour
            
            data = {
                'user_id': profile.user_id,
                'channel_preferences': profile.channel_preferences,
                'category_preferences': profile.category_preferences,
                'timing_preferences': profile.timing_preferences,
                'frequency_preferences': profile.frequency_preferences,
                'content_preferences': profile.content_preferences,
                'last_updated': profile.last_updated.isoformat(),
                'confidence_score': profile.confidence_score
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    async def _clear_user_cache(self, user_id: str):
        """Clear user cache"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"profile:{user_id}"
            self.redis_client.delete(cache_key)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache clear error: {e}")
    
    # Status and management methods
    def get_manager_status(self) -> Dict[str, Any]:
        """Get preference manager status"""
        return {
            "status": "operational",
            "preferences_updated": self.preferences_updated,
            "feedback_processed": self.feedback_processed,
            "profiles_generated": self.profiles_generated,
            "default_preferences": self.default_preferences,
            "feedback_weights": {
                fb.value: weight for fb, weight in self.feedback_weights.items()
            },
            "learning_rates": self.learning_rates,
            "configuration": {
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the preference manager
    async def test_preference_manager():
        print("⚙️ Testing Preference Manager")
        print("=" * 50)
        
        try:
            manager = PreferenceManager()
            
            # Test getting preferences
            print("Getting user preferences...")
            profile = await manager.get_user_preferences("user_123")
            print(f"Preference Profile: {profile.confidence_score:.2f} confidence")
            
            # Test updating preference
            print("Updating preference...")
            success = await manager.update_user_preference(
                "user_123",
                PreferenceType.CHANNEL,
                "push",
                0.9
            )
            print(f"Update Success: {success}")
            
            # Test recording feedback
            print("Recording feedback...")
            feedback_success = await manager.record_feedback(
                "test_notification_1",
                "user_123",
                FeedbackType.CLICKED,
                channel=DeliveryChannel.PUSH,
                category=NotificationCategory.OPPORTUNITY
            )
            print(f"Feedback Success: {feedback_success}")
            
            # Test delivery check
            print("Checking delivery preferences...")
            should_deliver, score = await manager.should_deliver_notification(
                "user_123",
                DeliveryChannel.PUSH,
                NotificationCategory.OPPORTUNITY,
                NotificationPriority.HIGH
            )
            print(f"Should Deliver: {should_deliver} (score: {score:.3f})")
            
            # Test status
            status = manager.get_manager_status()
            print(f"Manager Status: {status['status']}")
            
            print("\n✅ Preference Manager test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_preference_manager())