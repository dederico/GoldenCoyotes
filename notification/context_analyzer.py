#!/usr/bin/env python3
"""
Context Analyzer
User context analysis for intelligent notification delivery

This analyzer:
- Monitors user activity patterns and availability
- Analyzes device and location context
- Tracks user preferences and behavior
- Provides context-aware notification timing
- Supports multi-device context synchronization

Following Task 9 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from intelligence.behavior_analyzer import BehaviorAnalyzer

logger = logging.getLogger(__name__)


class ActivityLevel(Enum):
    """User activity levels"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INACTIVE = "inactive"


class DeviceType(Enum):
    """Device types"""
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    WEARABLE = "wearable"


class LocationType(Enum):
    """Location types"""
    HOME = "home"
    WORK = "work"
    COMMUTE = "commute"
    TRAVEL = "travel"
    UNKNOWN = "unknown"


@dataclass
class UserContext:
    """Current user context"""
    
    user_id: str
    activity_level: ActivityLevel
    device_context: Dict[str, Any]
    location_context: Dict[str, Any]
    time_context: Dict[str, Any]
    availability_score: float
    attention_score: float
    preferences: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    last_updated: datetime
    expires_at: datetime


@dataclass
class DeviceContext:
    """Device context information"""
    
    device_id: str
    device_type: DeviceType
    is_active: bool
    is_primary: bool
    battery_level: float
    network_status: str
    app_usage: Dict[str, float]
    last_interaction: datetime


@dataclass
class LocationContext:
    """Location context information"""
    
    location_id: str
    location_type: LocationType
    is_familiar: bool
    stay_duration: int  # minutes
    movement_pattern: str
    proximity_to_others: bool
    environmental_factors: Dict[str, Any]


@dataclass
class TimeContext:
    """Time-based context information"""
    
    hour_of_day: int
    day_of_week: int
    is_business_hours: bool
    is_weekend: bool
    time_zone: str
    typical_activity: str
    schedule_conflicts: List[Dict[str, Any]]


@dataclass
class ContextAnalysis:
    """Context analysis result"""
    
    user_id: str
    context_score: float
    availability_probability: float
    attention_probability: float
    interruption_tolerance: float
    optimal_delivery_window: Tuple[datetime, datetime]
    context_factors: Dict[str, float]
    recommendations: List[str]
    confidence: float
    analyzed_at: datetime


class ContextAnalyzer:
    """
    User context analysis system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Context Analyzer
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔍 Initializing Context Analyzer")
        
        # Initialize behavior analyzer
        self.behavior_analyzer = BehaviorAnalyzer(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize context models
        self._setup_context_models()
        
        # Performance tracking
        self.contexts_analyzed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Context scoring weights
        self.context_weights = {
            'activity_level': 0.25,
            'device_context': 0.20,
            'location_context': 0.15,
            'time_context': 0.20,
            'behavior_patterns': 0.20
        }
        
        self.logger.info("✅ Context Analyzer initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for context analysis")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching context data"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for context caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for context caching: {e}")
            self.redis_enabled = False
    
    def _setup_context_models(self):
        """Setup context analysis models"""
        try:
            # Activity level thresholds
            self.activity_thresholds = {
                ActivityLevel.VERY_HIGH: 0.8,
                ActivityLevel.HIGH: 0.6,
                ActivityLevel.MEDIUM: 0.4,
                ActivityLevel.LOW: 0.2,
                ActivityLevel.INACTIVE: 0.0
            }
            
            # Time context patterns
            self.time_patterns = {
                'business_hours': (9, 17),
                'evening_hours': (18, 22),
                'night_hours': (23, 6),
                'morning_hours': (7, 8)
            }
            
            # Device priority weights
            self.device_priorities = {
                DeviceType.MOBILE: 0.9,
                DeviceType.DESKTOP: 0.8,
                DeviceType.TABLET: 0.7,
                DeviceType.WEARABLE: 0.6
            }
            
            # Location context weights
            self.location_weights = {
                LocationType.WORK: 0.9,
                LocationType.HOME: 0.8,
                LocationType.COMMUTE: 0.4,
                LocationType.TRAVEL: 0.3,
                LocationType.UNKNOWN: 0.5
            }
            
            self.logger.info("✅ Context analysis models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up context models: {e}")
            raise
    
    async def analyze_user_context(
        self,
        user_id: str,
        device_data: Dict[str, Any] = None,
        location_data: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> UserContext:
        """
        Analyze current user context
        
        Args:
            user_id: User identifier
            device_data: Current device information
            location_data: Current location information
            use_cache: Whether to use cached context
            
        Returns:
            UserContext with current context analysis
        """
        try:
            self.logger.info(f"🔍 Analyzing context for user {user_id}")
            
            # Check cache first
            if use_cache:
                cached_context = await self._get_cached_context(user_id)
                if cached_context:
                    self.cache_hits += 1
                    return cached_context
            
            self.cache_misses += 1
            
            # Analyze activity level
            activity_level = await self._analyze_activity_level(user_id)
            
            # Analyze device context
            device_context = await self._analyze_device_context(user_id, device_data)
            
            # Analyze location context
            location_context = await self._analyze_location_context(user_id, location_data)
            
            # Analyze time context
            time_context = await self._analyze_time_context(user_id)
            
            # Calculate availability score
            availability_score = await self._calculate_availability_score(
                activity_level, device_context, location_context, time_context
            )
            
            # Calculate attention score
            attention_score = await self._calculate_attention_score(
                activity_level, device_context, location_context, time_context
            )
            
            # Get user preferences
            preferences = await self._get_user_preferences(user_id)
            
            # Get behavior patterns
            behavior_patterns = await self._get_behavior_patterns(user_id)
            
            # Create user context
            user_context = UserContext(
                user_id=user_id,
                activity_level=activity_level,
                device_context=device_context,
                location_context=location_context,
                time_context=time_context,
                availability_score=availability_score,
                attention_score=attention_score,
                preferences=preferences,
                behavior_patterns=behavior_patterns,
                last_updated=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=5)
            )
            
            # Cache the context
            if use_cache:
                await self._cache_context(user_context)
            
            # Store in database
            await self._store_context(user_context)
            
            self.contexts_analyzed += 1
            self.logger.info(f"✅ Context analyzed for user {user_id}")
            
            return user_context
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing user context: {e}")
            raise
    
    async def perform_context_analysis(
        self,
        user_id: str,
        context_data: Dict[str, Any] = None
    ) -> ContextAnalysis:
        """
        Perform comprehensive context analysis
        
        Args:
            user_id: User identifier
            context_data: Additional context data
            
        Returns:
            ContextAnalysis with detailed analysis
        """
        try:
            self.logger.info(f"🔍 Performing context analysis for user {user_id}")
            
            # Get user context
            user_context = await self.analyze_user_context(user_id)
            
            # Calculate context score
            context_score = await self._calculate_context_score(user_context)
            
            # Calculate availability probability
            availability_probability = await self._calculate_availability_probability(user_context)
            
            # Calculate attention probability
            attention_probability = await self._calculate_attention_probability(user_context)
            
            # Calculate interruption tolerance
            interruption_tolerance = await self._calculate_interruption_tolerance(user_context)
            
            # Find optimal delivery window
            optimal_window = await self._find_optimal_delivery_window(user_context)
            
            # Analyze context factors
            context_factors = await self._analyze_context_factors(user_context)
            
            # Generate recommendations
            recommendations = await self._generate_context_recommendations(user_context)
            
            # Calculate confidence
            confidence = await self._calculate_analysis_confidence(user_context)
            
            # Create analysis result
            analysis = ContextAnalysis(
                user_id=user_id,
                context_score=context_score,
                availability_probability=availability_probability,
                attention_probability=attention_probability,
                interruption_tolerance=interruption_tolerance,
                optimal_delivery_window=optimal_window,
                context_factors=context_factors,
                recommendations=recommendations,
                confidence=confidence,
                analyzed_at=datetime.now()
            )
            
            # Store analysis
            await self._store_context_analysis(analysis)
            
            self.logger.info(f"✅ Context analysis completed for user {user_id}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error performing context analysis: {e}")
            raise
    
    async def _analyze_activity_level(self, user_id: str) -> ActivityLevel:
        """Analyze user activity level"""
        try:
            # Get recent user interactions
            recent_interactions = await self.behavior_analyzer.get_recent_interactions(
                user_id, hours=1
            )
            
            if not recent_interactions:
                return ActivityLevel.INACTIVE
            
            # Calculate activity score
            activity_score = len(recent_interactions) / 10.0  # Normalize to 0-1
            activity_score = min(1.0, activity_score)
            
            # Determine activity level
            for level, threshold in sorted(self.activity_thresholds.items(), 
                                         key=lambda x: x[1], reverse=True):
                if activity_score >= threshold:
                    return level
            
            return ActivityLevel.INACTIVE
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing activity level: {e}")
            return ActivityLevel.MEDIUM
    
    async def _analyze_device_context(
        self,
        user_id: str,
        device_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze device context"""
        try:
            device_context = {
                'is_active': False,
                'is_primary_device': False,
                'device_type': 'unknown',
                'battery_level': 1.0,
                'network_status': 'unknown',
                'app_usage': {}
            }
            
            if device_data:
                device_context.update(device_data)
            
            # Get device usage patterns
            device_patterns = await self._get_device_patterns(user_id)
            if device_patterns:
                device_context['usage_patterns'] = device_patterns
            
            return device_context
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing device context: {e}")
            return {}
    
    async def _analyze_location_context(
        self,
        user_id: str,
        location_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Analyze location context"""
        try:
            location_context = {
                'location_type': 'unknown',
                'is_familiar': False,
                'stay_duration': 0,
                'movement_pattern': 'stationary',
                'proximity_to_others': False
            }
            
            if location_data:
                location_context.update(location_data)
            
            # Get location patterns
            location_patterns = await self._get_location_patterns(user_id)
            if location_patterns:
                location_context['patterns'] = location_patterns
            
            return location_context
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing location context: {e}")
            return {}
    
    async def _analyze_time_context(self, user_id: str) -> Dict[str, Any]:
        """Analyze time context"""
        try:
            now = datetime.now()
            
            time_context = {
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),
                'is_business_hours': 9 <= now.hour <= 17,
                'is_weekend': now.weekday() >= 5,
                'time_zone': 'UTC',
                'typical_activity': 'unknown'
            }
            
            # Get typical activity for this time
            typical_activity = await self._get_typical_activity(user_id, now.hour, now.weekday())
            if typical_activity:
                time_context['typical_activity'] = typical_activity
            
            # Check for schedule conflicts
            schedule_conflicts = await self._check_schedule_conflicts(user_id, now)
            time_context['schedule_conflicts'] = schedule_conflicts
            
            return time_context
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing time context: {e}")
            return {}
    
    async def _calculate_availability_score(
        self,
        activity_level: ActivityLevel,
        device_context: Dict[str, Any],
        location_context: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> float:
        """Calculate availability score"""
        try:
            availability_score = 0.0
            
            # Activity level factor
            activity_factor = self.activity_thresholds.get(activity_level, 0.5)
            availability_score += activity_factor * 0.3
            
            # Device factor
            if device_context.get('is_active', False):
                availability_score += 0.2
            
            # Location factor
            location_type = location_context.get('location_type', 'unknown')
            if location_type == 'work':
                availability_score += 0.3
            elif location_type == 'home':
                availability_score += 0.25
            
            # Time factor
            if time_context.get('is_business_hours', False):
                availability_score += 0.2
            
            return min(1.0, availability_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating availability score: {e}")
            return 0.5
    
    async def _calculate_attention_score(
        self,
        activity_level: ActivityLevel,
        device_context: Dict[str, Any],
        location_context: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> float:
        """Calculate attention score"""
        try:
            attention_score = 0.0
            
            # Activity level factor (inverse - lower activity = higher attention)
            activity_factor = 1.0 - self.activity_thresholds.get(activity_level, 0.5)
            attention_score += activity_factor * 0.3
            
            # Device factor
            if device_context.get('is_primary_device', False):
                attention_score += 0.2
            
            # Location factor
            location_type = location_context.get('location_type', 'unknown')
            if location_type in ['work', 'home']:
                attention_score += 0.3
            
            # Time factor
            if not time_context.get('is_business_hours', False):
                attention_score += 0.2
            
            return min(1.0, attention_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating attention score: {e}")
            return 0.5
    
    async def _calculate_context_score(self, user_context: UserContext) -> float:
        """Calculate overall context score"""
        try:
            context_score = 0.0
            
            # Activity level score
            activity_score = self.activity_thresholds.get(user_context.activity_level, 0.5)
            context_score += activity_score * self.context_weights['activity_level']
            
            # Device context score
            device_score = 0.5
            if user_context.device_context.get('is_active', False):
                device_score += 0.3
            if user_context.device_context.get('is_primary_device', False):
                device_score += 0.2
            context_score += device_score * self.context_weights['device_context']
            
            # Location context score
            location_type = user_context.location_context.get('location_type', 'unknown')
            location_score = 0.5
            if location_type in ['work', 'home']:
                location_score = 0.8
            context_score += location_score * self.context_weights['location_context']
            
            # Time context score
            time_score = 0.5
            if user_context.time_context.get('is_business_hours', False):
                time_score += 0.3
            context_score += time_score * self.context_weights['time_context']
            
            # Behavior patterns score
            behavior_score = 0.5  # Default
            context_score += behavior_score * self.context_weights['behavior_patterns']
            
            return min(1.0, context_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating context score: {e}")
            return 0.5
    
    async def _calculate_availability_probability(self, user_context: UserContext) -> float:
        """Calculate availability probability"""
        try:
            # Base on availability score with some randomness
            base_probability = user_context.availability_score
            
            # Adjust based on historical patterns
            historical_availability = await self._get_historical_availability(user_context.user_id)
            if historical_availability:
                base_probability = (base_probability + historical_availability) / 2
            
            return min(1.0, base_probability)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating availability probability: {e}")
            return 0.5
    
    async def _calculate_attention_probability(self, user_context: UserContext) -> float:
        """Calculate attention probability"""
        try:
            # Base on attention score
            base_probability = user_context.attention_score
            
            # Adjust based on current activity
            if user_context.activity_level in [ActivityLevel.VERY_HIGH, ActivityLevel.HIGH]:
                base_probability *= 0.7  # Reduce if very active
            
            return min(1.0, base_probability)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating attention probability: {e}")
            return 0.5
    
    async def _calculate_interruption_tolerance(self, user_context: UserContext) -> float:
        """Calculate interruption tolerance"""
        try:
            tolerance = 0.5
            
            # Adjust based on activity level
            if user_context.activity_level == ActivityLevel.LOW:
                tolerance += 0.3
            elif user_context.activity_level == ActivityLevel.INACTIVE:
                tolerance += 0.5
            
            # Adjust based on location
            location_type = user_context.location_context.get('location_type', 'unknown')
            if location_type == 'home':
                tolerance += 0.2
            elif location_type == 'work':
                tolerance -= 0.1
            
            return min(1.0, max(0.0, tolerance))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating interruption tolerance: {e}")
            return 0.5
    
    async def _find_optimal_delivery_window(
        self,
        user_context: UserContext
    ) -> Tuple[datetime, datetime]:
        """Find optimal delivery window"""
        try:
            now = datetime.now()
            
            # Default window: next hour
            start_time = now + timedelta(minutes=5)
            end_time = now + timedelta(hours=1)
            
            # Adjust based on availability
            if user_context.availability_score > 0.8:
                # High availability - deliver soon
                start_time = now + timedelta(minutes=1)
                end_time = now + timedelta(minutes=30)
            elif user_context.availability_score < 0.3:
                # Low availability - delay delivery
                start_time = now + timedelta(hours=1)
                end_time = now + timedelta(hours=4)
            
            # Adjust based on time of day
            if user_context.time_context.get('is_business_hours', False):
                # Business hours - more flexible
                end_time = min(end_time, now.replace(hour=17, minute=0, second=0))
            
            return (start_time, end_time)
            
        except Exception as e:
            self.logger.error(f"❌ Error finding optimal delivery window: {e}")
            now = datetime.now()
            return (now + timedelta(minutes=5), now + timedelta(hours=1))
    
    async def _analyze_context_factors(self, user_context: UserContext) -> Dict[str, float]:
        """Analyze individual context factors"""
        try:
            factors = {}
            
            # Activity factor
            factors['activity_level'] = self.activity_thresholds.get(
                user_context.activity_level, 0.5
            )
            
            # Device factors
            factors['device_active'] = 1.0 if user_context.device_context.get('is_active', False) else 0.0
            factors['primary_device'] = 1.0 if user_context.device_context.get('is_primary_device', False) else 0.0
            
            # Location factors
            location_type = user_context.location_context.get('location_type', 'unknown')
            factors['location_work'] = 1.0 if location_type == 'work' else 0.0
            factors['location_home'] = 1.0 if location_type == 'home' else 0.0
            
            # Time factors
            factors['business_hours'] = 1.0 if user_context.time_context.get('is_business_hours', False) else 0.0
            factors['weekend'] = 1.0 if user_context.time_context.get('is_weekend', False) else 0.0
            
            return factors
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing context factors: {e}")
            return {}
    
    async def _generate_context_recommendations(self, user_context: UserContext) -> List[str]:
        """Generate context-based recommendations"""
        try:
            recommendations = []
            
            # Activity-based recommendations
            if user_context.activity_level == ActivityLevel.VERY_HIGH:
                recommendations.append("User is very active - consider batching notifications")
            elif user_context.activity_level == ActivityLevel.INACTIVE:
                recommendations.append("User is inactive - good time for important notifications")
            
            # Device-based recommendations
            if not user_context.device_context.get('is_active', False):
                recommendations.append("User device is not active - consider alternative delivery methods")
            
            # Location-based recommendations
            location_type = user_context.location_context.get('location_type', 'unknown')
            if location_type == 'work':
                recommendations.append("User at work - prioritize business-related notifications")
            elif location_type == 'home':
                recommendations.append("User at home - good time for personal notifications")
            
            # Time-based recommendations
            if user_context.time_context.get('is_business_hours', False):
                recommendations.append("Business hours - appropriate for work notifications")
            else:
                recommendations.append("Outside business hours - use for urgent notifications only")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recommendations: {e}")
            return []
    
    async def _calculate_analysis_confidence(self, user_context: UserContext) -> float:
        """Calculate confidence in context analysis"""
        try:
            confidence = 0.0
            
            # Base confidence on data availability
            if user_context.device_context:
                confidence += 0.3
            
            if user_context.location_context:
                confidence += 0.3
            
            if user_context.behavior_patterns:
                confidence += 0.2
            
            if user_context.preferences:
                confidence += 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating analysis confidence: {e}")
            return 0.5
    
    # Helper methods
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user preferences: {e}")
            return {}
    
    async def _get_behavior_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get user behavior patterns"""
        try:
            patterns = await self.behavior_analyzer.get_behavior_patterns(user_id)
            return patterns.to_dict() if patterns else {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting behavior patterns: {e}")
            return {}
    
    async def _get_device_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get device usage patterns"""
        try:
            # This would typically query device usage data
            return {
                'primary_device': 'mobile',
                'usage_hours': [9, 10, 11, 14, 15, 16, 19, 20, 21],
                'app_preferences': ['email', 'messaging', 'social']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting device patterns: {e}")
            return {}
    
    async def _get_location_patterns(self, user_id: str) -> Dict[str, Any]:
        """Get location patterns"""
        try:
            # This would typically query location data
            return {
                'work_location': 'office',
                'home_location': 'home',
                'commute_pattern': 'regular',
                'frequent_locations': ['office', 'home', 'gym']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting location patterns: {e}")
            return {}
    
    async def _get_typical_activity(self, user_id: str, hour: int, day_of_week: int) -> str:
        """Get typical activity for time"""
        try:
            # This would typically query historical activity data
            if 9 <= hour <= 17 and day_of_week < 5:
                return 'work'
            elif 18 <= hour <= 22:
                return 'leisure'
            elif 23 <= hour or hour <= 6:
                return 'sleep'
            else:
                return 'personal'
                
        except Exception as e:
            self.logger.error(f"❌ Error getting typical activity: {e}")
            return 'unknown'
    
    async def _check_schedule_conflicts(self, user_id: str, time: datetime) -> List[Dict[str, Any]]:
        """Check for schedule conflicts"""
        try:
            # This would typically query calendar/schedule data
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Error checking schedule conflicts: {e}")
            return []
    
    async def _get_historical_availability(self, user_id: str) -> Optional[float]:
        """Get historical availability for this time"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(availability_score) 
                FROM user_contexts 
                WHERE user_id = ? AND last_updated > ?
            """, (user_id, (datetime.now() - timedelta(days=30)).isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result and result[0] else None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting historical availability: {e}")
            return None
    
    # Cache management methods
    async def _get_cached_context(self, user_id: str) -> Optional[UserContext]:
        """Get cached context"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"context:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return UserContext(
                    user_id=data['user_id'],
                    activity_level=ActivityLevel(data['activity_level']),
                    device_context=data['device_context'],
                    location_context=data['location_context'],
                    time_context=data['time_context'],
                    availability_score=data['availability_score'],
                    attention_score=data['attention_score'],
                    preferences=data['preferences'],
                    behavior_patterns=data['behavior_patterns'],
                    last_updated=datetime.fromisoformat(data['last_updated']),
                    expires_at=datetime.fromisoformat(data['expires_at'])
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_context(self, user_context: UserContext):
        """Cache user context"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"context:{user_context.user_id}"
            ttl = 300  # 5 minutes
            
            data = {
                'user_id': user_context.user_id,
                'activity_level': user_context.activity_level.value,
                'device_context': user_context.device_context,
                'location_context': user_context.location_context,
                'time_context': user_context.time_context,
                'availability_score': user_context.availability_score,
                'attention_score': user_context.attention_score,
                'preferences': user_context.preferences,
                'behavior_patterns': user_context.behavior_patterns,
                'last_updated': user_context.last_updated.isoformat(),
                'expires_at': user_context.expires_at.isoformat()
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Database methods
    async def _store_context(self, user_context: UserContext):
        """Store user context in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_contexts (
                    user_id TEXT,
                    activity_level TEXT,
                    device_context TEXT,
                    location_context TEXT,
                    time_context TEXT,
                    availability_score REAL,
                    attention_score REAL,
                    preferences TEXT,
                    behavior_patterns TEXT,
                    last_updated TEXT,
                    expires_at TEXT,
                    PRIMARY KEY (user_id, last_updated)
                )
            """)
            
            # Insert context
            cursor.execute("""
                INSERT OR REPLACE INTO user_contexts
                (user_id, activity_level, device_context, location_context, time_context,
                 availability_score, attention_score, preferences, behavior_patterns,
                 last_updated, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_context.user_id,
                user_context.activity_level.value,
                json.dumps(user_context.device_context),
                json.dumps(user_context.location_context),
                json.dumps(user_context.time_context),
                user_context.availability_score,
                user_context.attention_score,
                json.dumps(user_context.preferences),
                json.dumps(user_context.behavior_patterns),
                user_context.last_updated.isoformat(),
                user_context.expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing context: {e}")
    
    async def _store_context_analysis(self, analysis: ContextAnalysis):
        """Store context analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_analyses (
                    user_id TEXT,
                    context_score REAL,
                    availability_probability REAL,
                    attention_probability REAL,
                    interruption_tolerance REAL,
                    optimal_window_start TEXT,
                    optimal_window_end TEXT,
                    context_factors TEXT,
                    recommendations TEXT,
                    confidence REAL,
                    analyzed_at TEXT,
                    PRIMARY KEY (user_id, analyzed_at)
                )
            """)
            
            # Insert analysis
            cursor.execute("""
                INSERT OR REPLACE INTO context_analyses
                (user_id, context_score, availability_probability, attention_probability,
                 interruption_tolerance, optimal_window_start, optimal_window_end,
                 context_factors, recommendations, confidence, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis.user_id,
                analysis.context_score,
                analysis.availability_probability,
                analysis.attention_probability,
                analysis.interruption_tolerance,
                analysis.optimal_delivery_window[0].isoformat(),
                analysis.optimal_delivery_window[1].isoformat(),
                json.dumps(analysis.context_factors),
                json.dumps(analysis.recommendations),
                analysis.confidence,
                analysis.analyzed_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing context analysis: {e}")
    
    # Status and management methods
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get context analyzer status"""
        return {
            "status": "operational",
            "contexts_analyzed": self.contexts_analyzed,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "context_weights": self.context_weights,
            "activity_thresholds": {
                level.value: threshold 
                for level, threshold in self.activity_thresholds.items()
            },
            "configuration": {
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_context_cache(self, user_id: str = None) -> int:
        """Clear context cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if user_id:
                pattern = f"context:{user_id}"
            else:
                pattern = "context:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing context cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the context analyzer
    async def test_context_analyzer():
        print("🔍 Testing Context Analyzer")
        print("=" * 50)
        
        try:
            analyzer = ContextAnalyzer()
            
            # Test context analysis
            print("Analyzing user context...")
            user_context = await analyzer.analyze_user_context(
                user_id="user_123",
                device_data={'is_active': True, 'device_type': 'mobile'},
                location_data={'location_type': 'work', 'is_familiar': True}
            )
            print(f"User Context: {user_context.activity_level.value}, availability: {user_context.availability_score:.3f}")
            
            # Test comprehensive analysis
            print("Performing context analysis...")
            analysis = await analyzer.perform_context_analysis("user_123")
            print(f"Context Analysis: score {analysis.context_score:.3f}, confidence {analysis.confidence:.3f}")
            print(f"Optimal window: {analysis.optimal_delivery_window[0].strftime('%H:%M')} - {analysis.optimal_delivery_window[1].strftime('%H:%M')}")
            
            # Test recommendations
            print(f"Recommendations: {len(analysis.recommendations)}")
            for rec in analysis.recommendations:
                print(f"  - {rec}")
            
            # Test status
            status = analyzer.get_analyzer_status()
            print(f"Analyzer Status: {status['status']}")
            
            print("\n✅ Context Analyzer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_context_analyzer())