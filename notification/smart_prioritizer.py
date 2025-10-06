#!/usr/bin/env python3
"""
Smart Prioritizer
Intelligent notification prioritization system

This prioritizer:
- Analyzes notification importance and urgency
- Considers user context and availability
- Implements priority scoring algorithms
- Manages notification queues and delivery timing
- Prevents notification overload and spam

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
from analytics.insight_generator import InsightGenerator

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NotificationCategory(Enum):
    """Notification categories"""
    OPPORTUNITY = "opportunity"
    INSIGHT = "insight"
    ALERT = "alert"
    RECOMMENDATION = "recommendation"
    SYSTEM = "system"


@dataclass
class NotificationRequest:
    """Request for notification prioritization"""
    
    notification_id: str
    user_id: str
    title: str
    message: str
    category: NotificationCategory
    source_data: Dict[str, Any]
    urgency_factors: Dict[str, float]
    context_factors: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class PriorityScore:
    """Priority scoring result"""
    
    notification_id: str
    user_id: str
    priority_level: NotificationPriority
    priority_score: float
    urgency_score: float
    relevance_score: float
    timing_score: float
    frequency_penalty: float
    context_boost: float
    calculated_at: datetime
    expires_at: datetime


@dataclass
class NotificationQueue:
    """Notification queue entry"""
    
    notification_id: str
    user_id: str
    priority_score: PriorityScore
    scheduled_delivery: datetime
    delivery_attempts: int
    status: str  # pending, delivered, failed, cancelled
    queue_position: int


@dataclass
class PriorityMetrics:
    """Priority system metrics"""
    
    total_notifications: int
    processed_notifications: int
    avg_priority_score: float
    priority_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    delivery_success_rate: float
    avg_processing_time: float
    last_updated: datetime


class SmartPrioritizer:
    """
    Intelligent notification prioritization system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Smart Prioritizer
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Initializing Smart Prioritizer")
        
        # Initialize insight generator for context
        self.insight_generator = InsightGenerator(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize priority scoring models
        self._setup_priority_models()
        
        # Performance tracking
        self.notifications_processed = 0
        self.priority_calculations = 0
        self.delivery_attempts = 0
        
        # Priority thresholds
        self.priority_thresholds = {
            NotificationPriority.CRITICAL: 0.8,
            NotificationPriority.HIGH: 0.6,
            NotificationPriority.MEDIUM: 0.4,
            NotificationPriority.LOW: 0.0
        }
        
        # Frequency limits per user
        self.frequency_limits = {
            'daily': 20,
            'hourly': 5,
            'per_15_minutes': 2
        }
        
        self.logger.info("✅ Smart Prioritizer initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for prioritization")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching priority data"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for priority caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for priority caching: {e}")
            self.redis_enabled = False
    
    def _setup_priority_models(self):
        """Setup priority scoring models"""
        try:
            # Category importance weights
            self.category_weights = {
                NotificationCategory.OPPORTUNITY: 0.9,
                NotificationCategory.INSIGHT: 0.8,
                NotificationCategory.ALERT: 0.85,
                NotificationCategory.RECOMMENDATION: 0.7,
                NotificationCategory.SYSTEM: 0.5
            }
            
            # Urgency factor weights
            self.urgency_weights = {
                'time_sensitivity': 0.3,
                'business_impact': 0.25,
                'user_relevance': 0.2,
                'action_required': 0.15,
                'deadline_proximity': 0.1
            }
            
            # Context factor weights
            self.context_weights = {
                'user_activity': 0.25,
                'device_context': 0.15,
                'location_context': 0.1,
                'time_of_day': 0.2,
                'user_preferences': 0.3
            }
            
            self.logger.info("✅ Priority scoring models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up priority models: {e}")
            raise
    
    async def calculate_priority(
        self,
        request: NotificationRequest,
        user_context: Dict[str, Any] = None
    ) -> PriorityScore:
        """
        Calculate priority score for a notification
        
        Args:
            request: Notification request
            user_context: Additional user context
            
        Returns:
            PriorityScore with detailed scoring
        """
        try:
            self.logger.info(f"🎯 Calculating priority for notification {request.notification_id}")
            
            # Calculate urgency score
            urgency_score = await self._calculate_urgency_score(request)
            
            # Calculate relevance score
            relevance_score = await self._calculate_relevance_score(request, user_context)
            
            # Calculate timing score
            timing_score = await self._calculate_timing_score(request, user_context)
            
            # Calculate frequency penalty
            frequency_penalty = await self._calculate_frequency_penalty(request)
            
            # Calculate context boost
            context_boost = await self._calculate_context_boost(request, user_context)
            
            # Calculate overall priority score
            priority_score = (
                urgency_score * 0.3 +
                relevance_score * 0.25 +
                timing_score * 0.2 +
                context_boost * 0.15 +
                (1.0 - frequency_penalty) * 0.1
            )
            
            # Determine priority level
            priority_level = self._determine_priority_level(priority_score)
            
            # Create priority score object
            result = PriorityScore(
                notification_id=request.notification_id,
                user_id=request.user_id,
                priority_level=priority_level,
                priority_score=priority_score,
                urgency_score=urgency_score,
                relevance_score=relevance_score,
                timing_score=timing_score,
                frequency_penalty=frequency_penalty,
                context_boost=context_boost,
                calculated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            
            # Cache the result
            await self._cache_priority_score(result)
            
            # Store in database
            await self._store_priority_score(result)
            
            self.priority_calculations += 1
            self.logger.info(f"✅ Priority calculated: {priority_level.value} ({priority_score:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating priority: {e}")
            raise
    
    async def prioritize_notifications(
        self,
        requests: List[NotificationRequest],
        user_context: Dict[str, Any] = None
    ) -> List[PriorityScore]:
        """
        Prioritize multiple notifications
        
        Args:
            requests: List of notification requests
            user_context: Additional user context
            
        Returns:
            List of PriorityScore objects sorted by priority
        """
        try:
            self.logger.info(f"🎯 Prioritizing {len(requests)} notifications")
            
            priority_scores = []
            
            # Calculate priority for each notification
            for request in requests:
                try:
                    priority_score = await self.calculate_priority(request, user_context)
                    priority_scores.append(priority_score)
                except Exception as e:
                    self.logger.error(f"❌ Error prioritizing notification {request.notification_id}: {e}")
                    continue
            
            # Sort by priority score (highest first)
            priority_scores.sort(key=lambda x: x.priority_score, reverse=True)
            
            self.logger.info(f"✅ Prioritized {len(priority_scores)} notifications")
            return priority_scores
            
        except Exception as e:
            self.logger.error(f"❌ Error prioritizing notifications: {e}")
            return []
    
    async def create_notification_queue(
        self,
        priority_scores: List[PriorityScore],
        user_preferences: Dict[str, Any] = None
    ) -> List[NotificationQueue]:
        """
        Create notification queue based on priority scores
        
        Args:
            priority_scores: List of priority scores
            user_preferences: User notification preferences
            
        Returns:
            List of NotificationQueue entries
        """
        try:
            self.logger.info(f"📋 Creating notification queue for {len(priority_scores)} items")
            
            queue_entries = []
            current_time = datetime.now()
            
            # Apply frequency limits
            filtered_scores = await self._apply_frequency_limits(priority_scores)
            
            # Schedule delivery times
            for i, priority_score in enumerate(filtered_scores):
                # Calculate delivery time based on priority
                delivery_delay = self._calculate_delivery_delay(priority_score, user_preferences)
                scheduled_delivery = current_time + timedelta(seconds=delivery_delay)
                
                queue_entry = NotificationQueue(
                    notification_id=priority_score.notification_id,
                    user_id=priority_score.user_id,
                    priority_score=priority_score,
                    scheduled_delivery=scheduled_delivery,
                    delivery_attempts=0,
                    status='pending',
                    queue_position=i + 1
                )
                
                queue_entries.append(queue_entry)
            
            # Store queue in database
            await self._store_notification_queue(queue_entries)
            
            self.logger.info(f"✅ Created notification queue with {len(queue_entries)} entries")
            return queue_entries
            
        except Exception as e:
            self.logger.error(f"❌ Error creating notification queue: {e}")
            return []
    
    async def _calculate_urgency_score(self, request: NotificationRequest) -> float:
        """Calculate urgency score based on notification factors"""
        try:
            urgency_score = 0.0
            
            # Category base urgency
            category_urgency = self.category_weights.get(request.category, 0.5)
            urgency_score += category_urgency * 0.3
            
            # Urgency factors
            for factor, weight in self.urgency_weights.items():
                factor_value = request.urgency_factors.get(factor, 0.5)
                urgency_score += factor_value * weight
            
            # Time sensitivity
            if request.expires_at:
                time_remaining = (request.expires_at - datetime.now()).total_seconds()
                if time_remaining < 3600:  # Less than 1 hour
                    urgency_score += 0.2
                elif time_remaining < 86400:  # Less than 24 hours
                    urgency_score += 0.1
            
            return min(1.0, urgency_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating urgency score: {e}")
            return 0.5
    
    async def _calculate_relevance_score(
        self,
        request: NotificationRequest,
        user_context: Dict[str, Any] = None
    ) -> float:
        """Calculate relevance score based on user context"""
        try:
            relevance_score = 0.0
            
            # Base relevance from category
            category_relevance = self.category_weights.get(request.category, 0.5)
            relevance_score += category_relevance * 0.3
            
            # User context relevance
            if user_context:
                # Activity-based relevance
                user_activity = user_context.get('activity_level', 0.5)
                relevance_score += user_activity * 0.2
                
                # Interest-based relevance
                user_interests = user_context.get('interests', [])
                notification_topics = request.source_data.get('topics', [])
                
                if user_interests and notification_topics:
                    topic_overlap = len(set(user_interests) & set(notification_topics))
                    topic_relevance = min(1.0, topic_overlap / len(user_interests))
                    relevance_score += topic_relevance * 0.25
                
                # Behavioral relevance
                user_behavior = user_context.get('behavior_patterns', {})
                if user_behavior:
                    behavior_match = self._calculate_behavior_match(
                        user_behavior, request.source_data
                    )
                    relevance_score += behavior_match * 0.25
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating relevance score: {e}")
            return 0.5
    
    async def _calculate_timing_score(
        self,
        request: NotificationRequest,
        user_context: Dict[str, Any] = None
    ) -> float:
        """Calculate timing score based on user availability"""
        try:
            timing_score = 0.0
            current_time = datetime.now()
            
            # Time of day factors
            hour = current_time.hour
            
            # Business hours (9 AM - 5 PM) get higher score
            if 9 <= hour <= 17:
                timing_score += 0.3
            # Evening hours (6 PM - 9 PM) get moderate score
            elif 18 <= hour <= 21:
                timing_score += 0.2
            # Night hours get lower score
            else:
                timing_score += 0.1
            
            # Day of week factors
            weekday = current_time.weekday()
            if weekday < 5:  # Monday to Friday
                timing_score += 0.2
            else:  # Weekend
                timing_score += 0.1
            
            # User context timing
            if user_context:
                # User availability
                availability = user_context.get('availability', 0.5)
                timing_score += availability * 0.3
                
                # Device context
                device_context = user_context.get('device_context', {})
                if device_context.get('is_active', False):
                    timing_score += 0.2
            
            return min(1.0, timing_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating timing score: {e}")
            return 0.5
    
    async def _calculate_frequency_penalty(self, request: NotificationRequest) -> float:
        """Calculate frequency penalty to prevent spam"""
        try:
            # Get recent notifications for user
            recent_notifications = await self._get_recent_notifications(
                request.user_id, hours=24
            )
            
            # Count notifications by time periods
            now = datetime.now()
            
            # Last 15 minutes
            last_15min = sum(1 for n in recent_notifications 
                           if (now - n['created_at']).total_seconds() < 900)
            
            # Last hour
            last_hour = sum(1 for n in recent_notifications 
                          if (now - n['created_at']).total_seconds() < 3600)
            
            # Last 24 hours
            last_day = len(recent_notifications)
            
            # Calculate penalties
            penalty = 0.0
            
            if last_15min >= self.frequency_limits['per_15_minutes']:
                penalty += 0.5
            
            if last_hour >= self.frequency_limits['hourly']:
                penalty += 0.3
            
            if last_day >= self.frequency_limits['daily']:
                penalty += 0.2
            
            return min(1.0, penalty)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating frequency penalty: {e}")
            return 0.0
    
    async def _calculate_context_boost(
        self,
        request: NotificationRequest,
        user_context: Dict[str, Any] = None
    ) -> float:
        """Calculate context boost based on current situation"""
        try:
            context_boost = 0.0
            
            if not user_context:
                return context_boost
            
            # Activity boost
            user_activity = user_context.get('activity_level', 0.5)
            context_boost += user_activity * 0.3
            
            # Location boost
            location_context = user_context.get('location_context', {})
            if location_context.get('is_work_location', False):
                context_boost += 0.2
            
            # Device boost
            device_context = user_context.get('device_context', {})
            if device_context.get('is_primary_device', False):
                context_boost += 0.15
            
            # Preference boost
            preferences = user_context.get('preferences', {})
            category_preference = preferences.get(request.category.value, 0.5)
            context_boost += category_preference * 0.35
            
            return min(1.0, context_boost)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating context boost: {e}")
            return 0.0
    
    def _determine_priority_level(self, priority_score: float) -> NotificationPriority:
        """Determine priority level from score"""
        if priority_score >= self.priority_thresholds[NotificationPriority.CRITICAL]:
            return NotificationPriority.CRITICAL
        elif priority_score >= self.priority_thresholds[NotificationPriority.HIGH]:
            return NotificationPriority.HIGH
        elif priority_score >= self.priority_thresholds[NotificationPriority.MEDIUM]:
            return NotificationPriority.MEDIUM
        else:
            return NotificationPriority.LOW
    
    def _calculate_behavior_match(
        self,
        user_behavior: Dict[str, Any],
        notification_data: Dict[str, Any]
    ) -> float:
        """Calculate behavior pattern match"""
        try:
            # Simple behavior matching
            user_patterns = user_behavior.get('patterns', [])
            notification_patterns = notification_data.get('patterns', [])
            
            if not user_patterns or not notification_patterns:
                return 0.5
            
            # Calculate overlap
            overlap = len(set(user_patterns) & set(notification_patterns))
            max_patterns = max(len(user_patterns), len(notification_patterns))
            
            return overlap / max_patterns if max_patterns > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def _apply_frequency_limits(
        self,
        priority_scores: List[PriorityScore]
    ) -> List[PriorityScore]:
        """Apply frequency limits to priority scores"""
        try:
            # Group by user
            user_notifications = {}
            for score in priority_scores:
                if score.user_id not in user_notifications:
                    user_notifications[score.user_id] = []
                user_notifications[score.user_id].append(score)
            
            # Apply limits per user
            filtered_scores = []
            for user_id, scores in user_notifications.items():
                # Sort by priority
                scores.sort(key=lambda x: x.priority_score, reverse=True)
                
                # Apply daily limit
                daily_limit = self.frequency_limits['daily']
                user_filtered = scores[:daily_limit]
                
                filtered_scores.extend(user_filtered)
            
            return filtered_scores
            
        except Exception as e:
            self.logger.error(f"❌ Error applying frequency limits: {e}")
            return priority_scores
    
    def _calculate_delivery_delay(
        self,
        priority_score: PriorityScore,
        user_preferences: Dict[str, Any] = None
    ) -> int:
        """Calculate delivery delay in seconds"""
        try:
            # Base delay by priority
            base_delays = {
                NotificationPriority.CRITICAL: 0,      # Immediate
                NotificationPriority.HIGH: 60,         # 1 minute
                NotificationPriority.MEDIUM: 300,      # 5 minutes
                NotificationPriority.LOW: 900          # 15 minutes
            }
            
            delay = base_delays.get(priority_score.priority_level, 300)
            
            # Adjust based on user preferences
            if user_preferences:
                delay_preference = user_preferences.get('delivery_delay', 1.0)
                delay = int(delay * delay_preference)
            
            return delay
            
        except Exception:
            return 300  # Default 5 minutes
    
    # Database methods
    async def _get_recent_notifications(self, user_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent notifications for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get notifications from last N hours
            since_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT notification_id, created_at, category, priority_score
                FROM notification_priorities
                WHERE user_id = ? AND created_at > ?
                ORDER BY created_at DESC
            """, (user_id, since_time.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'notification_id': row[0],
                    'created_at': datetime.fromisoformat(row[1]),
                    'category': row[2],
                    'priority_score': row[3]
                }
                for row in results
            ]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recent notifications: {e}")
            return []
    
    async def _store_priority_score(self, priority_score: PriorityScore):
        """Store priority score in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notification_priorities (
                    notification_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    priority_level TEXT,
                    priority_score REAL,
                    urgency_score REAL,
                    relevance_score REAL,
                    timing_score REAL,
                    frequency_penalty REAL,
                    context_boost REAL,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            
            # Insert priority score
            cursor.execute("""
                INSERT OR REPLACE INTO notification_priorities
                (notification_id, user_id, priority_level, priority_score, urgency_score,
                 relevance_score, timing_score, frequency_penalty, context_boost,
                 created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                priority_score.notification_id,
                priority_score.user_id,
                priority_score.priority_level.value,
                priority_score.priority_score,
                priority_score.urgency_score,
                priority_score.relevance_score,
                priority_score.timing_score,
                priority_score.frequency_penalty,
                priority_score.context_boost,
                priority_score.calculated_at.isoformat(),
                priority_score.expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing priority score: {e}")
    
    async def _store_notification_queue(self, queue_entries: List[NotificationQueue]):
        """Store notification queue in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notification_queue (
                    notification_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    priority_score REAL,
                    scheduled_delivery TEXT,
                    delivery_attempts INTEGER,
                    status TEXT,
                    queue_position INTEGER,
                    created_at TEXT
                )
            """)
            
            # Insert queue entries
            for entry in queue_entries:
                cursor.execute("""
                    INSERT OR REPLACE INTO notification_queue
                    (notification_id, user_id, priority_score, scheduled_delivery,
                     delivery_attempts, status, queue_position, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.notification_id,
                    entry.user_id,
                    entry.priority_score.priority_score,
                    entry.scheduled_delivery.isoformat(),
                    entry.delivery_attempts,
                    entry.status,
                    entry.queue_position,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing notification queue: {e}")
    
    # Cache management methods
    async def _cache_priority_score(self, priority_score: PriorityScore):
        """Cache priority score"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"priority:{priority_score.notification_id}"
            ttl = 1800  # 30 minutes
            
            data = {
                'notification_id': priority_score.notification_id,
                'user_id': priority_score.user_id,
                'priority_level': priority_score.priority_level.value,
                'priority_score': priority_score.priority_score,
                'urgency_score': priority_score.urgency_score,
                'relevance_score': priority_score.relevance_score,
                'timing_score': priority_score.timing_score,
                'frequency_penalty': priority_score.frequency_penalty,
                'context_boost': priority_score.context_boost,
                'calculated_at': priority_score.calculated_at.isoformat(),
                'expires_at': priority_score.expires_at.isoformat()
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Status and management methods
    def get_prioritizer_status(self) -> Dict[str, Any]:
        """Get prioritizer status"""
        return {
            "status": "operational",
            "notifications_processed": self.notifications_processed,
            "priority_calculations": self.priority_calculations,
            "delivery_attempts": self.delivery_attempts,
            "priority_thresholds": {
                level.value: threshold 
                for level, threshold in self.priority_thresholds.items()
            },
            "frequency_limits": self.frequency_limits,
            "configuration": {
                "redis_enabled": self.redis_enabled,
                "category_weights": {
                    cat.value: weight 
                    for cat, weight in self.category_weights.items()
                }
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_priority_metrics(self) -> PriorityMetrics:
        """Get priority system metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total notifications
            cursor.execute("SELECT COUNT(*) FROM notification_priorities")
            total_notifications = cursor.fetchone()[0]
            
            # Get processed notifications
            cursor.execute("SELECT COUNT(*) FROM notification_queue")
            processed_notifications = cursor.fetchone()[0]
            
            # Get average priority score
            cursor.execute("SELECT AVG(priority_score) FROM notification_priorities")
            avg_priority_score = cursor.fetchone()[0] or 0.0
            
            # Get priority distribution
            cursor.execute("""
                SELECT priority_level, COUNT(*) 
                FROM notification_priorities 
                GROUP BY priority_level
            """)
            priority_distribution = dict(cursor.fetchall())
            
            # Get delivery success rate
            cursor.execute("SELECT COUNT(*) FROM notification_queue WHERE status = 'delivered'")
            delivered_count = cursor.fetchone()[0]
            
            delivery_success_rate = (delivered_count / max(processed_notifications, 1)) * 100
            
            conn.close()
            
            return PriorityMetrics(
                total_notifications=total_notifications,
                processed_notifications=processed_notifications,
                avg_priority_score=avg_priority_score,
                priority_distribution=priority_distribution,
                category_distribution={},  # Would need additional query
                delivery_success_rate=delivery_success_rate,
                avg_processing_time=0.0,  # Would need timing data
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting priority metrics: {e}")
            return PriorityMetrics(
                total_notifications=0,
                processed_notifications=0,
                avg_priority_score=0.0,
                priority_distribution={},
                category_distribution={},
                delivery_success_rate=0.0,
                avg_processing_time=0.0,
                last_updated=datetime.now()
            )
    
    def clear_priority_cache(self, user_id: str = None) -> int:
        """Clear priority cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if user_id:
                pattern = f"priority:*:{user_id}"
            else:
                pattern = "priority:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing priority cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the smart prioritizer
    async def test_smart_prioritizer():
        print("🎯 Testing Smart Prioritizer")
        print("=" * 50)
        
        try:
            prioritizer = SmartPrioritizer()
            
            # Test notification request
            print("Creating notification request...")
            request = NotificationRequest(
                notification_id="test_notification_1",
                user_id="user_123",
                title="High Priority Opportunity",
                message="A high-value opportunity has been identified",
                category=NotificationCategory.OPPORTUNITY,
                source_data={"opportunity_id": "opp_456", "value": 10000},
                urgency_factors={
                    "time_sensitivity": 0.8,
                    "business_impact": 0.9,
                    "user_relevance": 0.7
                },
                context_factors={},
                created_at=datetime.now()
            )
            
            # Test priority calculation
            print("Calculating priority...")
            priority_score = await prioritizer.calculate_priority(request)
            print(f"Priority Score: {priority_score.priority_level.value} ({priority_score.priority_score:.3f})")
            
            # Test batch prioritization
            print("Testing batch prioritization...")
            requests = [request]  # Would normally have multiple requests
            priority_scores = await prioritizer.prioritize_notifications(requests)
            print(f"Batch Prioritization: {len(priority_scores)} notifications prioritized")
            
            # Test queue creation
            print("Creating notification queue...")
            queue = await prioritizer.create_notification_queue(priority_scores)
            print(f"Queue Created: {len(queue)} entries")
            
            # Test metrics
            print("Getting priority metrics...")
            metrics = await prioritizer.get_priority_metrics()
            print(f"Metrics: {metrics.total_notifications} total, {metrics.avg_priority_score:.3f} avg score")
            
            # Test status
            status = prioritizer.get_prioritizer_status()
            print(f"Prioritizer Status: {status['status']}")
            
            print("\n✅ Smart Prioritizer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_smart_prioritizer())