#!/usr/bin/env python3
"""
Delivery Optimizer
Notification delivery optimization and scheduling system

This optimizer:
- Optimizes notification delivery timing and channels
- Manages delivery queues and retry logic
- Tracks delivery success rates and performance
- Implements intelligent batching and throttling
- Supports multi-channel delivery strategies

Following Task 9 from the PRP implementation blueprint.
"""

import json
import sqlite3
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from .smart_prioritizer import SmartPrioritizer, NotificationPriority
from .context_analyzer import ContextAnalyzer

logger = logging.getLogger(__name__)


class DeliveryChannel(Enum):
    """Delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    SLACK = "slack"
    WEBHOOK = "webhook"


class DeliveryStatus(Enum):
    """Delivery status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    THROTTLED = "throttled"


@dataclass
class DeliveryRequest:
    """Delivery request"""
    
    notification_id: str
    user_id: str
    channel: DeliveryChannel
    priority: NotificationPriority
    content: Dict[str, Any]
    delivery_time: datetime
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 30
    metadata: Dict[str, Any] = None


@dataclass
class DeliveryResult:
    """Delivery result"""
    
    notification_id: str
    user_id: str
    channel: DeliveryChannel
    status: DeliveryStatus
    delivered_at: Optional[datetime]
    retry_count: int
    error_message: Optional[str]
    response_time: float
    metadata: Dict[str, Any]


@dataclass
class DeliveryMetrics:
    """Delivery metrics"""
    
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    avg_response_time: float
    success_rate: float
    channel_performance: Dict[str, float]
    retry_rate: float
    last_updated: datetime


class DeliveryOptimizer:
    """
    Notification delivery optimization system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Delivery Optimizer
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Delivery Optimizer")
        
        # Initialize components
        self.prioritizer = SmartPrioritizer(config, ml_config)
        self.context_analyzer = ContextAnalyzer(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client
        self._setup_redis_client()
        
        # Initialize delivery channels
        self._setup_delivery_channels()
        
        # Performance tracking
        self.deliveries_attempted = 0
        self.deliveries_successful = 0
        self.deliveries_failed = 0
        
        # Delivery queues
        self.delivery_queues = {
            channel: asyncio.Queue() for channel in DeliveryChannel
        }
        
        # Throttling limits
        self.throttling_limits = {
            DeliveryChannel.EMAIL: 100,     # per hour
            DeliveryChannel.SMS: 50,        # per hour
            DeliveryChannel.PUSH: 1000,     # per hour
            DeliveryChannel.IN_APP: 5000,   # per hour
            DeliveryChannel.SLACK: 200,     # per hour
            DeliveryChannel.WEBHOOK: 500    # per hour
        }
        
        self.logger.info("✅ Delivery Optimizer initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for delivery optimization")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for delivery queues"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for delivery queues initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for delivery queues: {e}")
            self.redis_enabled = False
    
    def _setup_delivery_channels(self):
        """Setup delivery channels"""
        try:
            # Channel configurations
            self.channel_configs = {
                DeliveryChannel.EMAIL: {
                    'timeout': 30,
                    'max_retries': 3,
                    'retry_delay': 300,  # 5 minutes
                    'batch_size': 10
                },
                DeliveryChannel.SMS: {
                    'timeout': 10,
                    'max_retries': 2,
                    'retry_delay': 60,   # 1 minute
                    'batch_size': 5
                },
                DeliveryChannel.PUSH: {
                    'timeout': 5,
                    'max_retries': 2,
                    'retry_delay': 30,   # 30 seconds
                    'batch_size': 100
                },
                DeliveryChannel.IN_APP: {
                    'timeout': 1,
                    'max_retries': 1,
                    'retry_delay': 10,   # 10 seconds
                    'batch_size': 50
                },
                DeliveryChannel.SLACK: {
                    'timeout': 15,
                    'max_retries': 3,
                    'retry_delay': 120,  # 2 minutes
                    'batch_size': 20
                },
                DeliveryChannel.WEBHOOK: {
                    'timeout': 30,
                    'max_retries': 5,
                    'retry_delay': 180,  # 3 minutes
                    'batch_size': 25
                }
            }
            
            self.logger.info("✅ Delivery channels configured")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up delivery channels: {e}")
            raise
    
    async def optimize_delivery(
        self,
        notification_id: str,
        user_id: str,
        content: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        preferred_channels: List[DeliveryChannel] = None
    ) -> List[DeliveryRequest]:
        """
        Optimize notification delivery
        
        Args:
            notification_id: Notification identifier
            user_id: User identifier
            content: Notification content
            priority: Notification priority
            preferred_channels: Preferred delivery channels
            
        Returns:
            List of optimized delivery requests
        """
        try:
            self.logger.info(f"🚀 Optimizing delivery for notification {notification_id}")
            
            # Analyze user context
            user_context = await self.context_analyzer.analyze_user_context(user_id)
            
            # Select optimal channels
            optimal_channels = await self._select_optimal_channels(
                user_id, priority, user_context, preferred_channels
            )
            
            # Calculate optimal delivery times
            delivery_requests = []
            
            for channel in optimal_channels:
                # Calculate delivery time based on priority and context
                delivery_time = await self._calculate_delivery_time(
                    channel, priority, user_context
                )
                
                # Create delivery request
                request = DeliveryRequest(
                    notification_id=notification_id,
                    user_id=user_id,
                    channel=channel,
                    priority=priority,
                    content=content,
                    delivery_time=delivery_time,
                    retry_count=0,
                    max_retries=self.channel_configs[channel]['max_retries'],
                    timeout=self.channel_configs[channel]['timeout'],
                    metadata={'context_score': user_context.availability_score}
                )
                
                delivery_requests.append(request)
            
            # Schedule deliveries
            for request in delivery_requests:
                await self._schedule_delivery(request)
            
            self.logger.info(f"✅ Delivery optimized for {len(delivery_requests)} channels")
            return delivery_requests
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing delivery: {e}")
            return []
    
    async def deliver_notification(self, request: DeliveryRequest) -> DeliveryResult:
        """
        Deliver a notification
        
        Args:
            request: Delivery request
            
        Returns:
            DeliveryResult with delivery outcome
        """
        try:
            self.logger.info(f"🚀 Delivering notification {request.notification_id} via {request.channel.value}")
            
            start_time = datetime.now()
            
            # Check throttling
            if await self._is_throttled(request):
                return DeliveryResult(
                    notification_id=request.notification_id,
                    user_id=request.user_id,
                    channel=request.channel,
                    status=DeliveryStatus.THROTTLED,
                    delivered_at=None,
                    retry_count=request.retry_count,
                    error_message="Delivery throttled",
                    response_time=0.0,
                    metadata={}
                )
            
            # Attempt delivery
            try:
                success = await self._deliver_via_channel(request)
                
                if success:
                    status = DeliveryStatus.DELIVERED
                    delivered_at = datetime.now()
                    error_message = None
                    self.deliveries_successful += 1
                else:
                    status = DeliveryStatus.FAILED
                    delivered_at = None
                    error_message = "Delivery failed"
                    self.deliveries_failed += 1
                    
            except Exception as e:
                status = DeliveryStatus.FAILED
                delivered_at = None
                error_message = str(e)
                self.deliveries_failed += 1
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = DeliveryResult(
                notification_id=request.notification_id,
                user_id=request.user_id,
                channel=request.channel,
                status=status,
                delivered_at=delivered_at,
                retry_count=request.retry_count,
                error_message=error_message,
                response_time=response_time,
                metadata=request.metadata or {}
            )
            
            # Store result
            await self._store_delivery_result(result)
            
            # Handle retries if needed
            if status == DeliveryStatus.FAILED and request.retry_count < request.max_retries:
                await self._schedule_retry(request)
            
            self.deliveries_attempted += 1
            self.logger.info(f"✅ Delivery result: {status.value} in {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error delivering notification: {e}")
            return DeliveryResult(
                notification_id=request.notification_id,
                user_id=request.user_id,
                channel=request.channel,
                status=DeliveryStatus.FAILED,
                delivered_at=None,
                retry_count=request.retry_count,
                error_message=str(e),
                response_time=0.0,
                metadata={}
            )
    
    async def _select_optimal_channels(
        self,
        user_id: str,
        priority: NotificationPriority,
        user_context,
        preferred_channels: List[DeliveryChannel] = None
    ) -> List[DeliveryChannel]:
        """Select optimal delivery channels"""
        try:
            # Get user preferences
            user_preferences = await self._get_user_channel_preferences(user_id)
            
            # Start with preferred channels or defaults
            if preferred_channels:
                candidate_channels = preferred_channels
            else:
                candidate_channels = list(DeliveryChannel)
            
            # Filter by user preferences
            if user_preferences:
                candidate_channels = [
                    ch for ch in candidate_channels
                    if user_preferences.get(ch.value, True)
                ]
            
            # Score channels based on context
            channel_scores = {}
            
            for channel in candidate_channels:
                score = await self._score_channel(channel, user_context, priority)
                channel_scores[channel] = score
            
            # Sort by score and select top channels
            sorted_channels = sorted(
                channel_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select channels based on priority
            if priority == NotificationPriority.CRITICAL:
                # Use all available channels
                return [ch for ch, score in sorted_channels if score > 0.3]
            elif priority == NotificationPriority.HIGH:
                # Use top 2-3 channels
                return [ch for ch, score in sorted_channels[:3] if score > 0.5]
            else:
                # Use top channel
                return [sorted_channels[0][0]] if sorted_channels else []
                
        except Exception as e:
            self.logger.error(f"❌ Error selecting optimal channels: {e}")
            return [DeliveryChannel.PUSH]  # Default fallback
    
    async def _score_channel(
        self,
        channel: DeliveryChannel,
        user_context,
        priority: NotificationPriority
    ) -> float:
        """Score a delivery channel"""
        try:
            score = 0.5  # Base score
            
            # Device context scoring
            if channel == DeliveryChannel.PUSH:
                if user_context.device_context.get('is_active', False):
                    score += 0.3
                if user_context.device_context.get('device_type') == 'mobile':
                    score += 0.2
            
            elif channel == DeliveryChannel.EMAIL:
                if user_context.device_context.get('device_type') == 'desktop':
                    score += 0.2
                if user_context.time_context.get('is_business_hours', False):
                    score += 0.1
            
            elif channel == DeliveryChannel.IN_APP:
                if user_context.device_context.get('is_active', False):
                    score += 0.4
                if user_context.activity_level.value in ['high', 'very_high']:
                    score += 0.2
            
            # Priority adjustments
            if priority == NotificationPriority.CRITICAL:
                if channel in [DeliveryChannel.SMS, DeliveryChannel.PUSH]:
                    score += 0.2
            
            # Time context adjustments
            if not user_context.time_context.get('is_business_hours', False):
                if channel == DeliveryChannel.EMAIL:
                    score -= 0.1
            
            # Availability adjustments
            if user_context.availability_score > 0.7:
                if channel == DeliveryChannel.IN_APP:
                    score += 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring channel: {e}")
            return 0.5
    
    async def _calculate_delivery_time(
        self,
        channel: DeliveryChannel,
        priority: NotificationPriority,
        user_context
    ) -> datetime:
        """Calculate optimal delivery time"""
        try:
            now = datetime.now()
            
            # Base delay by priority
            if priority == NotificationPriority.CRITICAL:
                base_delay = 0
            elif priority == NotificationPriority.HIGH:
                base_delay = 30  # 30 seconds
            elif priority == NotificationPriority.MEDIUM:
                base_delay = 300  # 5 minutes
            else:
                base_delay = 900  # 15 minutes
            
            # Channel-specific adjustments
            if channel == DeliveryChannel.EMAIL:
                # Email can be delayed more
                base_delay *= 2
            elif channel == DeliveryChannel.SMS:
                # SMS should be immediate for urgent notifications
                if priority in [NotificationPriority.CRITICAL, NotificationPriority.HIGH]:
                    base_delay = min(base_delay, 60)
            
            # Context-based adjustments
            if user_context.availability_score > 0.8:
                # User is highly available - reduce delay
                base_delay = int(base_delay * 0.5)
            elif user_context.availability_score < 0.3:
                # User is not available - increase delay
                base_delay = int(base_delay * 2)
            
            # Time-based adjustments
            if not user_context.time_context.get('is_business_hours', False):
                # Outside business hours - delay non-critical notifications
                if priority not in [NotificationPriority.CRITICAL, NotificationPriority.HIGH]:
                    base_delay = max(base_delay, 3600)  # At least 1 hour
            
            return now + timedelta(seconds=base_delay)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating delivery time: {e}")
            return datetime.now() + timedelta(minutes=5)
    
    async def _schedule_delivery(self, request: DeliveryRequest):
        """Schedule a delivery request"""
        try:
            # Store in database
            await self._store_delivery_request(request)
            
            # Add to appropriate queue
            await self.delivery_queues[request.channel].put(request)
            
            self.logger.info(f"✅ Scheduled delivery for {request.notification_id} via {request.channel.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Error scheduling delivery: {e}")
    
    async def _is_throttled(self, request: DeliveryRequest) -> bool:
        """Check if delivery is throttled"""
        try:
            # Check hourly limits
            limit = self.throttling_limits.get(request.channel, 100)
            
            # Get recent deliveries for this channel
            recent_deliveries = await self._get_recent_deliveries(
                request.channel, hours=1
            )
            
            return len(recent_deliveries) >= limit
            
        except Exception as e:
            self.logger.error(f"❌ Error checking throttling: {e}")
            return False
    
    async def _deliver_via_channel(self, request: DeliveryRequest) -> bool:
        """Deliver notification via specific channel"""
        try:
            # Simulate delivery based on channel
            if request.channel == DeliveryChannel.EMAIL:
                return await self._deliver_email(request)
            elif request.channel == DeliveryChannel.SMS:
                return await self._deliver_sms(request)
            elif request.channel == DeliveryChannel.PUSH:
                return await self._deliver_push(request)
            elif request.channel == DeliveryChannel.IN_APP:
                return await self._deliver_in_app(request)
            elif request.channel == DeliveryChannel.SLACK:
                return await self._deliver_slack(request)
            elif request.channel == DeliveryChannel.WEBHOOK:
                return await self._deliver_webhook(request)
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error delivering via channel: {e}")
            return False
    
    async def _deliver_email(self, request: DeliveryRequest) -> bool:
        """Deliver via email"""
        try:
            # Simulate email delivery
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _deliver_sms(self, request: DeliveryRequest) -> bool:
        """Deliver via SMS"""
        try:
            # Simulate SMS delivery
            await asyncio.sleep(0.05)
            return True
        except Exception:
            return False
    
    async def _deliver_push(self, request: DeliveryRequest) -> bool:
        """Deliver via push notification"""
        try:
            # Simulate push delivery
            await asyncio.sleep(0.02)
            return True
        except Exception:
            return False
    
    async def _deliver_in_app(self, request: DeliveryRequest) -> bool:
        """Deliver via in-app notification"""
        try:
            # Simulate in-app delivery
            await asyncio.sleep(0.01)
            return True
        except Exception:
            return False
    
    async def _deliver_slack(self, request: DeliveryRequest) -> bool:
        """Deliver via Slack"""
        try:
            # Simulate Slack delivery
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _deliver_webhook(self, request: DeliveryRequest) -> bool:
        """Deliver via webhook"""
        try:
            # Simulate webhook delivery
            await asyncio.sleep(0.1)
            return True
        except Exception:
            return False
    
    async def _schedule_retry(self, request: DeliveryRequest):
        """Schedule retry for failed delivery"""
        try:
            retry_delay = self.channel_configs[request.channel]['retry_delay']
            retry_time = datetime.now() + timedelta(seconds=retry_delay)
            
            # Create retry request
            retry_request = DeliveryRequest(
                notification_id=request.notification_id,
                user_id=request.user_id,
                channel=request.channel,
                priority=request.priority,
                content=request.content,
                delivery_time=retry_time,
                retry_count=request.retry_count + 1,
                max_retries=request.max_retries,
                timeout=request.timeout,
                metadata=request.metadata
            )
            
            # Schedule retry
            await self._schedule_delivery(retry_request)
            
            self.logger.info(f"✅ Scheduled retry {retry_request.retry_count} for {request.notification_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error scheduling retry: {e}")
    
    # Helper methods
    async def _get_user_channel_preferences(self, user_id: str) -> Dict[str, bool]:
        """Get user channel preferences"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT channel_preferences 
                FROM user_preferences 
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user preferences: {e}")
            return {}
    
    async def _get_recent_deliveries(
        self,
        channel: DeliveryChannel,
        hours: int = 1
    ) -> List[Dict[str, Any]]:
        """Get recent deliveries for a channel"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since_time = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT * FROM delivery_results
                WHERE channel = ? AND delivered_at > ?
            """, (channel.value, since_time.isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting recent deliveries: {e}")
            return []
    
    # Database methods
    async def _store_delivery_request(self, request: DeliveryRequest):
        """Store delivery request in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS delivery_requests (
                    notification_id TEXT,
                    user_id TEXT,
                    channel TEXT,
                    priority TEXT,
                    content TEXT,
                    delivery_time TEXT,
                    retry_count INTEGER,
                    max_retries INTEGER,
                    timeout INTEGER,
                    metadata TEXT,
                    created_at TEXT,
                    PRIMARY KEY (notification_id, channel)
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO delivery_requests
                (notification_id, user_id, channel, priority, content, delivery_time,
                 retry_count, max_retries, timeout, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.notification_id,
                request.user_id,
                request.channel.value,
                request.priority.value,
                json.dumps(request.content),
                request.delivery_time.isoformat(),
                request.retry_count,
                request.max_retries,
                request.timeout,
                json.dumps(request.metadata or {}),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing delivery request: {e}")
    
    async def _store_delivery_result(self, result: DeliveryResult):
        """Store delivery result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS delivery_results (
                    notification_id TEXT,
                    user_id TEXT,
                    channel TEXT,
                    status TEXT,
                    delivered_at TEXT,
                    retry_count INTEGER,
                    error_message TEXT,
                    response_time REAL,
                    metadata TEXT,
                    created_at TEXT,
                    PRIMARY KEY (notification_id, channel, retry_count)
                )
            """)
            
            cursor.execute("""
                INSERT OR REPLACE INTO delivery_results
                (notification_id, user_id, channel, status, delivered_at,
                 retry_count, error_message, response_time, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.notification_id,
                result.user_id,
                result.channel.value,
                result.status.value,
                result.delivered_at.isoformat() if result.delivered_at else None,
                result.retry_count,
                result.error_message,
                result.response_time,
                json.dumps(result.metadata),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing delivery result: {e}")
    
    # Status and management methods
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get delivery optimizer status"""
        return {
            "status": "operational",
            "deliveries_attempted": self.deliveries_attempted,
            "deliveries_successful": self.deliveries_successful,
            "deliveries_failed": self.deliveries_failed,
            "success_rate": self.deliveries_successful / max(self.deliveries_attempted, 1),
            "queue_sizes": {
                channel.value: queue.qsize()
                for channel, queue in self.delivery_queues.items()
            },
            "throttling_limits": {
                channel.value: limit
                for channel, limit in self.throttling_limits.items()
            },
            "channel_configs": {
                channel.value: config
                for channel, config in self.channel_configs.items()
            },
            "configuration": {
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_delivery_metrics(self) -> DeliveryMetrics:
        """Get delivery metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic metrics
            cursor.execute("SELECT COUNT(*) FROM delivery_results")
            total_deliveries = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM delivery_results WHERE status = 'delivered'")
            successful_deliveries = cursor.fetchone()[0]
            
            failed_deliveries = total_deliveries - successful_deliveries
            
            # Get average response time
            cursor.execute("SELECT AVG(response_time) FROM delivery_results WHERE status = 'delivered'")
            avg_response_time = cursor.fetchone()[0] or 0.0
            
            # Get channel performance
            cursor.execute("""
                SELECT channel, 
                       COUNT(*) as total,
                       SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as successful
                FROM delivery_results
                GROUP BY channel
            """)
            
            channel_performance = {}
            for row in cursor.fetchall():
                channel, total, successful = row
                channel_performance[channel] = successful / max(total, 1)
            
            # Get retry rate
            cursor.execute("SELECT AVG(retry_count) FROM delivery_results")
            retry_rate = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return DeliveryMetrics(
                total_deliveries=total_deliveries,
                successful_deliveries=successful_deliveries,
                failed_deliveries=failed_deliveries,
                avg_response_time=avg_response_time,
                success_rate=successful_deliveries / max(total_deliveries, 1),
                channel_performance=channel_performance,
                retry_rate=retry_rate,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting delivery metrics: {e}")
            return DeliveryMetrics(
                total_deliveries=0,
                successful_deliveries=0,
                failed_deliveries=0,
                avg_response_time=0.0,
                success_rate=0.0,
                channel_performance={},
                retry_rate=0.0,
                last_updated=datetime.now()
            )


if __name__ == "__main__":
    # Test the delivery optimizer
    async def test_delivery_optimizer():
        print("🚀 Testing Delivery Optimizer")
        print("=" * 50)
        
        try:
            optimizer = DeliveryOptimizer()
            
            # Test delivery optimization
            print("Optimizing delivery...")
            delivery_requests = await optimizer.optimize_delivery(
                notification_id="test_notification_1",
                user_id="user_123",
                content={"title": "Test", "message": "Test message"},
                priority=NotificationPriority.HIGH
            )
            print(f"Delivery Requests: {len(delivery_requests)} optimized")
            
            # Test delivery
            if delivery_requests:
                print("Testing delivery...")
                result = await optimizer.deliver_notification(delivery_requests[0])
                print(f"Delivery Result: {result.status.value} in {result.response_time:.2f}s")
            
            # Test metrics
            print("Getting delivery metrics...")
            metrics = await optimizer.get_delivery_metrics()
            print(f"Metrics: {metrics.success_rate:.2%} success rate")
            
            # Test status
            status = optimizer.get_optimizer_status()
            print(f"Optimizer Status: {status['status']}")
            
            print("\n✅ Delivery Optimizer test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_delivery_optimizer())