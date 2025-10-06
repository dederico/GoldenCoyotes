#!/usr/bin/env python3
"""
Intelligence System Configuration
Central configuration management for the Business Dealer Intelligence System

Environment Variables:
- OPENAI_API_KEY: OpenAI API key for embeddings and content analysis
- REDIS_URL: Redis connection URL for caching
- DATABASE_URL: Primary database connection URL
- INTELLIGENCE_DB_PATH: Path to intelligence database file
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
- ENVIRONMENT: Application environment (development, staging, production)
"""

import os
import logging
from typing import Dict
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings"""

    intelligence_db_path: str = field(default="database/intelligence.db")
    backup_interval_hours: int = field(default=24)
    max_connection_pool_size: int = field(default=10)
    query_timeout_seconds: int = field(default=30)
    enable_foreign_keys: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create database config from environment variables"""
        return cls(
            intelligence_db_path=os.getenv(
                "INTELLIGENCE_DB_PATH", "database/intelligence.db"
            ),
            backup_interval_hours=int(os.getenv("DB_BACKUP_INTERVAL_HOURS", "24")),
            max_connection_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
            query_timeout_seconds=int(os.getenv("DB_QUERY_TIMEOUT", "30")),
            enable_foreign_keys=os.getenv("DB_ENABLE_FOREIGN_KEYS", "true").lower()
            == "true",
        )


@dataclass
class RedisConfig:
    """Redis configuration settings"""

    url: str = field(default="redis://localhost:6379")
    max_connections: int = field(default=20)
    socket_timeout: int = field(default=5)
    socket_connect_timeout: int = field(default=5)
    retry_on_timeout: bool = field(default=True)
    health_check_interval: int = field(default=30)
    default_ttl: int = field(default=3600)  # 1 hour

    # Cache TTL settings for different data types
    embedding_cache_ttl: int = field(default=86400)  # 24 hours
    user_behavior_cache_ttl: int = field(default=1800)  # 30 minutes
    recommendation_cache_ttl: int = field(default=300)  # 5 minutes
    analytics_cache_ttl: int = field(default=600)  # 10 minutes

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create Redis config from environment variables"""
        return cls(
            url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
            retry_on_timeout=os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower()
            == "true",
            health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
            default_ttl=int(os.getenv("REDIS_DEFAULT_TTL", "3600")),
            embedding_cache_ttl=int(os.getenv("REDIS_EMBEDDING_CACHE_TTL", "86400")),
            user_behavior_cache_ttl=int(
                os.getenv("REDIS_USER_BEHAVIOR_CACHE_TTL", "1800")
            ),
            recommendation_cache_ttl=int(
                os.getenv("REDIS_RECOMMENDATION_CACHE_TTL", "300")
            ),
            analytics_cache_ttl=int(os.getenv("REDIS_ANALYTICS_CACHE_TTL", "600")),
        )


@dataclass
class OpenAIConfig:
    """OpenAI API configuration settings"""

    api_key: str = field(default="")
    max_retries: int = field(default=3)
    timeout: int = field(default=30)
    max_tokens: int = field(default=4096)
    temperature: float = field(default=0.1)

    # Embedding model settings
    embedding_model: str = field(default="text-embedding-3-large")
    embedding_dimensions: int = field(default=3072)
    embedding_batch_size: int = field(default=100)

    # Chat completion settings
    chat_model: str = field(default="gpt-4o-mini")
    max_conversation_context: int = field(default=8192)

    # Rate limiting
    requests_per_minute: int = field(default=1000)
    tokens_per_minute: int = field(default=150000)

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create OpenAI config from environment variables"""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "3072")),
            embedding_batch_size=int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "100")),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            max_conversation_context=int(
                os.getenv("OPENAI_MAX_CONVERSATION_CONTEXT", "8192")
            ),
            requests_per_minute=int(os.getenv("OPENAI_REQUESTS_PER_MINUTE", "1000")),
            tokens_per_minute=int(os.getenv("OPENAI_TOKENS_PER_MINUTE", "150000")),
        )


@dataclass
class BehaviorAnalysisConfig:
    """Behavior analysis configuration"""

    min_interactions_for_analysis: int = field(default=5)
    analysis_window_days: int = field(default=30)
    engagement_threshold: float = field(default=0.1)
    interaction_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "view": 1.0,
            "click": 2.0,
            "share": 3.0,
            "contact": 5.0,
            "save": 4.0,
        }
    )

    # Segmentation settings
    segment_update_frequency_hours: int = field(default=6)
    min_segment_size: int = field(default=10)
    max_segments: int = field(default=20)

    @classmethod
    def from_env(cls) -> "BehaviorAnalysisConfig":
        """Create behavior analysis config from environment variables"""
        return cls(
            min_interactions_for_analysis=int(
                os.getenv("BEHAVIOR_MIN_INTERACTIONS", "5")
            ),
            analysis_window_days=int(os.getenv("BEHAVIOR_ANALYSIS_WINDOW_DAYS", "30")),
            engagement_threshold=float(
                os.getenv("BEHAVIOR_ENGAGEMENT_THRESHOLD", "0.1")
            ),
            segment_update_frequency_hours=int(
                os.getenv("BEHAVIOR_SEGMENT_UPDATE_HOURS", "6")
            ),
            min_segment_size=int(os.getenv("BEHAVIOR_MIN_SEGMENT_SIZE", "10")),
            max_segments=int(os.getenv("BEHAVIOR_MAX_SEGMENTS", "20")),
        )


@dataclass
class NotificationConfig:
    """Notification system configuration"""

    max_notifications_per_user_per_day: int = field(default=10)
    notification_cooldown_hours: int = field(default=2)
    priority_threshold: float = field(default=0.7)

    # Channel settings
    email_enabled: bool = field(default=True)
    push_enabled: bool = field(default=True)
    in_app_enabled: bool = field(default=True)

    # Timing optimization
    optimal_send_hours: tuple = field(default=(9, 17))  # 9 AM to 5 PM
    timezone_aware: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Create notification config from environment variables"""
        return cls(
            max_notifications_per_user_per_day=int(
                os.getenv("NOTIFICATION_MAX_PER_DAY", "10")
            ),
            notification_cooldown_hours=int(
                os.getenv("NOTIFICATION_COOLDOWN_HOURS", "2")
            ),
            priority_threshold=float(
                os.getenv("NOTIFICATION_PRIORITY_THRESHOLD", "0.7")
            ),
            email_enabled=os.getenv("NOTIFICATION_EMAIL_ENABLED", "true").lower()
            == "true",
            push_enabled=os.getenv("NOTIFICATION_PUSH_ENABLED", "true").lower()
            == "true",
            in_app_enabled=os.getenv("NOTIFICATION_IN_APP_ENABLED", "true").lower()
            == "true",
            timezone_aware=os.getenv("NOTIFICATION_TIMEZONE_AWARE", "true").lower()
            == "true",
        )


@dataclass
class AnalyticsConfig:
    """Analytics system configuration"""

    real_time_window_minutes: int = field(default=15)
    batch_processing_interval_minutes: int = field(default=60)
    metrics_retention_days: int = field(default=90)

    # Dashboard settings
    dashboard_refresh_seconds: int = field(default=30)
    max_dashboard_items: int = field(default=100)

    # Performance monitoring
    slow_query_threshold_ms: int = field(default=1000)
    enable_performance_logging: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "AnalyticsConfig":
        """Create analytics config from environment variables"""
        return cls(
            real_time_window_minutes=int(os.getenv("ANALYTICS_REAL_TIME_WINDOW", "15")),
            batch_processing_interval_minutes=int(
                os.getenv("ANALYTICS_BATCH_INTERVAL", "60")
            ),
            metrics_retention_days=int(os.getenv("ANALYTICS_RETENTION_DAYS", "90")),
            dashboard_refresh_seconds=int(
                os.getenv("ANALYTICS_DASHBOARD_REFRESH", "30")
            ),
            max_dashboard_items=int(os.getenv("ANALYTICS_MAX_DASHBOARD_ITEMS", "100")),
            slow_query_threshold_ms=int(
                os.getenv("ANALYTICS_SLOW_QUERY_THRESHOLD", "1000")
            ),
            enable_performance_logging=os.getenv(
                "ANALYTICS_PERFORMANCE_LOGGING", "true"
            ).lower()
            == "true",
        )


@dataclass
class IntelligenceConfig:
    """Main intelligence system configuration"""

    environment: str = field(default="development")
    debug: bool = field(default=False)
    log_level: str = field(default="INFO")

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    behavior_analysis: BehaviorAnalysisConfig = field(
        default_factory=BehaviorAnalysisConfig
    )
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    # Performance settings
    max_concurrent_requests: int = field(default=100)
    request_timeout_seconds: int = field(default=30)
    enable_request_logging: bool = field(default=True)

    # Security settings
    enable_rate_limiting: bool = field(default=True)
    rate_limit_requests_per_minute: int = field(default=1000)
    require_authentication: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "IntelligenceConfig":
        """Create complete intelligence config from environment variables"""
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            behavior_analysis=BehaviorAnalysisConfig.from_env(),
            notification=NotificationConfig.from_env(),
            analytics=AnalyticsConfig.from_env(),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
            enable_request_logging=os.getenv("ENABLE_REQUEST_LOGGING", "true").lower()
            == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower()
            == "true",
            rate_limit_requests_per_minute=int(
                os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "1000")
            ),
            require_authentication=os.getenv("REQUIRE_AUTHENTICATION", "true").lower()
            == "true",
        )

    def setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"logs/intelligence_{self.environment}.log"),
            ],
        )

        # Set specific logger levels
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def validate(self) -> None:
        """Validate configuration settings"""
        if not self.openai.api_key:
            raise ValueError("OpenAI API key is required")

        if self.behavior_analysis.min_interactions_for_analysis < 1:
            raise ValueError("Minimum interactions for analysis must be at least 1")

        if (
            self.notification.priority_threshold < 0
            or self.notification.priority_threshold > 1
        ):
            raise ValueError("Notification priority threshold must be between 0 and 1")

        if self.analytics.real_time_window_minutes < 1:
            raise ValueError("Real-time window must be at least 1 minute")

        print(
            f"✅ Intelligence configuration validated for {self.environment} environment"
        )


# Global configuration instance
config = IntelligenceConfig.from_env()


def get_config() -> IntelligenceConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> IntelligenceConfig:
    """Reload configuration from environment variables"""
    global config
    config = IntelligenceConfig.from_env()
    return config


if __name__ == "__main__":
    # Test configuration
    print("🧠 Intelligence System Configuration Test")
    print("=" * 50)

    try:
        config = IntelligenceConfig.from_env()
        config.validate()

        print(f"Environment: {config.environment}")
        print(f"Debug: {config.debug}")
        print(f"Log Level: {config.log_level}")
        print(f"Database: {config.database.intelligence_db_path}")
        print(f"Redis: {config.redis.url}")
        print(f"OpenAI Model: {config.openai.chat_model}")
        print(f"Embedding Model: {config.openai.embedding_model}")
        print(f"Max Concurrent Requests: {config.max_concurrent_requests}")

        print("\n✅ Configuration loaded successfully!")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
