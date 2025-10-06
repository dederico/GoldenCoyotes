#!/usr/bin/env python3
"""
Notification Package
Intelligent notification system for the Business Dealer Intelligence System

This package contains:
- SmartPrioritizer: Intelligent notification prioritization
- ContextAnalyzer: User context analysis for delivery optimization
- DeliveryOptimizer: Multi-channel delivery optimization
- PreferenceManager: User preference management and learning

All components support:
- Context-aware notification delivery
- Multi-channel delivery strategies
- User preference learning and adaptation
- Intelligent prioritization and scheduling
- Performance monitoring and optimization
"""

from .smart_prioritizer import (
    SmartPrioritizer,
    NotificationPriority,
    NotificationCategory,
    NotificationRequest,
    PriorityScore,
    NotificationQueue,
    PriorityMetrics
)
from .context_analyzer import (
    ContextAnalyzer,
    UserContext,
    DeviceContext,
    LocationContext,
    TimeContext,
    ContextAnalysis,
    ActivityLevel,
    DeviceType,
    LocationType
)
from .delivery_optimizer import (
    DeliveryOptimizer,
    DeliveryChannel,
    DeliveryStatus,
    DeliveryRequest,
    DeliveryResult,
    DeliveryMetrics
)
from .preference_manager import (
    PreferenceManager,
    UserPreference,
    NotificationFeedback,
    PreferenceProfile,
    PreferenceType,
    FeedbackType
)

__all__ = [
    # Smart Prioritizer
    'SmartPrioritizer',
    'NotificationPriority',
    'NotificationCategory',
    'NotificationRequest',
    'PriorityScore',
    'NotificationQueue',
    'PriorityMetrics',
    
    # Context Analyzer
    'ContextAnalyzer',
    'UserContext',
    'DeviceContext',
    'LocationContext',
    'TimeContext',
    'ContextAnalysis',
    'ActivityLevel',
    'DeviceType',
    'LocationType',
    
    # Delivery Optimizer
    'DeliveryOptimizer',
    'DeliveryChannel',
    'DeliveryStatus',
    'DeliveryRequest',
    'DeliveryResult',
    'DeliveryMetrics',
    
    # Preference Manager
    'PreferenceManager',
    'UserPreference',
    'NotificationFeedback',
    'PreferenceProfile',
    'PreferenceType',
    'FeedbackType'
]