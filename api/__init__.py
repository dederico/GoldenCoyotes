#!/usr/bin/env python3
"""
API Package for Business Dealer Intelligence System
Flask blueprints for all API endpoints

This package contains:
- Intelligence API: Core intelligence operations
- Analytics API: Metrics, insights, and dashboards 
- Notification API: Smart notification management
- Main API: Combined service with all endpoints

Following Task 10 from the PRP implementation blueprint.
"""

from .intelligence_api import intelligence_bp
from .analytics_api import analytics_bp
from .notification_api import notification_bp

__all__ = [
    'intelligence_bp',
    'analytics_bp', 
    'notification_bp'
]