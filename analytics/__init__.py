#!/usr/bin/env python3
"""
Analytics Package
Analytics and insights system for the Business Dealer Intelligence System

This package contains:
- MetricsCalculator: Comprehensive metrics calculation engine
- InsightGenerator: AI-powered insights generation
- DashboardDataProvider: Dashboard data aggregation and formatting
- PredictiveAnalytics: Advanced predictive analytics and forecasting

All components support:
- Real-time and historical data analysis
- Caching for improved performance
- AI-powered insights and recommendations
- Multiple visualization formats
- Predictive modeling and forecasting
"""

from .metrics_calculator import (
    MetricsCalculator, 
    MetricDefinition, 
    MetricResult, 
    MetricsSummary
)
from .insight_generator import (
    InsightGenerator,
    InsightData,
    Insight,
    InsightSummary
)
from .dashboard_data import (
    DashboardDataProvider,
    DashboardWidget,
    DashboardData,
    ChartData,
    TableData
)
from .predictive_analytics import (
    PredictiveAnalytics,
    ForecastRequest,
    ForecastResult,
    TrendAnalysis,
    AnomalyPrediction
)

__all__ = [
    # Metrics Calculator
    'MetricsCalculator',
    'MetricDefinition',
    'MetricResult',
    'MetricsSummary',
    
    # Insight Generator
    'InsightGenerator',
    'InsightData',
    'Insight',
    'InsightSummary',
    
    # Dashboard Data Provider
    'DashboardDataProvider',
    'DashboardWidget',
    'DashboardData',
    'ChartData',
    'TableData',
    
    # Predictive Analytics
    'PredictiveAnalytics',
    'ForecastRequest',
    'ForecastResult',
    'TrendAnalysis',
    'AnomalyPrediction'
]