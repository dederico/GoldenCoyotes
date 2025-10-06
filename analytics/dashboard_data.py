#!/usr/bin/env python3
"""
Dashboard Data Provider
Data aggregation and formatting for dashboard visualization

This provider:
- Aggregates metrics and insights for dashboard display
- Formats data for various visualization components
- Provides real-time and historical data views
- Supports filtering and customization options
- Handles data caching for performance optimization

Following Task 8 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from analytics.metrics_calculator import MetricsCalculator, MetricResult
from analytics.insight_generator import InsightGenerator, Insight, InsightSummary

logger = logging.getLogger(__name__)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    
    widget_id: str
    widget_type: str  # metric, chart, insight, table
    title: str
    data: Any
    config: Dict[str, Any]
    position: Dict[str, int]
    size: Dict[str, int]
    last_updated: datetime


@dataclass
class DashboardData:
    """Complete dashboard data"""
    
    dashboard_id: str
    title: str
    widgets: List[DashboardWidget]
    filters: Dict[str, Any]
    time_range: Dict[str, str]
    refresh_interval: int
    generated_at: datetime
    expires_at: datetime


@dataclass
class ChartData:
    """Chart visualization data"""
    
    chart_type: str  # line, bar, pie, scatter, heatmap
    title: str
    data: List[Dict[str, Any]]
    labels: List[str]
    series: List[Dict[str, Any]]
    config: Dict[str, Any]


@dataclass
class TableData:
    """Table visualization data"""
    
    title: str
    columns: List[Dict[str, str]]
    rows: List[List[Any]]
    pagination: Dict[str, Any]
    sorting: Dict[str, str]
    filtering: Dict[str, Any]


class DashboardDataProvider:
    """
    Dashboard data aggregation and formatting system
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Dashboard Data Provider
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 Initializing Dashboard Data Provider")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config, ml_config)
        
        # Initialize insight generator
        self.insight_generator = InsightGenerator(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Performance tracking
        self.dashboards_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Widget templates
        self.widget_templates = {
            'metric_card': {
                'type': 'metric',
                'config': {
                    'display_type': 'card',
                    'show_trend': True,
                    'show_comparison': True
                }
            },
            'line_chart': {
                'type': 'chart',
                'config': {
                    'chart_type': 'line',
                    'show_points': True,
                    'show_grid': True
                }
            },
            'bar_chart': {
                'type': 'chart',
                'config': {
                    'chart_type': 'bar',
                    'show_values': True,
                    'orientation': 'vertical'
                }
            },
            'insights_panel': {
                'type': 'insight',
                'config': {
                    'max_insights': 5,
                    'show_urgency': True,
                    'show_recommendations': True
                }
            }
        }
        
        self.logger.info("✅ Dashboard Data Provider initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for dashboard data")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching dashboard data"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for dashboard caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for dashboard caching: {e}")
            self.redis_enabled = False
    
    async def generate_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Dict[str, str] = None,
        filters: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> DashboardData:
        """
        Generate complete dashboard data
        
        Args:
            dashboard_id: Dashboard identifier
            time_range: Time range for data
            filters: Additional filters
            use_cache: Whether to use cached data
            
        Returns:
            DashboardData with all widgets
        """
        try:
            self.logger.info(f"📊 Generating dashboard data for {dashboard_id}")
            
            # Set default time range
            if time_range is None:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                time_range = {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d')
                }
            
            # Set default filters
            if filters is None:
                filters = {}
            
            # Check cache first
            if use_cache:
                cached_data = await self._get_cached_dashboard_data(dashboard_id, time_range)
                if cached_data:
                    self.cache_hits += 1
                    return cached_data
            
            self.cache_misses += 1
            
            # Generate widgets based on dashboard type
            widgets = []
            
            if dashboard_id == 'overview':
                widgets = await self._generate_overview_widgets(time_range, filters)
            elif dashboard_id == 'user_analytics':
                widgets = await self._generate_user_analytics_widgets(time_range, filters)
            elif dashboard_id == 'opportunity_performance':
                widgets = await self._generate_opportunity_widgets(time_range, filters)
            elif dashboard_id == 'insights':
                widgets = await self._generate_insights_widgets(time_range, filters)
            else:
                # Custom dashboard
                widgets = await self._generate_custom_widgets(dashboard_id, time_range, filters)
            
            # Create dashboard data
            dashboard_data = DashboardData(
                dashboard_id=dashboard_id,
                title=self._get_dashboard_title(dashboard_id),
                widgets=widgets,
                filters=filters,
                time_range=time_range,
                refresh_interval=300,  # 5 minutes
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=5)
            )
            
            # Cache the data
            if use_cache:
                await self._cache_dashboard_data(dashboard_id, time_range, dashboard_data)
            
            self.dashboards_generated += 1
            self.logger.info(f"✅ Dashboard data generated with {len(widgets)} widgets")
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"❌ Error generating dashboard data: {e}")
            raise
    
    async def _generate_overview_widgets(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[DashboardWidget]:
        """Generate overview dashboard widgets"""
        try:
            widgets = []
            
            # Key metrics cards
            key_metrics = [
                'daily_active_users',
                'user_engagement_rate',
                'opportunity_conversion_rate',
                'recommendation_click_rate'
            ]
            
            for i, metric_name in enumerate(key_metrics):
                metric_data = await self._generate_metric_widget_data(
                    metric_name, time_range, filters
                )
                
                widget = DashboardWidget(
                    widget_id=f"metric_{metric_name}",
                    widget_type='metric',
                    title=metric_name.replace('_', ' ').title(),
                    data=metric_data,
                    config=self.widget_templates['metric_card']['config'],
                    position={'x': i * 3, 'y': 0},
                    size={'width': 3, 'height': 2}
                )
                widgets.append(widget)
            
            # Activity trend chart
            activity_chart = await self._generate_activity_trend_chart(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='activity_trend',
                widget_type='chart',
                title='User Activity Trend',
                data=activity_chart,
                config=self.widget_templates['line_chart']['config'],
                position={'x': 0, 'y': 2},
                size={'width': 6, 'height': 4}
            ))
            
            # Conversion funnel
            funnel_chart = await self._generate_conversion_funnel(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='conversion_funnel',
                widget_type='chart',
                title='Conversion Funnel',
                data=funnel_chart,
                config=self.widget_templates['bar_chart']['config'],
                position={'x': 6, 'y': 2},
                size={'width': 6, 'height': 4}
            ))
            
            # Recent insights
            insights_data = await self._generate_insights_widget_data(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='recent_insights',
                widget_type='insight',
                title='Recent Insights',
                data=insights_data,
                config=self.widget_templates['insights_panel']['config'],
                position={'x': 0, 'y': 6},
                size={'width': 12, 'height': 3}
            ))
            
            return widgets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating overview widgets: {e}")
            return []
    
    async def _generate_user_analytics_widgets(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[DashboardWidget]:
        """Generate user analytics widgets"""
        try:
            widgets = []
            
            # User engagement metrics
            engagement_metrics = [
                'daily_active_users',
                'session_duration',
                'user_engagement_rate',
                'user_retention_rate'
            ]
            
            for i, metric_name in enumerate(engagement_metrics):
                metric_data = await self._generate_metric_widget_data(
                    metric_name, time_range, filters
                )
                
                widget = DashboardWidget(
                    widget_id=f"user_metric_{metric_name}",
                    widget_type='metric',
                    title=metric_name.replace('_', ' ').title(),
                    data=metric_data,
                    config=self.widget_templates['metric_card']['config'],
                    position={'x': i * 3, 'y': 0},
                    size={'width': 3, 'height': 2}
                )
                widgets.append(widget)
            
            # User behavior heatmap
            behavior_heatmap = await self._generate_user_behavior_heatmap(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='user_behavior_heatmap',
                widget_type='chart',
                title='User Behavior Heatmap',
                data=behavior_heatmap,
                config={'chart_type': 'heatmap', 'show_values': True},
                position={'x': 0, 'y': 2},
                size={'width': 8, 'height': 4}
            ))
            
            # Top users table
            top_users_table = await self._generate_top_users_table(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='top_users',
                widget_type='table',
                title='Top Active Users',
                data=top_users_table,
                config={'sortable': True, 'paginated': True},
                position={'x': 8, 'y': 2},
                size={'width': 4, 'height': 4}
            ))
            
            return widgets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating user analytics widgets: {e}")
            return []
    
    async def _generate_opportunity_widgets(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[DashboardWidget]:
        """Generate opportunity performance widgets"""
        try:
            widgets = []
            
            # Opportunity metrics
            opportunity_metrics = [
                'opportunity_conversion_rate',
                'avg_opportunity_score',
                'opportunity_match_accuracy',
                'opportunity_response_time'
            ]
            
            for i, metric_name in enumerate(opportunity_metrics):
                metric_data = await self._generate_metric_widget_data(
                    metric_name, time_range, filters
                )
                
                widget = DashboardWidget(
                    widget_id=f"opp_metric_{metric_name}",
                    widget_type='metric',
                    title=metric_name.replace('_', ' ').title(),
                    data=metric_data,
                    config=self.widget_templates['metric_card']['config'],
                    position={'x': i * 3, 'y': 0},
                    size={'width': 3, 'height': 2}
                )
                widgets.append(widget)
            
            # Opportunity pipeline
            pipeline_chart = await self._generate_opportunity_pipeline(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='opportunity_pipeline',
                widget_type='chart',
                title='Opportunity Pipeline',
                data=pipeline_chart,
                config=self.widget_templates['bar_chart']['config'],
                position={'x': 0, 'y': 2},
                size={'width': 6, 'height': 4}
            ))
            
            # Success rate by category
            success_by_category = await self._generate_success_by_category(time_range, filters)
            widgets.append(DashboardWidget(
                widget_id='success_by_category',
                widget_type='chart',
                title='Success Rate by Category',
                data=success_by_category,
                config={'chart_type': 'pie', 'show_percentages': True},
                position={'x': 6, 'y': 2},
                size={'width': 6, 'height': 4}
            ))
            
            return widgets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating opportunity widgets: {e}")
            return []
    
    async def _generate_insights_widgets(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[DashboardWidget]:
        """Generate insights dashboard widgets"""
        try:
            widgets = []
            
            # Insights summary
            insights_summary = await self.insight_generator.generate_insight_summary(
                time_period=time_range['end'],
                use_cache=True
            )
            
            # Insights by type
            insights_by_type_chart = ChartData(
                chart_type='pie',
                title='Insights by Type',
                data=[
                    {'name': insight_type, 'value': count}
                    for insight_type, count in insights_summary.insights_by_type.items()
                ],
                labels=list(insights_summary.insights_by_type.keys()),
                series=[{
                    'name': 'Insights',
                    'data': list(insights_summary.insights_by_type.values())
                }],
                config={'show_percentages': True, 'show_legend': True}
            )
            
            widgets.append(DashboardWidget(
                widget_id='insights_by_type',
                widget_type='chart',
                title='Insights by Type',
                data=insights_by_type_chart,
                config={'chart_type': 'pie'},
                position={'x': 0, 'y': 0},
                size={'width': 6, 'height': 4}
            ))
            
            # Insights by urgency
            insights_by_urgency_chart = ChartData(
                chart_type='bar',
                title='Insights by Urgency',
                data=[
                    {'name': urgency, 'value': count}
                    for urgency, count in insights_summary.insights_by_urgency.items()
                ],
                labels=list(insights_summary.insights_by_urgency.keys()),
                series=[{
                    'name': 'Insights',
                    'data': list(insights_summary.insights_by_urgency.values())
                }],
                config={'show_values': True, 'color_scheme': 'urgency'}
            )
            
            widgets.append(DashboardWidget(
                widget_id='insights_by_urgency',
                widget_type='chart',
                title='Insights by Urgency',
                data=insights_by_urgency_chart,
                config={'chart_type': 'bar'},
                position={'x': 6, 'y': 0},
                size={'width': 6, 'height': 4}
            ))
            
            # Key insights table
            key_insights_table = TableData(
                title='Key Insights',
                columns=[
                    {'key': 'title', 'name': 'Title'},
                    {'key': 'type', 'name': 'Type'},
                    {'key': 'urgency', 'name': 'Urgency'},
                    {'key': 'confidence', 'name': 'Confidence'},
                    {'key': 'generated_at', 'name': 'Generated'}
                ],
                rows=[
                    [
                        insight.title,
                        insight.insight_type,
                        insight.urgency,
                        f"{insight.confidence:.2f}",
                        insight.generated_at.strftime('%Y-%m-%d %H:%M')
                    ]
                    for insight in insights_summary.key_insights
                ],
                pagination={'page': 1, 'per_page': 10, 'total': len(insights_summary.key_insights)},
                sorting={'column': 'generated_at', 'direction': 'desc'},
                filtering={}
            )
            
            widgets.append(DashboardWidget(
                widget_id='key_insights_table',
                widget_type='table',
                title='Key Insights',
                data=key_insights_table,
                config={'sortable': True, 'filterable': True},
                position={'x': 0, 'y': 4},
                size={'width': 12, 'height': 4}
            ))
            
            return widgets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating insights widgets: {e}")
            return []
    
    async def _generate_custom_widgets(
        self,
        dashboard_id: str,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[DashboardWidget]:
        """Generate custom dashboard widgets"""
        try:
            # Load custom dashboard configuration
            config = await self._load_custom_dashboard_config(dashboard_id)
            
            if not config:
                self.logger.warning(f"⚠️ No configuration found for dashboard {dashboard_id}")
                return []
            
            widgets = []
            
            for widget_config in config.get('widgets', []):
                widget_data = await self._generate_widget_data(
                    widget_config, time_range, filters
                )
                
                widget = DashboardWidget(
                    widget_id=widget_config['id'],
                    widget_type=widget_config['type'],
                    title=widget_config['title'],
                    data=widget_data,
                    config=widget_config.get('config', {}),
                    position=widget_config.get('position', {'x': 0, 'y': 0}),
                    size=widget_config.get('size', {'width': 6, 'height': 4})
                )
                widgets.append(widget)
            
            return widgets
            
        except Exception as e:
            self.logger.error(f"❌ Error generating custom widgets: {e}")
            return []
    
    async def _generate_metric_widget_data(
        self,
        metric_name: str,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data for metric widget"""
        try:
            # Get current metric value
            current_metric = await self.metrics_calculator.calculate_metric(
                metric_name, time_range['end']
            )
            
            # Get historical data for trend
            historical_data = await self.metrics_calculator.get_time_series_metrics(
                metric_name,
                time_range['start'],
                time_range['end']
            )
            
            # Calculate trend
            trend = 'stable'
            trend_percentage = 0
            
            if len(historical_data) > 1:
                recent_values = [metric.value for metric in historical_data[-7:]]
                if len(recent_values) > 1:
                    trend_percentage = ((recent_values[-1] - recent_values[0]) / recent_values[0]) * 100
                    if trend_percentage > 5:
                        trend = 'increasing'
                    elif trend_percentage < -5:
                        trend = 'decreasing'
            
            return {
                'value': current_metric.value,
                'unit': current_metric.unit,
                'trend': trend,
                'trend_percentage': trend_percentage,
                'historical_data': [
                    {
                        'date': metric.time_period,
                        'value': metric.value
                    }
                    for metric in historical_data
                ],
                'last_updated': current_metric.calculation_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating metric widget data: {e}")
            return {
                'value': 0,
                'unit': 'count',
                'trend': 'stable',
                'trend_percentage': 0,
                'historical_data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    async def _generate_activity_trend_chart(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> ChartData:
        """Generate activity trend chart data"""
        try:
            # Get activity metrics
            activity_data = await self.metrics_calculator.get_time_series_metrics(
                'daily_active_users',
                time_range['start'],
                time_range['end']
            )
            
            return ChartData(
                chart_type='line',
                title='Daily Active Users',
                data=[
                    {
                        'date': metric.time_period,
                        'value': metric.value
                    }
                    for metric in activity_data
                ],
                labels=[metric.time_period for metric in activity_data],
                series=[{
                    'name': 'Active Users',
                    'data': [metric.value for metric in activity_data]
                }],
                config={
                    'show_points': True,
                    'show_grid': True,
                    'interpolation': 'smooth'
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating activity trend chart: {e}")
            return ChartData(
                chart_type='line',
                title='Daily Active Users',
                data=[],
                labels=[],
                series=[],
                config={}
            )
    
    async def _generate_conversion_funnel(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> ChartData:
        """Generate conversion funnel chart data"""
        try:
            # Get funnel metrics
            funnel_steps = [
                ('Views', 'content_views'),
                ('Clicks', 'recommendation_clicks'),
                ('Engagements', 'user_engagements'),
                ('Conversions', 'opportunity_conversions')
            ]
            
            funnel_data = []
            for step_name, metric_name in funnel_steps:
                try:
                    metric = await self.metrics_calculator.calculate_metric(
                        metric_name, time_range['end']
                    )
                    funnel_data.append({
                        'step': step_name,
                        'value': metric.value
                    })
                except:
                    funnel_data.append({
                        'step': step_name,
                        'value': 0
                    })
            
            return ChartData(
                chart_type='funnel',
                title='Conversion Funnel',
                data=funnel_data,
                labels=[step['step'] for step in funnel_data],
                series=[{
                    'name': 'Conversions',
                    'data': [step['value'] for step in funnel_data]
                }],
                config={
                    'show_percentages': True,
                    'show_values': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error generating conversion funnel: {e}")
            return ChartData(
                chart_type='funnel',
                title='Conversion Funnel',
                data=[],
                labels=[],
                series=[],
                config={}
            )
    
    async def _generate_insights_widget_data(
        self,
        time_range: Dict[str, str],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights widget data"""
        try:
            # Get recent insights
            insights = await self.insight_generator.generate_insights(
                insight_types=['trend', 'anomaly', 'opportunity', 'warning'],
                time_period=time_range['end'],
                use_cache=True
            )
            
            # Format insights for display
            formatted_insights = []
            for insight in insights[:5]:  # Show top 5 insights
                formatted_insights.append({
                    'title': insight.title,
                    'description': insight.description,
                    'type': insight.insight_type,
                    'urgency': insight.urgency,
                    'confidence': insight.confidence,
                    'recommendations': insight.recommendations,
                    'generated_at': insight.generated_at.isoformat()
                })
            
            return formatted_insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating insights widget data: {e}")
            return []
    
    def _get_dashboard_title(self, dashboard_id: str) -> str:
        """Get dashboard title"""
        titles = {
            'overview': 'Business Intelligence Overview',
            'user_analytics': 'User Analytics Dashboard',
            'opportunity_performance': 'Opportunity Performance',
            'insights': 'AI Insights Dashboard'
        }
        return titles.get(dashboard_id, dashboard_id.replace('_', ' ').title())
    
    # Cache management methods
    async def _get_cached_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Dict[str, str]
    ) -> Optional[DashboardData]:
        """Get cached dashboard data"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"dashboard:{dashboard_id}:{time_range['start']}:{time_range['end']}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                
                # Deserialize widgets
                widgets = []
                for widget_data in data['widgets']:
                    widget = DashboardWidget(
                        widget_id=widget_data['widget_id'],
                        widget_type=widget_data['widget_type'],
                        title=widget_data['title'],
                        data=widget_data['data'],
                        config=widget_data['config'],
                        position=widget_data['position'],
                        size=widget_data['size'],
                        last_updated=datetime.fromisoformat(widget_data['last_updated'])
                    )
                    widgets.append(widget)
                
                return DashboardData(
                    dashboard_id=data['dashboard_id'],
                    title=data['title'],
                    widgets=widgets,
                    filters=data['filters'],
                    time_range=data['time_range'],
                    refresh_interval=data['refresh_interval'],
                    generated_at=datetime.fromisoformat(data['generated_at']),
                    expires_at=datetime.fromisoformat(data['expires_at'])
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_dashboard_data(
        self,
        dashboard_id: str,
        time_range: Dict[str, str],
        dashboard_data: DashboardData
    ):
        """Cache dashboard data"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"dashboard:{dashboard_id}:{time_range['start']}:{time_range['end']}"
            ttl = self.config.redis.dashboard_cache_ttl
            
            # Serialize dashboard data
            serialized_data = {
                'dashboard_id': dashboard_data.dashboard_id,
                'title': dashboard_data.title,
                'widgets': [
                    {
                        'widget_id': widget.widget_id,
                        'widget_type': widget.widget_type,
                        'title': widget.title,
                        'data': widget.data,
                        'config': widget.config,
                        'position': widget.position,
                        'size': widget.size,
                        'last_updated': widget.last_updated.isoformat()
                    }
                    for widget in dashboard_data.widgets
                ],
                'filters': dashboard_data.filters,
                'time_range': dashboard_data.time_range,
                'refresh_interval': dashboard_data.refresh_interval,
                'generated_at': dashboard_data.generated_at.isoformat(),
                'expires_at': dashboard_data.expires_at.isoformat()
            }
            
            # Store in cache
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(serialized_data)
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Status and management methods
    def get_provider_status(self) -> Dict[str, Any]:
        """Get dashboard provider status"""
        return {
            "status": "operational",
            "dashboards_generated": self.dashboards_generated,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "widget_templates": list(self.widget_templates.keys()),
            "configuration": {
                "redis_enabled": self.redis_enabled,
                "refresh_interval": 300
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_dashboard_cache(self, dashboard_id: str = None) -> int:
        """Clear dashboard cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if dashboard_id:
                pattern = f"dashboard:{dashboard_id}:*"
            else:
                pattern = "dashboard:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing dashboard cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the dashboard data provider
    async def test_dashboard_provider():
        print("📊 Testing Dashboard Data Provider")
        print("=" * 50)
        
        try:
            provider = DashboardDataProvider()
            
            # Test overview dashboard
            print("Generating overview dashboard...")
            overview = await provider.generate_dashboard_data(
                dashboard_id='overview',
                time_range={
                    'start': '2024-01-01',
                    'end': '2024-01-31'
                }
            )
            print(f"Overview Dashboard: {overview.title}, {len(overview.widgets)} widgets")
            
            # Test user analytics dashboard
            print("Generating user analytics dashboard...")
            user_analytics = await provider.generate_dashboard_data(
                dashboard_id='user_analytics',
                time_range={
                    'start': '2024-01-01',
                    'end': '2024-01-31'
                }
            )
            print(f"User Analytics Dashboard: {user_analytics.title}, {len(user_analytics.widgets)} widgets")
            
            # Test insights dashboard
            print("Generating insights dashboard...")
            insights = await provider.generate_dashboard_data(
                dashboard_id='insights',
                time_range={
                    'start': '2024-01-01',
                    'end': '2024-01-31'
                }
            )
            print(f"Insights Dashboard: {insights.title}, {len(insights.widgets)} widgets")
            
            # Test status
            status = provider.get_provider_status()
            print(f"Provider Status: {status['status']}")
            
            print("\n✅ Dashboard Data Provider test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_dashboard_provider())