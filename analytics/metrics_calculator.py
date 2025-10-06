#!/usr/bin/env python3
"""
Metrics Calculator
Comprehensive metrics calculation for the Business Dealer Intelligence System

This calculator:
- Computes user engagement and behavior metrics
- Calculates opportunity performance and success rates
- Tracks recommendation effectiveness and conversion rates
- Provides real-time and historical analytics
- Supports custom metric definitions and calculations

Following Task 8 from the PRP implementation blueprint.
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    
    name: str
    description: str
    calculation_method: str
    aggregation_type: str  # sum, avg, count, rate, etc.
    time_window: str  # daily, weekly, monthly
    filters: Dict[str, Any]
    unit: str


@dataclass
class MetricResult:
    """Result of metric calculation"""
    
    metric_name: str
    value: float
    unit: str
    time_period: str
    calculation_time: datetime
    metadata: Dict[str, Any]
    breakdown: Dict[str, Any] = None


@dataclass
class MetricsSummary:
    """Summary of multiple metrics"""
    
    summary_type: str
    time_period: str
    metrics: List[MetricResult]
    total_metrics: int
    calculated_at: datetime
    performance_indicators: Dict[str, str]


class MetricsCalculator:
    """
    Comprehensive metrics calculation engine
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Metrics Calculator
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("📊 Initializing Metrics Calculator")
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Define standard metrics
        self._define_standard_metrics()
        
        # Performance tracking
        self.metrics_calculated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info("✅ Metrics Calculator initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for metrics calculation")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching metrics"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for metrics caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for metrics caching: {e}")
            self.redis_enabled = False
    
    def _define_standard_metrics(self):
        """Define standard metrics for the system"""
        self.standard_metrics = {
            # User Engagement Metrics
            'daily_active_users': MetricDefinition(
                name='daily_active_users',
                description='Number of unique users active per day',
                calculation_method='count_distinct',
                aggregation_type='count',
                time_window='daily',
                filters={},
                unit='users'
            ),
            
            'user_engagement_rate': MetricDefinition(
                name='user_engagement_rate',
                description='Average user engagement score',
                calculation_method='avg_engagement',
                aggregation_type='avg',
                time_window='daily',
                filters={},
                unit='score'
            ),
            
            'interaction_volume': MetricDefinition(
                name='interaction_volume',
                description='Total number of user interactions',
                calculation_method='count_interactions',
                aggregation_type='sum',
                time_window='daily',
                filters={},
                unit='interactions'
            ),
            
            # Opportunity Metrics
            'opportunity_conversion_rate': MetricDefinition(
                name='opportunity_conversion_rate',
                description='Rate of opportunities leading to conversions',
                calculation_method='conversion_rate',
                aggregation_type='rate',
                time_window='daily',
                filters={'interaction_type': ['contact', 'save']},
                unit='percentage'
            ),
            
            'opportunity_success_rate': MetricDefinition(
                name='opportunity_success_rate',
                description='Success rate of opportunities',
                calculation_method='success_rate',
                aggregation_type='rate',
                time_window='daily',
                filters={},
                unit='percentage'
            ),
            
            # Recommendation Metrics
            'recommendation_click_rate': MetricDefinition(
                name='recommendation_click_rate',
                description='Click-through rate for recommendations',
                calculation_method='click_rate',
                aggregation_type='rate',
                time_window='daily',
                filters={},
                unit='percentage'
            ),
            
            'recommendation_conversion_rate': MetricDefinition(
                name='recommendation_conversion_rate',
                description='Conversion rate from recommendations',
                calculation_method='recommendation_conversion',
                aggregation_type='rate',
                time_window='daily',
                filters={},
                unit='percentage'
            ),
            
            # System Performance Metrics
            'response_time_avg': MetricDefinition(
                name='response_time_avg',
                description='Average system response time',
                calculation_method='avg_response_time',
                aggregation_type='avg',
                time_window='hourly',
                filters={},
                unit='milliseconds'
            ),
            
            'cache_hit_rate': MetricDefinition(
                name='cache_hit_rate',
                description='Cache hit rate across the system',
                calculation_method='cache_performance',
                aggregation_type='rate',
                time_window='hourly',
                filters={},
                unit='percentage'
            )
        }
    
    async def calculate_metric(
        self,
        metric_name: str,
        time_period: str = None,
        filters: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> MetricResult:
        """
        Calculate a specific metric
        
        Args:
            metric_name: Name of the metric to calculate
            time_period: Time period for calculation (e.g., "2024-01-01")
            filters: Additional filters for calculation
            use_cache: Whether to use cached results
            
        Returns:
            MetricResult with calculated value
        """
        try:
            start_time = datetime.now()
            self.logger.info(f"📊 Calculating metric: {metric_name}")
            
            # Get metric definition
            if metric_name not in self.standard_metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
            
            metric_def = self.standard_metrics[metric_name]
            
            # Set default time period
            if time_period is None:
                time_period = datetime.now().strftime('%Y-%m-%d')
            
            # Check cache first
            if use_cache:
                cached_result = await self._get_cached_metric(metric_name, time_period)
                if cached_result:
                    self.cache_hits += 1
                    return cached_result
            
            self.cache_misses += 1
            
            # Calculate metric based on type
            value, metadata = await self._calculate_metric_value(
                metric_def, time_period, filters
            )
            
            # Create result
            result = MetricResult(
                metric_name=metric_name,
                value=value,
                unit=metric_def.unit,
                time_period=time_period,
                calculation_time=datetime.now(),
                metadata=metadata
            )
            
            # Cache result
            if use_cache:
                await self._cache_metric_result(metric_name, time_period, result)
            
            # Store in database
            await self._store_metric_result(result)
            
            self.metrics_calculated += 1
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"✅ Metric calculated: {metric_name} = {value} {metric_def.unit} in {calculation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating metric {metric_name}: {e}")
            raise
    
    async def calculate_metrics_batch(
        self,
        metric_names: List[str],
        time_period: str = None,
        use_cache: bool = True
    ) -> List[MetricResult]:
        """
        Calculate multiple metrics in batch
        
        Args:
            metric_names: List of metric names to calculate
            time_period: Time period for calculation
            use_cache: Whether to use cached results
            
        Returns:
            List of MetricResult objects
        """
        try:
            self.logger.info(f"📊 Calculating {len(metric_names)} metrics in batch")
            
            results = []
            
            # Process metrics concurrently where possible
            for metric_name in metric_names:
                try:
                    result = await self.calculate_metric(
                        metric_name, time_period, use_cache=use_cache
                    )
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Error calculating {metric_name}: {e}")
                    continue
            
            self.logger.info(f"✅ Batch calculation completed: {len(results)} metrics")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch metrics calculation: {e}")
            return []
    
    async def get_metrics_summary(
        self,
        summary_type: str = "dashboard",
        time_period: str = None,
        use_cache: bool = True
    ) -> MetricsSummary:
        """
        Get summary of key metrics
        
        Args:
            summary_type: Type of summary (dashboard, weekly, monthly)
            time_period: Time period for summary
            use_cache: Whether to use cached results
            
        Returns:
            MetricsSummary with key metrics
        """
        try:
            self.logger.info(f"📊 Generating metrics summary: {summary_type}")
            
            # Define metrics for different summary types
            summary_metrics = {
                'dashboard': [
                    'daily_active_users',
                    'user_engagement_rate',
                    'interaction_volume',
                    'opportunity_conversion_rate',
                    'recommendation_click_rate'
                ],
                'weekly': [
                    'daily_active_users',
                    'user_engagement_rate',
                    'interaction_volume',
                    'opportunity_conversion_rate',
                    'opportunity_success_rate',
                    'recommendation_click_rate',
                    'recommendation_conversion_rate'
                ],
                'monthly': [
                    'daily_active_users',
                    'user_engagement_rate',
                    'interaction_volume',
                    'opportunity_conversion_rate',
                    'opportunity_success_rate',
                    'recommendation_click_rate',
                    'recommendation_conversion_rate',
                    'response_time_avg',
                    'cache_hit_rate'
                ]
            }
            
            if summary_type not in summary_metrics:
                summary_type = 'dashboard'
            
            # Calculate metrics
            metric_results = await self.calculate_metrics_batch(
                summary_metrics[summary_type], time_period, use_cache
            )
            
            # Generate performance indicators
            performance_indicators = self._generate_performance_indicators(metric_results)
            
            summary = MetricsSummary(
                summary_type=summary_type,
                time_period=time_period or datetime.now().strftime('%Y-%m-%d'),
                metrics=metric_results,
                total_metrics=len(metric_results),
                calculated_at=datetime.now(),
                performance_indicators=performance_indicators
            )
            
            self.logger.info(f"✅ Metrics summary generated: {len(metric_results)} metrics")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating metrics summary: {e}")
            raise
    
    async def get_time_series_metrics(
        self,
        metric_name: str,
        start_date: str,
        end_date: str,
        granularity: str = 'daily'
    ) -> List[MetricResult]:
        """
        Get time series data for a metric
        
        Args:
            metric_name: Name of the metric
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            granularity: Time granularity (daily, weekly, monthly)
            
        Returns:
            List of MetricResult objects for time series
        """
        try:
            self.logger.info(f"📊 Generating time series for {metric_name}")
            
            # Generate date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            dates = []
            current_date = start_dt
            
            while current_date <= end_dt:
                dates.append(current_date.strftime('%Y-%m-%d'))
                
                if granularity == 'daily':
                    current_date += timedelta(days=1)
                elif granularity == 'weekly':
                    current_date += timedelta(weeks=1)
                elif granularity == 'monthly':
                    # Add month (approximate)
                    current_date = current_date.replace(day=1)
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)
            
            # Calculate metrics for each date
            results = []
            for date in dates:
                try:
                    result = await self.calculate_metric(metric_name, date, use_cache=True)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error calculating {metric_name} for {date}: {e}")
                    continue
            
            self.logger.info(f"✅ Time series generated: {len(results)} data points")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error generating time series: {e}")
            return []
    
    async def _calculate_metric_value(
        self,
        metric_def: MetricDefinition,
        time_period: str,
        additional_filters: Dict[str, Any] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the actual metric value"""
        try:
            # Parse time period
            period_date = datetime.strptime(time_period, '%Y-%m-%d')
            
            # Calculate date range based on time window
            if metric_def.time_window == 'daily':
                start_date = period_date
                end_date = period_date + timedelta(days=1)
            elif metric_def.time_window == 'weekly':
                start_date = period_date - timedelta(days=period_date.weekday())
                end_date = start_date + timedelta(weeks=1)
            elif metric_def.time_window == 'monthly':
                start_date = period_date.replace(day=1)
                if start_date.month == 12:
                    end_date = start_date.replace(year=start_date.year + 1, month=1)
                else:
                    end_date = start_date.replace(month=start_date.month + 1)
            else:
                start_date = period_date
                end_date = period_date + timedelta(days=1)
            
            # Route to specific calculation method
            if metric_def.calculation_method == 'count_distinct':
                return await self._calculate_count_distinct(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'avg_engagement':
                return await self._calculate_avg_engagement(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'count_interactions':
                return await self._calculate_count_interactions(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'conversion_rate':
                return await self._calculate_conversion_rate(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'success_rate':
                return await self._calculate_success_rate(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'click_rate':
                return await self._calculate_click_rate(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'recommendation_conversion':
                return await self._calculate_recommendation_conversion(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'avg_response_time':
                return await self._calculate_avg_response_time(start_date, end_date, metric_def)
            elif metric_def.calculation_method == 'cache_performance':
                return await self._calculate_cache_performance(start_date, end_date, metric_def)
            else:
                raise ValueError(f"Unknown calculation method: {metric_def.calculation_method}")
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating metric value: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_count_distinct(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate count distinct metric (e.g., daily active users)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(DISTINCT user_id) as count
                FROM user_interactions
                WHERE timestamp >= ? AND timestamp < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            count = result[0] if result else 0
            
            metadata = {
                "calculation_method": "count_distinct",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "query_type": "user_interactions"
            }
            
            return float(count), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating count distinct: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_avg_engagement(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate average engagement metric"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(engagement_score) as avg_engagement
                FROM user_profiles
                WHERE last_active >= ? AND last_active < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            avg_engagement = result[0] if result and result[0] else 0.0
            
            metadata = {
                "calculation_method": "avg_engagement",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "query_type": "user_profiles"
            }
            
            return float(avg_engagement), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating avg engagement: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_count_interactions(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate interaction count metric"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM user_interactions
                WHERE timestamp >= ? AND timestamp < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            count = result[0] if result else 0
            
            metadata = {
                "calculation_method": "count_interactions",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "query_type": "user_interactions"
            }
            
            return float(count), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating interaction count: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_conversion_rate(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate conversion rate metric"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total interactions
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM user_interactions
                WHERE timestamp >= ? AND timestamp < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            total_result = cursor.fetchone()
            total_interactions = total_result[0] if total_result else 0
            
            # Get conversion interactions
            conversion_types = metric_def.filters.get('interaction_type', ['contact', 'save'])
            placeholders = ','.join(['?' for _ in conversion_types])
            
            cursor.execute(f"""
                SELECT COUNT(*) as conversions
                FROM user_interactions
                WHERE timestamp >= ? AND timestamp < ?
                AND interaction_type IN ({placeholders})
            """, [start_date.isoformat(), end_date.isoformat()] + conversion_types)
            
            conversion_result = cursor.fetchone()
            conversions = conversion_result[0] if conversion_result else 0
            
            conn.close()
            
            conversion_rate = (conversions / total_interactions * 100) if total_interactions > 0 else 0.0
            
            metadata = {
                "calculation_method": "conversion_rate",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "total_interactions": total_interactions,
                "conversions": conversions,
                "conversion_types": conversion_types
            }
            
            return float(conversion_rate), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating conversion rate: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_success_rate(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate success rate metric"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get opportunities with interactions
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT ui.opportunity_id) as total_opportunities,
                    COUNT(DISTINCT CASE WHEN ui.interaction_type IN ('contact', 'save') 
                                   THEN ui.opportunity_id END) as successful_opportunities
                FROM user_interactions ui
                JOIN opportunities o ON ui.opportunity_id = o.id
                WHERE ui.timestamp >= ? AND ui.timestamp < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            total_opportunities = result[0] if result else 0
            successful_opportunities = result[1] if result else 0
            
            success_rate = (successful_opportunities / total_opportunities * 100) if total_opportunities > 0 else 0.0
            
            metadata = {
                "calculation_method": "success_rate",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "total_opportunities": total_opportunities,
                "successful_opportunities": successful_opportunities
            }
            
            return float(success_rate), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating success rate: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_click_rate(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate recommendation click rate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total recommendations
            cursor.execute("""
                SELECT COUNT(*) as total
                FROM recommendations
                WHERE generated_at >= ? AND generated_at < ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            total_result = cursor.fetchone()
            total_recommendations = total_result[0] if total_result else 0
            
            # Get clicked recommendations
            cursor.execute("""
                SELECT COUNT(*) as clicked
                FROM recommendations
                WHERE generated_at >= ? AND generated_at < ?
                AND clicked_at IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))
            
            clicked_result = cursor.fetchone()
            clicked_recommendations = clicked_result[0] if clicked_result else 0
            
            conn.close()
            
            click_rate = (clicked_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0.0
            
            metadata = {
                "calculation_method": "click_rate",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "total_recommendations": total_recommendations,
                "clicked_recommendations": clicked_recommendations
            }
            
            return float(click_rate), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating click rate: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_recommendation_conversion(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate recommendation conversion rate"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recommendations that were clicked
            cursor.execute("""
                SELECT COUNT(*) as clicked
                FROM recommendations
                WHERE generated_at >= ? AND generated_at < ?
                AND clicked_at IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))
            
            clicked_result = cursor.fetchone()
            clicked_recommendations = clicked_result[0] if clicked_result else 0
            
            # Get conversions from clicked recommendations
            cursor.execute("""
                SELECT COUNT(*) as conversions
                FROM recommendations r
                JOIN user_interactions ui ON r.opportunity_id = ui.opportunity_id 
                                          AND r.user_id = ui.user_id
                WHERE r.generated_at >= ? AND r.generated_at < ?
                AND r.clicked_at IS NOT NULL
                AND ui.interaction_type IN ('contact', 'save')
                AND ui.timestamp > r.clicked_at
            """, (start_date.isoformat(), end_date.isoformat()))
            
            conversion_result = cursor.fetchone()
            conversions = conversion_result[0] if conversion_result else 0
            
            conn.close()
            
            conversion_rate = (conversions / clicked_recommendations * 100) if clicked_recommendations > 0 else 0.0
            
            metadata = {
                "calculation_method": "recommendation_conversion",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "clicked_recommendations": clicked_recommendations,
                "conversions": conversions
            }
            
            return float(conversion_rate), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating recommendation conversion: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_avg_response_time(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate average response time (placeholder)"""
        try:
            # This would typically come from application logs or monitoring
            # For now, return a placeholder value
            avg_response_time = 250.0  # milliseconds
            
            metadata = {
                "calculation_method": "avg_response_time",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "note": "Placeholder implementation"
            }
            
            return float(avg_response_time), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating avg response time: {e}")
            return 0.0, {"error": str(e)}
    
    async def _calculate_cache_performance(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_def: MetricDefinition
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate cache hit rate (placeholder)"""
        try:
            # This would typically come from cache monitoring
            # For now, return a placeholder value
            cache_hit_rate = 85.0  # percentage
            
            metadata = {
                "calculation_method": "cache_performance",
                "date_range": f"{start_date.date()} to {end_date.date()}",
                "note": "Placeholder implementation"
            }
            
            return float(cache_hit_rate), metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating cache performance: {e}")
            return 0.0, {"error": str(e)}
    
    def _generate_performance_indicators(self, metrics: List[MetricResult]) -> Dict[str, str]:
        """Generate performance indicators based on metrics"""
        indicators = {}
        
        for metric in metrics:
            if metric.metric_name == 'user_engagement_rate':
                if metric.value >= 0.8:
                    indicators['engagement'] = 'excellent'
                elif metric.value >= 0.6:
                    indicators['engagement'] = 'good'
                elif metric.value >= 0.4:
                    indicators['engagement'] = 'moderate'
                else:
                    indicators['engagement'] = 'needs_improvement'
            
            elif metric.metric_name == 'opportunity_conversion_rate':
                if metric.value >= 15:
                    indicators['conversion'] = 'excellent'
                elif metric.value >= 10:
                    indicators['conversion'] = 'good'
                elif metric.value >= 5:
                    indicators['conversion'] = 'moderate'
                else:
                    indicators['conversion'] = 'needs_improvement'
            
            elif metric.metric_name == 'recommendation_click_rate':
                if metric.value >= 20:
                    indicators['recommendations'] = 'excellent'
                elif metric.value >= 15:
                    indicators['recommendations'] = 'good'
                elif metric.value >= 10:
                    indicators['recommendations'] = 'moderate'
                else:
                    indicators['recommendations'] = 'needs_improvement'
        
        return indicators
    
    # Cache management methods
    async def _get_cached_metric(self, metric_name: str, time_period: str) -> Optional[MetricResult]:
        """Get cached metric result"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"metric:{metric_name}:{time_period}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return MetricResult(
                    metric_name=data['metric_name'],
                    value=data['value'],
                    unit=data['unit'],
                    time_period=data['time_period'],
                    calculation_time=datetime.fromisoformat(data['calculation_time']),
                    metadata=data['metadata']
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_metric_result(self, metric_name: str, time_period: str, result: MetricResult):
        """Cache metric result"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"metric:{metric_name}:{time_period}"
            ttl = self.config.redis.analytics_cache_ttl
            
            data = {
                'metric_name': result.metric_name,
                'value': result.value,
                'unit': result.unit,
                'time_period': result.time_period,
                'calculation_time': result.calculation_time.isoformat(),
                'metadata': result.metadata
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Database methods
    async def _store_metric_result(self, result: MetricResult):
        """Store metric result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    value REAL,
                    unit TEXT,
                    time_period TEXT,
                    calculation_time TEXT,
                    metadata TEXT
                )
            """)
            
            # Insert metric result
            cursor.execute("""
                INSERT OR REPLACE INTO metrics 
                (id, metric_name, value, unit, time_period, calculation_time, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                f"{result.metric_name}_{result.time_period}",
                result.metric_name,
                result.value,
                result.unit,
                result.time_period,
                result.calculation_time.isoformat(),
                json.dumps(result.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing metric result: {e}")
    
    # Status and management methods
    def get_calculator_status(self) -> Dict[str, Any]:
        """Get metrics calculator status"""
        return {
            "status": "operational",
            "metrics_calculated": self.metrics_calculated,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "available_metrics": list(self.standard_metrics.keys()),
            "configuration": {
                "redis_enabled": self.redis_enabled,
                "analytics_cache_ttl": self.config.redis.analytics_cache_ttl
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_metrics_cache(self, metric_name: str = None) -> int:
        """Clear metrics cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if metric_name:
                pattern = f"metric:{metric_name}:*"
            else:
                pattern = "metric:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing metrics cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the metrics calculator
    async def test_metrics_calculator():
        print("📊 Testing Metrics Calculator")
        print("=" * 50)
        
        try:
            calculator = MetricsCalculator()
            
            # Test single metric calculation
            print("Calculating single metric...")
            result = await calculator.calculate_metric(
                metric_name="daily_active_users",
                time_period="2024-01-01"
            )
            print(f"Single Metric Result: {result.metric_name} = {result.value} {result.unit}")
            
            # Test batch metrics calculation
            print("Calculating batch metrics...")
            batch_results = await calculator.calculate_metrics_batch([
                "daily_active_users",
                "user_engagement_rate",
                "interaction_volume"
            ])
            print(f"Batch Results: {len(batch_results)} metrics")
            
            # Test metrics summary
            print("Generating metrics summary...")
            summary = await calculator.get_metrics_summary(summary_type="dashboard")
            print(f"Summary: {summary.total_metrics} metrics, indicators: {summary.performance_indicators}")
            
            # Test time series metrics
            print("Generating time series...")
            time_series = await calculator.get_time_series_metrics(
                metric_name="daily_active_users",
                start_date="2024-01-01",
                end_date="2024-01-07"
            )
            print(f"Time Series: {len(time_series)} data points")
            
            # Test status
            status = calculator.get_calculator_status()
            print(f"Calculator Status: {status['status']}")
            
            print("✅ Metrics Calculator test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_metrics_calculator())