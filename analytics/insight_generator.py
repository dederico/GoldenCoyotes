#!/usr/bin/env python3
"""
Insight Generator
AI-powered insights generation for the Business Dealer Intelligence System

This generator:
- Analyzes metrics and patterns to generate business insights
- Uses OpenAI to create human-readable insights and recommendations
- Identifies trends, anomalies, and opportunities
- Provides actionable recommendations based on data analysis
- Supports different insight types and urgency levels

Following Task 8 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from openai import OpenAI
from statistics import mean, stdev
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from analytics.metrics_calculator import MetricsCalculator, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class InsightData:
    """Data context for insight generation"""
    
    metrics: List[MetricResult]
    time_series: List[MetricResult]
    comparisons: Dict[str, float]
    trends: Dict[str, str]
    anomalies: List[Dict[str, Any]]


@dataclass
class Insight:
    """Generated insight"""
    
    id: str
    title: str
    description: str
    insight_type: str  # trend, anomaly, opportunity, warning
    urgency: str  # high, medium, low
    confidence: float
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class InsightSummary:
    """Summary of insights"""
    
    total_insights: int
    insights_by_type: Dict[str, int]
    insights_by_urgency: Dict[str, int]
    key_insights: List[Insight]
    generated_at: datetime


class InsightGenerator:
    """
    AI-powered insights generation engine
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Insight Generator
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔍 Initializing Insight Generator")
        
        # Initialize OpenAI client
        self._setup_openai_client()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Performance tracking
        self.insights_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Insight types and templates
        self.insight_types = {
            'trend': 'Trend Analysis',
            'anomaly': 'Anomaly Detection',
            'opportunity': 'Business Opportunity',
            'warning': 'Performance Warning',
            'recommendation': 'Strategic Recommendation'
        }
        
        self.logger.info("✅ Insight Generator initialized successfully")
    
    def _setup_openai_client(self):
        """Setup OpenAI client for insight generation"""
        try:
            self.openai_client = OpenAI(
                api_key=self.config.openai.api_key,
                timeout=self.config.openai.timeout,
                max_retries=self.config.openai.max_retries
            )
            
            # Test connection
            self.openai_client.models.list()
            self.logger.info("✅ OpenAI client initialized for insight generation")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            raise
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for insight generation")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching insights"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for insight caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for insight caching: {e}")
            self.redis_enabled = False
    
    async def generate_insights(
        self,
        insight_types: List[str] = None,
        time_period: str = None,
        use_cache: bool = True
    ) -> List[Insight]:
        """
        Generate insights from current data
        
        Args:
            insight_types: Types of insights to generate
            time_period: Time period for analysis
            use_cache: Whether to use cached insights
            
        Returns:
            List of generated insights
        """
        try:
            self.logger.info("🔍 Generating insights from data")
            
            # Set default insight types
            if insight_types is None:
                insight_types = ['trend', 'anomaly', 'opportunity', 'warning']
            
            # Set default time period
            if time_period is None:
                time_period = datetime.now().strftime('%Y-%m-%d')
            
            # Check cache first
            if use_cache:
                cached_insights = await self._get_cached_insights(time_period)
                if cached_insights:
                    self.cache_hits += 1
                    return cached_insights
            
            self.cache_misses += 1
            
            # Gather data for insight generation
            insight_data = await self._gather_insight_data(time_period)
            
            # Generate insights by type
            all_insights = []
            
            for insight_type in insight_types:
                try:
                    insights = await self._generate_insights_by_type(
                        insight_type, insight_data, time_period
                    )
                    all_insights.extend(insights)
                except Exception as e:
                    self.logger.error(f"❌ Error generating {insight_type} insights: {e}")
                    continue
            
            # Sort by urgency and confidence
            all_insights.sort(key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}[x.urgency],
                x.confidence
            ), reverse=True)
            
            # Cache insights
            if use_cache:
                await self._cache_insights(time_period, all_insights)
            
            # Store insights in database
            await self._store_insights(all_insights)
            
            self.insights_generated += len(all_insights)
            self.logger.info(f"✅ Generated {len(all_insights)} insights")
            
            return all_insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating insights: {e}")
            return []
    
    async def generate_insight_summary(
        self,
        time_period: str = None,
        use_cache: bool = True
    ) -> InsightSummary:
        """
        Generate summary of insights
        
        Args:
            time_period: Time period for analysis
            use_cache: Whether to use cached results
            
        Returns:
            InsightSummary with key insights
        """
        try:
            self.logger.info("🔍 Generating insight summary")
            
            # Generate insights
            insights = await self.generate_insights(
                time_period=time_period,
                use_cache=use_cache
            )
            
            # Count insights by type
            insights_by_type = {}
            for insight in insights:
                insights_by_type[insight.insight_type] = insights_by_type.get(insight.insight_type, 0) + 1
            
            # Count insights by urgency
            insights_by_urgency = {}
            for insight in insights:
                insights_by_urgency[insight.urgency] = insights_by_urgency.get(insight.urgency, 0) + 1
            
            # Get key insights (top 5 by urgency and confidence)
            key_insights = insights[:5]
            
            summary = InsightSummary(
                total_insights=len(insights),
                insights_by_type=insights_by_type,
                insights_by_urgency=insights_by_urgency,
                key_insights=key_insights,
                generated_at=datetime.now()
            )
            
            self.logger.info(f"✅ Generated insight summary with {len(insights)} insights")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error generating insight summary: {e}")
            raise
    
    async def _gather_insight_data(self, time_period: str) -> InsightData:
        """Gather data needed for insight generation"""
        try:
            # Get current metrics
            metrics = await self.metrics_calculator.calculate_metrics_batch([
                'daily_active_users',
                'user_engagement_rate',
                'interaction_volume',
                'opportunity_conversion_rate',
                'recommendation_click_rate'
            ], time_period)
            
            # Get time series data (last 7 days)
            end_date = datetime.strptime(time_period, '%Y-%m-%d')
            start_date = end_date - timedelta(days=7)
            
            time_series = []
            for metric in metrics:
                series = await self.metrics_calculator.get_time_series_metrics(
                    metric.metric_name,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                time_series.extend(series)
            
            # Calculate comparisons (current vs previous period)
            comparisons = {}
            for metric in metrics:
                prev_period = (end_date - timedelta(days=1)).strftime('%Y-%m-%d')
                try:
                    prev_metric = await self.metrics_calculator.calculate_metric(
                        metric.metric_name, prev_period
                    )
                    comparisons[metric.metric_name] = {
                        'current': metric.value,
                        'previous': prev_metric.value,
                        'change': ((metric.value - prev_metric.value) / prev_metric.value * 100) if prev_metric.value != 0 else 0
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Error calculating comparison for {metric.metric_name}: {e}")
                    comparisons[metric.metric_name] = {
                        'current': metric.value,
                        'previous': metric.value,
                        'change': 0
                    }
            
            # Identify trends
            trends = {}
            for metric in metrics:
                metric_series = [ts for ts in time_series if ts.metric_name == metric.metric_name]
                if len(metric_series) > 1:
                    values = [ts.value for ts in metric_series]
                    if len(values) > 2:
                        # Simple trend detection
                        if values[-1] > values[-2] and values[-2] > values[-3]:
                            trends[metric.metric_name] = 'increasing'
                        elif values[-1] < values[-2] and values[-2] < values[-3]:
                            trends[metric.metric_name] = 'decreasing'
                        else:
                            trends[metric.metric_name] = 'stable'
                    else:
                        trends[metric.metric_name] = 'stable'
            
            # Detect anomalies
            anomalies = []
            for metric in metrics:
                metric_series = [ts for ts in time_series if ts.metric_name == metric.metric_name]
                if len(metric_series) > 3:
                    values = [ts.value for ts in metric_series]
                    
                    # Calculate z-score for anomaly detection
                    if len(values) > 1:
                        mean_val = mean(values[:-1])  # Exclude current value
                        std_val = stdev(values[:-1]) if len(values) > 2 else 0
                        
                        if std_val > 0:
                            z_score = abs(values[-1] - mean_val) / std_val
                            if z_score > 2:  # Anomaly threshold
                                anomalies.append({
                                    'metric': metric.metric_name,
                                    'current_value': values[-1],
                                    'expected_value': mean_val,
                                    'z_score': z_score,
                                    'severity': 'high' if z_score > 3 else 'medium'
                                })
            
            return InsightData(
                metrics=metrics,
                time_series=time_series,
                comparisons=comparisons,
                trends=trends,
                anomalies=anomalies
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error gathering insight data: {e}")
            raise
    
    async def _generate_insights_by_type(
        self,
        insight_type: str,
        insight_data: InsightData,
        time_period: str
    ) -> List[Insight]:
        """Generate insights for a specific type"""
        try:
            if insight_type == 'trend':
                return await self._generate_trend_insights(insight_data, time_period)
            elif insight_type == 'anomaly':
                return await self._generate_anomaly_insights(insight_data, time_period)
            elif insight_type == 'opportunity':
                return await self._generate_opportunity_insights(insight_data, time_period)
            elif insight_type == 'warning':
                return await self._generate_warning_insights(insight_data, time_period)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"❌ Error generating {insight_type} insights: {e}")
            return []
    
    async def _generate_trend_insights(
        self,
        insight_data: InsightData,
        time_period: str
    ) -> List[Insight]:
        """Generate trend-based insights"""
        try:
            insights = []
            
            for metric_name, trend in insight_data.trends.items():
                if trend in ['increasing', 'decreasing']:
                    comparison = insight_data.comparisons.get(metric_name, {})
                    change = comparison.get('change', 0)
                    
                    # Generate insight using AI
                    ai_insight = await self._generate_ai_insight(
                        f"Trend Analysis: {metric_name}",
                        f"The {metric_name} metric is showing a {trend} trend with a {change:.1f}% change from the previous period. Current value: {comparison.get('current', 0)}, Previous value: {comparison.get('previous', 0)}.",
                        'trend',
                        {'metric': metric_name, 'trend': trend, 'change': change}
                    )
                    
                    if ai_insight:
                        insights.append(ai_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating trend insights: {e}")
            return []
    
    async def _generate_anomaly_insights(
        self,
        insight_data: InsightData,
        time_period: str
    ) -> List[Insight]:
        """Generate anomaly-based insights"""
        try:
            insights = []
            
            for anomaly in insight_data.anomalies:
                # Generate insight using AI
                ai_insight = await self._generate_ai_insight(
                    f"Anomaly Detected: {anomaly['metric']}",
                    f"An anomaly has been detected in {anomaly['metric']}. Current value: {anomaly['current_value']:.2f}, Expected value: {anomaly['expected_value']:.2f}, Z-score: {anomaly['z_score']:.2f}. Severity: {anomaly['severity']}.",
                    'anomaly',
                    anomaly
                )
                
                if ai_insight:
                    insights.append(ai_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating anomaly insights: {e}")
            return []
    
    async def _generate_opportunity_insights(
        self,
        insight_data: InsightData,
        time_period: str
    ) -> List[Insight]:
        """Generate opportunity-based insights"""
        try:
            insights = []
            
            # Look for positive trends that indicate opportunities
            for metric_name, trend in insight_data.trends.items():
                if trend == 'increasing':
                    comparison = insight_data.comparisons.get(metric_name, {})
                    change = comparison.get('change', 0)
                    
                    if change > 10:  # Significant positive change
                        # Generate insight using AI
                        ai_insight = await self._generate_ai_insight(
                            f"Growth Opportunity: {metric_name}",
                            f"There's a significant growth opportunity in {metric_name} with a {change:.1f}% increase. Current value: {comparison.get('current', 0)}. This positive trend suggests potential for optimization and expansion.",
                            'opportunity',
                            {'metric': metric_name, 'trend': trend, 'change': change}
                        )
                        
                        if ai_insight:
                            insights.append(ai_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating opportunity insights: {e}")
            return []
    
    async def _generate_warning_insights(
        self,
        insight_data: InsightData,
        time_period: str
    ) -> List[Insight]:
        """Generate warning-based insights"""
        try:
            insights = []
            
            # Look for negative trends that indicate warnings
            for metric_name, trend in insight_data.trends.items():
                if trend == 'decreasing':
                    comparison = insight_data.comparisons.get(metric_name, {})
                    change = comparison.get('change', 0)
                    
                    if change < -10:  # Significant negative change
                        # Generate insight using AI
                        ai_insight = await self._generate_ai_insight(
                            f"Performance Warning: {metric_name}",
                            f"Warning: {metric_name} is showing a declining trend with a {change:.1f}% decrease. Current value: {comparison.get('current', 0)}. This negative trend requires immediate attention.",
                            'warning',
                            {'metric': metric_name, 'trend': trend, 'change': change}
                        )
                        
                        if ai_insight:
                            insights.append(ai_insight)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating warning insights: {e}")
            return []
    
    async def _generate_ai_insight(
        self,
        title: str,
        context: str,
        insight_type: str,
        supporting_data: Dict[str, Any]
    ) -> Optional[Insight]:
        """Generate AI-powered insight"""
        try:
            # Create prompt for AI insight generation
            prompt = f"""
            Analyze the following business intelligence data and generate a comprehensive insight:

            Title: {title}
            Context: {context}
            Type: {insight_type}

            Please provide:
            1. A clear, concise description of the insight
            2. The urgency level (high, medium, low)
            3. Confidence level (0.0 to 1.0)
            4. 3-5 actionable recommendations
            5. Business impact assessment

            Format your response as JSON with the following structure:
            {{
                "description": "Detailed description of the insight",
                "urgency": "high/medium/low",
                "confidence": 0.0-1.0,
                "recommendations": ["recommendation1", "recommendation2", ...],
                "business_impact": "Assessment of business impact"
            }}
            """
            
            # Get response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.config.openai.chat_model,
                messages=[
                    {"role": "system", "content": "You are a business intelligence analyst specialized in generating actionable insights from data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.openai.temperature,
                max_tokens=self.config.openai.max_tokens
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Find JSON in response
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = ai_response[start_idx:end_idx]
                    ai_data = json.loads(json_str)
                    
                    # Create insight
                    insight = Insight(
                        id=f"insight_{int(datetime.now().timestamp())}_{insight_type}",
                        title=title,
                        description=ai_data.get('description', 'No description available'),
                        insight_type=insight_type,
                        urgency=ai_data.get('urgency', 'medium'),
                        confidence=float(ai_data.get('confidence', 0.5)),
                        recommendations=ai_data.get('recommendations', []),
                        supporting_data=supporting_data,
                        generated_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(days=7)
                    )
                    
                    return insight
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ Error parsing AI response JSON: {e}")
                
                # Fallback: create basic insight
                return Insight(
                    id=f"insight_{int(datetime.now().timestamp())}_{insight_type}",
                    title=title,
                    description=context,
                    insight_type=insight_type,
                    urgency='medium',
                    confidence=0.5,
                    recommendations=["Review the data and investigate further"],
                    supporting_data=supporting_data,
                    generated_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(days=7)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating AI insight: {e}")
            return None
    
    # Cache management methods
    async def _get_cached_insights(self, time_period: str) -> Optional[List[Insight]]:
        """Get cached insights"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"insights:{time_period}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                insights = []
                
                for insight_data in data:
                    insight = Insight(
                        id=insight_data['id'],
                        title=insight_data['title'],
                        description=insight_data['description'],
                        insight_type=insight_data['insight_type'],
                        urgency=insight_data['urgency'],
                        confidence=insight_data['confidence'],
                        recommendations=insight_data['recommendations'],
                        supporting_data=insight_data['supporting_data'],
                        generated_at=datetime.fromisoformat(insight_data['generated_at']),
                        expires_at=datetime.fromisoformat(insight_data['expires_at']) if insight_data.get('expires_at') else None
                    )
                    insights.append(insight)
                
                return insights
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_insights(self, time_period: str, insights: List[Insight]):
        """Cache insights"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"insights:{time_period}"
            ttl = self.config.redis.analytics_cache_ttl
            
            # Serialize insights
            serialized_insights = []
            for insight in insights:
                serialized_insights.append({
                    'id': insight.id,
                    'title': insight.title,
                    'description': insight.description,
                    'insight_type': insight.insight_type,
                    'urgency': insight.urgency,
                    'confidence': insight.confidence,
                    'recommendations': insight.recommendations,
                    'supporting_data': insight.supporting_data,
                    'generated_at': insight.generated_at.isoformat(),
                    'expires_at': insight.expires_at.isoformat() if insight.expires_at else None
                })
            
            # Store in cache
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(serialized_insights)
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    # Database methods
    async def _store_insights(self, insights: List[Insight]):
        """Store insights in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create insights table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    insight_type TEXT,
                    urgency TEXT,
                    confidence REAL,
                    recommendations TEXT,
                    supporting_data TEXT,
                    generated_at TEXT,
                    expires_at TEXT
                )
            """)
            
            # Insert insights
            for insight in insights:
                cursor.execute("""
                    INSERT OR REPLACE INTO insights 
                    (id, title, description, insight_type, urgency, confidence,
                     recommendations, supporting_data, generated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.id,
                    insight.title,
                    insight.description,
                    insight.insight_type,
                    insight.urgency,
                    insight.confidence,
                    json.dumps(insight.recommendations),
                    json.dumps(insight.supporting_data),
                    insight.generated_at.isoformat(),
                    insight.expires_at.isoformat() if insight.expires_at else None
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing insights: {e}")
    
    # Status and management methods
    def get_generator_status(self) -> Dict[str, Any]:
        """Get insight generator status"""
        return {
            "status": "operational",
            "insights_generated": self.insights_generated,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "insight_types": self.insight_types,
            "configuration": {
                "openai_model": self.config.openai.chat_model,
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_insights_cache(self, time_period: str = None) -> int:
        """Clear insights cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if time_period:
                pattern = f"insights:{time_period}"
            else:
                pattern = "insights:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing insights cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the insight generator
    async def test_insight_generator():
        print("🔍 Testing Insight Generator")
        print("=" * 50)
        
        try:
            generator = InsightGenerator()
            
            # Test insight generation
            print("Generating insights...")
            insights = await generator.generate_insights(
                insight_types=['trend', 'anomaly', 'opportunity'],
                time_period="2024-01-01"
            )
            print(f"Generated {len(insights)} insights")
            
            # Display insights
            for insight in insights[:3]:  # Show first 3 insights
                print(f"\n📊 {insight.title}")
                print(f"   Type: {insight.insight_type}, Urgency: {insight.urgency}")
                print(f"   Confidence: {insight.confidence:.2f}")
                print(f"   Description: {insight.description[:100]}...")
                print(f"   Recommendations: {len(insight.recommendations)}")
            
            # Test insight summary
            print("\nGenerating insight summary...")
            summary = await generator.generate_insight_summary()
            print(f"Summary: {summary.total_insights} insights")
            print(f"By type: {summary.insights_by_type}")
            print(f"By urgency: {summary.insights_by_urgency}")
            
            # Test status
            status = generator.get_generator_status()
            print(f"\nGenerator Status: {status['status']}")
            
            print("\n✅ Insight Generator test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_insight_generator())