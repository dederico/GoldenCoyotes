#!/usr/bin/env python3
"""
Predictive Analytics Engine
Advanced predictive analytics for business intelligence forecasting

This engine:
- Provides time-series forecasting for business metrics
- Predicts user behavior patterns and trends
- Forecasts opportunity success rates and market conditions
- Generates predictive insights and recommendations
- Supports multiple forecasting algorithms and models

Following Task 8 from the PRP implementation blueprint.
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config
from analytics.metrics_calculator import MetricsCalculator, MetricResult
from ml_models.prediction_model import PredictionModel

logger = logging.getLogger(__name__)


@dataclass
class ForecastRequest:
    """Request for forecasting"""
    
    metric_name: str
    forecast_horizon: int  # days
    historical_days: int  # days of history to use
    confidence_level: float  # 0.95 for 95% confidence interval
    seasonality: bool  # whether to account for seasonality
    external_factors: Dict[str, Any]  # external variables


@dataclass
class ForecastResult:
    """Result of forecasting"""
    
    metric_name: str
    forecast_dates: List[str]
    forecast_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    forecast_horizon: int
    confidence_level: float
    generated_at: datetime
    expires_at: datetime


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0-1
    seasonal_patterns: Dict[str, float]
    trend_change_points: List[Dict[str, Any]]
    forecast_confidence: float
    analysis_period: str


@dataclass
class AnomalyPrediction:
    """Anomaly prediction result"""
    
    metric_name: str
    predicted_anomalies: List[Dict[str, Any]]
    anomaly_probability: List[float]
    prediction_dates: List[str]
    threshold_values: Dict[str, float]
    confidence_score: float


class PredictiveAnalytics:
    """
    Advanced predictive analytics engine
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Predictive Analytics Engine
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔮 Initializing Predictive Analytics Engine")
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(config, ml_config)
        
        # Initialize prediction model
        self.prediction_model = PredictionModel(config, ml_config)
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        # Initialize forecasting models
        self._setup_forecasting_models()
        
        # Performance tracking
        self.forecasts_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Model performance tracking
        self.model_performance = {}
        
        self.logger.info("✅ Predictive Analytics Engine initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for predictive analytics")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for caching forecasts"""
        try:
            import redis
            
            self.redis_client = redis.Redis.from_url(
                self.config.redis.url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.redis_enabled = True
            self.logger.info("✅ Redis client for forecast caching initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis not available for forecast caching: {e}")
            self.redis_enabled = False
    
    def _setup_forecasting_models(self):
        """Setup forecasting models"""
        try:
            self.forecasting_models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42
                )
            }
            
            # Scalers for features
            self.scalers = {
                model_name: StandardScaler() 
                for model_name in self.forecasting_models.keys()
            }
            
            self.logger.info("✅ Forecasting models initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error setting up forecasting models: {e}")
            raise
    
    async def generate_forecast(
        self,
        request: ForecastRequest,
        use_cache: bool = True
    ) -> ForecastResult:
        """
        Generate forecast for a metric
        
        Args:
            request: Forecast request parameters
            use_cache: Whether to use cached results
            
        Returns:
            ForecastResult with predictions and confidence intervals
        """
        try:
            self.logger.info(f"🔮 Generating forecast for {request.metric_name}")
            
            # Check cache first
            if use_cache:
                cached_result = await self._get_cached_forecast(request)
                if cached_result:
                    self.cache_hits += 1
                    return cached_result
            
            self.cache_misses += 1
            
            # Prepare historical data
            historical_data = await self._prepare_historical_data(request)
            
            if len(historical_data) < request.historical_days * 0.5:
                raise ValueError(f"Insufficient historical data for {request.metric_name}")
            
            # Generate features
            features, target = await self._generate_features(historical_data, request)
            
            # Select best model
            best_model = await self._select_best_model(features, target, request)
            
            # Generate forecast
            forecast_values, confidence_intervals = await self._generate_forecast_values(
                best_model, features, target, request
            )
            
            # Generate forecast dates
            forecast_dates = self._generate_forecast_dates(
                historical_data, request.forecast_horizon
            )
            
            # Calculate model performance
            model_performance = await self._calculate_model_performance(
                best_model, features, target
            )
            
            # Calculate feature importance
            feature_importance = await self._calculate_feature_importance(
                best_model, features.columns if isinstance(features, pd.DataFrame) else []
            )
            
            # Create forecast result
            result = ForecastResult(
                metric_name=request.metric_name,
                forecast_dates=forecast_dates,
                forecast_values=forecast_values,
                confidence_intervals=confidence_intervals,
                model_performance=model_performance,
                feature_importance=feature_importance,
                forecast_horizon=request.forecast_horizon,
                confidence_level=request.confidence_level,
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=6)
            )
            
            # Cache the result
            if use_cache:
                await self._cache_forecast(request, result)
            
            # Store in database
            await self._store_forecast(result)
            
            self.forecasts_generated += 1
            self.logger.info(f"✅ Forecast generated for {request.metric_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating forecast: {e}")
            raise
    
    async def analyze_trends(
        self,
        metric_name: str,
        analysis_period: str = "30d",
        use_cache: bool = True
    ) -> TrendAnalysis:
        """
        Analyze trends in a metric
        
        Args:
            metric_name: Name of the metric to analyze
            analysis_period: Period for analysis (e.g., "30d", "90d")
            use_cache: Whether to use cached results
            
        Returns:
            TrendAnalysis with trend information
        """
        try:
            self.logger.info(f"📈 Analyzing trends for {metric_name}")
            
            # Parse analysis period
            days = int(analysis_period.replace('d', ''))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get historical data
            historical_data = await self.metrics_calculator.get_time_series_metrics(
                metric_name,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(historical_data) < 7:
                raise ValueError(f"Insufficient data for trend analysis of {metric_name}")
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'date': datetime.strptime(metric.time_period, '%Y-%m-%d'),
                    'value': metric.value
                }
                for metric in historical_data
            ])
            df = df.sort_values('date')
            
            # Analyze trend direction
            trend_direction = self._analyze_trend_direction(df['value'])
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(df['value'])
            
            # Detect seasonal patterns
            seasonal_patterns = self._detect_seasonal_patterns(df)
            
            # Find trend change points
            trend_change_points = self._find_trend_change_points(df)
            
            # Calculate forecast confidence
            forecast_confidence = self._calculate_forecast_confidence(df['value'])
            
            result = TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_patterns=seasonal_patterns,
                trend_change_points=trend_change_points,
                forecast_confidence=forecast_confidence,
                analysis_period=analysis_period
            )
            
            self.logger.info(f"✅ Trend analysis completed for {metric_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing trends: {e}")
            raise
    
    async def predict_anomalies(
        self,
        metric_name: str,
        prediction_horizon: int = 7,
        use_cache: bool = True
    ) -> AnomalyPrediction:
        """
        Predict potential anomalies in a metric
        
        Args:
            metric_name: Name of the metric
            prediction_horizon: Days ahead to predict
            use_cache: Whether to use cached results
            
        Returns:
            AnomalyPrediction with anomaly predictions
        """
        try:
            self.logger.info(f"🚨 Predicting anomalies for {metric_name}")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            historical_data = await self.metrics_calculator.get_time_series_metrics(
                metric_name,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if len(historical_data) < 14:
                raise ValueError(f"Insufficient data for anomaly prediction of {metric_name}")
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'date': datetime.strptime(metric.time_period, '%Y-%m-%d'),
                    'value': metric.value
                }
                for metric in historical_data
            ])
            df = df.sort_values('date')
            
            # Calculate statistical thresholds
            threshold_values = self._calculate_anomaly_thresholds(df['value'])
            
            # Generate forecast for anomaly prediction
            forecast_request = ForecastRequest(
                metric_name=metric_name,
                forecast_horizon=prediction_horizon,
                historical_days=30,
                confidence_level=0.95,
                seasonality=True,
                external_factors={}
            )
            
            forecast_result = await self.generate_forecast(forecast_request, use_cache)
            
            # Predict anomalies
            predicted_anomalies = []
            anomaly_probability = []
            
            for i, (date, value, confidence_interval) in enumerate(zip(
                forecast_result.forecast_dates,
                forecast_result.forecast_values,
                forecast_result.confidence_intervals
            )):
                # Check if predicted value is outside normal bounds
                is_anomaly = (
                    value < threshold_values['lower_bound'] or
                    value > threshold_values['upper_bound']
                )
                
                # Calculate anomaly probability
                if is_anomaly:
                    if value < threshold_values['lower_bound']:
                        probability = min(1.0, (threshold_values['lower_bound'] - value) / threshold_values['std'])
                        anomaly_type = 'below_threshold'
                    else:
                        probability = min(1.0, (value - threshold_values['upper_bound']) / threshold_values['std'])
                        anomaly_type = 'above_threshold'
                    
                    predicted_anomalies.append({
                        'date': date,
                        'predicted_value': value,
                        'anomaly_type': anomaly_type,
                        'severity': 'high' if probability > 0.7 else 'medium' if probability > 0.3 else 'low',
                        'confidence_interval': confidence_interval
                    })
                else:
                    probability = 0.0
                
                anomaly_probability.append(probability)
            
            # Calculate overall confidence
            confidence_score = np.mean([
                1.0 - abs(ci[1] - ci[0]) / (2 * threshold_values['std'])
                for ci in forecast_result.confidence_intervals
            ])
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            result = AnomalyPrediction(
                metric_name=metric_name,
                predicted_anomalies=predicted_anomalies,
                anomaly_probability=anomaly_probability,
                prediction_dates=forecast_result.forecast_dates,
                threshold_values=threshold_values,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"✅ Anomaly prediction completed for {metric_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error predicting anomalies: {e}")
            raise
    
    async def _prepare_historical_data(self, request: ForecastRequest) -> List[MetricResult]:
        """Prepare historical data for forecasting"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.historical_days)
            
            return await self.metrics_calculator.get_time_series_metrics(
                request.metric_name,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing historical data: {e}")
            raise
    
    async def _generate_features(
        self,
        historical_data: List[MetricResult],
        request: ForecastRequest
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate features for forecasting"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'date': datetime.strptime(metric.time_period, '%Y-%m-%d'),
                    'value': metric.value
                }
                for metric in historical_data
            ])
            df = df.sort_values('date')
            
            # Generate time-based features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Generate lag features
            for lag in [1, 3, 7, 14]:
                df[f'lag_{lag}'] = df['value'].shift(lag)
            
            # Generate rolling statistics
            for window in [3, 7, 14]:
                df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
            
            # Generate trend features
            df['trend'] = np.arange(len(df))
            
            # Seasonality features
            if request.seasonality:
                df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
                df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
                df['sin_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['cos_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # External factors
            for factor_name, factor_value in request.external_factors.items():
                df[f'external_{factor_name}'] = factor_value
            
            # Remove rows with NaN values
            df = df.dropna()
            
            # Prepare features and target
            feature_columns = [col for col in df.columns if col not in ['date', 'value']]
            features = df[feature_columns]
            target = df['value']
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"❌ Error generating features: {e}")
            raise
    
    async def _select_best_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        request: ForecastRequest
    ) -> str:
        """Select the best model for forecasting"""
        try:
            # Split data for validation
            split_index = int(len(features) * 0.8)
            X_train, X_test = features[:split_index], features[split_index:]
            y_train, y_test = target[:split_index], target[split_index:]
            
            best_model = None
            best_score = float('inf')
            
            for model_name, model in self.forecasting_models.items():
                try:
                    # Scale features
                    scaler = self.scalers[model_name]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    predictions = model.predict(X_test_scaled)
                    
                    # Calculate error
                    mae = mean_absolute_error(y_test, predictions)
                    
                    # Update best model
                    if mae < best_score:
                        best_score = mae
                        best_model = model_name
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error training model {model_name}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("No model could be trained successfully")
            
            self.logger.info(f"✅ Selected best model: {best_model} (MAE: {best_score:.4f})")
            return best_model
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting best model: {e}")
            raise
    
    async def _generate_forecast_values(
        self,
        best_model: str,
        features: pd.DataFrame,
        target: pd.Series,
        request: ForecastRequest
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast values and confidence intervals"""
        try:
            model = self.forecasting_models[best_model]
            scaler = self.scalers[best_model]
            
            # Train on all data
            X_scaled = scaler.fit_transform(features)
            model.fit(X_scaled, target)
            
            # Generate future features
            future_features = self._generate_future_features(
                features, request.forecast_horizon, request
            )
            
            # Scale future features
            future_features_scaled = scaler.transform(future_features)
            
            # Make predictions
            predictions = model.predict(future_features_scaled)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                model, X_scaled, target, future_features_scaled, request.confidence_level
            )
            
            return predictions.tolist(), confidence_intervals
            
        except Exception as e:
            self.logger.error(f"❌ Error generating forecast values: {e}")
            raise
    
    def _generate_future_features(
        self,
        historical_features: pd.DataFrame,
        forecast_horizon: int,
        request: ForecastRequest
    ) -> pd.DataFrame:
        """Generate features for future dates"""
        try:
            future_features = []
            
            # Get last date and values
            last_date = datetime.now()
            last_values = historical_features.tail(14)  # Last 14 days for lag features
            
            for i in range(forecast_horizon):
                future_date = last_date + timedelta(days=i + 1)
                
                features = {}
                
                # Time-based features
                features['day_of_week'] = future_date.weekday()
                features['day_of_month'] = future_date.day
                features['month'] = future_date.month
                features['quarter'] = (future_date.month - 1) // 3 + 1
                features['day_of_year'] = future_date.timetuple().tm_yday
                
                # Lag features (use last known values)
                for lag in [1, 3, 7, 14]:
                    if len(last_values) >= lag:
                        features[f'lag_{lag}'] = last_values.iloc[-lag]['value'] if 'value' in last_values.columns else 0
                    else:
                        features[f'lag_{lag}'] = 0
                
                # Rolling statistics (use last known values)
                for window in [3, 7, 14]:
                    if len(last_values) >= window:
                        values = last_values.tail(window)['value'] if 'value' in last_values.columns else [0] * window
                        features[f'rolling_mean_{window}'] = np.mean(values)
                        features[f'rolling_std_{window}'] = np.std(values)
                    else:
                        features[f'rolling_mean_{window}'] = 0
                        features[f'rolling_std_{window}'] = 0
                
                # Trend feature
                features['trend'] = len(historical_features) + i + 1
                
                # Seasonality features
                if request.seasonality:
                    features['sin_day'] = np.sin(2 * np.pi * features['day_of_year'] / 365.25)
                    features['cos_day'] = np.cos(2 * np.pi * features['day_of_year'] / 365.25)
                    features['sin_week'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                    features['cos_week'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                
                # External factors
                for factor_name, factor_value in request.external_factors.items():
                    features[f'external_{factor_name}'] = factor_value
                
                future_features.append(features)
            
            return pd.DataFrame(future_features)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating future features: {e}")
            raise
    
    def _calculate_confidence_intervals(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_future: np.ndarray,
        confidence_level: float
    ) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        try:
            # Calculate prediction intervals using bootstrapping
            n_bootstrap = 100
            predictions_bootstrap = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_bootstrap = X_train[indices]
                y_bootstrap = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
                
                # Train model on bootstrap sample
                model.fit(X_bootstrap, y_bootstrap)
                
                # Make predictions
                pred = model.predict(X_future)
                predictions_bootstrap.append(pred)
            
            # Calculate confidence intervals
            predictions_bootstrap = np.array(predictions_bootstrap)
            alpha = 1 - confidence_level
            
            confidence_intervals = []
            for i in range(X_future.shape[0]):
                pred_values = predictions_bootstrap[:, i]
                lower = np.percentile(pred_values, (alpha/2) * 100)
                upper = np.percentile(pred_values, (1 - alpha/2) * 100)
                confidence_intervals.append((lower, upper))
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence intervals: {e}")
            # Fallback to simple confidence intervals
            predictions = model.predict(X_future)
            std_error = np.std(predictions) * 0.1  # Simple approximation
            
            return [
                (pred - 1.96 * std_error, pred + 1.96 * std_error)
                for pred in predictions
            ]
    
    def _generate_forecast_dates(
        self,
        historical_data: List[MetricResult],
        forecast_horizon: int
    ) -> List[str]:
        """Generate forecast dates"""
        try:
            last_date = datetime.now()
            
            forecast_dates = []
            for i in range(forecast_horizon):
                forecast_date = last_date + timedelta(days=i + 1)
                forecast_dates.append(forecast_date.strftime('%Y-%m-%d'))
            
            return forecast_dates
            
        except Exception as e:
            self.logger.error(f"❌ Error generating forecast dates: {e}")
            return []
    
    async def _calculate_model_performance(
        self,
        best_model: str,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            model = self.forecasting_models[best_model]
            scaler = self.scalers[best_model]
            
            # Use cross-validation for performance assessment
            X_scaled = scaler.transform(features)
            
            # Split for validation
            split_index = int(len(features) * 0.8)
            X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
            y_train, y_test = target[:split_index], target[split_index:]
            
            # Train and predict
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'model_name': best_model
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating model performance: {e}")
            return {
                'mae': 0.0,
                'mse': 0.0,
                'rmse': 0.0,
                'r2_score': 0.0,
                'model_name': best_model
            }
    
    async def _calculate_feature_importance(
        self,
        best_model: str,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            model = self.forecasting_models[best_model]
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {
                    name: float(importance)
                    for name, importance in zip(feature_names, importances)
                }
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coefficients = np.abs(model.coef_)
                return {
                    name: float(coef)
                    for name, coef in zip(feature_names, coefficients)
                }
            else:
                return {name: 0.0 for name in feature_names}
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating feature importance: {e}")
            return {name: 0.0 for name in feature_names}
    
    def _analyze_trend_direction(self, values: pd.Series) -> str:
        """Analyze trend direction"""
        try:
            # Use linear regression to determine trend
            X = np.arange(len(values)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, values)
            
            slope = model.coef_[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _calculate_trend_strength(self, values: pd.Series) -> float:
        """Calculate trend strength"""
        try:
            # Use R-squared from linear regression
            X = np.arange(len(values)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, values)
            
            return max(0.0, min(1.0, model.score(X, values)))
            
        except Exception:
            return 0.0
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect seasonal patterns"""
        try:
            patterns = {}
            
            # Daily patterns
            if 'day_of_week' in df.columns:
                daily_avg = df.groupby('day_of_week')['value'].mean()
                patterns['daily_variation'] = daily_avg.std() / daily_avg.mean()
            
            # Monthly patterns
            if 'month' in df.columns:
                monthly_avg = df.groupby('month')['value'].mean()
                patterns['monthly_variation'] = monthly_avg.std() / monthly_avg.mean()
            
            return patterns
            
        except Exception:
            return {}
    
    def _find_trend_change_points(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find trend change points"""
        try:
            change_points = []
            
            # Simple change point detection using moving averages
            window = min(7, len(df) // 4)
            if window < 2:
                return change_points
            
            moving_avg = df['value'].rolling(window=window).mean()
            
            for i in range(window, len(moving_avg) - window):
                before = moving_avg.iloc[i-window:i].mean()
                after = moving_avg.iloc[i:i+window].mean()
                
                # Check for significant change
                if abs(after - before) > df['value'].std() * 0.5:
                    change_points.append({
                        'date': df.iloc[i]['date'].strftime('%Y-%m-%d'),
                        'value_before': before,
                        'value_after': after,
                        'change_magnitude': abs(after - before)
                    })
            
            return change_points
            
        except Exception:
            return []
    
    def _calculate_forecast_confidence(self, values: pd.Series) -> float:
        """Calculate forecast confidence"""
        try:
            # Based on data stability and trend consistency
            stability = 1.0 / (1.0 + values.std() / values.mean())
            trend_consistency = self._calculate_trend_strength(values)
            
            return (stability + trend_consistency) / 2.0
            
        except Exception:
            return 0.5
    
    def _calculate_anomaly_thresholds(self, values: pd.Series) -> Dict[str, float]:
        """Calculate anomaly detection thresholds"""
        try:
            mean_val = values.mean()
            std_val = values.std()
            
            # Use 2 standard deviations as threshold
            lower_bound = mean_val - 2 * std_val
            upper_bound = mean_val + 2 * std_val
            
            return {
                'mean': mean_val,
                'std': std_val,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception:
            return {
                'mean': 0.0,
                'std': 1.0,
                'lower_bound': -2.0,
                'upper_bound': 2.0
            }
    
    # Cache management methods
    async def _get_cached_forecast(self, request: ForecastRequest) -> Optional[ForecastResult]:
        """Get cached forecast"""
        if not self.redis_enabled:
            return None
        
        try:
            cache_key = f"forecast:{request.metric_name}:{request.forecast_horizon}:{request.historical_days}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return ForecastResult(
                    metric_name=data['metric_name'],
                    forecast_dates=data['forecast_dates'],
                    forecast_values=data['forecast_values'],
                    confidence_intervals=[tuple(ci) for ci in data['confidence_intervals']],
                    model_performance=data['model_performance'],
                    feature_importance=data['feature_importance'],
                    forecast_horizon=data['forecast_horizon'],
                    confidence_level=data['confidence_level'],
                    generated_at=datetime.fromisoformat(data['generated_at']),
                    expires_at=datetime.fromisoformat(data['expires_at'])
                )
                
        except Exception as e:
            self.logger.warning(f"⚠️ Cache get error: {e}")
        
        return None
    
    async def _cache_forecast(self, request: ForecastRequest, result: ForecastResult):
        """Cache forecast result"""
        if not self.redis_enabled:
            return
        
        try:
            cache_key = f"forecast:{request.metric_name}:{request.forecast_horizon}:{request.historical_days}"
            ttl = self.config.redis.forecast_cache_ttl
            
            data = {
                'metric_name': result.metric_name,
                'forecast_dates': result.forecast_dates,
                'forecast_values': result.forecast_values,
                'confidence_intervals': result.confidence_intervals,
                'model_performance': result.model_performance,
                'feature_importance': result.feature_importance,
                'forecast_horizon': result.forecast_horizon,
                'confidence_level': result.confidence_level,
                'generated_at': result.generated_at.isoformat(),
                'expires_at': result.expires_at.isoformat()
            }
            
            self.redis_client.setex(cache_key, ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cache store error: {e}")
    
    async def _store_forecast(self, result: ForecastResult):
        """Store forecast in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create forecasts table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    forecast_dates TEXT,
                    forecast_values TEXT,
                    confidence_intervals TEXT,
                    model_performance TEXT,
                    feature_importance TEXT,
                    forecast_horizon INTEGER,
                    confidence_level REAL,
                    generated_at TEXT,
                    expires_at TEXT
                )
            """)
            
            # Insert forecast
            cursor.execute("""
                INSERT OR REPLACE INTO forecasts 
                (id, metric_name, forecast_dates, forecast_values, confidence_intervals,
                 model_performance, feature_importance, forecast_horizon, confidence_level,
                 generated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"forecast_{result.metric_name}_{int(result.generated_at.timestamp())}",
                result.metric_name,
                json.dumps(result.forecast_dates),
                json.dumps(result.forecast_values),
                json.dumps(result.confidence_intervals),
                json.dumps(result.model_performance),
                json.dumps(result.feature_importance),
                result.forecast_horizon,
                result.confidence_level,
                result.generated_at.isoformat(),
                result.expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing forecast: {e}")
    
    # Status and management methods
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get predictive analytics status"""
        return {
            "status": "operational",
            "forecasts_generated": self.forecasts_generated,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            },
            "model_performance": self.model_performance,
            "available_models": list(self.forecasting_models.keys()),
            "configuration": {
                "redis_enabled": self.redis_enabled
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def clear_forecast_cache(self, metric_name: str = None) -> int:
        """Clear forecast cache"""
        if not self.redis_enabled:
            return 0
        
        try:
            if metric_name:
                pattern = f"forecast:{metric_name}:*"
            else:
                pattern = "forecast:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"❌ Error clearing forecast cache: {e}")
            return 0


if __name__ == "__main__":
    # Test the predictive analytics engine
    async def test_predictive_analytics():
        print("🔮 Testing Predictive Analytics Engine")
        print("=" * 50)
        
        try:
            analytics = PredictiveAnalytics()
            
            # Test forecast generation
            print("Generating forecast...")
            forecast_request = ForecastRequest(
                metric_name='daily_active_users',
                forecast_horizon=7,
                historical_days=30,
                confidence_level=0.95,
                seasonality=True,
                external_factors={}
            )
            
            forecast_result = await analytics.generate_forecast(forecast_request)
            print(f"Forecast Result: {len(forecast_result.forecast_values)} predictions")
            print(f"Model Performance: {forecast_result.model_performance}")
            
            # Test trend analysis
            print("Analyzing trends...")
            trend_analysis = await analytics.analyze_trends(
                'daily_active_users',
                analysis_period='30d'
            )
            print(f"Trend Analysis: {trend_analysis.trend_direction} (strength: {trend_analysis.trend_strength:.2f})")
            
            # Test anomaly prediction
            print("Predicting anomalies...")
            anomaly_prediction = await analytics.predict_anomalies(
                'daily_active_users',
                prediction_horizon=7
            )
            print(f"Anomaly Prediction: {len(anomaly_prediction.predicted_anomalies)} anomalies predicted")
            
            # Test status
            status = analytics.get_analytics_status()
            print(f"Analytics Status: {status['status']}")
            
            print("\n✅ Predictive Analytics Engine test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_predictive_analytics())