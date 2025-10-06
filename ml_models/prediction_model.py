#!/usr/bin/env python3
"""
Prediction Model
ML-based prediction model for user behavior and opportunity outcomes

This model:
- Predicts user behavior patterns and preferences
- Forecasts opportunity success rates and engagement
- Provides time-series predictions for user activity
- Supports multiple algorithms (Gradient Boosting, Neural Networks, etc.)
- Handles model training, validation, and deployment

Following Task 7 from the PRP implementation blueprint.
"""

import os
import pickle
import json
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


@dataclass
class PredictionFeatures:
    """Features for prediction model"""
    
    user_id: str
    opportunity_id: Optional[str]
    historical_behavior: Dict[str, float]
    market_conditions: Dict[str, float]
    seasonal_factors: Dict[str, float]
    competitive_landscape: Dict[str, float]
    user_network_strength: float
    features_vector: np.ndarray
    feature_names: List[str]


@dataclass
class PredictionResult:
    """Result of prediction"""
    
    user_id: str
    opportunity_id: Optional[str]
    prediction_type: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_horizon: int  # days
    model_version: str
    predicted_at: datetime


@dataclass
class PredictionMetrics:
    """Prediction model metrics"""
    
    model_type: str
    mse: float
    mae: float
    r2_score: float
    cross_val_score: float
    training_samples: int
    validation_samples: int
    training_time: float
    model_version: str


class PredictionModel:
    """
    ML-based prediction model for user behavior and opportunity outcomes
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Prediction Model
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔮 Initializing Prediction Model")
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Model versioning
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Performance tracking
        self.predictions_generated = 0
        self.model_trained = False
        self.last_training_time = None
        self.current_metrics = None
        
        # Prediction types
        self.prediction_types = {
            'user_engagement': 'Predict user engagement levels',
            'opportunity_success': 'Predict opportunity success probability',
            'interaction_volume': 'Predict interaction volume',
            'conversion_rate': 'Predict conversion rates',
            'user_churn': 'Predict user churn probability'
        }
        
        self.logger.info("✅ Prediction Model initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for prediction model")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_ml_components(self):
        """Setup ML components"""
        config = self.ml_config.prediction_model
        
        # Initialize models based on configuration
        self.models = {}
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            subsample=config.subsample,
            random_state=config.random_state
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=self.ml_config.training.n_jobs
        )
        
        # Neural Network
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=config.hidden_layer_sizes,
            activation=config.activation,
            solver=config.solver,
            alpha=config.alpha,
            max_iter=config.max_iter,
            early_stopping=config.early_stopping,
            validation_fraction=config.validation_fraction,
            n_iter_no_change=config.n_iter_no_change,
            random_state=config.random_state
        )
        
        # Current model
        self.current_model = None
        self.current_model_type = config.model_type
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Feature names
        self.feature_names = config.feature_columns
        
        self.logger.info(f"✅ ML components initialized with {len(self.models)} prediction models")
    
    async def train_prediction_model(
        self,
        prediction_type: str,
        model_type: str = None,
        force_retrain: bool = False
    ) -> PredictionMetrics:
        """
        Train the prediction model
        
        Args:
            prediction_type: Type of prediction to train for
            model_type: ML model type to use
            force_retrain: Force retraining even if model exists
            
        Returns:
            PredictionMetrics with training results
        """
        try:
            start_time = datetime.now()
            
            # Use specified model type or default
            if model_type is None:
                model_type = self.current_model_type
            
            self.logger.info(f"🔮 Training {model_type} model for {prediction_type}")
            
            # Check if model exists and is recent
            if not force_retrain and self._model_exists(prediction_type, model_type) and self._model_is_recent():
                self.logger.info("ℹ️ Using existing recent prediction model")
                return self._load_model_metrics(prediction_type, model_type)
            
            # Load training data
            training_data = await self._load_training_data(prediction_type)
            
            if len(training_data) < self.ml_config.training.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # Prepare features and labels
            X, y = await self._prepare_training_data(training_data, prediction_type)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.2,
                random_state=self.ml_config.random_seed
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            if model_type not in self.models:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = self.models[model_type]
            model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = np.mean(cv_scores)
            
            metrics = PredictionMetrics(
                model_type=model_type,
                mse=mse,
                mae=mae,
                r2_score=r2,
                cross_val_score=cv_mean,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                training_time=(datetime.now() - start_time).total_seconds(),
                model_version=self.model_version
            )
            
            # Validate model performance
            if not self._validate_model_performance(metrics):
                raise ValueError("Model performance below acceptable thresholds")
            
            # Save model
            await self._save_model(prediction_type, model_type, metrics)
            
            self.current_model = model
            self.current_model_type = model_type
            self.model_trained = True
            self.last_training_time = datetime.now()
            self.current_metrics = metrics
            
            self.logger.info(f"✅ Prediction model trained successfully - R²: {metrics.r2_score:.3f}, MAE: {metrics.mae:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error training prediction model: {e}")
            raise
    
    async def generate_prediction(
        self,
        user_id: str,
        prediction_type: str,
        opportunity_id: str = None,
        prediction_horizon: int = 7,
        features: PredictionFeatures = None
    ) -> PredictionResult:
        """
        Generate prediction for user
        
        Args:
            user_id: ID of the user
            prediction_type: Type of prediction
            opportunity_id: Optional opportunity ID
            prediction_horizon: Prediction horizon in days
            features: Optional pre-computed features
            
        Returns:
            PredictionResult with prediction
        """
        try:
            self.logger.info(f"🔮 Generating {prediction_type} prediction for user {user_id}")
            
            # Ensure model is trained
            if not self.model_trained:
                await self.train_prediction_model(prediction_type)
            
            # Get or compute features
            if features is None:
                features = await self._extract_prediction_features(
                    user_id, prediction_type, opportunity_id, prediction_horizon
                )
            
            # Scale features
            features_scaled = self.scaler.transform(features.features_vector.reshape(1, -1))
            
            # Generate prediction
            predicted_value = self.current_model.predict(features_scaled)[0]
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                features_scaled, predicted_value
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features.feature_names)
            
            # Create result
            result = PredictionResult(
                user_id=user_id,
                opportunity_id=opportunity_id,
                prediction_type=prediction_type,
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                prediction_horizon=prediction_horizon,
                model_version=self.model_version,
                predicted_at=datetime.now()
            )
            
            # Store prediction in database
            await self._store_prediction(result)
            
            self.predictions_generated += 1
            self.logger.info(f"✅ Prediction generated: {predicted_value:.3f} for {prediction_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error generating prediction: {e}")
            raise
    
    async def batch_generate_predictions(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[PredictionResult]:
        """
        Generate multiple predictions in batch
        
        Args:
            requests: List of prediction requests
            
        Returns:
            List of PredictionResult objects
        """
        try:
            self.logger.info(f"🔮 Batch generating {len(requests)} predictions")
            
            results = []
            
            # Group requests by prediction type for efficiency
            grouped_requests = {}
            for request in requests:
                pred_type = request.get('prediction_type')
                if pred_type not in grouped_requests:
                    grouped_requests[pred_type] = []
                grouped_requests[pred_type].append(request)
            
            # Process each prediction type
            for prediction_type, type_requests in grouped_requests.items():
                # Ensure model is trained for this type
                if not self.model_trained:
                    await self.train_prediction_model(prediction_type)
                
                # Extract features for all requests
                batch_features = []
                request_metadata = []
                
                for request in type_requests:
                    try:
                        features = await self._extract_prediction_features(
                            request['user_id'],
                            prediction_type,
                            request.get('opportunity_id'),
                            request.get('prediction_horizon', 7)
                        )
                        batch_features.append(features)
                        request_metadata.append(request)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error extracting features for request: {e}")
                        continue
                
                if not batch_features:
                    continue
                
                # Create feature matrix
                feature_matrix = np.vstack([f.features_vector for f in batch_features])
                
                # Scale features
                feature_matrix_scaled = self.scaler.transform(feature_matrix)
                
                # Generate predictions
                predicted_values = self.current_model.predict(feature_matrix_scaled)
                
                # Create results
                for i, (features, request, predicted_value) in enumerate(
                    zip(batch_features, request_metadata, predicted_values)
                ):
                    confidence_interval = self._calculate_confidence_interval(
                        feature_matrix_scaled[i:i+1], predicted_value
                    )
                    
                    feature_importance = self._get_feature_importance(features.feature_names)
                    
                    result = PredictionResult(
                        user_id=request['user_id'],
                        opportunity_id=request.get('opportunity_id'),
                        prediction_type=prediction_type,
                        predicted_value=predicted_value,
                        confidence_interval=confidence_interval,
                        feature_importance=feature_importance,
                        prediction_horizon=request.get('prediction_horizon', 7),
                        model_version=self.model_version,
                        predicted_at=datetime.now()
                    )
                    
                    results.append(result)
            
            # Store all predictions in database
            await self._store_batch_predictions(results)
            
            self.predictions_generated += len(results)
            self.logger.info(f"✅ Batch prediction completed for {len(results)} requests")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch prediction: {e}")
            raise
    
    async def evaluate_prediction_accuracy(
        self,
        prediction_type: str,
        days_back: int = 30
    ) -> Dict[str, float]:
        """
        Evaluate prediction accuracy against actual outcomes
        
        Args:
            prediction_type: Type of prediction to evaluate
            days_back: Number of days back to evaluate
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            self.logger.info(f"📊 Evaluating {prediction_type} prediction accuracy")
            
            # Get predictions from the specified period
            predictions = await self._get_historical_predictions(prediction_type, days_back)
            
            if not predictions:
                return {"error": "No historical predictions found"}
            
            # Get actual outcomes
            actual_outcomes = await self._get_actual_outcomes(predictions, prediction_type)
            
            if not actual_outcomes:
                return {"error": "No actual outcomes available"}
            
            # Calculate accuracy metrics
            predicted_values = [p['predicted_value'] for p in predictions]
            actual_values = [actual_outcomes.get(p['user_id'], 0) for p in predictions]
            
            # Remove pairs where actual outcome is not available
            valid_pairs = [(p, a) for p, a in zip(predicted_values, actual_values) if a is not None]
            
            if not valid_pairs:
                return {"error": "No valid prediction-outcome pairs"}
            
            predicted_values, actual_values = zip(*valid_pairs)
            
            # Calculate metrics
            mse = mean_squared_error(actual_values, predicted_values)
            mae = mean_absolute_error(actual_values, predicted_values)
            r2 = r2_score(actual_values, predicted_values)
            
            # Calculate correlation
            correlation = np.corrcoef(predicted_values, actual_values)[0, 1]
            
            accuracy_metrics = {
                "prediction_type": prediction_type,
                "evaluation_period_days": days_back,
                "sample_size": len(valid_pairs),
                "mse": mse,
                "mae": mae,
                "r2_score": r2,
                "correlation": correlation,
                "evaluated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Prediction accuracy evaluated - R²: {r2:.3f}, MAE: {mae:.3f}")
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating prediction accuracy: {e}")
            return {"error": str(e)}
    
    async def _load_training_data(self, prediction_type: str) -> List[Dict[str, Any]]:
        """Load training data for specific prediction type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if prediction_type == 'user_engagement':
                query = """
                    SELECT 
                        up.user_id,
                        up.engagement_score,
                        COUNT(ui.id) as interaction_count,
                        AVG(ui.duration) as avg_duration,
                        COUNT(DISTINCT ui.opportunity_id) as unique_opportunities,
                        SUM(CASE WHEN ui.interaction_type IN ('contact', 'save') THEN 1 ELSE 0 END) as conversions,
                        up.industry,
                        up.location
                    FROM user_profiles up
                    LEFT JOIN user_interactions ui ON up.user_id = ui.user_id
                    WHERE ui.timestamp > ? OR ui.timestamp IS NULL
                    GROUP BY up.user_id
                    HAVING up.engagement_score IS NOT NULL
                """
                
            elif prediction_type == 'opportunity_success':
                query = """
                    SELECT 
                        ui.user_id,
                        ui.opportunity_id,
                        o.industry,
                        o.location,
                        o.opportunity_type,
                        COUNT(ui.id) as interaction_count,
                        SUM(CASE WHEN ui.interaction_type IN ('contact', 'save') THEN 1 ELSE 0 END) as success_count,
                        AVG(ui.duration) as avg_duration,
                        up.engagement_score
                    FROM user_interactions ui
                    JOIN opportunities o ON ui.opportunity_id = o.id
                    JOIN user_profiles up ON ui.user_id = up.user_id
                    WHERE ui.timestamp > ?
                    GROUP BY ui.user_id, ui.opportunity_id
                    HAVING interaction_count > 0
                """
                
            elif prediction_type == 'interaction_volume':
                query = """
                    SELECT 
                        ui.user_id,
                        DATE(ui.timestamp) as interaction_date,
                        COUNT(ui.id) as daily_interactions,
                        up.engagement_score,
                        up.industry,
                        up.location
                    FROM user_interactions ui
                    JOIN user_profiles up ON ui.user_id = up.user_id
                    WHERE ui.timestamp > ?
                    GROUP BY ui.user_id, DATE(ui.timestamp)
                    HAVING daily_interactions > 0
                """
                
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")
            
            cutoff_date = (datetime.now() - timedelta(days=90)).isoformat()
            cursor.execute(query, (cutoff_date,))
            results = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            training_data = []
            for row in results:
                if prediction_type == 'user_engagement':
                    training_data.append({
                        'user_id': row[0],
                        'target': row[1],  # engagement_score
                        'interaction_count': row[2],
                        'avg_duration': row[3] or 0,
                        'unique_opportunities': row[4],
                        'conversions': row[5],
                        'industry': row[6],
                        'location': row[7]
                    })
                elif prediction_type == 'opportunity_success':
                    training_data.append({
                        'user_id': row[0],
                        'opportunity_id': row[1],
                        'industry': row[2],
                        'location': row[3],
                        'opportunity_type': row[4],
                        'interaction_count': row[5],
                        'target': row[6],  # success_count
                        'avg_duration': row[7] or 0,
                        'engagement_score': row[8]
                    })
                elif prediction_type == 'interaction_volume':
                    training_data.append({
                        'user_id': row[0],
                        'interaction_date': row[1],
                        'target': row[2],  # daily_interactions
                        'engagement_score': row[3],
                        'industry': row[4],
                        'location': row[5]
                    })
            
            self.logger.info(f"✅ Loaded {len(training_data)} training samples for {prediction_type}")
            return training_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading training data: {e}")
            raise
    
    async def _prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
        prediction_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        try:
            features_list = []
            labels = []
            
            for sample in training_data:
                # Extract features for each sample
                features = await self._extract_prediction_features_from_sample(
                    sample, prediction_type
                )
                
                features_list.append(features)
                labels.append(sample['target'])
            
            X = np.vstack(features_list)
            y = np.array(labels)
            
            self.logger.info(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing training data: {e}")
            raise
    
    async def _extract_prediction_features(
        self,
        user_id: str,
        prediction_type: str,
        opportunity_id: str = None,
        prediction_horizon: int = 7
    ) -> PredictionFeatures:
        """Extract features for prediction"""
        try:
            # Get user profile
            user_profile = await self._get_user_profile(user_id)
            
            # Get historical behavior
            historical_behavior = await self._get_historical_behavior(user_id, prediction_horizon)
            
            # Get market conditions
            market_conditions = await self._get_market_conditions(prediction_type)
            
            # Get seasonal factors
            seasonal_factors = self._get_seasonal_factors()
            
            # Get competitive landscape
            competitive_landscape = await self._get_competitive_landscape(prediction_type)
            
            # Get network strength
            network_strength = await self._get_network_strength(user_id)
            
            # Create feature vector
            features_vector = np.array([
                historical_behavior.get('avg_engagement', 0.0),
                historical_behavior.get('interaction_frequency', 0.0),
                historical_behavior.get('conversion_rate', 0.0),
                market_conditions.get('market_activity', 0.5),
                seasonal_factors.get('seasonal_multiplier', 1.0),
                competitive_landscape.get('competition_level', 0.5),
                network_strength
            ])
            
            return PredictionFeatures(
                user_id=user_id,
                opportunity_id=opportunity_id,
                historical_behavior=historical_behavior,
                market_conditions=market_conditions,
                seasonal_factors=seasonal_factors,
                competitive_landscape=competitive_landscape,
                user_network_strength=network_strength,
                features_vector=features_vector,
                feature_names=self.feature_names
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting prediction features: {e}")
            raise
    
    async def _extract_prediction_features_from_sample(
        self,
        sample: Dict[str, Any],
        prediction_type: str
    ) -> np.ndarray:
        """Extract features from training sample"""
        try:
            # Basic features from sample
            engagement_score = sample.get('engagement_score', 0.0)
            interaction_count = sample.get('interaction_count', 0)
            avg_duration = sample.get('avg_duration', 0.0)
            conversions = sample.get('conversions', 0)
            
            # Calculate derived features
            conversion_rate = conversions / max(interaction_count, 1)
            interaction_frequency = interaction_count / 30.0  # Normalize by 30 days
            
            # Market conditions (simplified)
            market_activity = 0.5  # Placeholder
            seasonal_multiplier = 1.0  # Placeholder
            competition_level = 0.5  # Placeholder
            
            features = np.array([
                engagement_score,
                interaction_frequency,
                conversion_rate,
                market_activity,
                seasonal_multiplier,
                competition_level,
                avg_duration / 300.0  # Normalize duration
            ])
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting features from sample: {e}")
            raise
    
    async def _get_historical_behavior(self, user_id: str, days_back: int) -> Dict[str, float]:
        """Get historical behavior metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    AVG(CASE WHEN interaction_type IN ('contact', 'save') THEN 1 ELSE 0 END) as conversion_rate,
                    COUNT(*) as total_interactions,
                    AVG(duration) as avg_duration,
                    COUNT(DISTINCT opportunity_id) as unique_opportunities
                FROM user_interactions
                WHERE user_id = ? AND timestamp > ?
            """, (user_id, (datetime.now() - timedelta(days=days_back)).isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'conversion_rate': result[0] or 0.0,
                    'interaction_frequency': result[1] / days_back,
                    'avg_duration': result[2] or 0.0,
                    'unique_opportunities': result[3] or 0,
                    'avg_engagement': (result[0] or 0.0) * 0.5 + (result[1] or 0) / 100.0
                }
            
            return {
                'conversion_rate': 0.0,
                'interaction_frequency': 0.0,
                'avg_duration': 0.0,
                'unique_opportunities': 0,
                'avg_engagement': 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting historical behavior: {e}")
            return {}
    
    async def _get_market_conditions(self, prediction_type: str) -> Dict[str, float]:
        """Get market conditions"""
        try:
            # Simplified market conditions
            # In a real implementation, this would analyze market data
            return {
                'market_activity': 0.6,
                'market_volatility': 0.3,
                'market_trend': 0.1
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market conditions: {e}")
            return {'market_activity': 0.5}
    
    def _get_seasonal_factors(self) -> Dict[str, float]:
        """Get seasonal factors"""
        try:
            # Simple seasonal analysis based on current date
            now = datetime.now()
            month = now.month
            
            # Business seasonality (simplified)
            if month in [1, 2]:  # Q1 - typically slower
                seasonal_multiplier = 0.8
            elif month in [3, 4, 5]:  # Q2 - moderate
                seasonal_multiplier = 1.0
            elif month in [6, 7, 8]:  # Q3 - summer slowdown
                seasonal_multiplier = 0.7
            else:  # Q4 - typically busy
                seasonal_multiplier = 1.2
            
            return {
                'seasonal_multiplier': seasonal_multiplier,
                'month_factor': month / 12.0,
                'quarter_factor': ((month - 1) // 3 + 1) / 4.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting seasonal factors: {e}")
            return {'seasonal_multiplier': 1.0}
    
    async def _get_competitive_landscape(self, prediction_type: str) -> Dict[str, float]:
        """Get competitive landscape metrics"""
        try:
            # Simplified competitive analysis
            # In a real implementation, this would analyze competitor data
            return {
                'competition_level': 0.5,
                'market_saturation': 0.4,
                'competitive_pressure': 0.3
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting competitive landscape: {e}")
            return {'competition_level': 0.5}
    
    async def _get_network_strength(self, user_id: str) -> float:
        """Get user network strength"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(connection_strength), COUNT(*)
                FROM network_analytics
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] is not None:
                return min(result[0], 1.0)
            
            return 0.3
            
        except Exception as e:
            self.logger.error(f"❌ Error getting network strength: {e}")
            return 0.3
    
    def _calculate_confidence_interval(
        self,
        features: np.ndarray,
        predicted_value: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        try:
            # Simplified confidence interval calculation
            # In a real implementation, this would use model-specific methods
            
            # Use standard error estimate
            if hasattr(self.current_model, 'predict_proba'):
                # For probabilistic models
                std_error = 0.1
            else:
                # For deterministic models, use a fixed percentage
                std_error = abs(predicted_value) * 0.1
            
            # Calculate confidence interval
            z_score = 1.96  # 95% confidence level
            margin_of_error = z_score * std_error
            
            lower_bound = predicted_value - margin_of_error
            upper_bound = predicted_value + margin_of_error
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence interval: {e}")
            return (predicted_value * 0.9, predicted_value * 1.1)
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(self.current_model, 'feature_importances_'):
                importance_scores = self.current_model.feature_importances_
                return dict(zip(feature_names, importance_scores))
            elif hasattr(self.current_model, 'coef_'):
                # For linear models
                importance_scores = np.abs(self.current_model.coef_)
                return dict(zip(feature_names, importance_scores))
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting feature importance: {e}")
            return {}
    
    # Helper methods for data retrieval
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id, industry, location, company_size, job_role, 
                       interests, preferences, engagement_score, last_active
                FROM user_profiles
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'user_id': result[0],
                    'industry': result[1],
                    'location': result[2],
                    'company_size': result[3],
                    'job_role': result[4],
                    'interests': json.loads(result[5] or '[]'),
                    'preferences': json.loads(result[6] or '{}'),
                    'engagement_score': result[7],
                    'last_active': result[8]
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user profile: {e}")
            return {}
    
    async def _get_historical_predictions(
        self,
        prediction_type: str,
        days_back: int
    ) -> List[Dict[str, Any]]:
        """Get historical predictions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    opportunity_id TEXT,
                    prediction_type TEXT,
                    predicted_value REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    feature_importance TEXT,
                    prediction_horizon INTEGER,
                    model_version TEXT,
                    predicted_at TEXT
                )
            """)
            
            cursor.execute("""
                SELECT user_id, opportunity_id, prediction_type, predicted_value, 
                       confidence_lower, confidence_upper, predicted_at
                FROM predictions
                WHERE prediction_type = ? AND predicted_at > ?
                ORDER BY predicted_at DESC
            """, (prediction_type, (datetime.now() - timedelta(days=days_back)).isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            predictions = []
            for result in results:
                predictions.append({
                    'user_id': result[0],
                    'opportunity_id': result[1],
                    'prediction_type': result[2],
                    'predicted_value': result[3],
                    'confidence_lower': result[4],
                    'confidence_upper': result[5],
                    'predicted_at': result[6]
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting historical predictions: {e}")
            return []
    
    async def _get_actual_outcomes(
        self,
        predictions: List[Dict[str, Any]],
        prediction_type: str
    ) -> Dict[str, float]:
        """Get actual outcomes for predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            actual_outcomes = {}
            
            for prediction in predictions:
                user_id = prediction['user_id']
                predicted_at = datetime.fromisoformat(prediction['predicted_at'])
                
                if prediction_type == 'user_engagement':
                    # Get actual engagement after prediction
                    cursor.execute("""
                        SELECT engagement_score FROM user_profiles
                        WHERE user_id = ?
                    """, (user_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        actual_outcomes[user_id] = result[0]
                
                elif prediction_type == 'interaction_volume':
                    # Get actual interaction volume
                    cursor.execute("""
                        SELECT COUNT(*) FROM user_interactions
                        WHERE user_id = ? AND timestamp > ?
                    """, (user_id, predicted_at.isoformat()))
                    
                    result = cursor.fetchone()
                    if result:
                        actual_outcomes[user_id] = result[0]
            
            conn.close()
            return actual_outcomes
            
        except Exception as e:
            self.logger.error(f"❌ Error getting actual outcomes: {e}")
            return {}
    
    # Model management methods
    def _model_exists(self, prediction_type: str, model_type: str) -> bool:
        """Check if model file exists"""
        model_path = os.path.join(
            self.ml_config.training.models_path,
            f"prediction_model_{prediction_type}_{model_type}_{self.model_version}.pkl"
        )
        return os.path.exists(model_path)
    
    def _model_is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if model is recent enough"""
        if not self.last_training_time:
            return False
        
        age_hours = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return age_hours < max_age_hours
    
    def _validate_model_performance(self, metrics: PredictionMetrics) -> bool:
        """Validate model performance meets thresholds"""
        config = self.ml_config.prediction_model
        
        return (
            metrics.r2_score >= config.min_r2_score and
            metrics.mae <= config.min_mae
        )
    
    async def _save_model(
        self,
        prediction_type: str,
        model_type: str,
        metrics: PredictionMetrics
    ):
        """Save model to disk"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.ml_config.training.models_path, exist_ok=True)
            
            # Save model
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"prediction_model_{prediction_type}_{model_type}_{self.model_version}.pkl"
            )
            
            model_data = {
                'model': self.current_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': metrics,
                'prediction_type': prediction_type,
                'model_type': model_type,
                'version': self.model_version,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"✅ Prediction model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            raise
    
    def _load_model_metrics(self, prediction_type: str, model_type: str) -> PredictionMetrics:
        """Load model metrics from disk"""
        try:
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"prediction_model_{prediction_type}_{model_type}_{self.model_version}.pkl"
            )
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            return model_data['metrics']
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model metrics: {e}")
            return PredictionMetrics(
                model_type=model_type,
                mse=0.0,
                mae=0.0,
                r2_score=0.0,
                cross_val_score=0.0,
                training_samples=0,
                validation_samples=0,
                training_time=0.0,
                model_version=self.model_version
            )
    
    # Database operations
    async def _store_prediction(self, prediction: PredictionResult):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    opportunity_id TEXT,
                    prediction_type TEXT,
                    predicted_value REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    feature_importance TEXT,
                    prediction_horizon INTEGER,
                    model_version TEXT,
                    predicted_at TEXT
                )
            """)
            
            # Insert prediction
            cursor.execute("""
                INSERT OR REPLACE INTO predictions 
                (id, user_id, opportunity_id, prediction_type, predicted_value, 
                 confidence_lower, confidence_upper, feature_importance, 
                 prediction_horizon, model_version, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"pred_{prediction.user_id}_{prediction.prediction_type}_{int(prediction.predicted_at.timestamp())}",
                prediction.user_id,
                prediction.opportunity_id,
                prediction.prediction_type,
                prediction.predicted_value,
                prediction.confidence_interval[0],
                prediction.confidence_interval[1],
                json.dumps(prediction.feature_importance),
                prediction.prediction_horizon,
                prediction.model_version,
                prediction.predicted_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing prediction: {e}")
    
    async def _store_batch_predictions(self, predictions: List[PredictionResult]):
        """Store batch predictions in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    opportunity_id TEXT,
                    prediction_type TEXT,
                    predicted_value REAL,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    feature_importance TEXT,
                    prediction_horizon INTEGER,
                    model_version TEXT,
                    predicted_at TEXT
                )
            """)
            
            # Prepare batch data
            batch_data = []
            for prediction in predictions:
                batch_data.append((
                    f"pred_{prediction.user_id}_{prediction.prediction_type}_{int(prediction.predicted_at.timestamp())}",
                    prediction.user_id,
                    prediction.opportunity_id,
                    prediction.prediction_type,
                    prediction.predicted_value,
                    prediction.confidence_interval[0],
                    prediction.confidence_interval[1],
                    json.dumps(prediction.feature_importance),
                    prediction.prediction_horizon,
                    prediction.model_version,
                    prediction.predicted_at.isoformat()
                ))
            
            # Insert batch data
            cursor.executemany("""
                INSERT OR REPLACE INTO predictions 
                (id, user_id, opportunity_id, prediction_type, predicted_value, 
                 confidence_lower, confidence_upper, feature_importance, 
                 prediction_horizon, model_version, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing batch predictions: {e}")
    
    # Status and management methods
    def get_model_status(self) -> Dict[str, Any]:
        """Get prediction model status"""
        return {
            "status": "operational" if self.model_trained else "not_trained",
            "model_version": self.model_version,
            "current_model_type": self.current_model_type,
            "predictions_generated": self.predictions_generated,
            "model_trained": self.model_trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "current_metrics": self.current_metrics.__dict__ if self.current_metrics else None,
            "prediction_types": self.prediction_types,
            "available_models": list(self.models.keys()),
            "feature_names": self.feature_names,
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the prediction model
    async def test_prediction_model():
        print("🔮 Testing Prediction Model")
        print("=" * 50)
        
        try:
            model = PredictionModel()
            
            # Test model training
            print("Training prediction model...")
            metrics = await model.train_prediction_model(
                prediction_type='user_engagement',
                model_type='gradient_boosting',
                force_retrain=True
            )
            print(f"Training Metrics: {metrics}")
            
            # Test prediction generation
            print("Generating prediction...")
            prediction = await model.generate_prediction(
                user_id="test_user_123",
                prediction_type='user_engagement',
                prediction_horizon=7
            )
            print(f"Prediction: {prediction}")
            
            # Test batch predictions
            print("Batch generating predictions...")
            requests = [
                {"user_id": "test_user_123", "prediction_type": "user_engagement"},
                {"user_id": "test_user_456", "prediction_type": "user_engagement"}
            ]
            batch_results = await model.batch_generate_predictions(requests)
            print(f"Batch Results: {len(batch_results)} predictions")
            
            # Test prediction accuracy evaluation
            print("Evaluating prediction accuracy...")
            accuracy = await model.evaluate_prediction_accuracy(
                prediction_type='user_engagement',
                days_back=30
            )
            print(f"Accuracy Metrics: {accuracy}")
            
            # Test status
            status = model.get_model_status()
            print(f"Model Status: {status}")
            
            print("✅ Prediction Model test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_prediction_model())