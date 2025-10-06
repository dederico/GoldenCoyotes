#!/usr/bin/env python3
"""
Scoring Model
ML-based opportunity scoring model using Random Forest and other algorithms

This model:
- Scores opportunities based on user preferences and behavior
- Predicts success probability for user-opportunity pairs
- Supports multiple algorithms (Random Forest, XGBoost, etc.)
- Handles feature engineering and model training
- Provides model versioning and performance tracking

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

from config.intelligence_config import get_config
from config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


@dataclass
class ScoringFeatures:
    """Features for opportunity scoring"""
    
    user_id: str
    opportunity_id: str
    industry_match: float
    location_proximity: float
    interaction_frequency: float
    historical_success_rate: float
    network_strength: float
    timing_score: float
    content_similarity: float
    behavioral_patterns: float
    features_vector: np.ndarray
    feature_names: List[str]


@dataclass
class ScoringResult:
    """Result of opportunity scoring"""
    
    opportunity_id: str
    user_id: str
    relevance_score: float
    success_probability: float
    confidence_score: float
    feature_importance: Dict[str, float]
    model_version: str
    scored_at: datetime


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_samples: int
    validation_samples: int
    training_time: float
    model_version: str


class ScoringModel:
    """
    ML-based opportunity scoring model
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Scoring Model
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Initializing Scoring Model")
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Model versioning
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Performance tracking
        self.scores_generated = 0
        self.model_trained = False
        self.last_training_time = None
        self.current_metrics = None
        
        self.logger.info("✅ Scoring Model initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for scoring model")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_ml_components(self):
        """Setup ML components"""
        # Initialize models based on configuration
        model_config = self.ml_config.scoring_model
        
        if model_config.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                min_samples_split=model_config.min_samples_split,
                min_samples_leaf=model_config.min_samples_leaf,
                max_features=model_config.max_features,
                random_state=model_config.random_state,
                class_weight=model_config.class_weight,
                n_jobs=self.ml_config.training.n_jobs
            )
        elif model_config.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=model_config.n_estimators,
                max_depth=model_config.max_depth,
                learning_rate=0.1,
                min_samples_split=model_config.min_samples_split,
                min_samples_leaf=model_config.min_samples_leaf,
                random_state=model_config.random_state
            )
        else:
            # Default to Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=self.ml_config.training.n_jobs
            )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Label encoder for categorical features
        self.label_encoders = {}
        
        # Feature names
        self.feature_names = self.ml_config.scoring_model.feature_columns
        
        self.logger.info(f"✅ ML components initialized with {model_config.model_type} model")
    
    async def train_model(self, force_retrain: bool = False) -> ModelMetrics:
        """
        Train the scoring model
        
        Args:
            force_retrain: Force retraining even if model exists
            
        Returns:
            ModelMetrics with training results
        """
        try:
            start_time = datetime.now()
            self.logger.info("🎯 Starting model training")
            
            # Check if model exists and is recent
            if not force_retrain and self._model_exists() and self._model_is_recent():
                self.logger.info("ℹ️ Using existing recent model")
                return self._load_model_metrics()
            
            # Load training data
            training_data = await self._load_training_data()
            
            if len(training_data) < self.ml_config.scoring_model.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # Prepare features and labels
            X, y = await self._prepare_training_data(training_data)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.ml_config.scoring_model.validation_split,
                random_state=self.ml_config.random_seed,
                stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred = self.model.predict(X_val_scaled)
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate metrics
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, average='weighted'),
                recall=recall_score(y_val, y_pred, average='weighted'),
                f1_score=f1_score(y_val, y_pred, average='weighted'),
                roc_auc=roc_auc_score(y_val, y_pred_proba),
                training_samples=len(X_train),
                validation_samples=len(X_val),
                training_time=(datetime.now() - start_time).total_seconds(),
                model_version=self.model_version
            )
            
            # Validate model performance
            if not self._validate_model_performance(metrics):
                raise ValueError("Model performance below acceptable thresholds")
            
            # Save model
            await self._save_model(metrics)
            
            self.model_trained = True
            self.last_training_time = datetime.now()
            self.current_metrics = metrics
            
            self.logger.info(f"✅ Model trained successfully - Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error training model: {e}")
            raise
    
    async def score_opportunity(
        self,
        user_id: str,
        opportunity_id: str,
        features: ScoringFeatures = None
    ) -> ScoringResult:
        """
        Score an opportunity for a user
        
        Args:
            user_id: ID of the user
            opportunity_id: ID of the opportunity
            features: Optional pre-computed features
            
        Returns:
            ScoringResult with scores and predictions
        """
        try:
            self.logger.info(f"📊 Scoring opportunity {opportunity_id} for user {user_id}")
            
            # Ensure model is trained
            if not self.model_trained:
                await self.train_model()
            
            # Get or compute features
            if features is None:
                features = await self._extract_features(user_id, opportunity_id)
            
            # Scale features
            features_scaled = self.scaler.transform(features.features_vector.reshape(1, -1))
            
            # Generate predictions
            relevance_score = self.model.predict_proba(features_scaled)[0][1]
            success_probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(features_scaled)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(features.feature_names)
            
            # Create result
            result = ScoringResult(
                opportunity_id=opportunity_id,
                user_id=user_id,
                relevance_score=relevance_score,
                success_probability=success_probability,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_version=self.model_version,
                scored_at=datetime.now()
            )
            
            self.scores_generated += 1
            self.logger.info(f"✅ Opportunity scored - Relevance: {relevance_score:.3f}, Success: {success_probability:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring opportunity: {e}")
            raise
    
    async def batch_score_opportunities(
        self,
        user_id: str,
        opportunity_ids: List[str]
    ) -> List[ScoringResult]:
        """
        Score multiple opportunities for a user
        
        Args:
            user_id: ID of the user
            opportunity_ids: List of opportunity IDs
            
        Returns:
            List of ScoringResult objects
        """
        try:
            self.logger.info(f"📊 Batch scoring {len(opportunity_ids)} opportunities for user {user_id}")
            
            # Ensure model is trained
            if not self.model_trained:
                await self.train_model()
            
            results = []
            
            # Process in batches for memory efficiency
            batch_size = 100
            for i in range(0, len(opportunity_ids), batch_size):
                batch_ids = opportunity_ids[i:i + batch_size]
                
                # Extract features for batch
                batch_features = []
                for opportunity_id in batch_ids:
                    features = await self._extract_features(user_id, opportunity_id)
                    batch_features.append(features)
                
                # Create feature matrix
                feature_matrix = np.vstack([f.features_vector for f in batch_features])
                
                # Scale features
                feature_matrix_scaled = self.scaler.transform(feature_matrix)
                
                # Generate predictions
                relevance_scores = self.model.predict_proba(feature_matrix_scaled)[:, 1]
                success_probabilities = self.model.predict_proba(feature_matrix_scaled)[:, 1]
                
                # Create results
                for j, opportunity_id in enumerate(batch_ids):
                    confidence_score = self._calculate_confidence_score(
                        feature_matrix_scaled[j:j+1]
                    )
                    
                    feature_importance = self._get_feature_importance(batch_features[j].feature_names)
                    
                    result = ScoringResult(
                        opportunity_id=opportunity_id,
                        user_id=user_id,
                        relevance_score=relevance_scores[j],
                        success_probability=success_probabilities[j],
                        confidence_score=confidence_score,
                        feature_importance=feature_importance,
                        model_version=self.model_version,
                        scored_at=datetime.now()
                    )
                    
                    results.append(result)
                
                self.logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(opportunity_ids) + batch_size - 1)//batch_size}")
            
            self.scores_generated += len(results)
            self.logger.info(f"✅ Batch scoring completed for {len(results)} opportunities")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch scoring: {e}")
            raise
    
    async def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get training data from user interactions and opportunity scores
            query = """
                SELECT 
                    ui.user_id,
                    ui.opportunity_id,
                    ui.interaction_type,
                    COUNT(ui.id) as interaction_count,
                    AVG(CASE WHEN ui.interaction_type IN ('contact', 'save') THEN 1 ELSE 0 END) as success_label,
                    up.industry,
                    up.location,
                    up.engagement_score,
                    o.industry as opp_industry,
                    o.location as opp_location,
                    o.opportunity_type,
                    o.created_at
                FROM user_interactions ui
                JOIN user_profiles up ON ui.user_id = up.user_id
                JOIN opportunities o ON ui.opportunity_id = o.id
                WHERE ui.timestamp > ?
                GROUP BY ui.user_id, ui.opportunity_id
                HAVING interaction_count > 0
            """
            
            cutoff_date = (datetime.now() - timedelta(
                days=self.ml_config.scoring_model.training_data_window_days
            )).isoformat()
            
            cursor.execute(query, (cutoff_date,))
            results = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            training_data = []
            for row in results:
                training_data.append({
                    'user_id': row[0],
                    'opportunity_id': row[1],
                    'interaction_type': row[2],
                    'interaction_count': row[3],
                    'success_label': row[4],
                    'user_industry': row[5],
                    'user_location': row[6],
                    'user_engagement': row[7],
                    'opp_industry': row[8],
                    'opp_location': row[9],
                    'opp_type': row[10],
                    'opp_created_at': row[11]
                })
            
            self.logger.info(f"✅ Loaded {len(training_data)} training samples")
            return training_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading training data: {e}")
            raise
    
    async def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model"""
        try:
            features_list = []
            labels = []
            
            for sample in training_data:
                # Extract features for each sample
                features = await self._extract_features(
                    sample['user_id'], 
                    sample['opportunity_id']
                )
                
                features_list.append(features.features_vector)
                labels.append(1 if sample['success_label'] > 0.5 else 0)
            
            X = np.vstack(features_list)
            y = np.array(labels)
            
            self.logger.info(f"✅ Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing training data: {e}")
            raise
    
    async def _extract_features(self, user_id: str, opportunity_id: str) -> ScoringFeatures:
        """Extract features for scoring"""
        try:
            # Get user profile
            user_profile = await self._get_user_profile(user_id)
            
            # Get opportunity data
            opportunity = await self._get_opportunity(opportunity_id)
            
            # Get user interactions
            user_interactions = await self._get_user_interactions(user_id)
            
            # Calculate features
            features = {}
            
            # 1. Industry match
            features['industry_match'] = self._calculate_industry_match(
                user_profile.get('industry', ''),
                opportunity.get('industry', '')
            )
            
            # 2. Location proximity
            features['location_proximity'] = self._calculate_location_proximity(
                user_profile.get('location', ''),
                opportunity.get('location', '')
            )
            
            # 3. Interaction frequency
            features['interaction_frequency'] = len(user_interactions) / 30  # Normalized by 30 days
            
            # 4. Historical success rate
            features['historical_success_rate'] = self._calculate_historical_success_rate(user_interactions)
            
            # 5. Network strength
            features['network_strength'] = await self._calculate_network_strength(user_id)
            
            # 6. Timing score
            features['timing_score'] = self._calculate_timing_score(opportunity)
            
            # 7. Content similarity
            features['content_similarity'] = await self._calculate_content_similarity(
                user_profile, opportunity
            )
            
            # 8. Behavioral patterns
            features['behavioral_patterns'] = await self._calculate_behavioral_patterns(user_id)
            
            # Create feature vector
            feature_vector = np.array([
                features['industry_match'],
                features['location_proximity'],
                features['interaction_frequency'],
                features['historical_success_rate'],
                features['network_strength'],
                features['timing_score'],
                features['content_similarity'],
                features['behavioral_patterns']
            ])
            
            return ScoringFeatures(
                user_id=user_id,
                opportunity_id=opportunity_id,
                industry_match=features['industry_match'],
                location_proximity=features['location_proximity'],
                interaction_frequency=features['interaction_frequency'],
                historical_success_rate=features['historical_success_rate'],
                network_strength=features['network_strength'],
                timing_score=features['timing_score'],
                content_similarity=features['content_similarity'],
                behavioral_patterns=features['behavioral_patterns'],
                features_vector=feature_vector,
                feature_names=self.feature_names
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting features: {e}")
            raise
    
    def _calculate_industry_match(self, user_industry: str, opp_industry: str) -> float:
        """Calculate industry match score"""
        if not user_industry or not opp_industry:
            return 0.5
        
        if user_industry.lower() == opp_industry.lower():
            return 1.0
        
        # Simple related industry matching
        industry_relations = {
            'technology': ['software', 'tech', 'it'],
            'finance': ['banking', 'fintech'],
            'healthcare': ['medical', 'pharma'],
            'retail': ['ecommerce', 'commerce']
        }
        
        for base, related in industry_relations.items():
            if (user_industry.lower() in related and opp_industry.lower() in related):
                return 0.7
        
        return 0.2
    
    def _calculate_location_proximity(self, user_location: str, opp_location: str) -> float:
        """Calculate location proximity score"""
        if not user_location or not opp_location:
            return 0.5
        
        if user_location.lower() == opp_location.lower():
            return 1.0
        
        # Simple location matching
        user_parts = user_location.lower().split(',')
        opp_parts = opp_location.lower().split(',')
        
        if len(user_parts) > 1 and len(opp_parts) > 1:
            if user_parts[0].strip() == opp_parts[0].strip():  # Same city
                return 0.8
            if user_parts[-1].strip() == opp_parts[-1].strip():  # Same country
                return 0.4
        
        return 0.1
    
    def _calculate_historical_success_rate(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate historical success rate"""
        if not interactions:
            return 0.5
        
        success_actions = ['contact', 'save']
        success_count = sum(1 for i in interactions if i.get('interaction_type') in success_actions)
        
        return success_count / len(interactions)
    
    async def _calculate_network_strength(self, user_id: str) -> float:
        """Calculate network strength score"""
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
            self.logger.warning(f"⚠️ Error calculating network strength: {e}")
            return 0.3
    
    def _calculate_timing_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate timing score"""
        try:
            created_at = opportunity.get('created_at')
            if not created_at:
                return 0.5
            
            created_date = datetime.fromisoformat(created_at)
            days_since_creation = (datetime.now() - created_date).days
            
            # Decay score over time
            timing_score = max(0, 1 - (days_since_creation / 30))
            
            return timing_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating timing score: {e}")
            return 0.5
    
    async def _calculate_content_similarity(
        self,
        user_profile: Dict[str, Any],
        opportunity: Dict[str, Any]
    ) -> float:
        """Calculate content similarity score"""
        try:
            # Simple text similarity (can be enhanced with embeddings)
            user_interests = user_profile.get('interests', [])
            opp_tags = opportunity.get('tags', [])
            
            if not user_interests or not opp_tags:
                return 0.5
            
            # Calculate intersection
            user_interests_lower = [i.lower() for i in user_interests]
            opp_tags_lower = [t.lower() for t in opp_tags]
            
            intersection = set(user_interests_lower) & set(opp_tags_lower)
            union = set(user_interests_lower) | set(opp_tags_lower)
            
            if not union:
                return 0.5
            
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating content similarity: {e}")
            return 0.5
    
    async def _calculate_behavioral_patterns(self, user_id: str) -> float:
        """Calculate behavioral patterns score"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user engagement score
            cursor.execute("""
                SELECT engagement_score FROM user_profiles
                WHERE user_id = ?
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] is not None:
                return min(result[0], 1.0)
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating behavioral patterns: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, features_scaled: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        try:
            # Use prediction probability as confidence measure
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Confidence is the maximum probability
            confidence = max(probabilities)
            
            return confidence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating confidence score: {e}")
            return 0.5
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_
                return dict(zip(feature_names, importance_scores))
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting feature importance: {e}")
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
    
    async def _get_opportunity(self, opportunity_id: str) -> Dict[str, Any]:
        """Get opportunity from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, description, opportunity_type, industry, 
                       location, budget_range, deadline, contact_info, tags, 
                       status, created_at, updated_at
                FROM opportunities
                WHERE id = ?
            """, (opportunity_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'title': result[1],
                    'description': result[2],
                    'opportunity_type': result[3],
                    'industry': result[4],
                    'location': result[5],
                    'budget_range': result[6],
                    'deadline': result[7],
                    'contact_info': json.loads(result[8] or '{}'),
                    'tags': json.loads(result[9] or '[]'),
                    'status': result[10],
                    'created_at': result[11],
                    'updated_at': result[12]
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting opportunity: {e}")
            return {}
    
    async def _get_user_interactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user interactions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id, opportunity_id, interaction_type, timestamp, 
                       duration, metadata
                FROM user_interactions
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, (datetime.now() - timedelta(days=30)).isoformat()))
            
            results = cursor.fetchall()
            conn.close()
            
            interactions = []
            for result in results:
                interactions.append({
                    'user_id': result[0],
                    'opportunity_id': result[1],
                    'interaction_type': result[2],
                    'timestamp': result[3],
                    'duration': result[4],
                    'metadata': json.loads(result[5] or '{}')
                })
            
            return interactions
            
        except Exception as e:
            self.logger.error(f"❌ Error getting user interactions: {e}")
            return []
    
    # Model management methods
    def _model_exists(self) -> bool:
        """Check if model file exists"""
        model_path = os.path.join(self.ml_config.training.models_path, f"scoring_model_{self.model_version}.pkl")
        return os.path.exists(model_path)
    
    def _model_is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if model is recent enough"""
        if not self.last_training_time:
            return False
        
        age_hours = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return age_hours < max_age_hours
    
    def _validate_model_performance(self, metrics: ModelMetrics) -> bool:
        """Validate model performance meets thresholds"""
        config = self.ml_config.scoring_model
        
        return (
            metrics.accuracy >= config.min_accuracy and
            metrics.precision >= config.min_precision and
            metrics.recall >= config.min_recall and
            metrics.f1_score >= config.min_f1_score
        )
    
    async def _save_model(self, metrics: ModelMetrics):
        """Save model to disk"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.ml_config.training.models_path, exist_ok=True)
            
            # Save model
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"scoring_model_{self.model_version}.pkl"
            )
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': metrics,
                'version': self.model_version,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"✅ Model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving model: {e}")
            raise
    
    def _load_model_metrics(self) -> ModelMetrics:
        """Load model metrics from disk"""
        try:
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"scoring_model_{self.model_version}.pkl"
            )
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            return model_data['metrics']
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model metrics: {e}")
            return ModelMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                roc_auc=0.0, training_samples=0, validation_samples=0,
                training_time=0.0, model_version=self.model_version
            )
    
    # Status and management methods
    def get_model_status(self) -> Dict[str, Any]:
        """Get scoring model status"""
        return {
            "status": "operational" if self.model_trained else "not_trained",
            "model_version": self.model_version,
            "scores_generated": self.scores_generated,
            "model_trained": self.model_trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "current_metrics": self.current_metrics.__dict__ if self.current_metrics else None,
            "feature_names": self.feature_names,
            "model_type": self.ml_config.scoring_model.model_type,
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the scoring model
    async def test_scoring_model():
        print("🎯 Testing Scoring Model")
        print("=" * 50)
        
        try:
            model = ScoringModel()
            
            # Test model training
            print("Training model...")
            metrics = await model.train_model(force_retrain=True)
            print(f"Training Metrics: {metrics}")
            
            # Test opportunity scoring
            print("Scoring opportunity...")
            score_result = await model.score_opportunity(
                user_id="test_user_123",
                opportunity_id="test_opp_456"
            )
            print(f"Score Result: {score_result}")
            
            # Test batch scoring
            print("Batch scoring...")
            batch_results = await model.batch_score_opportunities(
                user_id="test_user_123",
                opportunity_ids=["test_opp_456", "test_opp_789"]
            )
            print(f"Batch Results: {len(batch_results)} scores")
            
            # Test status
            status = model.get_model_status()
            print(f"Model Status: {status}")
            
            print("✅ Scoring Model test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_scoring_model())