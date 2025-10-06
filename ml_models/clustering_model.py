#!/usr/bin/env python3
"""
Clustering Model
User segmentation and clustering using K-means, DBSCAN, and HDBSCAN

This model:
- Segments users based on behavior patterns and preferences
- Supports multiple clustering algorithms (K-means, DBSCAN, HDBSCAN)
- Provides cluster analysis and user segment insights
- Handles dynamic cluster updates and optimization
- Tracks cluster stability and quality metrics

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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

# Optional HDBSCAN import (may not be available in all environments)
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from config.intelligence_config import get_config
from config.ml_config import get_ml_config

logger = logging.getLogger(__name__)


@dataclass
class ClusteringFeatures:
    """Features for user clustering"""
    
    user_id: str
    engagement_score: float
    interaction_diversity: float
    session_duration: float
    content_preferences: List[str]
    timing_patterns: List[int]
    network_activity: float
    conversion_rate: float
    features_vector: np.ndarray
    feature_names: List[str]


@dataclass
class UserCluster:
    """User cluster assignment"""
    
    user_id: str
    cluster_id: int
    cluster_label: str
    cluster_probability: float
    distance_to_centroid: float
    cluster_characteristics: Dict[str, Any]
    assigned_at: datetime


@dataclass
class ClusterAnalysis:
    """Cluster analysis results"""
    
    cluster_id: int
    cluster_label: str
    user_count: int
    characteristics: Dict[str, Any]
    center_point: np.ndarray
    variance: float
    silhouette_score: float
    representative_users: List[str]
    key_features: List[str]


@dataclass
class ClusteringMetrics:
    """Clustering model metrics"""
    
    algorithm: str
    n_clusters: int
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    training_samples: int
    clustering_time: float
    model_version: str


class ClusteringModel:
    """
    User clustering and segmentation model
    """
    
    def __init__(self, config=None, ml_config=None):
        """
        Initialize the Clustering Model
        
        Args:
            config: Intelligence configuration
            ml_config: ML configuration
        """
        self.config = config or get_config()
        self.ml_config = ml_config or get_ml_config()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("👥 Initializing Clustering Model")
        
        # Initialize database connection
        self._setup_database()
        
        # Initialize ML components
        self._setup_ml_components()
        
        # Model versioning
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Performance tracking
        self.users_clustered = 0
        self.model_trained = False
        self.last_training_time = None
        self.current_metrics = None
        self.cluster_analyses = []
        
        self.logger.info("✅ Clustering Model initialized successfully")
    
    def _setup_database(self):
        """Setup database connection"""
        try:
            self.db_path = self.config.database.intelligence_db_path
            self.logger.info("✅ Database connection established for clustering model")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    def _setup_ml_components(self):
        """Setup ML components"""
        config = self.ml_config.clustering_model
        
        # Initialize clustering algorithms
        self.models = {}
        
        # K-means
        self.models['kmeans'] = KMeans(
            n_clusters=config.n_clusters,
            init=config.init,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tol,
            random_state=config.random_state,
            n_jobs=self.ml_config.training.n_jobs
        )
        
        # DBSCAN
        self.models['dbscan'] = DBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples,
            n_jobs=self.ml_config.training.n_jobs
        )
        
        # HDBSCAN (if available)
        if HDBSCAN_AVAILABLE:
            self.models['hdbscan'] = HDBSCAN(
                min_cluster_size=config.hdbscan_min_cluster_size,
                min_samples=config.hdbscan_min_samples,
                cluster_selection_epsilon=config.hdbscan_cluster_selection_epsilon
            )
        
        # Current model
        self.current_model = None
        self.current_algorithm = config.model_type
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95, random_state=config.random_state)
        
        # Feature names
        self.feature_names = config.feature_columns
        
        # Cluster labels
        self.cluster_labels = {}
        
        self.logger.info(f"✅ ML components initialized with {len(self.models)} clustering algorithms")
    
    async def train_clustering_model(
        self,
        algorithm: str = None,
        force_retrain: bool = False
    ) -> ClusteringMetrics:
        """
        Train the clustering model
        
        Args:
            algorithm: Clustering algorithm to use
            force_retrain: Force retraining even if model exists
            
        Returns:
            ClusteringMetrics with training results
        """
        try:
            start_time = datetime.now()
            
            # Use specified algorithm or default
            if algorithm is None:
                algorithm = self.current_algorithm
            
            self.logger.info(f"👥 Training clustering model with {algorithm}")
            
            # Check if model exists and is recent
            if not force_retrain and self._model_exists(algorithm) and self._model_is_recent():
                self.logger.info("ℹ️ Using existing recent clustering model")
                return self._load_model_metrics(algorithm)
            
            # Load clustering data
            clustering_data = await self._load_clustering_data()
            
            if len(clustering_data) < self.config.behavior_analysis.min_segment_size:
                raise ValueError(f"Insufficient data for clustering: {len(clustering_data)} users")
            
            # Prepare features
            X = await self._prepare_clustering_features(clustering_data)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            X_pca = self.pca.fit_transform(X_scaled)
            
            # Train clustering model
            if algorithm not in self.models:
                raise ValueError(f"Unknown clustering algorithm: {algorithm}")
            
            model = self.models[algorithm]
            
            # Fit model
            if algorithm == 'kmeans':
                cluster_labels = model.fit_predict(X_pca)
                inertia = model.inertia_
            elif algorithm == 'dbscan':
                cluster_labels = model.fit_predict(X_pca)
                inertia = None
            elif algorithm == 'hdbscan' and HDBSCAN_AVAILABLE:
                cluster_labels = model.fit_predict(X_pca)
                inertia = None
            else:
                raise ValueError(f"Clustering algorithm {algorithm} not supported")
            
            # Calculate metrics
            metrics = self._calculate_clustering_metrics(
                X_pca, cluster_labels, algorithm, inertia,
                len(clustering_data), (datetime.now() - start_time).total_seconds()
            )
            
            # Validate clustering quality
            if not self._validate_clustering_quality(metrics):
                raise ValueError("Clustering quality below acceptable thresholds")
            
            # Analyze clusters
            self.cluster_analyses = await self._analyze_clusters(
                clustering_data, cluster_labels, X_pca
            )
            
            # Generate cluster labels
            self.cluster_labels = self._generate_cluster_labels(self.cluster_analyses)
            
            # Save model
            await self._save_clustering_model(algorithm, metrics)
            
            # Update state
            self.current_model = model
            self.current_algorithm = algorithm
            self.model_trained = True
            self.last_training_time = datetime.now()
            self.current_metrics = metrics
            
            self.logger.info(f"✅ Clustering model trained successfully - {metrics.n_clusters} clusters, Silhouette: {metrics.silhouette_score:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error training clustering model: {e}")
            raise
    
    async def assign_user_to_cluster(self, user_id: str) -> UserCluster:
        """
        Assign a user to a cluster
        
        Args:
            user_id: ID of the user
            
        Returns:
            UserCluster with assignment details
        """
        try:
            self.logger.info(f"👥 Assigning user {user_id} to cluster")
            
            # Ensure model is trained
            if not self.model_trained:
                await self.train_clustering_model()
            
            # Extract features for user
            user_features = await self._extract_user_features(user_id)
            
            # Scale and transform features
            features_scaled = self.scaler.transform(user_features.features_vector.reshape(1, -1))
            features_pca = self.pca.transform(features_scaled)
            
            # Predict cluster
            if self.current_algorithm == 'kmeans':
                cluster_id = self.current_model.predict(features_pca)[0]
                cluster_probability = self._calculate_cluster_probability(features_pca, cluster_id)
                distance_to_centroid = self._calculate_distance_to_centroid(features_pca, cluster_id)
            elif self.current_algorithm in ['dbscan', 'hdbscan']:
                cluster_id = self.current_model.fit_predict(features_pca)[0]
                cluster_probability = 1.0 if cluster_id != -1 else 0.0
                distance_to_centroid = 0.0
            else:
                raise ValueError(f"Unsupported algorithm: {self.current_algorithm}")
            
            # Get cluster characteristics
            cluster_characteristics = self._get_cluster_characteristics(cluster_id)
            
            # Get cluster label
            cluster_label = self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
            
            # Create user cluster assignment
            user_cluster = UserCluster(
                user_id=user_id,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                cluster_probability=cluster_probability,
                distance_to_centroid=distance_to_centroid,
                cluster_characteristics=cluster_characteristics,
                assigned_at=datetime.now()
            )
            
            # Store assignment in database
            await self._store_user_cluster_assignment(user_cluster)
            
            self.users_clustered += 1
            self.logger.info(f"✅ User {user_id} assigned to cluster {cluster_id} ({cluster_label})")
            
            return user_cluster
            
        except Exception as e:
            self.logger.error(f"❌ Error assigning user to cluster: {e}")
            raise
    
    async def batch_assign_users_to_clusters(self, user_ids: List[str]) -> List[UserCluster]:
        """
        Assign multiple users to clusters
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            List of UserCluster assignments
        """
        try:
            self.logger.info(f"👥 Batch assigning {len(user_ids)} users to clusters")
            
            # Ensure model is trained
            if not self.model_trained:
                await self.train_clustering_model()
            
            results = []
            
            # Process in batches for memory efficiency
            batch_size = 100
            for i in range(0, len(user_ids), batch_size):
                batch_ids = user_ids[i:i + batch_size]
                
                # Extract features for batch
                batch_features = []
                for user_id in batch_ids:
                    try:
                        features = await self._extract_user_features(user_id)
                        batch_features.append((user_id, features))
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error extracting features for user {user_id}: {e}")
                        continue
                
                if not batch_features:
                    continue
                
                # Create feature matrix
                feature_matrix = np.vstack([features.features_vector for _, features in batch_features])
                
                # Scale and transform features
                feature_matrix_scaled = self.scaler.transform(feature_matrix)
                feature_matrix_pca = self.pca.transform(feature_matrix_scaled)
                
                # Predict clusters
                if self.current_algorithm == 'kmeans':
                    cluster_ids = self.current_model.predict(feature_matrix_pca)
                elif self.current_algorithm in ['dbscan', 'hdbscan']:
                    cluster_ids = self.current_model.fit_predict(feature_matrix_pca)
                else:
                    raise ValueError(f"Unsupported algorithm: {self.current_algorithm}")
                
                # Create user cluster assignments
                for j, (user_id, features) in enumerate(batch_features):
                    cluster_id = cluster_ids[j]
                    
                    # Calculate cluster metrics
                    if self.current_algorithm == 'kmeans':
                        cluster_probability = self._calculate_cluster_probability(
                            feature_matrix_pca[j:j+1], cluster_id
                        )
                        distance_to_centroid = self._calculate_distance_to_centroid(
                            feature_matrix_pca[j:j+1], cluster_id
                        )
                    else:
                        cluster_probability = 1.0 if cluster_id != -1 else 0.0
                        distance_to_centroid = 0.0
                    
                    # Get cluster characteristics
                    cluster_characteristics = self._get_cluster_characteristics(cluster_id)
                    
                    # Get cluster label
                    cluster_label = self.cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
                    
                    # Create user cluster assignment
                    user_cluster = UserCluster(
                        user_id=user_id,
                        cluster_id=cluster_id,
                        cluster_label=cluster_label,
                        cluster_probability=cluster_probability,
                        distance_to_centroid=distance_to_centroid,
                        cluster_characteristics=cluster_characteristics,
                        assigned_at=datetime.now()
                    )
                    
                    results.append(user_cluster)
                
                self.logger.info(f"✅ Processed batch {i//batch_size + 1}/{(len(user_ids) + batch_size - 1)//batch_size}")
            
            # Store all assignments in database
            await self._store_batch_user_cluster_assignments(results)
            
            self.users_clustered += len(results)
            self.logger.info(f"✅ Batch clustering completed for {len(results)} users")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch clustering: {e}")
            raise
    
    async def get_cluster_insights(self, cluster_id: int = None) -> List[ClusterAnalysis]:
        """
        Get insights about clusters
        
        Args:
            cluster_id: Specific cluster ID (optional)
            
        Returns:
            List of ClusterAnalysis objects
        """
        try:
            if not self.model_trained:
                await self.train_clustering_model()
            
            if cluster_id is not None:
                # Return specific cluster analysis
                cluster_analysis = next(
                    (ca for ca in self.cluster_analyses if ca.cluster_id == cluster_id),
                    None
                )
                return [cluster_analysis] if cluster_analysis else []
            
            # Return all cluster analyses
            return self.cluster_analyses
            
        except Exception as e:
            self.logger.error(f"❌ Error getting cluster insights: {e}")
            return []
    
    async def update_cluster_assignments(self) -> Dict[str, Any]:
        """
        Update cluster assignments for all users
        
        Returns:
            Dictionary with update statistics
        """
        try:
            self.logger.info("🔄 Updating cluster assignments for all users")
            
            # Get all users
            all_users = await self._get_all_users()
            
            if not all_users:
                return {"updated_users": 0, "total_users": 0}
            
            # Retrain model with latest data
            await self.train_clustering_model(force_retrain=True)
            
            # Batch assign all users
            updated_assignments = await self.batch_assign_users_to_clusters(all_users)
            
            # Calculate update statistics
            cluster_distribution = {}
            for assignment in updated_assignments:
                cluster_id = assignment.cluster_id
                cluster_distribution[cluster_id] = cluster_distribution.get(cluster_id, 0) + 1
            
            update_stats = {
                "updated_users": len(updated_assignments),
                "total_users": len(all_users),
                "cluster_distribution": cluster_distribution,
                "updated_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Updated cluster assignments for {len(updated_assignments)} users")
            return update_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error updating cluster assignments: {e}")
            return {"error": str(e)}
    
    async def _load_clustering_data(self) -> List[Dict[str, Any]]:
        """Load data for clustering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user profiles and aggregated interaction data
            query = """
                SELECT 
                    up.user_id,
                    up.industry,
                    up.location,
                    up.engagement_score,
                    up.interests,
                    up.preferences,
                    COUNT(ui.id) as interaction_count,
                    COUNT(DISTINCT ui.opportunity_id) as unique_opportunities,
                    AVG(ui.duration) as avg_duration,
                    COUNT(DISTINCT ui.interaction_type) as interaction_diversity,
                    SUM(CASE WHEN ui.interaction_type IN ('contact', 'save') THEN 1 ELSE 0 END) as conversions
                FROM user_profiles up
                LEFT JOIN user_interactions ui ON up.user_id = ui.user_id
                WHERE ui.timestamp > ? OR ui.timestamp IS NULL
                GROUP BY up.user_id
                HAVING interaction_count > 0 OR up.engagement_score > 0
            """
            
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute(query, (cutoff_date,))
            results = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            clustering_data = []
            for row in results:
                clustering_data.append({
                    'user_id': row[0],
                    'industry': row[1],
                    'location': row[2],
                    'engagement_score': row[3] or 0.0,
                    'interests': json.loads(row[4] or '[]'),
                    'preferences': json.loads(row[5] or '{}'),
                    'interaction_count': row[6],
                    'unique_opportunities': row[7],
                    'avg_duration': row[8] or 0.0,
                    'interaction_diversity': row[9],
                    'conversions': row[10]
                })
            
            self.logger.info(f"✅ Loaded {len(clustering_data)} users for clustering")
            return clustering_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading clustering data: {e}")
            raise
    
    async def _prepare_clustering_features(self, clustering_data: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for clustering"""
        try:
            features_list = []
            
            for user_data in clustering_data:
                # Calculate feature values
                engagement_score = user_data.get('engagement_score', 0.0)
                interaction_diversity = user_data.get('interaction_diversity', 0) / 5.0  # Normalize
                session_duration = min(user_data.get('avg_duration', 0) / 300.0, 1.0)  # Normalize
                network_activity = user_data.get('interaction_count', 0) / 100.0  # Normalize
                conversion_rate = user_data.get('conversions', 0) / max(user_data.get('interaction_count', 1), 1)
                
                # Content preferences (simplified)
                content_preferences_score = len(user_data.get('interests', [])) / 10.0
                
                # Timing patterns (simplified - would need more sophisticated analysis)
                timing_patterns_score = 0.5  # Placeholder
                
                features = [
                    engagement_score,
                    interaction_diversity,
                    session_duration,
                    content_preferences_score,
                    timing_patterns_score,
                    network_activity,
                    conversion_rate
                ]
                
                features_list.append(features)
            
            X = np.array(features_list)
            
            self.logger.info(f"✅ Prepared clustering features: {X.shape[0]} users, {X.shape[1]} features")
            return X
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing clustering features: {e}")
            raise
    
    async def _extract_user_features(self, user_id: str) -> ClusteringFeatures:
        """Extract features for a single user"""
        try:
            # Get user profile
            user_profile = await self._get_user_profile(user_id)
            
            # Get user interactions
            user_interactions = await self._get_user_interactions(user_id)
            
            # Calculate features
            engagement_score = user_profile.get('engagement_score', 0.0)
            
            # Interaction diversity
            interaction_types = set(i.get('interaction_type', '') for i in user_interactions)
            interaction_diversity = len(interaction_types) / 5.0  # Normalize
            
            # Session duration
            avg_duration = np.mean([i.get('duration', 0) for i in user_interactions if i.get('duration')])
            session_duration = min(avg_duration / 300.0, 1.0) if avg_duration else 0.0
            
            # Content preferences
            content_preferences = user_profile.get('interests', [])
            content_preferences_score = len(content_preferences) / 10.0
            
            # Timing patterns (simplified)
            timing_patterns = [9, 12, 15]  # Placeholder
            timing_patterns_score = 0.5  # Placeholder
            
            # Network activity
            network_activity = len(user_interactions) / 100.0
            
            # Conversion rate
            conversions = sum(1 for i in user_interactions if i.get('interaction_type') in ['contact', 'save'])
            conversion_rate = conversions / max(len(user_interactions), 1)
            
            # Create feature vector
            features_vector = np.array([
                engagement_score,
                interaction_diversity,
                session_duration,
                content_preferences_score,
                timing_patterns_score,
                network_activity,
                conversion_rate
            ])
            
            return ClusteringFeatures(
                user_id=user_id,
                engagement_score=engagement_score,
                interaction_diversity=interaction_diversity,
                session_duration=session_duration,
                content_preferences=content_preferences,
                timing_patterns=timing_patterns,
                network_activity=network_activity,
                conversion_rate=conversion_rate,
                features_vector=features_vector,
                feature_names=self.feature_names
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting user features: {e}")
            raise
    
    def _calculate_clustering_metrics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        algorithm: str,
        inertia: Optional[float],
        n_samples: int,
        training_time: float
    ) -> ClusteringMetrics:
        """Calculate clustering metrics"""
        try:
            # Filter out noise points for metrics calculation
            mask = labels != -1
            X_filtered = X[mask]
            labels_filtered = labels[mask]
            
            if len(X_filtered) == 0:
                raise ValueError("All points classified as noise")
            
            # Calculate metrics
            silhouette = silhouette_score(X_filtered, labels_filtered)
            calinski_harabasz = calinski_harabasz_score(X_filtered, labels_filtered)
            davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
            
            # Number of clusters
            n_clusters = len(np.unique(labels_filtered))
            
            return ClusteringMetrics(
                algorithm=algorithm,
                n_clusters=n_clusters,
                silhouette_score=silhouette,
                calinski_harabasz_score=calinski_harabasz,
                davies_bouldin_score=davies_bouldin,
                inertia=inertia,
                training_samples=n_samples,
                clustering_time=training_time,
                model_version=self.model_version
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating clustering metrics: {e}")
            raise
    
    def _validate_clustering_quality(self, metrics: ClusteringMetrics) -> bool:
        """Validate clustering quality"""
        config = self.ml_config.clustering_model
        
        return (
            metrics.silhouette_score >= config.min_silhouette_score and
            metrics.calinski_harabasz_score >= config.min_calinski_harabasz_score and
            metrics.n_clusters >= 2
        )
    
    async def _analyze_clusters(
        self,
        clustering_data: List[Dict[str, Any]],
        labels: np.ndarray,
        X: np.ndarray
    ) -> List[ClusterAnalysis]:
        """Analyze clusters and generate insights"""
        try:
            cluster_analyses = []
            unique_labels = np.unique(labels)
            
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise points
                    continue
                
                # Get cluster mask
                cluster_mask = labels == cluster_id
                cluster_data = [clustering_data[i] for i in range(len(clustering_data)) if cluster_mask[i]]
                cluster_points = X[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate cluster characteristics
                characteristics = self._calculate_cluster_characteristics(cluster_data)
                
                # Get center point
                center_point = np.mean(cluster_points, axis=0)
                
                # Calculate variance
                variance = np.mean(np.var(cluster_points, axis=0))
                
                # Calculate cluster silhouette score
                if len(cluster_points) > 1:
                    cluster_silhouette = silhouette_score(X, labels, metric='euclidean')
                else:
                    cluster_silhouette = 0.0
                
                # Get representative users
                representative_users = self._get_representative_users(cluster_data, cluster_points, center_point)
                
                # Identify key features
                key_features = self._identify_key_features(cluster_points, center_point)
                
                # Generate cluster label
                cluster_label = self._generate_cluster_label(characteristics)
                
                cluster_analysis = ClusterAnalysis(
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    user_count=len(cluster_data),
                    characteristics=characteristics,
                    center_point=center_point,
                    variance=variance,
                    silhouette_score=cluster_silhouette,
                    representative_users=representative_users,
                    key_features=key_features
                )
                
                cluster_analyses.append(cluster_analysis)
            
            return cluster_analyses
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing clusters: {e}")
            return []
    
    def _calculate_cluster_characteristics(self, cluster_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate cluster characteristics"""
        try:
            if not cluster_data:
                return {}
            
            # Calculate averages
            avg_engagement = np.mean([user.get('engagement_score', 0) for user in cluster_data])
            avg_interactions = np.mean([user.get('interaction_count', 0) for user in cluster_data])
            avg_conversions = np.mean([user.get('conversions', 0) for user in cluster_data])
            
            # Most common industry
            industries = [user.get('industry', '') for user in cluster_data if user.get('industry')]
            most_common_industry = max(set(industries), key=industries.count) if industries else 'Unknown'
            
            # Most common interests
            all_interests = []
            for user in cluster_data:
                all_interests.extend(user.get('interests', []))
            
            if all_interests:
                interest_counts = {}
                for interest in all_interests:
                    interest_counts[interest] = interest_counts.get(interest, 0) + 1
                
                top_interests = sorted(interest_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                top_interests = [interest for interest, count in top_interests]
            else:
                top_interests = []
            
            return {
                'avg_engagement_score': avg_engagement,
                'avg_interactions': avg_interactions,
                'avg_conversions': avg_conversions,
                'most_common_industry': most_common_industry,
                'top_interests': top_interests,
                'user_count': len(cluster_data)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating cluster characteristics: {e}")
            return {}
    
    def _get_representative_users(
        self,
        cluster_data: List[Dict[str, Any]],
        cluster_points: np.ndarray,
        center_point: np.ndarray
    ) -> List[str]:
        """Get representative users for cluster"""
        try:
            if len(cluster_data) == 0:
                return []
            
            # Calculate distances to center
            distances = np.linalg.norm(cluster_points - center_point, axis=1)
            
            # Get indices of closest users
            closest_indices = np.argsort(distances)[:3]
            
            # Get user IDs
            representative_users = [cluster_data[i]['user_id'] for i in closest_indices]
            
            return representative_users
            
        except Exception as e:
            self.logger.error(f"❌ Error getting representative users: {e}")
            return []
    
    def _identify_key_features(self, cluster_points: np.ndarray, center_point: np.ndarray) -> List[str]:
        """Identify key features for cluster"""
        try:
            if len(cluster_points) == 0:
                return []
            
            # Calculate feature importance based on deviation from global mean
            global_mean = np.mean(cluster_points, axis=0)
            feature_importance = np.abs(center_point - global_mean)
            
            # Get top features
            top_feature_indices = np.argsort(feature_importance)[-3:]
            
            key_features = [self.feature_names[i] for i in top_feature_indices if i < len(self.feature_names)]
            
            return key_features
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying key features: {e}")
            return []
    
    def _generate_cluster_label(self, characteristics: Dict[str, Any]) -> str:
        """Generate human-readable cluster label"""
        try:
            avg_engagement = characteristics.get('avg_engagement_score', 0.0)
            avg_interactions = characteristics.get('avg_interactions', 0)
            avg_conversions = characteristics.get('avg_conversions', 0)
            
            # Simple rule-based labeling
            if avg_engagement > 0.8 and avg_conversions > 5:
                return "High Value Users"
            elif avg_engagement > 0.6 and avg_interactions > 50:
                return "Engaged Explorers"
            elif avg_conversions > 3:
                return "Conversion Focused"
            elif avg_interactions > 30:
                return "Active Users"
            elif avg_engagement < 0.3:
                return "Low Engagement"
            else:
                return "Moderate Users"
                
        except Exception as e:
            self.logger.error(f"❌ Error generating cluster label: {e}")
            return "Unknown Cluster"
    
    def _generate_cluster_labels(self, cluster_analyses: List[ClusterAnalysis]) -> Dict[int, str]:
        """Generate labels for all clusters"""
        labels = {}
        for analysis in cluster_analyses:
            labels[analysis.cluster_id] = analysis.cluster_label
        return labels
    
    def _calculate_cluster_probability(self, features: np.ndarray, cluster_id: int) -> float:
        """Calculate probability of cluster assignment"""
        try:
            if self.current_algorithm == 'kmeans':
                # Use distance to centroid as probability measure
                distances = np.linalg.norm(features - self.current_model.cluster_centers_, axis=1)
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                
                if max_distance > min_distance:
                    probability = 1.0 - (distances[cluster_id] - min_distance) / (max_distance - min_distance)
                else:
                    probability = 1.0
                
                return max(0.0, min(1.0, probability))
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating cluster probability: {e}")
            return 0.5
    
    def _calculate_distance_to_centroid(self, features: np.ndarray, cluster_id: int) -> float:
        """Calculate distance to cluster centroid"""
        try:
            if self.current_algorithm == 'kmeans':
                centroid = self.current_model.cluster_centers_[cluster_id]
                distance = np.linalg.norm(features - centroid)
                return float(distance)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating distance to centroid: {e}")
            return 0.0
    
    def _get_cluster_characteristics(self, cluster_id: int) -> Dict[str, Any]:
        """Get characteristics for a specific cluster"""
        try:
            cluster_analysis = next(
                (ca for ca in self.cluster_analyses if ca.cluster_id == cluster_id),
                None
            )
            
            if cluster_analysis:
                return cluster_analysis.characteristics
            
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting cluster characteristics: {e}")
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
    
    async def _get_all_users(self) -> List[str]:
        """Get all user IDs from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT user_id FROM user_profiles")
            results = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting all users: {e}")
            return []
    
    # Model persistence methods
    def _model_exists(self, algorithm: str) -> bool:
        """Check if model file exists"""
        model_path = os.path.join(
            self.ml_config.training.models_path,
            f"clustering_model_{algorithm}_{self.model_version}.pkl"
        )
        return os.path.exists(model_path)
    
    def _model_is_recent(self, max_age_hours: int = 24) -> bool:
        """Check if model is recent enough"""
        if not self.last_training_time:
            return False
        
        age_hours = (datetime.now() - self.last_training_time).total_seconds() / 3600
        return age_hours < max_age_hours
    
    async def _save_clustering_model(self, algorithm: str, metrics: ClusteringMetrics):
        """Save clustering model to disk"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(self.ml_config.training.models_path, exist_ok=True)
            
            # Save model
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"clustering_model_{algorithm}_{self.model_version}.pkl"
            )
            
            model_data = {
                'model': self.current_model,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_names': self.feature_names,
                'cluster_analyses': self.cluster_analyses,
                'cluster_labels': self.cluster_labels,
                'metrics': metrics,
                'algorithm': algorithm,
                'version': self.model_version,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"✅ Clustering model saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving clustering model: {e}")
            raise
    
    def _load_model_metrics(self, algorithm: str) -> ClusteringMetrics:
        """Load model metrics from disk"""
        try:
            model_path = os.path.join(
                self.ml_config.training.models_path,
                f"clustering_model_{algorithm}_{self.model_version}.pkl"
            )
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            return model_data['metrics']
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model metrics: {e}")
            return ClusteringMetrics(
                algorithm=algorithm,
                n_clusters=0,
                silhouette_score=0.0,
                calinski_harabasz_score=0.0,
                davies_bouldin_score=0.0,
                inertia=None,
                training_samples=0,
                clustering_time=0.0,
                model_version=self.model_version
            )
    
    # Database operations
    async def _store_user_cluster_assignment(self, user_cluster: UserCluster):
        """Store user cluster assignment in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create user_clusters table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_clusters (
                    user_id TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_label TEXT,
                    cluster_probability REAL,
                    distance_to_centroid REAL,
                    cluster_characteristics TEXT,
                    assigned_at TEXT,
                    model_version TEXT
                )
            """)
            
            # Insert or update user cluster assignment
            cursor.execute("""
                INSERT OR REPLACE INTO user_clusters 
                (user_id, cluster_id, cluster_label, cluster_probability, 
                 distance_to_centroid, cluster_characteristics, assigned_at, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_cluster.user_id,
                user_cluster.cluster_id,
                user_cluster.cluster_label,
                user_cluster.cluster_probability,
                user_cluster.distance_to_centroid,
                json.dumps(user_cluster.cluster_characteristics),
                user_cluster.assigned_at.isoformat(),
                self.model_version
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing user cluster assignment: {e}")
    
    async def _store_batch_user_cluster_assignments(self, user_clusters: List[UserCluster]):
        """Store batch user cluster assignments in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create user_clusters table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_clusters (
                    user_id TEXT PRIMARY KEY,
                    cluster_id INTEGER,
                    cluster_label TEXT,
                    cluster_probability REAL,
                    distance_to_centroid REAL,
                    cluster_characteristics TEXT,
                    assigned_at TEXT,
                    model_version TEXT
                )
            """)
            
            # Prepare batch data
            batch_data = []
            for user_cluster in user_clusters:
                batch_data.append((
                    user_cluster.user_id,
                    user_cluster.cluster_id,
                    user_cluster.cluster_label,
                    user_cluster.cluster_probability,
                    user_cluster.distance_to_centroid,
                    json.dumps(user_cluster.cluster_characteristics),
                    user_cluster.assigned_at.isoformat(),
                    self.model_version
                ))
            
            # Insert batch data
            cursor.executemany("""
                INSERT OR REPLACE INTO user_clusters 
                (user_id, cluster_id, cluster_label, cluster_probability, 
                 distance_to_centroid, cluster_characteristics, assigned_at, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error storing batch user cluster assignments: {e}")
    
    # Status and management methods
    def get_model_status(self) -> Dict[str, Any]:
        """Get clustering model status"""
        return {
            "status": "operational" if self.model_trained else "not_trained",
            "model_version": self.model_version,
            "current_algorithm": self.current_algorithm,
            "users_clustered": self.users_clustered,
            "model_trained": self.model_trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "current_metrics": self.current_metrics.__dict__ if self.current_metrics else None,
            "cluster_count": len(self.cluster_analyses),
            "cluster_labels": self.cluster_labels,
            "available_algorithms": list(self.models.keys()),
            "feature_names": self.feature_names,
            "last_updated": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test the clustering model
    async def test_clustering_model():
        print("👥 Testing Clustering Model")
        print("=" * 50)
        
        try:
            model = ClusteringModel()
            
            # Test model training
            print("Training clustering model...")
            metrics = await model.train_clustering_model(algorithm='kmeans', force_retrain=True)
            print(f"Training Metrics: {metrics}")
            
            # Test user assignment
            print("Assigning user to cluster...")
            user_cluster = await model.assign_user_to_cluster("test_user_123")
            print(f"User Cluster: {user_cluster}")
            
            # Test batch assignment
            print("Batch assigning users...")
            batch_results = await model.batch_assign_users_to_clusters(
                ["test_user_123", "test_user_456"]
            )
            print(f"Batch Results: {len(batch_results)} assignments")
            
            # Test cluster insights
            print("Getting cluster insights...")
            insights = await model.get_cluster_insights()
            print(f"Cluster Insights: {len(insights)} clusters")
            
            # Test status
            status = model.get_model_status()
            print(f"Model Status: {status}")
            
            print("✅ Clustering Model test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run test
    import asyncio
    asyncio.run(test_clustering_model())