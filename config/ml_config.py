#!/usr/bin/env python3
"""
Machine Learning Configuration
Configuration for ML models, training, and prediction systems in the Business Dealer Intelligence System

Environment Variables:
- ML_MODELS_PATH: Path to store trained models
- ML_TRAINING_DATA_PATH: Path to training data
- ML_ENABLE_GPU: Enable GPU acceleration if available
- ML_RANDOM_SEED: Random seed for reproducibility
- ML_VALIDATION_SPLIT: Validation split ratio
- ML_BATCH_SIZE: Batch size for training
"""

import os
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class ModelType(Enum):
    """Supported ML model types"""

    SCORING = "scoring"
    CLUSTERING = "clustering"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TrainingMode(Enum):
    """Training modes"""

    BATCH = "batch"
    INCREMENTAL = "incremental"
    ONLINE = "online"


@dataclass
class ScoringModelConfig:
    """Configuration for opportunity scoring models"""

    model_type: str = field(default="random_forest")
    n_estimators: int = field(default=100)
    max_depth: int = field(default=10)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=1)
    max_features: str = field(default="sqrt")
    random_state: int = field(default=42)

    # Feature engineering
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "user_industry_match",
            "location_proximity",
            "interaction_frequency",
            "historical_success_rate",
            "network_strength",
            "timing_score",
            "content_similarity",
            "behavioral_patterns",
        ]
    )

    # Training parameters
    training_data_window_days: int = field(default=90)
    min_training_samples: int = field(default=1000)
    validation_split: float = field(default=0.2)
    class_weight: str = field(default="balanced")

    # Performance thresholds
    min_accuracy: float = field(default=0.75)
    min_precision: float = field(default=0.70)
    min_recall: float = field(default=0.70)
    min_f1_score: float = field(default=0.70)

    @classmethod
    def from_env(cls) -> "ScoringModelConfig":
        """Create scoring model config from environment variables"""
        return cls(
            model_type=os.getenv("SCORING_MODEL_TYPE", "random_forest"),
            n_estimators=int(os.getenv("SCORING_N_ESTIMATORS", "100")),
            max_depth=int(os.getenv("SCORING_MAX_DEPTH", "10")),
            min_samples_split=int(os.getenv("SCORING_MIN_SAMPLES_SPLIT", "2")),
            min_samples_leaf=int(os.getenv("SCORING_MIN_SAMPLES_LEAF", "1")),
            max_features=os.getenv("SCORING_MAX_FEATURES", "sqrt"),
            random_state=int(os.getenv("ML_RANDOM_SEED", "42")),
            training_data_window_days=int(
                os.getenv("SCORING_TRAINING_WINDOW_DAYS", "90")
            ),
            min_training_samples=int(os.getenv("SCORING_MIN_TRAINING_SAMPLES", "1000")),
            validation_split=float(os.getenv("ML_VALIDATION_SPLIT", "0.2")),
            class_weight=os.getenv("SCORING_CLASS_WEIGHT", "balanced"),
            min_accuracy=float(os.getenv("SCORING_MIN_ACCURACY", "0.75")),
            min_precision=float(os.getenv("SCORING_MIN_PRECISION", "0.70")),
            min_recall=float(os.getenv("SCORING_MIN_RECALL", "0.70")),
            min_f1_score=float(os.getenv("SCORING_MIN_F1_SCORE", "0.70")),
        )


@dataclass
class ClusteringModelConfig:
    """Configuration for clustering models"""

    model_type: str = field(default="kmeans")
    n_clusters: int = field(default=8)
    init: str = field(default="k-means++")
    n_init: int = field(default=10)
    max_iter: int = field(default=300)
    tol: float = field(default=1e-4)
    random_state: int = field(default=42)

    # Alternative clustering models
    dbscan_eps: float = field(default=0.5)
    dbscan_min_samples: int = field(default=5)

    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = field(default=10)
    hdbscan_min_samples: int = field(default=5)
    hdbscan_cluster_selection_epsilon: float = field(default=0.0)

    # Feature selection
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "engagement_score",
            "interaction_diversity",
            "session_duration",
            "content_preferences",
            "timing_patterns",
            "network_activity",
            "conversion_rate",
        ]
    )

    # Data preprocessing
    normalize_features: bool = field(default=True)
    remove_outliers: bool = field(default=True)
    outlier_threshold: float = field(default=3.0)

    # Model evaluation
    min_silhouette_score: float = field(default=0.5)
    min_calinski_harabasz_score: float = field(default=100.0)

    @classmethod
    def from_env(cls) -> "ClusteringModelConfig":
        """Create clustering model config from environment variables"""
        return cls(
            model_type=os.getenv("CLUSTERING_MODEL_TYPE", "kmeans"),
            n_clusters=int(os.getenv("CLUSTERING_N_CLUSTERS", "8")),
            init=os.getenv("CLUSTERING_INIT", "k-means++"),
            n_init=int(os.getenv("CLUSTERING_N_INIT", "10")),
            max_iter=int(os.getenv("CLUSTERING_MAX_ITER", "300")),
            tol=float(os.getenv("CLUSTERING_TOL", "1e-4")),
            random_state=int(os.getenv("ML_RANDOM_SEED", "42")),
            dbscan_eps=float(os.getenv("CLUSTERING_DBSCAN_EPS", "0.5")),
            dbscan_min_samples=int(os.getenv("CLUSTERING_DBSCAN_MIN_SAMPLES", "5")),
            hdbscan_min_cluster_size=int(
                os.getenv("CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE", "10")
            ),
            hdbscan_min_samples=int(os.getenv("CLUSTERING_HDBSCAN_MIN_SAMPLES", "5")),
            hdbscan_cluster_selection_epsilon=float(
                os.getenv("CLUSTERING_HDBSCAN_EPSILON", "0.0")
            ),
            normalize_features=os.getenv(
                "CLUSTERING_NORMALIZE_FEATURES", "true"
            ).lower()
            == "true",
            remove_outliers=os.getenv("CLUSTERING_REMOVE_OUTLIERS", "true").lower()
            == "true",
            outlier_threshold=float(os.getenv("CLUSTERING_OUTLIER_THRESHOLD", "3.0")),
            min_silhouette_score=float(
                os.getenv("CLUSTERING_MIN_SILHOUETTE_SCORE", "0.5")
            ),
            min_calinski_harabasz_score=float(
                os.getenv("CLUSTERING_MIN_CALINSKI_HARABASZ_SCORE", "100.0")
            ),
        )


@dataclass
class PredictionModelConfig:
    """Configuration for prediction models"""

    model_type: str = field(default="gradient_boosting")
    n_estimators: int = field(default=100)
    learning_rate: float = field(default=0.1)
    max_depth: int = field(default=3)
    min_samples_split: int = field(default=2)
    min_samples_leaf: int = field(default=1)
    subsample: float = field(default=1.0)
    random_state: int = field(default=42)

    # Neural network parameters (if using neural networks)
    hidden_layer_sizes: Tuple[int, ...] = field(default=(100, 50))
    activation: str = field(default="relu")
    solver: str = field(default="adam")
    alpha: float = field(default=0.0001)
    max_iter: int = field(default=200)

    # Training parameters
    early_stopping: bool = field(default=True)
    validation_fraction: float = field(default=0.1)
    n_iter_no_change: int = field(default=10)

    # Feature engineering
    feature_columns: List[str] = field(
        default_factory=lambda: [
            "user_behavior_score",
            "opportunity_quality",
            "market_conditions",
            "seasonal_factors",
            "competitive_landscape",
            "user_network_strength",
            "historical_performance",
        ]
    )

    # Performance thresholds
    min_accuracy: float = field(default=0.75)
    min_mae: float = field(default=0.1)  # Mean Absolute Error
    min_r2_score: float = field(default=0.6)  # R² Score

    @classmethod
    def from_env(cls) -> "PredictionModelConfig":
        """Create prediction model config from environment variables"""
        return cls(
            model_type=os.getenv("PREDICTION_MODEL_TYPE", "gradient_boosting"),
            n_estimators=int(os.getenv("PREDICTION_N_ESTIMATORS", "100")),
            learning_rate=float(os.getenv("PREDICTION_LEARNING_RATE", "0.1")),
            max_depth=int(os.getenv("PREDICTION_MAX_DEPTH", "3")),
            min_samples_split=int(os.getenv("PREDICTION_MIN_SAMPLES_SPLIT", "2")),
            min_samples_leaf=int(os.getenv("PREDICTION_MIN_SAMPLES_LEAF", "1")),
            subsample=float(os.getenv("PREDICTION_SUBSAMPLE", "1.0")),
            random_state=int(os.getenv("ML_RANDOM_SEED", "42")),
            activation=os.getenv("PREDICTION_ACTIVATION", "relu"),
            solver=os.getenv("PREDICTION_SOLVER", "adam"),
            alpha=float(os.getenv("PREDICTION_ALPHA", "0.0001")),
            max_iter=int(os.getenv("PREDICTION_MAX_ITER", "200")),
            early_stopping=os.getenv("PREDICTION_EARLY_STOPPING", "true").lower()
            == "true",
            validation_fraction=float(
                os.getenv("PREDICTION_VALIDATION_FRACTION", "0.1")
            ),
            n_iter_no_change=int(os.getenv("PREDICTION_N_ITER_NO_CHANGE", "10")),
            min_accuracy=float(os.getenv("PREDICTION_MIN_ACCURACY", "0.75")),
            min_mae=float(os.getenv("PREDICTION_MIN_MAE", "0.1")),
            min_r2_score=float(os.getenv("PREDICTION_MIN_R2_SCORE", "0.6")),
        )


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models"""

    cache_embeddings: bool = field(default=True)
    embedding_dimension: int = field(default=3072)
    batch_size: int = field(default=100)
    max_text_length: int = field(default=8192)

    # Similarity thresholds
    similarity_threshold: float = field(default=0.7)
    min_similarity_for_match: float = field(default=0.5)

    # Vector search parameters
    vector_index_type: str = field(default="faiss")
    search_k: int = field(default=100)
    ef_construction: int = field(default=200)
    ef_search: int = field(default=50)

    @classmethod
    def from_env(cls) -> "EmbeddingModelConfig":
        """Create embedding model config from environment variables"""
        return cls(
            cache_embeddings=os.getenv("EMBEDDING_CACHE_EMBEDDINGS", "true").lower()
            == "true",
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "3072")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            max_text_length=int(os.getenv("EMBEDDING_MAX_TEXT_LENGTH", "8192")),
            similarity_threshold=float(
                os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.7")
            ),
            min_similarity_for_match=float(
                os.getenv("EMBEDDING_MIN_SIMILARITY_FOR_MATCH", "0.5")
            ),
            vector_index_type=os.getenv("EMBEDDING_VECTOR_INDEX_TYPE", "faiss"),
            search_k=int(os.getenv("EMBEDDING_SEARCH_K", "100")),
            ef_construction=int(os.getenv("EMBEDDING_EF_CONSTRUCTION", "200")),
            ef_search=int(os.getenv("EMBEDDING_EF_SEARCH", "50")),
        )


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    models_path: str = field(default="models/")
    training_data_path: str = field(default="data/training/")
    validation_data_path: str = field(default="data/validation/")

    # Training schedule
    training_mode: TrainingMode = field(default=TrainingMode.BATCH)
    batch_training_schedule: str = field(default="daily")  # daily, weekly, monthly
    incremental_training_interval_hours: int = field(default=6)

    # Data requirements
    min_training_samples: int = field(default=1000)
    max_training_samples: int = field(default=100000)
    data_freshness_days: int = field(default=30)

    # Model versioning
    enable_model_versioning: bool = field(default=True)
    max_model_versions: int = field(default=10)
    model_performance_threshold: float = field(
        default=0.05
    )  # 5% improvement to replace model

    # Monitoring
    enable_performance_monitoring: bool = field(default=True)
    performance_check_interval_hours: int = field(default=24)
    alert_on_performance_degradation: bool = field(default=True)
    performance_degradation_threshold: float = field(default=0.1)  # 10% degradation

    # Hardware/Performance
    enable_gpu: bool = field(default=False)
    max_memory_usage_gb: int = field(default=4)
    n_jobs: int = field(default=-1)  # Use all available cores

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Create training config from environment variables"""
        return cls(
            models_path=os.getenv("ML_MODELS_PATH", "models/"),
            training_data_path=os.getenv("ML_TRAINING_DATA_PATH", "data/training/"),
            validation_data_path=os.getenv(
                "ML_VALIDATION_DATA_PATH", "data/validation/"
            ),
            training_mode=TrainingMode(os.getenv("ML_TRAINING_MODE", "batch")),
            batch_training_schedule=os.getenv("ML_BATCH_TRAINING_SCHEDULE", "daily"),
            incremental_training_interval_hours=int(
                os.getenv("ML_INCREMENTAL_TRAINING_INTERVAL", "6")
            ),
            min_training_samples=int(os.getenv("ML_MIN_TRAINING_SAMPLES", "1000")),
            max_training_samples=int(os.getenv("ML_MAX_TRAINING_SAMPLES", "100000")),
            data_freshness_days=int(os.getenv("ML_DATA_FRESHNESS_DAYS", "30")),
            enable_model_versioning=os.getenv(
                "ML_ENABLE_MODEL_VERSIONING", "true"
            ).lower()
            == "true",
            max_model_versions=int(os.getenv("ML_MAX_MODEL_VERSIONS", "10")),
            model_performance_threshold=float(
                os.getenv("ML_MODEL_PERFORMANCE_THRESHOLD", "0.05")
            ),
            enable_performance_monitoring=os.getenv(
                "ML_ENABLE_PERFORMANCE_MONITORING", "true"
            ).lower()
            == "true",
            performance_check_interval_hours=int(
                os.getenv("ML_PERFORMANCE_CHECK_INTERVAL", "24")
            ),
            alert_on_performance_degradation=os.getenv(
                "ML_ALERT_ON_PERFORMANCE_DEGRADATION", "true"
            ).lower()
            == "true",
            performance_degradation_threshold=float(
                os.getenv("ML_PERFORMANCE_DEGRADATION_THRESHOLD", "0.1")
            ),
            enable_gpu=os.getenv("ML_ENABLE_GPU", "false").lower() == "true",
            max_memory_usage_gb=int(os.getenv("ML_MAX_MEMORY_USAGE_GB", "4")),
            n_jobs=int(os.getenv("ML_N_JOBS", "-1")),
        )


@dataclass
class MLConfig:
    """Main ML configuration"""

    random_seed: int = field(default=42)
    reproducible_training: bool = field(default=True)

    # Sub-configurations
    scoring_model: ScoringModelConfig = field(default_factory=ScoringModelConfig)
    clustering_model: ClusteringModelConfig = field(
        default_factory=ClusteringModelConfig
    )
    prediction_model: PredictionModelConfig = field(
        default_factory=PredictionModelConfig
    )
    embedding_model: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_env(cls) -> "MLConfig":
        """Create complete ML config from environment variables"""
        return cls(
            random_seed=int(os.getenv("ML_RANDOM_SEED", "42")),
            reproducible_training=os.getenv("ML_REPRODUCIBLE_TRAINING", "true").lower()
            == "true",
            scoring_model=ScoringModelConfig.from_env(),
            clustering_model=ClusteringModelConfig.from_env(),
            prediction_model=PredictionModelConfig.from_env(),
            embedding_model=EmbeddingModelConfig.from_env(),
            training=TrainingConfig.from_env(),
        )

    def setup_reproducibility(self) -> None:
        """Setup reproducible training environment"""
        if self.reproducible_training:
            # Set random seeds
            np.random.seed(self.random_seed)
            os.environ["PYTHONHASHSEED"] = str(self.random_seed)

            # Set sklearn random state
            import sklearn

            sklearn.utils.check_random_state(self.random_seed)

            print(f"✅ Reproducibility setup with seed: {self.random_seed}")

    def validate(self) -> None:
        """Validate ML configuration"""
        if not os.path.exists(self.training.models_path):
            os.makedirs(self.training.models_path, exist_ok=True)
            print(f"📁 Created models directory: {self.training.models_path}")

        if not os.path.exists(self.training.training_data_path):
            os.makedirs(self.training.training_data_path, exist_ok=True)
            print(
                f"📁 Created training data directory: {self.training.training_data_path}"
            )

        if self.scoring_model.min_accuracy < 0 or self.scoring_model.min_accuracy > 1:
            raise ValueError("Scoring model minimum accuracy must be between 0 and 1")

        if self.clustering_model.n_clusters < 2:
            raise ValueError("Number of clusters must be at least 2")

        if self.prediction_model.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if (
            self.embedding_model.similarity_threshold < 0
            or self.embedding_model.similarity_threshold > 1
        ):
            raise ValueError("Similarity threshold must be between 0 and 1")

        print("✅ ML configuration validated successfully")


# Global ML configuration instance
ml_config = MLConfig.from_env()


def get_ml_config() -> MLConfig:
    """Get the global ML configuration instance"""
    return ml_config


def reload_ml_config() -> MLConfig:
    """Reload ML configuration from environment variables"""
    global ml_config
    ml_config = MLConfig.from_env()
    return ml_config


if __name__ == "__main__":
    # Test ML configuration
    print("🤖 Machine Learning Configuration Test")
    print("=" * 50)

    try:
        config = MLConfig.from_env()
        config.setup_reproducibility()
        config.validate()

        print(f"Random Seed: {config.random_seed}")
        print(f"Reproducible Training: {config.reproducible_training}")
        print(f"Models Path: {config.training.models_path}")
        print(f"Training Data Path: {config.training.training_data_path}")
        print(f"Scoring Model: {config.scoring_model.model_type}")
        print(f"Clustering Model: {config.clustering_model.model_type}")
        print(f"Prediction Model: {config.prediction_model.model_type}")
        print(f"Embedding Dimension: {config.embedding_model.embedding_dimension}")
        print(f"GPU Enabled: {config.training.enable_gpu}")
        print(f"Training Mode: {config.training.training_mode.value}")

        print("\n✅ ML Configuration loaded successfully!")

    except Exception as e:
        print(f"❌ ML Configuration error: {e}")
        exit(1)
