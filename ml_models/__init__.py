#!/usr/bin/env python3
"""
ML Models Package
Machine Learning models for the Business Dealer Intelligence System

This package contains:
- ScoringModel: ML-based opportunity scoring
- ClusteringModel: User segmentation and clustering
- PredictionModel: Behavior and outcome prediction
- EmbeddingManager: Centralized embedding management

All models support:
- Multiple algorithms and model types
- Batch processing for efficiency
- Model persistence and versioning
- Performance monitoring and metrics
- Caching for improved performance
"""

from .scoring_model import ScoringModel, ScoringFeatures, ScoringResult, ModelMetrics
from .clustering_model import ClusteringModel, ClusteringFeatures, UserCluster, ClusterAnalysis, ClusteringMetrics
from .prediction_model import PredictionModel, PredictionFeatures, PredictionResult, PredictionMetrics
from .embedding_manager import EmbeddingManager, EmbeddingRequest, EmbeddingResult, SimilarityResult, EmbeddingStats

__all__ = [
    # Scoring Model
    'ScoringModel',
    'ScoringFeatures', 
    'ScoringResult',
    'ModelMetrics',
    
    # Clustering Model
    'ClusteringModel',
    'ClusteringFeatures',
    'UserCluster',
    'ClusterAnalysis', 
    'ClusteringMetrics',
    
    # Prediction Model
    'PredictionModel',
    'PredictionFeatures',
    'PredictionResult',
    'PredictionMetrics',
    
    # Embedding Manager
    'EmbeddingManager',
    'EmbeddingRequest',
    'EmbeddingResult',
    'SimilarityResult',
    'EmbeddingStats'
]