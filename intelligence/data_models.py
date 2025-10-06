#!/usr/bin/env python3
"""
Core Data Models for Business Dealer Intelligence System
Pydantic models for data validation and serialization throughout the system

Following the data models specification from the PRP implementation blueprint.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator as validator, field_validator
#from pydantic import BaseModel, Field, model_validator, field_validator

from uuid import uuid4


class OpportunityType(str, Enum):
    """Types of business opportunities"""

    BUYER = "buyer"
    SELLER = "seller"
    SERVICE = "service"
    PARTNERSHIP = "partnership"


class InteractionType(str, Enum):
    """Types of user interactions with opportunities"""

    VIEW = "view"
    CLICK = "click"
    SHARE = "share"
    CONTACT = "contact"
    SAVE = "save"


class RecommendationType(str, Enum):
    """Types of recommendations"""

    PERSONALIZED = "personalized"
    TRENDING = "trending"
    SIMILAR_USERS = "similar_users"
    LOCATION_BASED = "location_based"
    INDUSTRY_MATCH = "industry_match"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""

    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"
    SMS = "sms"


class ModelType(str, Enum):
    """ML Model types"""

    SCORING = "scoring"
    CLUSTERING = "clustering"
    PREDICTION = "prediction"
    EMBEDDING = "embedding"


class UserInteraction(BaseModel):
    """User interaction with opportunities"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique interaction ID"
    )
    user_id: str = Field(..., description="ID of the user")
    opportunity_id: str = Field(..., description="ID of the opportunity")
    interaction_type: InteractionType = Field(..., description="Type of interaction")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the interaction occurred"
    )
    duration: Optional[int] = Field(None, ge=0, description="Duration in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("duration")
    def validate_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError("Duration must be non-negative")
        return v

    @field_validator("metadata")
    def validate_metadata(cls, v):
        if v is None:
            return {}
        # Ensure all keys are strings and values are JSON serializable
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("Metadata keys must be strings")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class OpportunityScore(BaseModel):
    """Opportunity relevance and success scoring"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique score ID"
    )
    opportunity_id: str = Field(..., description="ID of the opportunity")
    user_id: str = Field(..., description="ID of the user")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0-1)"
    )
    success_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Success probability (0-1)"
    )
    factors: Dict[str, float] = Field(
        default_factory=dict, description="Factors that influenced the score"
    )
    calculated_at: datetime = Field(
        default_factory=datetime.now, description="When the score was calculated"
    )
    expires_at: Optional[datetime] = Field(None, description="When the score expires")

    @field_validator("factors")
    def validate_factors(cls, v):
        if v is None:
            return {}
        # Ensure all factor values are between 0 and 1
        for factor, value in v.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError(f"Factor {factor} must be a number between 0 and 1")
        return v

    @field_validator("expiry")
    def validate_expiry(cls, values):
        calculated_at = values.get("calculated_at")
        expires_at = values.get("expires_at")

        if expires_at and calculated_at and expires_at <= calculated_at:
            raise ValueError("expires_at must be after calculated_at")

        return values

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class Recommendation(BaseModel):
    """Generated recommendation for a user"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique recommendation ID"
    )
    user_id: str = Field(..., description="ID of the user")
    opportunity_id: str = Field(..., description="ID of the opportunity")
    recommendation_type: RecommendationType = Field(
        ..., description="Type of recommendation"
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score (0-1)")
    reasoning: str = Field(..., description="Explanation for the recommendation")
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the recommendation was generated",
    )
    clicked_at: Optional[datetime] = Field(
        None, description="When the recommendation was clicked"
    )
    dismissed_at: Optional[datetime] = Field(
        None, description="When the recommendation was dismissed"
    )
    expires_at: Optional[datetime] = Field(
        None, description="When the recommendation expires"
    )

    @validator("reasoning")
    def validate_reasoning(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters long")
        return v

    @root_validator
    def validate_interaction_times(cls, values):
        generated_at = values.get("generated_at")
        clicked_at = values.get("clicked_at")
        dismissed_at = values.get("dismissed_at")
        expires_at = values.get("expires_at")

        if clicked_at and generated_at and clicked_at < generated_at:
            raise ValueError("clicked_at must be after generated_at")

        if dismissed_at and generated_at and dismissed_at < generated_at:
            raise ValueError("dismissed_at must be after generated_at")

        if expires_at and generated_at and expires_at <= generated_at:
            raise ValueError("expires_at must be after generated_at")

        return values

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class BehaviorMetrics(BaseModel):
    """User behavior metrics"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique metrics ID"
    )
    user_id: str = Field(..., description="ID of the user")
    metric_type: str = Field(..., description="Type of metric")
    value: float = Field(..., description="Metric value")
    period: str = Field(..., description="Period (daily, weekly, monthly)")
    period_start: datetime = Field(..., description="Start of the period")
    period_end: datetime = Field(..., description="End of the period")
    calculated_at: datetime = Field(
        default_factory=datetime.now, description="When the metric was calculated"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @validator("period")
    def validate_period(cls, v):
        allowed_periods = ["daily", "weekly", "monthly", "yearly"]
        if v not in allowed_periods:
            raise ValueError(f"Period must be one of: {allowed_periods}")
        return v

    @root_validator
    def validate_period_dates(cls, values):
        period_start = values.get("period_start")
        period_end = values.get("period_end")

        if period_start and period_end and period_start >= period_end:
            raise ValueError("period_start must be before period_end")

        return values

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class NetworkAnalytics(BaseModel):
    """Network connection analytics"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique analytics ID"
    )
    user_id: str = Field(..., description="ID of the user")
    contact_id: str = Field(..., description="ID of the contact")
    connection_strength: float = Field(
        ..., ge=0.0, le=1.0, description="Connection strength (0-1)"
    )
    interaction_frequency: float = Field(
        default=0.0, ge=0.0, description="Interaction frequency"
    )
    last_interaction: Optional[datetime] = Field(
        None, description="Last interaction timestamp"
    )
    total_interactions: int = Field(
        default=0, ge=0, description="Total number of interactions"
    )
    successful_connections: int = Field(
        default=0, ge=0, description="Number of successful connections"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the record was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the record was last updated"
    )

    @validator("successful_connections")
    def validate_successful_connections(cls, v, values):
        total_interactions = values.get("total_interactions", 0)
        if v > total_interactions:
            raise ValueError("successful_connections cannot exceed total_interactions")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class EmbeddingCache(BaseModel):
    """Cached embeddings for content"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique cache ID"
    )
    content_id: str = Field(..., description="ID of the content")
    content_type: str = Field(..., description="Type of content")
    embedding_model: str = Field(..., description="Model used for embedding")
    embedding_vector: List[float] = Field(..., description="Embedding vector")
    content_hash: str = Field(..., description="Hash of the content")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the embedding was created"
    )
    expires_at: Optional[datetime] = Field(
        None, description="When the embedding expires"
    )

    @validator("content_type")
    def validate_content_type(cls, v):
        allowed_types = [
            "opportunity",
            "user_profile",
            "content",
            "text",
            "image",
            "video",
        ]
        if v not in allowed_types:
            raise ValueError(f"Content type must be one of: {allowed_types}")
        return v

    @validator("embedding_vector")
    def validate_embedding_vector(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding vector must contain only numbers")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class NotificationHistory(BaseModel):
    """Notification history tracking"""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), description="Unique notification ID"
    )
    user_id: str = Field(..., description="ID of the user")
    notification_type: str = Field(..., description="Type of notification")
    content: str = Field(..., description="Notification content")
    priority_score: float = Field(
        ..., ge=0.0, le=1.0, description="Priority score (0-1)"
    )
    sent_at: datetime = Field(
        default_factory=datetime.now, description="When the notification was sent"
    )
    opened_at: Optional[datetime] = Field(
        None, description="When the notification was opened"
    )
    clicked_at: Optional[datetime] = Field(
        None, description="When the notification was clicked"
    )
    dismissed_at: Optional[datetime] = Field(
        None, description="When the notification was dismissed"
    )
    delivery_channel: NotificationChannel = Field(..., description="Delivery channel")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @validator("content")
    def validate_content(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Notification content must be at least 5 characters long")
        return v

    @root_validator
    def validate_notification_times(cls, values):
        sent_at = values.get("sent_at")
        opened_at = values.get("opened_at")
        clicked_at = values.get("clicked_at")
        dismissed_at = values.get("dismissed_at")

        if opened_at and sent_at and opened_at < sent_at:
            raise ValueError("opened_at must be after sent_at")

        if clicked_at and sent_at and clicked_at < sent_at:
            raise ValueError("clicked_at must be after sent_at")

        if dismissed_at and sent_at and dismissed_at < sent_at:
            raise ValueError("dismissed_at must be after sent_at")

        return values

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class UserProfile(BaseModel):
    """User profile for intelligence system"""

    user_id: str = Field(..., description="ID of the user")
    industry: Optional[str] = Field(None, description="User's industry")
    location: Optional[str] = Field(None, description="User's location")
    company_size: Optional[str] = Field(None, description="Company size")
    job_role: Optional[str] = Field(None, description="Job role")
    interests: List[str] = Field(default_factory=list, description="User interests")
    preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User preferences"
    )
    engagement_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Engagement score (0-1)"
    )
    last_active: Optional[datetime] = Field(None, description="Last activity timestamp")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the profile was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="When the profile was last updated"
    )

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class Opportunity(BaseModel):
    """Business opportunity"""

    id: str = Field(..., description="Unique opportunity ID")
    title: str = Field(..., description="Opportunity title")
    description: str = Field(..., description="Opportunity description")
    opportunity_type: OpportunityType = Field(..., description="Type of opportunity")
    industry: Optional[str] = Field(None, description="Industry category")
    location: Optional[str] = Field(None, description="Location")
    budget_range: Optional[str] = Field(None, description="Budget range")
    deadline: Optional[datetime] = Field(None, description="Deadline")
    contact_info: Dict[str, str] = Field(
        default_factory=dict, description="Contact information"
    )
    tags: List[str] = Field(default_factory=list, description="Tags")
    status: str = Field(default="active", description="Opportunity status")
    created_at: datetime = Field(
        default_factory=datetime.now, description="When the opportunity was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the opportunity was last updated",
    )

    @validator("title")
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Title must be at least 5 characters long")
        return v

    @validator("description")
    def validate_description(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("Description must be at least 20 characters long")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class IntelligenceRequest(BaseModel):
    """Request to intelligence system"""

    user_id: str = Field(..., description="ID of the user making the request")
    request_type: str = Field(..., description="Type of intelligence request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Request context")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the request was made"
    )

    @validator("request_type")
    def validate_request_type(cls, v):
        allowed_types = [
            "recommendations",
            "opportunity_match",
            "behavior_analysis",
            "analytics",
            "prediction",
        ]
        if v not in allowed_types:
            raise ValueError(f"Request type must be one of: {allowed_types}")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class IntelligenceResponse(BaseModel):
    """Response from intelligence system"""

    request_id: Optional[str] = Field(None, description="ID of the original request")
    user_id: str = Field(..., description="ID of the user")
    response_type: str = Field(..., description="Type of response")
    data: Dict[str, Any] = Field(..., description="Response data")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the response was generated"
    )

    @validator("processing_time_ms")
    def validate_processing_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("Processing time must be non-negative")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


# Utility functions for model validation and conversion
def validate_model_data(model_class: BaseModel, data: Dict[str, Any]) -> BaseModel:
    """
    Validate and convert dictionary data to Pydantic model

    Args:
        model_class: Pydantic model class
        data: Dictionary data to validate

    Returns:
        Validated model instance
    """
    try:
        return model_class(**data)
    except Exception as e:
        raise ValueError(f"Invalid data for {model_class.__name__}: {str(e)}")


def model_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Convert Pydantic model to dictionary

    Args:
        model: Pydantic model instance
        exclude_none: Whether to exclude None values

    Returns:
        Dictionary representation
    """
    return model.dict(exclude_none=exclude_none)


def models_to_dict_list(
    models: List[BaseModel], exclude_none: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert list of Pydantic models to list of dictionaries

    Args:
        models: List of Pydantic model instances
        exclude_none: Whether to exclude None values

    Returns:
        List of dictionary representations
    """
    return [model_to_dict(model, exclude_none) for model in models]


if __name__ == "__main__":
    # Test data models
    print("🧠 Testing Business Dealer Intelligence Data Models")
    print("=" * 60)

    # Test UserInteraction
    interaction = UserInteraction(
        user_id="user123",
        opportunity_id="opp456",
        interaction_type=InteractionType.CLICK,
        duration=45,
        metadata={"source": "mobile_app", "session_id": "sess789"},
    )
    print(f"✅ UserInteraction: {interaction.dict()}")

    # Test OpportunityScore
    score = OpportunityScore(
        opportunity_id="opp456",
        user_id="user123",
        relevance_score=0.85,
        success_probability=0.72,
        factors={"industry_match": 0.9, "location_match": 0.8, "behavior_match": 0.7},
    )
    print(f"✅ OpportunityScore: {score.dict()}")

    # Test Recommendation
    recommendation = Recommendation(
        user_id="user123",
        opportunity_id="opp456",
        recommendation_type=RecommendationType.PERSONALIZED,
        score=0.88,
        reasoning="Recommended based on user's industry preference and past interaction patterns",
    )
    print(f"✅ Recommendation: {recommendation.dict()}")

    # Test BehaviorMetrics
    metrics = BehaviorMetrics(
        user_id="user123",
        metric_type="engagement_rate",
        value=0.65,
        period="weekly",
        period_start=datetime.now(),
        period_end=datetime.now(),
    )
    print(f"✅ BehaviorMetrics: {metrics.dict()}")

    print("\n🎉 All data models validated successfully!")
