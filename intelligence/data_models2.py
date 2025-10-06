#!/usr/bin/env python3
"""
Core Data Models for Business Dealer Intelligence System
Pydantic models for data validation and serialization throughout the system

Compatible with Pydantic v2.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from pydantic import model_validator

class OpportunityType(str, Enum):
    BUYER = "buyer"
    SELLER = "seller"
    SERVICE = "service"
    PARTNERSHIP = "partnership"


class InteractionType(str, Enum):
    VIEW = "view"
    CLICK = "click"
    SHARE = "share"
    CONTACT = "contact"
    SAVE = "save"


class RecommendationType(str, Enum):
    PERSONALIZED = "personalized"
    TRENDING = "trending"
    SIMILAR_USERS = "similar_users"
    LOCATION_BASED = "location_based"
    INDUSTRY_MATCH = "industry_match"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class NotificationChannel(str, Enum):
    EMAIL = "email"
    PUSH = "push"
    IN_APP = "in_app"
    SMS = "sms"


class ModelType(str, Enum):
    SCORING = "scoring"
    CLUSTERING = "clustering"
    PREDICTION = "prediction"
    EMBEDDING = "embedding"


class UserInteraction(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    opportunity_id: str
    interaction_type: InteractionType
    timestamp: datetime = Field(default_factory=datetime.now)
    duration: Optional[int] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v):
        if v is not None and v < 0:
            raise ValueError("Duration must be non-negative")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        if v is None:
            return {}
        for key in v:
            if not isinstance(key, str):
                raise ValueError("Metadata keys must be strings")
        return v

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class OpportunityScore(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    opportunity_id: str
    user_id: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    success_probability: float = Field(..., ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict)
    calculated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, v):
        if v is None:
            return {}
        for factor, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Factor {factor} must be a number between 0 and 1")
        return v

    
    @model_validator(mode="after")
    @classmethod
    def validate_expiry(cls, data):
        if data.expires_at and data.expires_at <= data.calculated_at:
            raise ValueError("expires_at must be after calculated_at")
        return data

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

class Recommendation(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    opportunity_id: str
    recommendation_type: RecommendationType
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    generated_at: datetime = Field(default_factory=datetime.now)
    clicked_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Reasoning must be at least 10 characters long")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_interaction_times(cls, data):
        if data.clicked_at and data.clicked_at < data.generated_at:
            raise ValueError("clicked_at must be after generated_at")
        if data.dismissed_at and data.dismissed_at < data.generated_at:
            raise ValueError("dismissed_at must be after generated_at")
        if data.expires_at and data.expires_at <= data.generated_at:
            raise ValueError("expires_at must be after generated_at")
        return data


class BehaviorMetrics(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    metric_type: str
    value: float
    period: str
    period_start: datetime
    period_end: datetime
    calculated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("period")
    @classmethod
    def validate_period(cls, v):
        allowed = ["daily", "weekly", "monthly", "yearly"]
        if v not in allowed:
            raise ValueError(f"Period must be one of: {allowed}")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_period_dates(cls, data):
        if data.period_start >= data.period_end:
            raise ValueError("period_start must be before period_end")
        return data

class NetworkAnalytics(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    contact_id: str
    connection_strength: float = Field(..., ge=0.0, le=1.0)
    interaction_frequency: float = Field(default=0.0, ge=0.0)
    last_interaction: Optional[datetime] = None
    total_interactions: int = Field(default=0, ge=0)
    successful_connections: int = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    @classmethod
    def validate_connection_count(cls, data):
        if data.successful_connections > data.total_interactions:
            raise ValueError("successful_connections cannot exceed total_interactions")
        return data


class EmbeddingCache(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    content_id: str
    content_type: str
    embedding_model: str
    embedding_vector: List[float]
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v):
        allowed = [
            "opportunity",
            "user_profile",
            "content",
            "text",
            "image",
            "video"
        ]
        if v not in allowed:
            raise ValueError(f"Content type must be one of: {allowed}")
        return v

    @field_validator("embedding_vector")
    @classmethod
    def validate_vector(cls, v):
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding vector must contain only numbers")
        return v

class NotificationHistory(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    notification_type: str
    content: str
    priority_score: float = Field(..., ge=0.0, le=1.0)
    sent_at: datetime = Field(default_factory=datetime.now)
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    dismissed_at: Optional[datetime] = None
    delivery_channel: NotificationChannel
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Notification content must be at least 5 characters long")
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_times(cls, data):
        if data.opened_at and data.opened_at < data.sent_at:
            raise ValueError("opened_at must be after sent_at")
        if data.clicked_at and data.clicked_at < data.sent_at:
            raise ValueError("clicked_at must be after sent_at")
        if data.dismissed_at and data.dismissed_at < data.sent_at:
            raise ValueError("dismissed_at must be after sent_at")
        return data


class UserProfile(BaseModel):
    user_id: str
    industry: Optional[str] = None
    location: Optional[str] = None
    company_size: Optional[str] = None
    job_role: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    last_active: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Opportunity(BaseModel):
    id: str
    title: str
    description: str
    opportunity_type: OpportunityType
    industry: Optional[str] = None
    location: Optional[str] = None
    budget_range: Optional[str] = None
    deadline: Optional[datetime] = None
    contact_info: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    status: str = Field(default="active")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Title must be at least 5 characters long")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        if not v or len(v.strip()) < 20:
            raise ValueError("Description must be at least 20 characters long")
        return v


class IntelligenceRequest(BaseModel):
    user_id: str
    request_type: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("request_type")
    @classmethod
    def validate_request_type(cls, v):
        allowed = [
            "recommendations",
            "opportunity_match",
            "behavior_analysis",
            "analytics",
            "prediction",
        ]
        if v not in allowed:
            raise ValueError(f"Request type must be one of: {allowed}")
        return v


class IntelligenceResponse(BaseModel):
    request_id: Optional[str] = None
    user_id: str
    response_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator("processing_time_ms")
    @classmethod
    def validate_processing_time(cls, v):
        if v is not None and v < 0:
            raise ValueError("Processing time must be non-negative")
        return v


# -------------------------------------------------------------------
# Utility functions for conversion and validation
# -------------------------------------------------------------------

def validate_model_data(model_class: type[BaseModel], data: Dict[str, Any]) -> BaseModel:
    try:
        return model_class(**data)
    except Exception as e:
        raise ValueError(f"Invalid data for {model_class.__name__}: {str(e)}")


def model_to_dict(model: BaseModel, exclude_none: bool = True) -> Dict[str, Any]:
    return model.model_dump(exclude_none=exclude_none)


def models_to_dict_list(models: List[BaseModel], exclude_none: bool = True) -> List[Dict[str, Any]]:
    return [model_to_dict(model, exclude_none) for model in models]
