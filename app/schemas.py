from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr


# ---- Auth ----
class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: str
    is_active: bool = True
    scan_enabled: bool = True

    class Config:
        from_attributes = True


# ---- Prediction / Analysis ----
class PredictionItem(BaseModel):
    label: str
    plant: str
    disease: str
    severity: str
    is_healthy: bool
    confidence: float
    cause: Optional[str] = None
    treatment: Optional[str] = None


class AnalysisResponse(BaseModel):
    id: int
    plant: str
    disease: str
    label: str
    severity: str
    confidence: float
    is_healthy: bool
    cause: Optional[str] = None
    treatment: Optional[str] = None
    backend: str = "mock"
    image_url: Optional[str] = None
    created_at: datetime
    top_k: Optional[List[PredictionItem]] = None

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    total_analyses: int
    distinct_diseases: int
    avg_confidence: float


TokenResponse.model_rebuild()
