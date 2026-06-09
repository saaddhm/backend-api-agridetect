from datetime import datetime
from typing import List, Optional
from sqlalchemy import JSON, Column
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str
    email: str = Field(index=True, unique=True)
    password_hash: str
    role: str = Field(default="USER")  # USER | ADMIN
    is_active: bool = Field(default=True, index=True)
    scan_enabled: bool = Field(default=True, index=True)
    email_verified: bool = Field(default=True, index=True)
    email_verification_token: Optional[str] = Field(default=None, index=True)
    email_verification_code: Optional[str] = Field(default=None, index=True)
    email_code_expires_at: Optional[datetime] = None
    email_verified_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Analysis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    image_name: str
    label: str
    plant: str
    disease: str
    severity: str
    confidence: float
    is_healthy: bool = False
    cause: Optional[str] = None
    treatment: Optional[str] = None
    backend: str = "mock"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdminNotification(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    title: str
    message: str
    level: str = Field(default="INFO")
    is_read: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlertRecord(SQLModel, table=True):
    """Alerte épidémiologique créée/gérée par l'admin.

    Ciblage : si is_global == True -> visible par tous les utilisateurs ;
    sinon visible uniquement par l'utilisateur user_id.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    plant: str = ""
    disease: str = ""
    severity: str = Field(default="MODEREE")  # FAIBLE | MODEREE | ELEVEE | CRITIQUE
    zone: str = "Zone globale"
    message: str = ""
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    is_global: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DiseaseTreatment(SQLModel, table=True):
    """Fiche de traitement détaillée par maladie (module Treatment Recommendations)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    label: str = Field(index=True, unique=True)        # class_label du modèle = clé de mapping
    disease_name: str = Field(index=True)
    disease_name_fr: Optional[str] = None
    disease_name_ar: Optional[str] = None
    plant_name: str = ""
    description: str = ""
    symptoms: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    causes: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    prevention: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    organic_treatment: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    chemical_treatment: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    recommended_products: List[dict] = Field(default_factory=list, sa_column=Column(JSON))
    severity: str = "MODEREE"
    recovery_time: str = ""
    expert_advice: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================================================
#  SUPPORT CHAT (agriculteur <-> administrateur)
# ===========================================================================

class ConversationStatus:
    OPEN = "open"
    PENDING = "pending"
    CLOSED = "closed"
    ALL = {OPEN, PENDING, CLOSED}


class SenderRole:
    USER = "user"
    ADMIN = "admin"


class Conversation(SQLModel, table=True):
    """Une conversation de support, unique par agriculteur."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    assigned_admin_id: Optional[int] = Field(
        default=None, foreign_key="user.id", index=True
    )
    status: str = Field(default=ConversationStatus.OPEN, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = Field(default=None, index=True)


class Message(SQLModel, table=True):
    """Un message dans une conversation de support."""
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(index=True, foreign_key="conversation.id")
    sender_id: int = Field(foreign_key="user.id")
    sender_role: str  # user | admin
    content: str
    is_read: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class DeviceToken(SQLModel, table=True):
    """Jeton FCM d'un appareil, pour le ciblage des notifications push."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    token: str = Field(index=True, unique=True)
    platform: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
