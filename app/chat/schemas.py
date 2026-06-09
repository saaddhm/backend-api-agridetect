"""Schémas Pydantic pour le module de chat de support."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class MessageRead(BaseModel):
    id: int
    conversation_id: int
    sender_id: int
    sender_role: str
    content: str
    is_read: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationRead(BaseModel):
    id: int
    user_id: int
    assigned_admin_id: Optional[int] = None
    status: str
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None

    # Champs enrichis (utiles surtout pour la vue administrateur).
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    last_message: Optional[str] = None
    unread_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationPage(BaseModel):
    items: List[ConversationRead]
    total: int
    page: int
    page_size: int


class MessagePage(BaseModel):
    items: List[MessageRead]
    total: int
    page: int
    page_size: int


class DeviceTokenCreate(BaseModel):
    token: str = Field(min_length=10, max_length=4096)
    platform: Optional[str] = None


class SimpleOk(BaseModel):
    ok: bool = True
