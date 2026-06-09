"""Endpoints REST du chat de support (préfixe complet : /api/chat)."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session, select

from ..auth import get_current_admin, get_current_user
from ..database import get_session
from ..models import Conversation, DeviceToken, Message, User
from .dependencies import get_accessible_conversation, get_accessible_message
from .schemas import (
    ConversationPage,
    ConversationRead,
    DeviceTokenCreate,
    MessageCreate,
    MessagePage,
    MessageRead,
    SimpleOk,
)
from .service import ChatService, ConversationService

router = APIRouter(prefix="/chat", tags=["Chat"])


# --------------------------------------------------------------------------- #
#  CONVERSATIONS
# --------------------------------------------------------------------------- #
@router.post("/conversations", response_model=ConversationRead, status_code=201)
def create_or_get_conversation(
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Crée (ou retourne) la conversation de l'utilisateur courant."""
    svc = ConversationService(session)
    conversation = svc.get_or_create_for_user(current.id)
    return svc.to_read(conversation, viewer_role=current.role)


@router.get("/conversations/me", response_model=ConversationRead)
def get_my_conversation(
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Retourne la conversation de l'utilisateur courant (la crée si absente)."""
    svc = ConversationService(session)
    conversation = svc.get_or_create_for_user(current.id)
    return svc.to_read(conversation, viewer_role=current.role)


@router.get("/conversations", response_model=ConversationPage)
def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Recherche par nom ou email"),
    session: Session = Depends(get_session),
    _admin: User = Depends(get_current_admin),  # ADMIN uniquement
):
    """Liste paginée de toutes les conversations (réservé aux administrateurs)."""
    svc = ConversationService(session)
    items, total = svc.list_all(page=page, page_size=page_size, search=search)
    return ConversationPage(
        items=[svc.to_read(c, viewer_role="ADMIN") for c in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationRead)
def get_conversation(
    conversation: Conversation = Depends(get_accessible_conversation),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Détails d'une conversation (propriétaire ou admin)."""
    return ConversationService(session).to_read(conversation, viewer_role=current.role)


@router.put("/conversations/{conversation_id}/close", response_model=ConversationRead)
def close_conversation(
    conversation: Conversation = Depends(get_accessible_conversation),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Ferme une conversation (propriétaire ou admin)."""
    svc = ConversationService(session)
    svc.close(conversation)
    return svc.to_read(conversation, viewer_role=current.role)


# --------------------------------------------------------------------------- #
#  MESSAGES
# --------------------------------------------------------------------------- #
@router.get("/conversations/{conversation_id}/messages", response_model=MessagePage)
def list_messages(
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
    conversation: Conversation = Depends(get_accessible_conversation),
    session: Session = Depends(get_session),
):
    """Messages paginés d'une conversation (propriétaire ou admin)."""
    chat = ChatService(session)
    items, total = chat.list_messages(conversation.id, page=page, page_size=page_size)
    return MessagePage(
        items=[MessageRead.model_validate(m) for m in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageRead,
    status_code=201,
)
def send_message(
    payload: MessageCreate,
    conversation: Conversation = Depends(get_accessible_conversation),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Envoie un message (propriétaire ou admin). Notifie l'autre partie."""
    if conversation.status == "closed" and current.role != "ADMIN":
        # Un message d'un user rouvre automatiquement (géré dans le service).
        pass
    message = ChatService(session).send_message(conversation, current, payload.content)
    return MessageRead.model_validate(message)


@router.put("/messages/{message_id}/read", response_model=MessageRead)
def mark_message_read(
    message: Message = Depends(get_accessible_message),
    session: Session = Depends(get_session),
):
    """Marque un message comme lu (propriétaire ou admin)."""
    updated = ChatService(session).mark_read(message)
    return MessageRead.model_validate(updated)


# --------------------------------------------------------------------------- #
#  JETONS D'APPAREIL (FCM) — utilitaire pour le ciblage des push
# --------------------------------------------------------------------------- #
@router.post("/devices", response_model=SimpleOk, status_code=201)
def register_device(
    payload: DeviceTokenCreate,
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Enregistre / met à jour le jeton FCM de l'appareil courant pour cet utilisateur."""
    existing = session.exec(
        select(DeviceToken).where(DeviceToken.token == payload.token)
    ).first()
    if existing:
        existing.user_id = current.id
        if payload.platform:
            existing.platform = payload.platform
        session.add(existing)
    else:
        session.add(
            DeviceToken(
                user_id=current.id, token=payload.token, platform=payload.platform
            )
        )
    session.commit()
    return SimpleOk()


@router.delete("/devices", response_model=SimpleOk)
def unregister_device(
    token: str = Query(..., min_length=10),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Supprime un jeton FCM (déconnexion / changement d'appareil)."""
    row = session.exec(
        select(DeviceToken)
        .where(DeviceToken.token == token)
        .where(DeviceToken.user_id == current.id)
    ).first()
    if row:
        session.delete(row)
        session.commit()
    return SimpleOk()
