"""Dépendances de résolution + contrôle d'accès pour le chat.

Règles :
    * L'authentification JWT est requise (get_current_user).
    * Un utilisateur n'accède qu'à SA conversation / SES messages.
    * Un administrateur accède à toutes les conversations.
"""
from fastapi import Depends, HTTPException, Path, status
from sqlmodel import Session

from ..auth import get_current_user
from ..database import get_session
from ..models import Conversation, Message, User


def _is_admin(user: User) -> bool:
    return user.role == "ADMIN"


def get_accessible_conversation(
    conversation_id: int = Path(..., ge=1),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation introuvable.",
        )
    if not _is_admin(current) and conversation.user_id != current.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé à cette conversation.",
        )
    return conversation


def get_accessible_message(
    message_id: int = Path(..., ge=1),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
) -> Message:
    message = session.get(Message, message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message introuvable."
        )
    conversation = session.get(Conversation, message.conversation_id)
    if not _is_admin(current) and (
        conversation is None or conversation.user_id != current.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé à ce message.",
        )
    return message
