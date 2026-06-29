"""Dépendances de résolution + contrôle d'accès pour le chat.

Règles :
    * L'authentification JWT est requise (get_current_user).
    * Un utilisateur n'accède qu'à SA conversation / SES messages.
    * Un administrateur accède à toutes les conversations.
"""
import logging

from fastapi import Depends, HTTPException, Path, status
from sqlmodel import Session

from ..auth import get_current_user
from ..database import get_session
from ..models import Conversation, Message, User

# Journal de securite : toute tentative d'acces non autorise est tracee.
security_logger = logging.getLogger("agridetect.security")


def _is_admin(user: User) -> bool:
    return user.role == "ADMIN"


def get_accessible_conversation(
    conversation_id: int = Path(..., ge=1),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
) -> Conversation:
    """Resout une conversation et verifie que l'appelant a le droit d'y acceder.

    SECURITE : l'identite provient EXCLUSIVEMENT du JWT (current.id / current.role).
    Aucun user_id du client n'est utilise pour decider de l'acces. Un non-admin
    ne peut acceder qu'a sa propre conversation -> bloque l'escalade horizontale.
    """
    conversation = session.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation introuvable.",
        )
    if not _is_admin(current) and conversation.user_id != current.id:
        # Tentative d'acces a la conversation d'un autre utilisateur.
        security_logger.warning(
            "ACCES REFUSE chat: user_id=%s (%s) a tente d'acceder a la "
            "conversation %s appartenant a user_id=%s",
            current.id, current.email, conversation.id, conversation.user_id,
        )
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
    """Resout un message et verifie l'acces a sa conversation parente."""
    message = session.get(Message, message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message introuvable."
        )
    conversation = session.get(Conversation, message.conversation_id)
    if not _is_admin(current) and (
        conversation is None or conversation.user_id != current.id
    ):
        security_logger.warning(
            "ACCES REFUSE message: user_id=%s (%s) a tente d'acceder au "
            "message %s (conversation %s)",
            current.id, current.email, message_id, message.conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Accès refusé à ce message.",
        )
    return message
