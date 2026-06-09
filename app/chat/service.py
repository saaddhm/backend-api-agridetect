"""Logique métier du chat de support : conversations et messages."""
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import or_
from sqlmodel import Session, func, select

from ..models import (
    Conversation,
    ConversationStatus,
    Message,
    SenderRole,
    User,
)
from .notifications import NotificationService
from .schemas import ConversationRead, MessageRead


class ConversationService:
    """Création, récupération et administration des conversations."""

    def __init__(self, session: Session):
        self.session = session

    # ----------------------------------------------------------------- lecture
    def get(self, conversation_id: int) -> Optional[Conversation]:
        return self.session.get(Conversation, conversation_id)

    def get_for_user(self, user_id: int) -> Optional[Conversation]:
        return self.session.exec(
            select(Conversation).where(Conversation.user_id == user_id)
        ).first()

    def get_or_create_for_user(self, user_id: int) -> Conversation:
        """Retourne la conversation de l'utilisateur, en la créant si besoin."""
        conv = self.get_for_user(user_id)
        if conv:
            return conv
        conv = Conversation(user_id=user_id, status=ConversationStatus.OPEN)
        self.session.add(conv)
        self.session.commit()
        self.session.refresh(conv)
        return conv

    def list_all(
        self, page: int = 1, page_size: int = 20, search: Optional[str] = None
    ) -> Tuple[List[Conversation], int]:
        """Liste paginée de toutes les conversations (admin), avec recherche user."""
        page = max(1, page)
        page_size = min(max(1, page_size), 100)

        stmt = select(Conversation).join(User, User.id == Conversation.user_id)
        count_stmt = (
            select(func.count())
            .select_from(Conversation)
            .join(User, User.id == Conversation.user_id)
        )
        if search:
            like = f"%{search.strip()}%"
            cond = or_(User.full_name.ilike(like), User.email.ilike(like))
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        total = self.session.exec(count_stmt).one()
        stmt = (
            stmt.order_by(Conversation.last_message_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = self.session.exec(stmt).all()
        return items, int(total)

    # ----------------------------------------------------------------- écriture
    def close(self, conversation: Conversation) -> Conversation:
        conversation.status = ConversationStatus.CLOSED
        conversation.updated_at = datetime.utcnow()
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def assign_admin(self, conversation: Conversation, admin_id: int) -> None:
        if conversation.assigned_admin_id is None:
            conversation.assigned_admin_id = admin_id
            conversation.updated_at = datetime.utcnow()
            self.session.add(conversation)
            self.session.commit()

    # ----------------------------------------------------------- sérialisation
    def to_read(self, conversation: Conversation, viewer_role: str) -> ConversationRead:
        """Construit le schéma de sortie enrichi (nom user, dernier message, non lus)."""
        user = self.session.get(User, conversation.user_id)

        last = self.session.exec(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(1)
        ).first()

        # Messages non lus = messages entrants pour le lecteur.
        incoming_role = (
            SenderRole.USER if viewer_role == "ADMIN" else SenderRole.ADMIN
        )
        unread = self.session.exec(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation.id)
            .where(Message.sender_role == incoming_role)
            .where(Message.is_read == False)  # noqa: E712
        ).one()

        return ConversationRead(
            id=conversation.id,
            user_id=conversation.user_id,
            assigned_admin_id=conversation.assigned_admin_id,
            status=conversation.status,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at,
            user_name=user.full_name if user else None,
            user_email=user.email if user else None,
            last_message=last.content if last else None,
            unread_count=int(unread),
        )


class ChatService:
    """Gestion des messages d'une conversation."""

    def __init__(self, session: Session):
        self.session = session
        self.notifications = NotificationService(session)

    def list_messages(
        self, conversation_id: int, page: int = 1, page_size: int = 30
    ) -> Tuple[List[Message], int]:
        """Messages paginés, du plus récent au plus ancien (page 1 = récents)."""
        page = max(1, page)
        page_size = min(max(1, page_size), 100)

        total = self.session.exec(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        ).one()

        items = self.session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
        return items, int(total)

    def send_message(
        self, conversation: Conversation, sender: User, content: str
    ) -> Message:
        """Crée un message, met à jour la conversation et déclenche la notification."""
        role = SenderRole.ADMIN if sender.role == "ADMIN" else SenderRole.USER
        now = datetime.utcnow()

        message = Message(
            conversation_id=conversation.id,
            sender_id=sender.id,
            sender_role=role,
            content=content.strip(),
            is_read=False,
            created_at=now,
        )
        self.session.add(message)

        # Mise à jour des métadonnées de la conversation.
        conversation.last_message_at = now
        conversation.updated_at = now
        if role == SenderRole.ADMIN:
            # Un admin qui répond s'attribue la conversation et la passe en "open".
            if conversation.assigned_admin_id is None:
                conversation.assigned_admin_id = sender.id
            conversation.status = ConversationStatus.OPEN
        else:
            # Un message user rouvre une conversation fermée et la met "pending".
            if conversation.status == ConversationStatus.CLOSED:
                conversation.status = ConversationStatus.OPEN
            else:
                conversation.status = ConversationStatus.PENDING
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(message)

        # Notification (DB + FCM) — tolérante aux pannes.
        try:
            self.notifications.notify_new_message(conversation, message, sender.full_name)
        except Exception:  # pragma: no cover
            pass
        return message

    def mark_read(self, message: Message) -> Message:
        if not message.is_read:
            message.is_read = True
            self.session.add(message)
            self.session.commit()
            self.session.refresh(message)
        return message
