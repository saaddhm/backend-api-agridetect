"""NotificationService — enregistrement en base + push FCM.

Stratégie :
    * Chaque notification est historisée dans la table `adminnotification`
      (réutilisée comme boîte de réception in-app, déjà exploitée par le mobile).
    * Un push Firebase Cloud Messaging est envoyé aux appareils du destinataire
      (jetons stockés dans `devicetoken`). L'envoi est tolérant aux pannes :
      si firebase-admin n'est pas configuré, on journalise et on continue.

Configuration FCM (variables d'environnement) :
    FIREBASE_CREDENTIALS = chemin vers le fichier serviceAccount.json
    (à défaut, on tente les Application Default Credentials).
"""
import logging
import os
from typing import Optional

from sqlmodel import Session, select

from ..models import AdminNotification, ConversationStatus, DeviceToken, SenderRole, User

logger = logging.getLogger("agridetect.chat")

_FB_READY: Optional[bool] = None


def _ensure_firebase() -> bool:
    """Initialise firebase-admin une seule fois. Retourne False si indisponible."""
    global _FB_READY
    if _FB_READY is not None:
        return _FB_READY
    try:
        import firebase_admin
        from firebase_admin import credentials

        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS")
            if cred_path and os.path.exists(cred_path):
                firebase_admin.initialize_app(credentials.Certificate(cred_path))
            else:
                firebase_admin.initialize_app()  # Application Default Credentials
        _FB_READY = True
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        logger.warning("FCM indisponible (push désactivé) : %s", exc)
        _FB_READY = False
    return _FB_READY


def _send_fcm(session: Session, user_id: int, title: str, body: str, data: dict) -> None:
    if not _ensure_firebase():
        return
    try:
        from firebase_admin import messaging

        tokens = [
            t.token
            for t in session.exec(
                select(DeviceToken).where(DeviceToken.user_id == user_id)
            ).all()
        ]
        if not tokens:
            return
        message = messaging.MulticastMessage(
            tokens=tokens,
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in data.items()},
        )
        resp = messaging.send_each_for_multicast(message)
        # Purge des jetons invalides.
        if resp.failure_count:
            for token, res in zip(tokens, resp.responses):
                if not res.success:
                    stale = session.exec(
                        select(DeviceToken).where(DeviceToken.token == token)
                    ).first()
                    if stale:
                        session.delete(stale)
            session.commit()
    except Exception as exc:  # pragma: no cover
        logger.warning("Echec d'envoi FCM : %s", exc)


class NotificationService:
    """Notifie les bonnes personnes lors de l'envoi d'un message."""

    def __init__(self, session: Session):
        self.session = session

    def _record(self, user_id: int, title: str, message: str, level: str = "CHAT") -> None:
        self.session.add(
            AdminNotification(
                user_id=user_id, title=title, message=message[:240], level=level
            )
        )

    def _admin_ids(self, conversation) -> list[int]:
        if conversation.assigned_admin_id:
            return [conversation.assigned_admin_id]
        return [
            u.id
            for u in self.session.exec(select(User).where(User.role == "ADMIN")).all()
            if u.id is not None
        ]

    def notify_new_message(self, conversation, message, sender_name: str) -> None:
        """Notifie le destinataire (admin si l'expéditeur est un user, et inversement)."""
        preview = message.content.strip()
        data = {
            "type": "chat",
            "conversation_id": conversation.id,
            "message_id": message.id,
        }

        if message.sender_role == SenderRole.USER:
            title = f"Nouveau message de {sender_name}"
            for admin_id in self._admin_ids(conversation):
                self._record(admin_id, title, preview)
                _send_fcm(self.session, admin_id, title, preview, data)
        else:
            title = "Réponse du support AgriDetect"
            self._record(conversation.user_id, title, preview)
            _send_fcm(self.session, conversation.user_id, title, preview, data)

        self.session.commit()
