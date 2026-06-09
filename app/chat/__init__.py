"""Module de chat de support (agriculteur <-> administrateur).

Sous-modules :
    schemas       — schémas Pydantic (entrées / sorties API)
    notifications — NotificationService (enregistrement DB + push FCM)
    service       — ConversationService et ChatService (logique métier)
    dependencies  — résolution de conversation + contrôle des permissions
    router        — endpoints FastAPI (/api/chat/...)
"""
