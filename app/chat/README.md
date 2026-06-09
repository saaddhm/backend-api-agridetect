# Support Chat — Backend (AgriDetect AI)

Système de messagerie de support entre **agriculteurs** et **administrateurs**,
intégré à l'API FastAPI existante (JWT, SQLModel/SQLAlchemy, MySQL/SQLite).

## Structure des fichiers

```
backend-api/
├── app/
│   ├── models.py                 # + Conversation, Message, DeviceToken
│   ├── main.py                   # include_router(chat_router)
│   └── chat/
│       ├── __init__.py
│       ├── schemas.py            # Schémas Pydantic (I/O API)
│       ├── notifications.py      # NotificationService (DB + FCM)
│       ├── service.py            # ConversationService, ChatService
│       ├── dependencies.py       # Permissions / résolution de conversation
│       └── router.py             # Endpoints REST (/api/chat/...)
└── migrations/
    └── 001_create_chat_tables.sql  # DDL MySQL (production)
```

## Modèle de données

- **Conversation** : `id, user_id, assigned_admin_id?, status(open|pending|closed),
  created_at, updated_at, last_message_at`. Une conversation par agriculteur.
- **Message** : `id, conversation_id, sender_id, sender_role(user|admin), content,
  is_read, created_at`.
- **DeviceToken** : jetons FCM par utilisateur (ciblage des push).

## Endpoints (préfixe `/api`)

| Méthode | Chemin | Accès | Rôle |
|--------|--------|-------|------|
| POST | `/chat/conversations` | user | Crée/retourne sa conversation |
| GET  | `/chat/conversations/me` | user | Sa conversation |
| GET  | `/chat/conversations` | **admin** | Liste paginée + recherche |
| GET  | `/chat/conversations/{id}` | propriétaire/admin | Détails |
| PUT  | `/chat/conversations/{id}/close` | propriétaire/admin | Fermer |
| GET  | `/chat/conversations/{id}/messages` | propriétaire/admin | Messages paginés |
| POST | `/chat/conversations/{id}/messages` | propriétaire/admin | Envoyer un message |
| PUT  | `/chat/messages/{id}/read` | propriétaire/admin | Marquer lu |
| POST | `/chat/devices` | user | Enregistrer un jeton FCM |
| DELETE | `/chat/devices?token=...` | user | Supprimer un jeton FCM |

### Sécurité
- JWT obligatoire sur tous les endpoints (`get_current_user`).
- Un utilisateur n'accède qu'à **sa** conversation/ses messages ; un **admin**
  accède à tout. Contrôle centralisé dans `dependencies.py` (403 sinon).
- `GET /chat/conversations` est protégé par `get_current_admin` (403 pour un user).

## Notifications (FCM)

`NotificationService.notify_new_message` :
1. enregistre une notification en base (`adminnotification`, boîte in-app) ;
2. envoie un push FCM aux appareils du destinataire :
   - message **d'un user** → notifie l'admin assigné (ou tous les admins) ;
   - message **d'un admin** → notifie l'agriculteur.

Configuration : variable d'environnement `FIREBASE_CREDENTIALS` = chemin du
`serviceAccount.json`. Si firebase-admin est absent ou non configuré, le push est
silencieusement ignoré (l'API ne tombe jamais à cause de la notification).

## Base de données

- **MySQL (prod)** : `DATABASE_URL=mysql+pymysql://user:pass@host:3306/agridetect`
  puis exécuter `migrations/001_create_chat_tables.sql`.
- **Dev (SQLite/MySQL)** : les tables sont créées automatiquement au démarrage
  par `SQLModel.metadata.create_all()`.

## Temps réel : WebSocket (Option A) vs Firestore (Option B)

**Recommandation : Option B (Firestore) pour le temps réel mobile, l'API REST
ci-dessus restant la source de vérité.**

| Critère | A. WebSocket FastAPI | B. Firestore |
|--------|----------------------|--------------|
| Scalabilité | Connexions persistantes en mémoire du process : limite par instance, état à répliquer (Redis pub/sub) derrière plusieurs workers/instances | Géré par Google, scaling automatique, pas d'état serveur à maintenir |
| Performance mobile | Reconnexions à gérer (réseau mobile instable, passage 4G/Wi‑Fi, arrière-plan) | SDK offline‑first : cache local, resync auto, listeners robustes |
| Push hors‑app | À coupler quand même avec FCM | FCM natif + listeners |
| Complexité backend | Endpoint WS + gestion des rooms + broker | Aucune (le client écoute Firestore) |
| Coût | Compute (connexions ouvertes) | Lectures/écritures Firestore |

Le projet utilise déjà Firebase/Firestore côté mobile : **le client écoute
Firestore en temps réel**, et le backend reste l'autorité (persistance MySQL,
permissions, historique, FCM). En option, une Cloud Function peut répliquer les
messages MySQL→Firestore, ou le mobile écrit dans Firestore et un worker
synchronise vers MySQL.

WebSocket (Option A) reste pertinent pour un back‑office web admin temps réel sur
une seule instance ; au‑delà, il faut un broker (Redis) pour diffuser entre
workers — surcoût d'infra non justifié ici face à Firestore + FCM.

> Un squelette WebSocket optionnel peut être ajouté
> (`/api/chat/ws/{conversation_id}`) avec un `ConnectionManager` en mémoire +
> Redis pub/sub pour le multi‑instance, si une démo temps réel sans Firestore
> est requise.
```
