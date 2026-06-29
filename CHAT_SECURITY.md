# Support Chat — Audit & correctif d'isolation des utilisateurs

## 1. Cause exacte du bug

Le chat du **mobile** et du **dashboard** lisait/écrivait **directement dans Firestore**
(`chats/{userId}/messages`). Tous les clients s'authentifient en **anonyme**, et la
règle Firestore déployée était :

```
match /chats/{userId} { allow read, write: if request.auth != null; }
```

`request.auth != null` est vrai pour **n'importe quel** client anonyme, et le
propriétaire de la conversation n'était que l'**id de document fourni par le client**.
→ N'importe quel utilisateur authentifié pouvait lire/écrire la conversation de
**n'importe quel autre** utilisateur : **escalade horizontale de privilèges**.

> Les endpoints REST `/api/chat` étaient déjà corrects, mais les applications ne les
> utilisaient pas.

## 2. Correctif — tout passe par le backend FastAPI (JWT)

| Couche | Avant | Après |
|--------|-------|-------|
| Mobile | Firestore direct (`chats/{userId}`) | API REST `/api/chat` (JWT) — polling |
| Dashboard | Firestore direct | API REST `/api/chat` (JWT) |
| Identité | id de doc fourni par le client | `current_user.id` extrait du JWT |
| Rôle expéditeur | champ client | dérivé de `current_user.role` côté serveur |
| Firestore `chats/**` | `if request.auth != null` | `if false` (bloqué) |

### Backend — contrôles d'accès (déjà en place, durcis + journalisés)

`app/chat/dependencies.py` :

```python
# AVANT (résumé) : récupère la conversation par id, sans vérifier le propriétaire.

# APRÈS : ownership obligatoire + log de sécurité.
if not _is_admin(current) and conversation.user_id != current.id:
    security_logger.warning("ACCES REFUSE chat: user_id=%s a tente d'acceder a la "
                            "conversation %s de user_id=%s", current.id, conversation.id,
                            conversation.user_id)
    raise HTTPException(403, "Accès refusé à cette conversation.")
```

Vérification de **tous** les endpoints (`app/chat/router.py`) :

| Méthode | Endpoint | Protection |
|--------|----------|------------|
| POST | `/chat/conversations` | JWT — crée/récupère **la conversation de current_user** |
| GET | `/chat/conversations/me` | JWT — current_user uniquement |
| GET | `/chat/conversations` | **`get_current_admin`** (403 si non-admin) |
| GET | `/chat/conversations/{id}` | `get_accessible_conversation` (owner/admin) |
| GET | `/chat/conversations/{id}/messages` | `get_accessible_conversation` |
| POST | `/chat/conversations/{id}/messages` | `get_accessible_conversation` |
| POST | `/chat/conversations/{id}/messages/media` | `get_accessible_conversation` |
| PUT | `/chat/messages/{id}/read` | `get_accessible_message` |
| PUT | `/chat/conversations/{id}/close` | `get_accessible_conversation` |

Le `sender_role` et le `sender_id` sont **toujours** dérivés du JWT
(`service.send_message`), jamais du corps de la requête.

### Base de données

`Message` : ajout de `type` (text|image|audio), `media_url`, `audio_ms`.
Relations : `conversation.user_id -> user.id`, `message.conversation_id -> conversation.id`,
`message.sender_id -> user.id`. Chaque conversation est **unique par utilisateur**
(`ConversationService.get_or_create_for_user(current.id)`).

### Mobile / Dashboard

- Le mobile ne manipule plus qu'un `conversationId` ; en mode utilisateur il est
  résolu via `POST /chat/conversations` (= **sa** conversation). Aucun `user_id`
  n'est envoyé par le client.
- Pas de cache global : à la **déconnexion**, le token JWT est effacé ; toute requête
  suivante renvoie 401 et aucune donnée de l'ancienne session n'est conservée. Un
  nouvel utilisateur ne voit donc que ses propres messages.
- Médias servis par le backend via une URL non-devinable (`/api/chat/media/{uuid}`).

## 3. Déploiement (VPS SQLite)

```bash
cd /apps/agridetect-ai/backend-api
# nouvelles colonnes Message (create_all n'altère pas une table existante)
sqlite3 agridetect.db "ALTER TABLE message ADD COLUMN type VARCHAR DEFAULT 'text';"   2>/dev/null
sqlite3 agridetect.db "ALTER TABLE message ADD COLUMN media_url VARCHAR;"             2>/dev/null
sqlite3 agridetect.db "ALTER TABLE message ADD COLUMN audio_ms INTEGER DEFAULT 0;"   2>/dev/null
systemctl restart agridetect
```
(Si les tables `conversation`/`message` n'existaient pas encore, `create_all` les crée
avec les bonnes colonnes — les `ALTER` échouent alors sans danger.)

Règles Firestore (depuis `firebase/`) :
```bash
firebase deploy --only firestore:rules --project agridetect-59edc
```

## 4. Scénario de test end-to-end

Pré-requis : 3 comptes — User A, User B (USER) et un ADMIN.

```bash
API=http://localhost:8000/api
TA=$(curl -s $API/auth/login -H 'Content-Type: application/json' -d '{"email":"a@x.com","password":"..."}' | jq -r .access_token)
TB=$(curl -s $API/auth/login -H 'Content-Type: application/json' -d '{"email":"b@x.com","password":"..."}' | jq -r .access_token)
TADM=$(curl -s $API/auth/login -H 'Content-Type: application/json' -d '{"email":"admin@agridetect.ai","password":"Admin@123"}' | jq -r .access_token)

# A et B créent/ouvrent LEUR conversation + envoient un message
CA=$(curl -s -X POST $API/chat/conversations -H "Authorization: Bearer $TA" | jq -r .id)
CB=$(curl -s -X POST $API/chat/conversations -H "Authorization: Bearer $TB" | jq -r .id)
curl -s -X POST $API/chat/conversations/$CA/messages -H "Authorization: Bearer $TA" -H 'Content-Type: application/json' -d '{"content":"Bonjour, message de A"}'
curl -s -X POST $API/chat/conversations/$CB/messages -H "Authorization: Bearer $TB" -H 'Content-Type: application/json' -d '{"content":"Bonjour, message de B"}'

# ✅ A ne voit QUE sa conversation
curl -s $API/chat/conversations/$CA/messages -H "Authorization: Bearer $TA"   # 200, messages de A

# ❌ A tente de lire la conversation de B  -> 403 (et log "ACCES REFUSE")
curl -s -o /dev/null -w "%{http_code}\n" $API/chat/conversations/$CB/messages -H "Authorization: Bearer $TA"   # 403

# ❌ B tente de lire la conversation de A  -> 403
curl -s -o /dev/null -w "%{http_code}\n" $API/chat/conversations/$CA/messages -H "Authorization: Bearer $TB"   # 403

# ❌ Un USER tente la liste admin -> 403
curl -s -o /dev/null -w "%{http_code}\n" $API/chat/conversations -H "Authorization: Bearer $TA"   # 403

# ✅ L'ADMIN voit toutes les conversations et peut répondre à chacune
curl -s $API/chat/conversations -H "Authorization: Bearer $TADM"                                   # 200, A + B
curl -s -X POST $API/chat/conversations/$CA/messages -H "Authorization: Bearer $TADM" -H 'Content-Type: application/json' -d '{"content":"Réponse support à A"}'
```

Résultat attendu :
- User A ne voit pas les messages de User B (403). ✅
- User B ne voit pas les messages de User A (403). ✅
- L'admin accède à toutes les conversations et répond. ✅
- Déconnexion/connexion : token effacé → aucune donnée de la session précédente. ✅
