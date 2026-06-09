# AgriDetect AI — Backend API (FastAPI)

API REST de détection des maladies des plantes. **Réutilise le modèle CNN Keras existant**
(`plant_disease_recog_model_pwp.keras`, entrée 160×160, 39 classes) issu du projet
*Plant-Disease-Recognition-System*, et l'expose à l'application mobile Flutter.

## Fonctionnalités

- 🔐 **Authentification JWT** (inscription, connexion, BCrypt, rôles)
- 🧠 **Détection IA** : `POST /api/predict` → plante, maladie, confiance, sévérité, cause, traitement + top-3
- 🗂️ **Historique** des analyses par utilisateur (SQLite par défaut, PostgreSQL en option)
- 📊 **Statistiques** (nombre d'analyses, maladies distinctes, confiance moyenne)
- 📚 **Catalogue** des 39 classes reconnues, métadonnées en français
- 🧪 **Mode mock déterministe** : l'API fonctionne sans le modèle de 204 Mo (démo / CI)

## Installation

```bash
cd backend-api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Brancher le vrai modèle (optionnel mais recommandé)

```bash
pip install tensorflow-cpu==2.16.1
# copier le modèle existant dans backend-api/models/
cp "/chemin/vers/Plant-Disease-Recognition-System/models/plant_disease_recog_model_pwp.keras" models/
```

Sans cette étape, l'API démarre en **mode mock** (prédictions déterministes basées sur l'image) :
pratique pour développer l'app Flutter sans charger le modèle.

## Lancement

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Documentation interactive : http://localhost:8000/docs
- Santé / backend actif : http://localhost:8000/health

### Verification e-mail

Configurez ces variables sur le VPS pour envoyer les e-mails de verification :

```bash
PUBLIC_API_URL=http://2.24.15.70:8000/api
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-user@example.com
SMTP_PASSWORD=your-password
SMTP_FROM=noreply@agridetect.ai
SMTP_TLS=true
```

Si `SMTP_HOST` n'est pas configure, le backend journalise le lien de verification dans les logs.

## Endpoints principaux

| Méthode | Endpoint | Auth | Description |
|---------|----------|------|-------------|
| POST | `/api/auth/register` | — | Inscription + envoi e-mail de verification |
| GET | `/api/auth/verify-email?token=...` | — | Verification de l'adresse e-mail |
| POST | `/api/auth/login` | — | Connexion (retourne un JWT) |
| GET | `/api/auth/me` | ✅ | Profil courant |
| POST | `/api/predict` | ✅ | Analyser une image (multipart `image`) |
| GET | `/api/analyses` | ✅ | Historique de l'utilisateur |
| GET | `/api/analyses/stats` | ✅ | Statistiques |
| GET | `/api/analyses/{id}` | ✅ | Détail d'une analyse |
| GET | `/api/analyses/{id}/image` | ✅ | Image de l'analyse |
| DELETE | `/api/analyses/{id}` | ✅ | Supprimer une analyse |
| GET | `/api/catalog/classes` | — | Les 39 classes du modèle |

### Exemple — prédiction

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/auth/login -H "Content-Type: application/json" \
  -d '{"email":"saad@test.io","password":"Secret@123"}' | jq -r .access_token)

curl -X POST localhost:8000/api/predict -H "Authorization: Bearer $TOKEN" \
  -F "image=@feuille.jpg"
```

Réponse :

```json
{
  "id": 1, "plant": "Tomate", "disease": "Mildiou", "label": "Tomato___Late_blight",
  "severity": "ELEVEE", "confidence": 0.94, "is_healthy": false,
  "cause": "Fungus Phytophthora infestans...",
  "treatment": "Bouillie bordelaise en préventif (10-20 g/L)...",
  "backend": "tensorflow",
  "top_k": [ { "disease": "Mildiou", "confidence": 0.94 }, ... ]
}
```

## Tests

```bash
PYTHONPATH=. python tests/test_api.py     # test d'intégration de bout en bout
```

## Docker

```bash
docker build -t agridetect-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models agridetect-api
```

## Stack

FastAPI · SQLModel · python-jose (JWT) · passlib/bcrypt · Pillow · NumPy · TensorFlow (modèle).
