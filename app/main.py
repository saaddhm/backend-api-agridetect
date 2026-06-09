import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .ml import predictor
from .chat.router import router as chat_router
from .routers import (admin_router, analyses_router, auth_router, catalog_router,
                      treatments_router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agridetect")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="API REST de détection des maladies des plantes (AgriDetect AI). "
                "Réutilise le modèle CNN Keras (39 classes, entrée 160×160).",
)

app.description = (
    "API REST de detection des maladies des plantes (AgriDetect AI). "
    "Pipeline: YOLO model.pt puis agridetect_model.keras."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  DEBUG : journalise précisément toute erreur de validation 422
#  (Content-Type reçu + corps brut + champs en erreur).
#  Très utile pour diagnostiquer un 422 ; à conserver ou retirer ensuite.
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw_body = await request.body()
    content_type = request.headers.get("content-type")
    logger.warning(
        "422 sur %s\n  Content-Type : %s\n  Corps reçu   : %s\n  Erreurs      : %s",
        request.url.path,
        content_type,
        raw_body.decode("utf-8", errors="replace")[:1000],
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "received_content_type": content_type,
            "hint": "Le corps doit être du JSON avec l'en-tête 'Content-Type: application/json'.",
        },
    )


app.include_router(auth_router.router, prefix=settings.API_PREFIX)
app.include_router(analyses_router.router, prefix=settings.API_PREFIX)
app.include_router(catalog_router.router, prefix=settings.API_PREFIX)
app.include_router(admin_router.router, prefix=settings.API_PREFIX)
app.include_router(treatments_router.router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)


def _seed_admin():
    """Crée un compte administrateur par défaut s'il n'en existe aucun."""
    from sqlmodel import Session, select
    from .auth import hash_password
    from .database import engine
    from .models import User
    with Session(engine) as s:
        exists = s.exec(select(User).where(User.role == "ADMIN")).first()
        if not exists:
            s.add(User(full_name="Administrateur", email="admin@agridetect.ai",
                       password_hash=hash_password("Admin@123"), role="ADMIN"))
            s.commit()
            logger.info("Compte admin par défaut créé : admin@agridetect.ai / Admin@123")


def _seed_treatments():
    from sqlmodel import Session
    from .database import engine
    from .treatments_seed import seed_treatments
    with Session(engine) as s:
        seed_treatments(s)


@app.on_event("startup")
def on_startup():
    init_db()
    _seed_admin()
    _seed_treatments()


@app.get("/health", tags=["Système"])
def health():
    predictor._try_load()
    from .ml import (
        LABELS,
        MIN_CENTER_LEAF_SIGNAL,
        MIN_LEAF_SIGNAL,
        MODEL_PATH,
        MODEL_VERSION,
        PLANT_DETECTOR_PATH,
    )
    import os
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_backend": predictor.backend,
        "disease_model_file": os.path.basename(MODEL_PATH),
        "disease_model_exists": os.path.exists(MODEL_PATH),
        "plant_detector_file": os.path.basename(PLANT_DETECTOR_PATH),
        "plant_detector_exists": os.path.exists(PLANT_DETECTOR_PATH),
        "plant_detector_classes": predictor.detector_names,
        "img_size": list(predictor.img_size),
        "num_classes": len(LABELS),
        "labels": LABELS,
        "leaf_signal_thresholds": {
            "whole": MIN_LEAF_SIGNAL,
            "center": MIN_CENTER_LEAF_SIGNAL,
        },
        "has_background_class": "Background_without_leaves" in LABELS,
        "model_error": getattr(predictor, "_model_error", None),
    }


@app.get("/", tags=["Système"])
def root():
    return {"app": settings.APP_NAME, "docs": "/docs", "health": "/health"}
