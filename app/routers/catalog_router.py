from fastapi import APIRouter, Header

from ..ml import LABELS, class_info

router = APIRouter(prefix="/catalog", tags=["Catalogue"])


@router.get("/classes")
def list_classes(accept_language: str | None = Header(default=None)):
    """Liste des classes reconnues par le modele, avec metadonnees traduites."""
    classes = [class_info(label, accept_language) for label in LABELS]
    return {"count": len(classes), "classes": classes}


@router.get("/diseases")
def list_diseases(accept_language: str | None = Header(default=None)):
    """Maladies hors classes saines."""
    diseases = [class_info(label, accept_language) for label in LABELS]
    diseases = [item for item in diseases if not item["is_healthy"]]
    return {"count": len(diseases), "diseases": diseases}
