from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..auth import get_current_admin
from ..database import get_session
from ..ml import CLASSES_INFO
from ..models import DiseaseTreatment, User

router = APIRouter(prefix="/treatments", tags=["Traitements"])

DISCLAIMER = (
    "Les recommandations de traitement sont fournies à titre informatif uniquement. "
    "Consultez toujours un expert agricole local et suivez les instructions de l'étiquette "
    "du produit avant toute application."
)


# --------------------------- Schémas ---------------------------
class TreatmentBody(BaseModel):
    label: str
    disease_name: str
    disease_name_fr: Optional[str] = None
    disease_name_ar: Optional[str] = None
    plant_name: str = ""
    description: str = ""
    symptoms: List[str] = []
    causes: List[str] = []
    prevention: List[str] = []
    organic_treatment: List[str] = []
    chemical_treatment: List[str] = []
    recommended_products: List[dict] = []
    severity: str = "MODEREE"
    recovery_time: str = ""
    expert_advice: str = ""


def _to_dict(t: DiseaseTreatment) -> dict:
    return {
        "id": t.id,
        "label": t.label,
        "disease": t.disease_name,
        "disease_fr": t.disease_name_fr,
        "disease_ar": t.disease_name_ar,
        "plant": t.plant_name,
        "description": t.description,
        "symptoms": t.symptoms or [],
        "causes": t.causes or [],
        "prevention": t.prevention or [],
        "organic_treatment": t.organic_treatment or [],
        "chemical_treatment": t.chemical_treatment or [],
        "recommended_products": t.recommended_products or [],
        "severity": t.severity,
        "recovery_time": t.recovery_time,
        "expert_advice": t.expert_advice,
        "disclaimer": DISCLAIMER,
    }


def _fallback(key: str) -> Optional[dict]:
    """Si la maladie n'est pas en base, déduit une fiche minimale des métadonnées IA."""
    k = key.lower().replace("_", " ").strip()
    for info in CLASSES_INFO:
        if info["is_healthy"]:
            continue
        if (info["label"].lower() == key.lower()
                or info["disease"].lower() == k
                or k in info["label"].lower().replace("_", " ")):
            return {
                "id": None, "label": info["label"], "disease": info["disease"],
                "disease_fr": info["disease"], "disease_ar": info["disease"],
                "plant": info["plant"], "description": info.get("cause", ""),
                "symptoms": [], "causes": [info["cause"]] if info.get("cause") else [],
                "prevention": [], "organic_treatment": [info["treatment"]] if info.get("treatment") else [],
                "chemical_treatment": [], "recommended_products": [],
                "severity": info["severity"], "recovery_time": "À préciser",
                "expert_advice": info.get("treatment", ""), "disclaimer": DISCLAIMER,
            }
    return None


# --------------------------- Public ---------------------------
@router.get("/{disease_key}")
def get_treatment(disease_key: str, session: Session = Depends(get_session)):
    """Fiche de traitement par label OU nom de maladie (FR / EN / AR)."""
    key = disease_key.strip()
    stmt = select(DiseaseTreatment).where(
        (DiseaseTreatment.label == key)
        | (DiseaseTreatment.disease_name == key)
        | (DiseaseTreatment.disease_name_fr == key)
        | (DiseaseTreatment.disease_name_ar == key)
    )
    t = session.exec(stmt).first()
    if t:
        return _to_dict(t)
    fb = _fallback(key)
    if fb:
        return fb
    raise HTTPException(status_code=404, detail="Aucune fiche de traitement pour cette maladie.")


@router.get("")
def list_treatments(session: Session = Depends(get_session)):
    rows = session.exec(select(DiseaseTreatment).order_by(DiseaseTreatment.disease_name)).all()
    return [_to_dict(t) for t in rows]


# --------------------------- Admin CRUD ---------------------------
@router.post("", status_code=201, dependencies=[Depends(get_current_admin)])
def create_treatment(body: TreatmentBody, session: Session = Depends(get_session)):
    if session.exec(select(DiseaseTreatment).where(DiseaseTreatment.label == body.label)).first():
        raise HTTPException(status_code=409, detail="Une fiche existe déjà pour ce label.")
    t = DiseaseTreatment(**body.model_dump())
    session.add(t)
    session.commit()
    session.refresh(t)
    return _to_dict(t)


@router.put("/{treatment_id}", dependencies=[Depends(get_current_admin)])
def update_treatment(treatment_id: int, body: TreatmentBody, session: Session = Depends(get_session)):
    t = session.get(DiseaseTreatment, treatment_id)
    if not t:
        raise HTTPException(status_code=404, detail="Fiche introuvable.")
    for k, v in body.model_dump().items():
        setattr(t, k, v)
    t.updated_at = datetime.utcnow()
    session.add(t)
    session.commit()
    session.refresh(t)
    return _to_dict(t)


@router.delete("/{treatment_id}", status_code=204, dependencies=[Depends(get_current_admin)])
def delete_treatment(treatment_id: int, session: Session = Depends(get_session)):
    t = session.get(DiseaseTreatment, treatment_id)
    if not t:
        raise HTTPException(status_code=404, detail="Fiche introuvable.")
    session.delete(t)
    session.commit()
