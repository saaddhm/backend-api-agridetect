import uuid
import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlmodel import Session, func, select

from ..auth import get_current_user
from ..config import settings
from ..database import get_session
from ..ml import LABELS, class_info, predictor
from ..models import AdminNotification, AlertRecord, Analysis, User
from ..schemas import AnalysisResponse, StatsResponse

router = APIRouter(tags=["Analyses"])
logger = logging.getLogger("agridetect")


def _to_response(a: Analysis, top_k=None, lang: str | None = None) -> AnalysisResponse:
    meta = class_info(a.label, lang) if a.label in LABELS else {}
    return AnalysisResponse(
        id=a.id, plant=meta.get("plant", a.plant), disease=meta.get("disease", a.disease),
        label=a.label, severity=meta.get("severity", a.severity),
        confidence=a.confidence, is_healthy=meta.get("is_healthy", a.is_healthy),
        cause=meta.get("cause", a.cause), treatment=meta.get("treatment", a.treatment),
        backend=a.backend, image_url=f"{settings.API_PREFIX}/analyses/{a.id}/image",
        created_at=a.created_at, top_k=top_k,
    )


@router.post("/predict", response_model=AnalysisResponse)
async def predict(
    image: UploadFile = File(...),
    plant_hint: str | None = Form(default=None),
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
    accept_language: str | None = Header(default=None),
):
    if not current.scan_enabled:
        raise HTTPException(status_code=403, detail="Le scan est desactive pour ce compte.")

    content = await image.read()
    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image trop volumineuse.")
    if not content:
        raise HTTPException(status_code=400, detail="Image vide.")

    await image.seek(0)
    debug_path = Path(settings.UPLOAD_DIR, "debug_received.jpg")
    debug_path.write_bytes(content)
    logger.info(
        "Image upload received filename=%s content_type=%s bytes=%s plant_hint=%s debug=%s",
        image.filename,
        image.content_type,
        len(content),
        plant_hint,
        debug_path,
    )

    try:
        result = predictor.predict(content, top_k=3, lang=accept_language, plant_hint=plant_hint)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result.get("status") == "no_plant":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "NO_PLANT",
                "message": "No clear leaf detected. Please capture one plant leaf clearly.",
            },
        )
    if result.get("status") == "low_confidence":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "LOW_CONFIDENCE",
                "message": "Unable to identify the plant disease. Please take a clearer picture of the plant leaf.",
            },
        )

    fname = f"{uuid.uuid4().hex}_{image.filename or 'capture.jpg'}"
    Path(settings.UPLOAD_DIR, fname).write_bytes(content)

    analysis = Analysis(
        user_id=current.id, image_name=fname, label=result["label"], plant=result["plant"],
        disease=result["disease"], severity=result["severity"], confidence=result["confidence"],
        is_healthy=result["is_healthy"], cause=result.get("cause"), treatment=result.get("treatment"),
        backend=result.get("backend", "mock"),
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return _to_response(analysis, top_k=result.get("top_k"), lang=accept_language)


@router.get("/analyses", response_model=list[AnalysisResponse])
def history(
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
    accept_language: str | None = Header(default=None),
):
    rows = session.exec(
        select(Analysis).where(Analysis.user_id == current.id).order_by(Analysis.created_at.desc())
    ).all()
    return [_to_response(a, lang=accept_language) for a in rows]


@router.get("/analyses/stats", response_model=StatsResponse)
def stats(session: Session = Depends(get_session), current: User = Depends(get_current_user)):
    rows = session.exec(select(Analysis).where(Analysis.user_id == current.id)).all()
    total = len(rows)
    diseases = {a.disease for a in rows if not a.is_healthy}
    avg = round(sum(a.confidence for a in rows) / total, 4) if total else 0.0
    return StatsResponse(total_analyses=total, distinct_diseases=len(diseases), avg_confidence=avg)


@router.get("/alerts/epidemiological")
def epidemiological_alerts(
    days: int = 7,
    threshold: int = 3,
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
):
    """Detecte les maladies recurrentes recentes.

    Le projet ne stocke pas encore de GPS/zone utilisateur. On retourne donc une
    zone globale, prete a etre remplacee par une zone reelle si la localisation
    est ajoutee plus tard.
    """
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.exec(
        select(Analysis)
        .where(Analysis.user_id == current.id)  # alertes propres à l'utilisateur connecté
        .where(Analysis.is_healthy == False)  # noqa: E712
        .where(Analysis.created_at >= since)
    ).all()
    counter = Counter((a.plant, a.disease, a.severity) for a in rows)
    alerts = []
    for (plant, disease, severity), count in counter.most_common():
        if count < threshold:
            continue
        alerts.append({
            "id": f"{plant}-{disease}-{severity}".lower().replace(" ", "-"),
            "plant": plant,
            "disease": disease,
            "severity": severity,
            "count": count,
            "days": days,
            "threshold": threshold,
            "zone": "Zone globale",
            "message": f"{count} detections de {disease} sur {plant} durant les {days} derniers jours.",
            "created_at": datetime.utcnow(),
        })

    # Alertes créées par l'admin, visibles si globales OU ciblées sur cet utilisateur.
    stored = session.exec(
        select(AlertRecord)
        .where((AlertRecord.is_global == True) | (AlertRecord.user_id == current.id))  # noqa: E712
        .order_by(AlertRecord.created_at.desc())
    ).all()
    for a in stored:
        alerts.append({
            "id": f"admin-{a.id}",
            "plant": a.plant,
            "disease": a.disease,
            "severity": a.severity,
            "count": 0,
            "days": days,
            "threshold": threshold,
            "zone": a.zone,
            "message": a.message,
            "created_at": a.created_at,
        })
    return alerts


@router.get("/notifications")
def notifications(session: Session = Depends(get_session), current: User = Depends(get_current_user)):
    rows = session.exec(
        select(AdminNotification)
        .where(AdminNotification.user_id == current.id)
        .order_by(AdminNotification.created_at.desc())
    ).all()
    return [
        {
            "id": notification.id,
            "title": notification.title,
            "message": notification.message,
            "level": notification.level,
            "is_read": notification.is_read,
            "created_at": notification.created_at,
        }
        for notification in rows
    ]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def get_one(
    analysis_id: int,
    session: Session = Depends(get_session),
    current: User = Depends(get_current_user),
    accept_language: str | None = Header(default=None),
):
    a = session.get(Analysis, analysis_id)
    if not a or a.user_id != current.id:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    return _to_response(a, lang=accept_language)


@router.get("/analyses/{analysis_id}/image")
def get_image(analysis_id: int, session: Session = Depends(get_session), current: User = Depends(get_current_user)):
    from fastapi.responses import FileResponse
    a = session.get(Analysis, analysis_id)
    if not a or a.user_id != current.id:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    path = Path(settings.UPLOAD_DIR, a.image_name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image introuvable.")
    return FileResponse(path)


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete(analysis_id: int, session: Session = Depends(get_session), current: User = Depends(get_current_user)):
    a = session.get(Analysis, analysis_id)
    if not a or a.user_id != current.id:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    session.delete(a)
    session.commit()
