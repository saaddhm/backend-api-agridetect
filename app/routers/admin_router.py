from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlmodel import Session, func, select

from ..auth import get_current_admin, hash_password
from ..config import settings
from ..database import get_session
from ..models import AdminNotification, AlertRecord, Analysis, User

router = APIRouter(prefix="/admin", tags=["Administration"], dependencies=[Depends(get_current_admin)])


class AdminUserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: str = "USER"
    is_active: bool = True
    scan_enabled: bool = True


class AdminUserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    scan_enabled: Optional[bool] = None


class AlertCreate(BaseModel):
    plant: str = ""
    disease: str = ""
    severity: str = "MODEREE"
    zone: str = "Zone globale"
    message: str = ""
    is_global: bool = False
    user_id: Optional[int] = None  # requis si is_global == False


class AlertUpdate(BaseModel):
    plant: Optional[str] = None
    disease: Optional[str] = None
    severity: Optional[str] = None
    zone: Optional[str] = None
    message: Optional[str] = None
    is_global: Optional[bool] = None
    user_id: Optional[int] = None


class AdminAnalysisCreate(BaseModel):
    user_id: int
    label: str
    plant: str
    disease: str
    severity: str = "FAIBLE"
    confidence: float = 1.0
    is_healthy: bool = False
    cause: Optional[str] = None
    treatment: Optional[str] = None
    backend: str = "admin"
    image_name: str = "admin-created.jpg"


class AdminAnalysisUpdate(BaseModel):
    user_id: Optional[int] = None
    label: Optional[str] = None
    plant: Optional[str] = None
    disease: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    is_healthy: Optional[bool] = None
    cause: Optional[str] = None
    treatment: Optional[str] = None
    backend: Optional[str] = None
    image_name: Optional[str] = None


class AdminNotificationCreate(BaseModel):
    email: EmailStr
    title: str
    message: str
    level: str = "INFO"


class AdminNotificationUpdate(BaseModel):
    email: Optional[EmailStr] = None
    title: Optional[str] = None
    message: Optional[str] = None
    level: Optional[str] = None
    is_read: Optional[bool] = None


def _validate_role(role: str) -> str:
    normalized = role.upper()
    if normalized not in {"USER", "ADMIN"}:
        raise HTTPException(status_code=400, detail="Role invalide.")
    return normalized


def _validate_confidence(confidence: float) -> float:
    if confidence < 0 or confidence > 1:
        raise HTTPException(status_code=400, detail="La confiance doit etre entre 0 et 1.")
    return confidence


def _validate_notification_level(level: str) -> str:
    normalized = level.upper()
    if normalized not in {"INFO", "ALERTE", "URGENT"}:
        raise HTTPException(status_code=400, detail="Priorite de notification invalide.")
    return normalized


def _serialize_user(user: User, analyses_count: int = 0) -> dict:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "scan_enabled": user.scan_enabled,
        "analyses_count": analyses_count,
        "created_at": user.created_at,
    }


def _serialize_analysis(analysis: Analysis, emails: dict[int, str]) -> dict:
    return {
        "id": analysis.id,
        "user_id": analysis.user_id,
        "user_email": emails.get(analysis.user_id, "-"),
        "plant": analysis.plant,
        "disease": analysis.disease,
        "label": analysis.label,
        "severity": analysis.severity,
        "confidence": analysis.confidence,
        "is_healthy": analysis.is_healthy,
        "cause": analysis.cause,
        "treatment": analysis.treatment,
        "backend": analysis.backend,
        "image_name": analysis.image_name,
        "image_url": f"{settings.API_PREFIX}/analyses/{analysis.id}/image",
        "created_at": analysis.created_at,
    }


def _serialize_notification(notification: AdminNotification, emails: dict[int, str]) -> dict:
    return {
        "id": notification.id,
        "user_id": notification.user_id,
        "user_email": emails.get(notification.user_id, "-"),
        "title": notification.title,
        "message": notification.message,
        "level": notification.level,
        "is_read": notification.is_read,
        "created_at": notification.created_at,
    }


@router.get("/overview")
def overview(session: Session = Depends(get_session)):
    """Indicateurs globaux pour le tableau de bord."""
    users = session.exec(select(User)).all()
    analyses = session.exec(select(Analysis)).all()
    total = len(analyses)
    diseased = [a for a in analyses if not a.is_healthy]
    avg_conf = round(sum(a.confidence for a in analyses) / total, 4) if total else 0.0
    return {
        "users_count": len(users),
        "analyses_count": total,
        "diseases_detected": len(diseased),
        "distinct_diseases": len({a.disease for a in diseased}),
        "healthy_count": total - len(diseased),
        "avg_confidence": avg_conf,
    }


@router.get("/users")
def list_users(session: Session = Depends(get_session)):
    """Liste des utilisateurs avec leur nombre d'analyses."""
    counts = dict(session.exec(select(Analysis.user_id, func.count(Analysis.id)).group_by(Analysis.user_id)).all())
    users = session.exec(select(User).order_by(User.created_at.desc())).all()
    return [_serialize_user(user, counts.get(user.id, 0)) for user in users]


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(payload: AdminUserCreate, session: Session = Depends(get_session)):
    """Cree un utilisateur depuis la console admin."""
    exists = session.exec(select(User).where(User.email == payload.email)).first()
    if exists:
        raise HTTPException(status_code=409, detail="Cet e-mail existe deja.")
    user = User(
        full_name=payload.full_name.strip(),
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
        role=_validate_role(payload.role),
        is_active=payload.is_active,
        scan_enabled=payload.scan_enabled,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return _serialize_user(user)


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    payload: AdminUserUpdate,
    session: Session = Depends(get_session),
    current_admin: User = Depends(get_current_admin),
):
    """Met a jour un utilisateur depuis la console admin."""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    if payload.email is not None:
        email = str(payload.email).lower()
        exists = session.exec(select(User).where(User.email == email, User.id != user_id)).first()
        if exists:
            raise HTTPException(status_code=409, detail="Cet e-mail existe deja.")
        user.email = email
    if payload.full_name is not None:
        user.full_name = payload.full_name.strip()
    if payload.password:
        user.password_hash = hash_password(payload.password)
    if payload.role is not None:
        user.role = _validate_role(payload.role)
    if payload.is_active is not None:
        if current_admin.id == user_id and payload.is_active is False:
            raise HTTPException(status_code=400, detail="Vous ne pouvez pas desactiver votre propre compte.")
        user.is_active = payload.is_active
    if payload.scan_enabled is not None:
        user.scan_enabled = payload.scan_enabled
    session.add(user)
    session.commit()
    session.refresh(user)
    count = session.exec(select(func.count(Analysis.id)).where(Analysis.user_id == user.id)).one()
    return _serialize_user(user, count)


@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int,
    session: Session = Depends(get_session),
    current_admin: User = Depends(get_current_admin),
):
    """Supprime un utilisateur et ses analyses."""
    if current_admin.id == user_id:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas supprimer votre propre compte.")
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    analyses = session.exec(select(Analysis).where(Analysis.user_id == user_id)).all()
    for analysis in analyses:
        path = Path(settings.UPLOAD_DIR, analysis.image_name)
        if path.exists():
            path.unlink()
        session.delete(analysis)
    notifications = session.exec(select(AdminNotification).where(AdminNotification.user_id == user_id)).all()
    for notification in notifications:
        session.delete(notification)
    session.delete(user)
    session.commit()


@router.get("/notifications")
def list_notifications(limit: int = 100, session: Session = Depends(get_session)):
    """Messages envoyes aux utilisateurs depuis la console admin."""
    rows = session.exec(
        select(AdminNotification).order_by(AdminNotification.created_at.desc()).limit(limit)
    ).all()
    emails = {user.id: user.email for user in session.exec(select(User)).all()}
    return [_serialize_notification(notification, emails) for notification in rows]


@router.post("/notifications", status_code=status.HTTP_201_CREATED)
def send_notification(payload: AdminNotificationCreate, session: Session = Depends(get_session)):
    """Envoie un message visible dans l'ecran Alertes du client mobile."""
    user = session.exec(select(User).where(User.email == str(payload.email).lower())).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable pour cet e-mail.")
    notification = AdminNotification(
        user_id=user.id,
        title=payload.title.strip(),
        message=payload.message.strip(),
        level=_validate_notification_level(payload.level),
    )
    session.add(notification)
    session.commit()
    session.refresh(notification)
    return _serialize_notification(notification, {user.id: user.email})


@router.put("/notifications/{notification_id}")
def update_notification(
    notification_id: int,
    payload: AdminNotificationUpdate,
    session: Session = Depends(get_session),
):
    """Met a jour une notification envoyee depuis la console admin."""
    notification = session.get(AdminNotification, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification introuvable.")

    if payload.email is not None:
        user = session.exec(select(User).where(User.email == str(payload.email).lower())).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable pour cet e-mail.")
        notification.user_id = user.id
    if payload.title is not None:
        notification.title = payload.title.strip()
    if payload.message is not None:
        notification.message = payload.message.strip()
    if payload.level is not None:
        notification.level = _validate_notification_level(payload.level)
    if payload.is_read is not None:
        notification.is_read = payload.is_read

    session.add(notification)
    session.commit()
    session.refresh(notification)
    emails = {user.id: user.email for user in session.exec(select(User)).all()}
    return _serialize_notification(notification, emails)


@router.delete("/notifications/{notification_id}", status_code=204)
def delete_notification(notification_id: int, session: Session = Depends(get_session)):
    """Supprime une notification depuis la console admin."""
    notification = session.get(AdminNotification, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification introuvable.")
    session.delete(notification)
    session.commit()


@router.get("/analyses")
def all_analyses(limit: int = 200, session: Session = Depends(get_session)):
    """Toutes les analyses, tous utilisateurs confondus."""
    rows = session.exec(select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)).all()
    emails = {user.id: user.email for user in session.exec(select(User)).all()}
    return [_serialize_analysis(analysis, emails) for analysis in rows]


@router.post("/analyses", status_code=status.HTTP_201_CREATED)
def create_analysis(payload: AdminAnalysisCreate, session: Session = Depends(get_session)):
    """Cree une analyse manuelle depuis la console admin."""
    user = session.get(User, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
    analysis = Analysis(
        user_id=payload.user_id,
        image_name=payload.image_name,
        label=payload.label,
        plant=payload.plant,
        disease=payload.disease,
        severity=payload.severity.upper(),
        confidence=_validate_confidence(payload.confidence),
        is_healthy=payload.is_healthy,
        cause=payload.cause,
        treatment=payload.treatment,
        backend=payload.backend,
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    return _serialize_analysis(analysis, {user.id: user.email})


@router.put("/analyses/{analysis_id}")
def update_analysis(analysis_id: int, payload: AdminAnalysisUpdate, session: Session = Depends(get_session)):
    """Met a jour une analyse depuis la console admin."""
    analysis = session.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    if payload.user_id is not None:
        user = session.get(User, payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur introuvable.")
        analysis.user_id = payload.user_id
    for field in ("label", "plant", "disease", "cause", "treatment", "backend", "image_name"):
        value = getattr(payload, field)
        if value is not None:
            setattr(analysis, field, value)
    if payload.severity is not None:
        analysis.severity = payload.severity.upper()
    if payload.confidence is not None:
        analysis.confidence = _validate_confidence(payload.confidence)
    if payload.is_healthy is not None:
        analysis.is_healthy = payload.is_healthy
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    emails = {user.id: user.email for user in session.exec(select(User)).all()}
    return _serialize_analysis(analysis, emails)


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: int, session: Session = Depends(get_session)):
    """Supprime une analyse depuis la console admin."""
    analysis = session.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analyse introuvable.")
    path = Path(settings.UPLOAD_DIR, analysis.image_name)
    if path.exists():
        path.unlink()
    session.delete(analysis)
    session.commit()


# ---------------------------------------------------------------------------
# CRUD des alertes épidémiologiques (admin)
# ---------------------------------------------------------------------------
def _serialize_alert(a: AlertRecord) -> dict:
    return {
        "id": a.id,
        "plant": a.plant,
        "disease": a.disease,
        "severity": a.severity,
        "zone": a.zone,
        "message": a.message,
        "user_id": a.user_id,
        "is_global": a.is_global,
        "created_at": a.created_at.isoformat(),
    }


@router.get("/alerts")
def list_alerts(session: Session = Depends(get_session)):
    rows = session.exec(select(AlertRecord).order_by(AlertRecord.created_at.desc())).all()
    return [_serialize_alert(a) for a in rows]


@router.post("/alerts", status_code=status.HTTP_201_CREATED)
def create_alert(payload: AlertCreate, session: Session = Depends(get_session)):
    if not payload.is_global and payload.user_id is None:
        raise HTTPException(status_code=400, detail="user_id requis pour une alerte ciblée (non globale).")
    if payload.user_id is not None and not session.get(User, payload.user_id):
        raise HTTPException(status_code=404, detail="Utilisateur cible introuvable.")
    alert = AlertRecord(
        plant=payload.plant.strip(),
        disease=payload.disease.strip(),
        severity=payload.severity,
        zone=payload.zone.strip() or "Zone globale",
        message=payload.message.strip(),
        is_global=payload.is_global,
        user_id=None if payload.is_global else payload.user_id,
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return _serialize_alert(alert)


@router.put("/alerts/{alert_id}")
def update_alert(alert_id: int, payload: AlertUpdate, session: Session = Depends(get_session)):
    alert = session.get(AlertRecord, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte introuvable.")
    if payload.plant is not None:
        alert.plant = payload.plant.strip()
    if payload.disease is not None:
        alert.disease = payload.disease.strip()
    if payload.severity is not None:
        alert.severity = payload.severity
    if payload.zone is not None:
        alert.zone = payload.zone.strip() or "Zone globale"
    if payload.message is not None:
        alert.message = payload.message.strip()
    if payload.is_global is not None:
        alert.is_global = payload.is_global
        if payload.is_global:
            alert.user_id = None
    if payload.user_id is not None and not alert.is_global:
        alert.user_id = payload.user_id
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return _serialize_alert(alert)


@router.delete("/alerts/{alert_id}", status_code=204)
def delete_alert(alert_id: int, session: Session = Depends(get_session)):
    alert = session.get(AlertRecord, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alerte introuvable.")
    session.delete(alert)
    session.commit()


@router.get("/analytics/top-diseases")
def top_diseases(session: Session = Depends(get_session)):
    """Maladies les plus frequentes (hors plantes saines)."""
    rows = session.exec(select(Analysis).where(Analysis.is_healthy == False)).all()  # noqa: E712
    counter = Counter(f"{analysis.plant} - {analysis.disease}" for analysis in rows)
    return [{"name": key, "count": value} for key, value in counter.most_common(8)]


@router.get("/analytics/timeline")
def timeline(days: int = 14, session: Session = Depends(get_session)):
    """Nombre d'analyses par jour sur les N derniers jours."""
    since = datetime.utcnow().date() - timedelta(days=days - 1)
    rows = session.exec(select(Analysis)).all()
    buckets = {(since + timedelta(days=i)).isoformat(): 0 for i in range(days)}
    for analysis in rows:
        key = analysis.created_at.date().isoformat()
        if key in buckets:
            buckets[key] += 1
    return [{"date": date, "count": count} for date, count in buckets.items()]
