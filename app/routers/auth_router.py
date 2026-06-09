import secrets
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from ..auth import create_access_token, get_current_user, hash_password, verify_password
from ..database import get_session
from ..email import send_verification_email
from ..models import User
from ..schemas import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentification"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    email = str(body.email).lower()
    exists = session.exec(select(User).where(User.email == email)).first()
    if exists:
        raise HTTPException(status_code=409, detail="Cet e-mail est deja utilise.")

    verification_token = secrets.token_urlsafe(32)
    user = User(
        full_name=body.full_name.strip(),
        email=email,
        password_hash=hash_password(body.password),
        email_verified=False,
        email_verification_token=verification_token,
        email_verified_at=None,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    background_tasks.add_task(
        send_verification_email,
        user.email,
        user.full_name,
        verification_token,
    )

    return RegisterResponse(
        message="Compte cree. Verifiez votre e-mail pour activer votre compte.",
        email=user.email,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, session: Session = Depends(get_session)):
    email = str(body.email).lower()
    user = session.exec(select(User).where(User.email == email)).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="E-mail ou mot de passe incorrect.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Compte desactive. Contactez l'administrateur.")
    if not user.email_verified:
        raise HTTPException(status_code=403, detail="Veuillez verifier votre e-mail avant de vous connecter.")

    token = create_access_token(user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get("/verify-email", response_class=HTMLResponse)
def verify_email(token: str, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email_verification_token == token)).first()
    if not user:
        return HTMLResponse(
            content="""
            <html><body style="font-family: Arial, sans-serif; padding: 32px;">
              <h2>Lien invalide</h2>
              <p>Ce lien de verification est invalide ou deja utilise.</p>
            </body></html>
            """,
            status_code=400,
        )

    user.email_verified = True
    user.email_verification_token = None
    user.email_verified_at = datetime.utcnow()
    session.add(user)
    session.commit()

    return """
    <html><body style="font-family: Arial, sans-serif; padding: 32px; color: #102018;">
      <h2>E-mail verifie</h2>
      <p>Votre compte AgriDetect AI est maintenant active. Vous pouvez vous connecter dans l'application mobile.</p>
    </body></html>
    """


@router.get("/me", response_model=UserResponse)
def me(current: User = Depends(get_current_user)):
    return UserResponse.model_validate(current)
