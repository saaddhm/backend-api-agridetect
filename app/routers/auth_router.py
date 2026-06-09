import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select

from ..auth import create_access_token, get_current_user, hash_password, verify_password
from ..database import get_session
from ..email import send_verification_code, send_verification_email
from ..models import User
from ..schemas import (
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    ResendCodeRequest,
    TokenResponse,
    UserResponse,
    VerifyCodeRequest,
)

router = APIRouter(prefix="/auth", tags=["Authentification"])

CODE_TTL_MINUTES = 15


def _new_code() -> str:
    """Code de verification a 6 chiffres."""
    return f"{secrets.randbelow(1_000_000):06d}"


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(
    body: RegisterRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    email = str(body.email).lower()
    exists = session.exec(select(User).where(User.email == email)).first()
    if exists:
        # Si le compte existe mais n'est pas verifie, on renvoie un nouveau code.
        if not exists.email_verified:
            code = _new_code()
            exists.email_verification_code = code
            exists.email_code_expires_at = datetime.utcnow() + timedelta(minutes=CODE_TTL_MINUTES)
            session.add(exists)
            session.commit()
            background_tasks.add_task(send_verification_code, exists.email, exists.full_name, code)
            return RegisterResponse(
                message="Un nouveau code de verification vous a ete envoye par e-mail.",
                email=exists.email,
            )
        raise HTTPException(status_code=409, detail="Cet e-mail est deja utilise.")

    code = _new_code()
    user = User(
        full_name=body.full_name.strip(),
        email=email,
        password_hash=hash_password(body.password),
        email_verified=False,
        email_verification_code=code,
        email_code_expires_at=datetime.utcnow() + timedelta(minutes=CODE_TTL_MINUTES),
        email_verified_at=None,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    background_tasks.add_task(send_verification_code, user.email, user.full_name, code)

    return RegisterResponse(
        message="Compte cree. Saisissez le code recu par e-mail pour activer votre compte.",
        email=user.email,
    )


@router.post("/verify-code", response_model=TokenResponse)
def verify_code(body: VerifyCodeRequest, session: Session = Depends(get_session)):
    email = str(body.email).lower()
    code = body.code.strip()
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="Compte introuvable.")
    if user.email_verified:
        # Deja verifie : on connecte directement.
        token = create_access_token(user.email)
        return TokenResponse(access_token=token, user=UserResponse.model_validate(user))
    if not user.email_verification_code or not user.email_code_expires_at:
        raise HTTPException(status_code=400, detail="Aucun code en attente. Renvoyez un code.")
    if datetime.utcnow() > user.email_code_expires_at:
        raise HTTPException(status_code=400, detail="Code expire. Renvoyez un nouveau code.")
    if code != user.email_verification_code:
        raise HTTPException(status_code=400, detail="Code incorrect.")

    user.email_verified = True
    user.email_verified_at = datetime.utcnow()
    user.email_verification_code = None
    user.email_code_expires_at = None
    user.email_verification_token = None
    session.add(user)
    session.commit()
    session.refresh(user)

    token = create_access_token(user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/resend-code", response_model=RegisterResponse)
def resend_code(
    body: ResendCodeRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
):
    email = str(body.email).lower()
    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="Compte introuvable.")
    if user.email_verified:
        return RegisterResponse(message="Ce compte est deja verifie.", email=user.email)

    code = _new_code()
    user.email_verification_code = code
    user.email_code_expires_at = datetime.utcnow() + timedelta(minutes=CODE_TTL_MINUTES)
    session.add(user)
    session.commit()
    background_tasks.add_task(send_verification_code, user.email, user.full_name, code)
    return RegisterResponse(
        message="Un nouveau code vous a ete envoye par e-mail.",
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
