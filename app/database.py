from sqlalchemy import inspect, text
from sqlmodel import SQLModel, create_engine, Session
from .config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=connect_args)


def init_db() -> None:
    # importer les modèles pour les enregistrer dans les métadonnées
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    _ensure_user_boolean_column("is_active")
    _ensure_user_boolean_column("scan_enabled")
    _ensure_user_boolean_column("email_verified")
    _ensure_user_nullable_column("email_verification_token", "VARCHAR(255)")
    _ensure_user_nullable_column("email_verified_at", "DATETIME")


def _ensure_user_boolean_column(name: str) -> None:
    """Ajoute les petites migrations necessaires aux bases existantes."""
    inspector = inspect(engine)
    if not inspector.has_table("user"):
        return
    columns = {column["name"] for column in inspector.get_columns("user")}
    if name in columns:
        return
    default = "1" if engine.dialect.name == "sqlite" else "TRUE"
    with engine.begin() as connection:
        connection.execute(text(f'ALTER TABLE "user" ADD COLUMN {name} BOOLEAN NOT NULL DEFAULT {default}'))


def _ensure_user_nullable_column(name: str, column_type: str) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("user"):
        return
    columns = {column["name"] for column in inspector.get_columns("user")}
    if name in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(f'ALTER TABLE "user" ADD COLUMN {name} {column_type} NULL'))


def get_session():
    with Session(engine) as session:
        yield session
