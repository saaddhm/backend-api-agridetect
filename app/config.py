import os

class Settings:
    APP_NAME = "AgriDetect AI — API"
    API_PREFIX = "/api"
    JWT_SECRET = os.getenv("JWT_SECRET", "change-this-very-long-secret-key-at-least-256-bits-long")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))  # 24 h
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./agridetect.db")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "http://localhost:8000/api").rstrip("/")
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "noreply@agridetect.ai")
    SMTP_TLS = os.getenv("SMTP_TLS", "true").lower() in {"1", "true", "yes", "on"}

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
