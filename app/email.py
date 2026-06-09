import logging
import smtplib
from email.message import EmailMessage
from html import escape

from .config import settings

logger = logging.getLogger("agridetect")


def send_verification_email(email: str, full_name: str, token: str) -> None:
    verify_url = f"{settings.PUBLIC_API_URL}/auth/verify-email?token={token}"
    display_name = escape(full_name or "AgriDetect user")

    if not settings.SMTP_HOST:
        logger.warning(
            "SMTP is not configured. Email verification link for %s: %s",
            email,
            verify_url,
        )
        return

    message = EmailMessage()
    message["Subject"] = "Verifiez votre adresse e-mail AgriDetect AI"
    message["From"] = settings.SMTP_FROM
    message["To"] = email
    message.set_content(
        "\n".join(
            [
                f"Bonjour {full_name},",
                "",
                "Merci de creer votre compte AgriDetect AI.",
                "Cliquez sur le lien ci-dessous pour verifier votre adresse e-mail :",
                verify_url,
                "",
                "Si vous n'avez pas cree ce compte, ignorez cet e-mail.",
            ]
        )
    )
    message.add_alternative(
        f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #102018;">
            <h2>Bienvenue sur AgriDetect AI</h2>
            <p>Bonjour {display_name},</p>
            <p>Merci de creer votre compte. Cliquez sur le bouton ci-dessous pour verifier votre adresse e-mail.</p>
            <p>
              <a href="{verify_url}" style="display:inline-block;padding:12px 18px;background:#167145;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:bold;">
                Verifier mon e-mail
              </a>
            </p>
            <p>Si le bouton ne fonctionne pas, copiez ce lien dans votre navigateur :</p>
            <p>{verify_url}</p>
          </body>
        </html>
        """,
        subtype="html",
    )

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
        if settings.SMTP_TLS:
            smtp.starttls()
        if settings.SMTP_USER:
            smtp.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        smtp.send_message(message)
