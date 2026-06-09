"""Test d'integration de bout en bout."""
import io
import uuid

from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session, select

from app.database import engine, init_db
from app.main import app
from app.models import User

init_db()
client = TestClient(app)


def _fake_image() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (60, 140, 70)).save(buf, format="JPEG")
    return buf.getvalue()


def test_full_flow():
    assert client.get("/health").json()["status"] == "ok"

    email = f"saad-{uuid.uuid4().hex[:8]}@test.io"
    r = client.post(
        "/api/auth/register",
        json={"full_name": "Saad", "email": email, "password": "Secret@123"},
    )
    assert r.status_code == 201, r.text

    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        assert user and user.email_verification_token
        verify_token = user.email_verification_token

    assert client.get(f"/api/auth/verify-email?token={verify_token}").status_code == 200

    r = client.post("/api/auth/login", json={"email": email, "password": "Secret@123"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    h = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/auth/me", headers=h).json()["email"] == email
    assert client.get("/api/catalog/classes").json()["count"] == 39
    r = client.post("/api/predict", headers=h, files={"image": ("leaf.jpg", _fake_image(), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"plant", "disease", "confidence", "severity", "treatment"} <= set(body)
    assert body["top_k"] and len(body["top_k"]) == 3
    aid = body["id"]
    assert len(client.get("/api/analyses", headers=h).json()) >= 1
    assert client.get("/api/analyses/stats", headers=h).json()["total_analyses"] >= 1
    assert client.get(f"/api/analyses/{aid}", headers=h).status_code == 200
    assert client.delete(f"/api/analyses/{aid}", headers=h).status_code == 204
    print(
        "OK - plante:",
        body["plant"],
        "| maladie:",
        body["disease"],
        "| confiance:",
        body["confidence"],
        "| backend:",
        body["backend"],
    )


if __name__ == "__main__":
    test_full_flow()
