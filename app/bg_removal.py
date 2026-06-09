"""
Suppression automatique de l'arrière-plan (isole la feuille sur fond blanc).

Stratégie en cascade, sans dépendance problématique :
  1. rembg (U2Net) s'il est installé           -> meilleure qualité
  2. sinon OpenCV GrabCut                       -> bon, AUCUN conflit numpy/TensorFlow
  3. sinon l'image d'origine est conservée      -> jamais d'erreur

Activé/désactivé via la variable d'environnement REMOVE_BG (true par défaut).
Pour activer la méthode OpenCV (recommandée) : pip install opencv-python-headless
"""
import io
import os

REMOVE_BG = os.getenv("REMOVE_BG", "true").lower() == "true"
_rembg_session = None


def _via_rembg(image_bytes: bytes) -> bytes:
    from rembg import new_session, remove
    from PIL import Image
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2netp")
    cut = remove(image_bytes, session=_rembg_session)
    fg = Image.open(io.BytesIO(cut)).convert("RGBA")
    white = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white, fg).convert("RGB")
    out = io.BytesIO()
    composed.save(out, format="JPEG", quality=90)
    return out.getvalue()


def _via_opencv(image_bytes: bytes) -> bytes:
    import cv2
    import numpy as np

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx, my = int(w * 0.06), int(h * 0.06)
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    white = np.full_like(img, 255)
    out = np.where(fg[:, :, None] == 255, img, white)
    ok, buf = cv2.imencode(".jpg", out)
    return buf.tobytes() if ok else image_bytes


def remove_background(image_bytes: bytes) -> bytes:
    if not REMOVE_BG:
        return image_bytes
    try:
        import rembg  # noqa: F401
        return _via_rembg(image_bytes)
    except Exception:
        pass
    try:
        import cv2  # noqa: F401
        return _via_opencv(image_bytes)
    except Exception as e:
        print(f"[bg] suppression indisponible ({e}); image conservee.")
        return image_bytes
