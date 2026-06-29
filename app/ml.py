"""
Inference pipeline for AgriDetect AI.

Workflow:
1. YOLO model.pt validates that a plant leaf is present.
2. agridetect_model.keras classifies the disease only if YOLO detected a leaf.

This avoids forcing PlantVillage disease predictions on laptops, screens,
walls, people, desks, and other non-plant images.
"""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

# Système de versioning : l'application charge TOUJOURS current_model.keras
# (copie du modèle de production en cours). Voir models/README.
MODEL_VERSION = os.getenv("MODEL_VERSION", "efficientnetv2b0")
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(MODELS_DIR / "agridetect_efficientnetv2b0.keras")
)
LABELS_PATH = os.getenv("LABELS_PATH", str(MODELS_DIR / "labels.txt"))
PLANT_DETECTOR_PATH = os.getenv("PLANT_DETECTOR_PATH", str(MODELS_DIR / "model.pt"))

DEFAULT_IMG_SIZE = (224, 224)
MIN_DISEASE_CONFIDENCE = float(os.getenv("MIN_DISEASE_CONFIDENCE", "0.70"))
MIN_PLANT_DETECTION_CONFIDENCE = float(os.getenv("MIN_PLANT_DETECTION_CONFIDENCE", "0.25"))
MIN_PLANT_HINT_CONFIDENCE = float(os.getenv("MIN_PLANT_HINT_CONFIDENCE", "0.05"))
MIN_LEAF_SIGNAL = float(os.getenv("MIN_LEAF_SIGNAL", "0.04"))
MIN_CENTER_LEAF_SIGNAL = float(os.getenv("MIN_CENTER_LEAF_SIGNAL", "0.04"))
ML_COMPARE_PREPROCESSING = os.getenv("ML_COMPARE_PREPROCESSING", "0").lower() in {
    "1",
    "true",
    "yes",
}

LOGGER = logging.getLogger("agridetect")

FALLBACK_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _load_labels() -> list[str]:
    """Load deployed class order from labels.txt, with the historical list as fallback."""
    path = Path(LABELS_PATH)
    if not path.exists():
        LOGGER.warning("Labels file not found at %s; using fallback labels.", path)
        return FALLBACK_LABELS

    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != len(FALLBACK_LABELS):
        raise ValueError(
            f"Labels file has {len(labels)} classes, expected {len(FALLBACK_LABELS)}: {path}"
        )
    if labels != FALLBACK_LABELS:
        LOGGER.warning(
            "Labels file order differs from fallback order; using deployed labels file: %s",
            path,
        )
    return labels


# Exact 38-class order used by the deployed model. Prefer backend-api/models/labels.txt
# so the API follows the exported training class order instead of a stale hard-coded list.
LABELS = _load_labels()

PLANT_KEYS = sorted({label.split("___")[0] for label in LABELS})

PLANT_ALIASES = {
    "cherry": "Cherry_(including_sour)",
    "corn": "Corn_(maize)",
    "maize": "Corn_(maize)",
    "bell pepper": "Pepper,_bell",
    "pepper": "Pepper,_bell",
}

PLANT_FR = {
    "Apple": "Pommier",
    "Blueberry": "Myrtille",
    "Cherry_(including_sour)": "Cerisier",
    "Corn_(maize)": "Mais",
    "Grape": "Vigne",
    "Orange": "Oranger",
    "Peach": "Pecher",
    "Pepper,_bell": "Poivron",
    "Potato": "Pomme de terre",
    "Raspberry": "Framboisier",
    "Soybean": "Soja",
    "Squash": "Courge",
    "Strawberry": "Fraisier",
    "Tomato": "Tomate",
}

DISEASE_FR = {
    "Apple___Apple_scab": ("Tavelure du pommier", "ELEVEE"),
    "Apple___Black_rot": ("Pourriture noire", "ELEVEE"),
    "Apple___Cedar_apple_rust": ("Rouille du pommier", "MODEREE"),
    "Apple___healthy": ("Saine", "FAIBLE"),
    "Blueberry___healthy": ("Saine", "FAIBLE"),
    "Cherry_(including_sour)___Powdery_mildew": ("Oidium", "MODEREE"),
    "Cherry_(including_sour)___healthy": ("Saine", "FAIBLE"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Cercosporiose (taches grises)",
        "MODEREE",
    ),
    "Corn_(maize)___Common_rust_": ("Rouille commune", "MODEREE"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Helminthosporiose du Nord", "ELEVEE"),
    "Corn_(maize)___healthy": ("Saine", "FAIBLE"),
    "Grape___Black_rot": ("Pourriture noire", "ELEVEE"),
    "Grape___Esca_(Black_Measles)": ("Esca (maladie du bois)", "CRITIQUE"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Brulure des feuilles", "MODEREE"),
    "Grape___healthy": ("Saine", "FAIBLE"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Huanglongbing (greening)", "CRITIQUE"),
    "Peach___Bacterial_spot": ("Tache bacterienne", "MODEREE"),
    "Peach___healthy": ("Saine", "FAIBLE"),
    "Pepper,_bell___Bacterial_spot": ("Tache bacterienne", "MODEREE"),
    "Pepper,_bell___healthy": ("Saine", "FAIBLE"),
    "Potato___Early_blight": ("Alternariose", "MODEREE"),
    "Potato___Late_blight": ("Mildiou", "CRITIQUE"),
    "Potato___healthy": ("Saine", "FAIBLE"),
    "Raspberry___healthy": ("Saine", "FAIBLE"),
    "Soybean___healthy": ("Saine", "FAIBLE"),
    "Squash___Powdery_mildew": ("Oidium", "MODEREE"),
    "Strawberry___Leaf_scorch": ("Brulure des feuilles", "MODEREE"),
    "Strawberry___healthy": ("Saine", "FAIBLE"),
    "Tomato___Bacterial_spot": ("Tache bacterienne", "MODEREE"),
    "Tomato___Early_blight": ("Alternariose", "MODEREE"),
    "Tomato___Late_blight": ("Mildiou", "ELEVEE"),
    "Tomato___Leaf_Mold": ("Cladosporiose (moisissure des feuilles)", "MODEREE"),
    "Tomato___Septoria_leaf_spot": ("Septoriose", "MODEREE"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Acariens", "MODEREE"),
    "Tomato___Target_Spot": ("Tache cible", "MODEREE"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Virus TYLCV", "CRITIQUE"),
    "Tomato___Tomato_mosaic_virus": ("Virus de la mosaique", "ELEVEE"),
    "Tomato___healthy": ("Saine", "FAIBLE"),
}

TREATMENT_FR = {
    "Apple___Apple_scab": "Utiliser des fongicides et des varietes resistantes. Tailler et detruire les feuilles infectees.",
    "Apple___Black_rot": "Tailler et retirer les parties atteintes. Appliquer un fongicide en saison de croissance.",
    "Apple___Cedar_apple_rust": "Eliminer les genevriers hotes a proximite. Utiliser des varietes resistantes si possible.",
    "Cherry_(including_sour)___Powdery_mildew": "Assurer une bonne circulation de l'air et utiliser un traitement adapte contre l'oidium.",
    "Corn_(maize)___Common_rust_": "Privilegier des hybrides resistants; appliquer un fongicide si la pression est forte.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotation des cultures, varietes resistantes et fongicides foliaires si necessaire.",
    "Grape___Black_rot": "Eliminer les debris infectes et appliquer des fongicides preventifs en periode sensible.",
    "Grape___Esca_(Black_Measles)": "Pas de traitement curatif fiable; eviter les plaies de taille et retirer les ceps fortement atteints.",
    "Orange___Haunglongbing_(Citrus_greening)": "Maladie incurable: arracher les arbres atteints et lutter contre le vecteur.",
    "Potato___Early_blight": "Retirer les feuilles atteintes, eviter le stress hydrique et utiliser un fongicide adapte.",
    "Potato___Late_blight": "Intervenir rapidement avec un programme fongicide adapte et detruire les tissus infectes.",
    "Tomato___Early_blight": "Retirer les feuilles basses atteintes, arroser au sol et utiliser un fongicide adapte.",
    "Tomato___Late_blight": "Eliminer les feuilles et fruits atteints; utiliser un traitement preventif adapte en climat humide.",
    "Tomato___Leaf_Mold": "Aerer la serre, reduire l'humidite et appliquer un fongicide adapte.",
    "Tomato___Septoria_leaf_spot": "Retirer les feuilles atteintes, eviter l'arrosage sur feuilles et pratiquer la rotation.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Utiliser des auxiliaires ou acaricides adaptes et surveiller les foyers.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Lutter contre l'aleurode vecteur, arracher les plants infectes et utiliser des varietes resistantes.",
    "Tomato___Tomato_mosaic_virus": "Aucun traitement curatif; arracher les plants infectes et desinfecter les outils.",
}

HEALTHY_TREATMENT = (
    "Aucune maladie detectee. Maintenir de bonnes pratiques culturales pour "
    "preserver la sante de la plante."
)


def _load_json_meta() -> dict:
    try:
        data = json.loads((BASE_DIR / "data" / "plant_disease.json").read_text(encoding="utf-8"))
        return {d["name"]: d for d in data}
    except Exception:
        return {}


_JSON_META = _load_json_meta()


def _normalize_lang(lang: Optional[str] = None) -> str:
    code = (lang or "fr").split(",")[0].split("-")[0].lower()
    return code if code in {"en", "fr", "ar"} else "fr"


def _translated(meta: dict, field: str, lang: str, fallback: str = "") -> str:
    values = meta.get(f"{field}_i18n") or {}
    return values.get(lang) or values.get("fr") or meta.get(field) or fallback


def class_info(label: str, lang: Optional[str] = "fr") -> dict:
    lang = _normalize_lang(lang)
    plant_key = label.split("___")[0]
    disease_fr, severity = DISEASE_FR.get(label, ("Inconnue", "MODEREE"))
    healthy = label.endswith("healthy")
    meta = _JSON_META.get(label, {})
    treatment = _translated(
        meta,
        "cure",
        lang,
        HEALTHY_TREATMENT if healthy else TREATMENT_FR.get(label, "Consulter un agronome."),
    )
    cause = _translated(meta, "cause", lang, "-")
    return {
        "label": label,
        "plant": _translated(meta, "plant", lang, PLANT_FR.get(plant_key, plant_key.replace("_", " "))),
        "disease": _translated(meta, "disease", lang, disease_fr),
        "severity": severity,
        "is_healthy": healthy,
        "cause": cause,
        "treatment": treatment,
    }


CLASSES_INFO = [class_info(label) for label in LABELS]


class _Predictor:
    def __init__(self):
        self._model = None
        self._plant_detector = None
        self._loaded = False
        self.backend = "not-loaded"
        self.img_size = DEFAULT_IMG_SIZE
        self._model_error = None
        self._model_has_rescaling = False
        self._model_has_normalization = False
        self._model_outputs_logits = False
        self._preprocess_mode = "unknown"
        self.detector_names: dict[int, str] = {}

    def _try_load(self):
        if self._loaded:
            return
        self._loaded = True

        if not Path(MODEL_PATH).exists():
            self.backend = "missing"
            self._model_error = f"Disease model not found: {MODEL_PATH}"
            return
        if not Path(PLANT_DETECTOR_PATH).exists():
            self.backend = "missing"
            self._model_error = f"YOLO plant detector not found: {PLANT_DETECTOR_PATH}"
            return

        try:
            try:
                import keras

                self._model = keras.models.load_model(MODEL_PATH)
            except Exception:
                import tensorflow as tf

                self._model = tf.keras.models.load_model(MODEL_PATH)

            self.img_size = self._detect_input_size()
            self._validate_model_output()
            self._model_has_rescaling = self._detect_rescaling_layer()
            self._model_has_normalization = self._detect_normalization_layer()
            self._model_outputs_logits = self._detect_logits_output()
            self._preprocess_mode = self._select_preprocess_mode()

            out_shape = getattr(self._model, "output_shape", None)
            LOGGER.info("=" * 52)
            LOGGER.info("Model loaded successfully")
            LOGGER.info("Model version : %s", MODEL_VERSION)
            LOGGER.info("Model file    : %s", os.path.basename(MODEL_PATH))
            LOGGER.info("Input shape   : (1, %d, %d, 3)", self.img_size[0], self.img_size[1])
            LOGGER.info("Output shape  : %s", out_shape)
            LOGGER.info("Num classes   : %d", len(LABELS))
            LOGGER.info("Labels file   : %s", LABELS_PATH)
            LOGGER.info("Rescaling     : %s (internal 1/255)", self._model_has_rescaling)
            LOGGER.info("Normalization : %s (internal mean/std)", self._model_has_normalization)
            LOGGER.info("Preprocess    : %s", self._preprocess_mode)
            LOGGER.info(
                "Output mode   : %s",
                "logits" if self._model_outputs_logits else "probabilities/softmax",
            )
            LOGGER.info("=" * 52)

            from ultralytics import YOLO

            self._plant_detector = YOLO(PLANT_DETECTOR_PATH)
            self.detector_names = dict(getattr(self._plant_detector, "names", {}) or {})
            self.backend = "yolo-pt+keras"
        except Exception as exc:
            LOGGER.exception("Failed to load ML pipeline")
            self._model = None
            self._plant_detector = None
            self.backend = "invalid"
            self._model_error = str(exc)

    def _detect_input_size(self) -> tuple[int, int]:
        shape = getattr(self._model, "input_shape", None)
        if isinstance(shape, list):
            shape = shape[0]
        if shape and len(shape) >= 4 and shape[1] and shape[2]:
            return int(shape[1]), int(shape[2])
        return DEFAULT_IMG_SIZE

    def _validate_model_output(self) -> None:
        shape = getattr(self._model, "output_shape", None)
        if isinstance(shape, list):
            shape = shape[0]
        output_classes = int(shape[-1]) if shape and shape[-1] else None
        if output_classes != len(LABELS):
            raise ValueError(
                f"Model output has {output_classes} classes, expected {len(LABELS)} classes."
            )

    def _iter_model_layers(self):
        stack = list(getattr(self._model, "layers", []))
        while stack:
            layer = stack.pop(0)
            yield layer
            nested = getattr(layer, "layers", None)
            if nested:
                stack[0:0] = list(nested)

    def _detect_rescaling_layer(self) -> bool:
        for layer in self._iter_model_layers():
            if layer.__class__.__name__.lower() == "rescaling":
                return True
        return False

    def _detect_normalization_layer(self) -> bool:
        for layer in self._iter_model_layers():
            if layer.__class__.__name__.lower() == "normalization":
                return True
        return False

    def _detect_logits_output(self) -> bool:
        layers = list(self._iter_model_layers())
        for layer in reversed(layers):
            activation = getattr(layer, "activation", None)
            activation_name = getattr(activation, "__name__", "")
            if activation_name == "softmax":
                return False
            if activation_name == "linear":
                return True
        return False

    def _select_preprocess_mode(self) -> str:
        if self._model_has_rescaling or self._model_has_normalization:
            return "raw_0_255_internal_preprocessing"
        return os.getenv("ML_PREPROCESS_MODE", "div_255").lower()

    def _image_from_bytes(self, image_bytes: bytes):
        from PIL import Image, ImageOps

        return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")

    def _detect_plant(self, image_bytes: bytes) -> dict:
        if self._plant_detector is None:
            raise RuntimeError("YOLO plant detector is not loaded.")

        image_obj = self._image_from_bytes(image_bytes)
        results = self._plant_detector.predict(
            source=image_obj,
            imgsz=640,
            conf=MIN_PLANT_DETECTION_CONFIDENCE,
            verbose=False,
        )
        detections = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf[0])
                    class_index = int(box.cls[0])
                    detections.append(
                        {
                            "label": self.detector_names.get(class_index, str(class_index)),
                            "confidence": confidence,
                        }
                    )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        best = detections[0] if detections else None
        return {
            "has_plant": best is not None,
            "confidence": float(best["confidence"]) if best else 0.0,
            "label": best["label"] if best else None,
            "detections": detections,
        }

    def _invalid_leaf_result(self, confidence: float, lang: Optional[str]) -> dict:
        lang = _normalize_lang(lang)
        messages = {
            "en": "No clear leaf detected. Please capture one plant leaf clearly.",
            "fr": "Aucune feuille claire detectee. Veuillez capturer une feuille de plante clairement.",
            "ar": "No clear leaf detected. Please capture one plant leaf clearly.",
        }
        message = messages[lang]
        return {
            "label": "no_plant",
            "plant": message,
            "disease": message,
            "severity": "FAIBLE",
            "is_healthy": False,
            "status": "no_plant",
            "cause": message,
            "treatment": message,
            "confidence": round(float(confidence), 4),
            "backend": self.backend,
            "top_k": [],
        }

    def _leaf_signal(self, image_bytes: bytes) -> dict:
        """Estimate whether the frame contains enough leaf-like pixels."""
        image_obj = self._image_from_bytes(image_bytes).resize((256, 256))
        arr = np.asarray(image_obj, dtype=np.float32)

        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        max_channel = np.max(arr, axis=2)
        min_channel = np.min(arr, axis=2)
        saturation = np.where(
            max_channel > 0,
            (max_channel - min_channel) / max_channel,
            0.0,
        )

        exg = (2 * green) - red - blue
        green_leaf = (green > 45) & (green >= red) & (green >= blue) & (exg > 18)
        yellow_leaf = (
            (red > 90)
            & (green > 90)
            & (blue < 120)
            & (np.abs(red - green) < 45)
            & (saturation > 0.18)
        )
        paper_or_sky = (max_channel > 220) & (saturation < 0.14)
        leaf_pixels = (green_leaf | yellow_leaf) & ~paper_or_sky

        h, w = leaf_pixels.shape
        x0, x1 = int(w * 0.2), int(w * 0.8)
        y0, y1 = int(h * 0.2), int(h * 0.8)
        center = leaf_pixels[y0:y1, x0:x1]

        return {
            "whole": float(np.mean(leaf_pixels)),
            "center": float(np.mean(center)) if center.size else 0.0,
        }

    def _has_enough_leaf_signal(self, leaf_signal: dict) -> bool:
        return (
            leaf_signal["whole"] >= MIN_LEAF_SIGNAL
            or leaf_signal["center"] >= MIN_CENTER_LEAF_SIGNAL
        )

    def _low_confidence_result(self, best: dict) -> dict:
        result = dict(best)
        result["status"] = "low_confidence"
        result["backend"] = self.backend
        return result

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        image_obj = self._image_from_bytes(image_bytes).resize(self.img_size)
        arr = np.asarray(image_obj, dtype="float32")
        arr = np.expand_dims(arr, axis=0)

        if self._preprocess_mode in {"raw_0_255", "raw", "raw_0_255_internal_preprocessing"}:
            return arr

        if self._preprocess_mode in {"div_255", "normalize", "rescale"}:
            return arr / 255.0

        if self._preprocess_mode in {"efficientnetv2", "efficientnet_v2", "preprocess_input"}:
            try:
                import tensorflow as tf

                processed = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
                return processed.numpy() if hasattr(processed, "numpy") else processed
            except Exception:
                LOGGER.exception("EfficientNetV2 preprocess_input failed; falling back to raw input.")
                return arr

        LOGGER.warning("Unknown ML_PREPROCESS_MODE=%s; using raw 0..255 input.", self._preprocess_mode)
        return arr

    def _preprocess_variants(self, image_bytes: bytes) -> dict[str, np.ndarray]:
        image_obj = self._image_from_bytes(image_bytes).resize(self.img_size)
        raw = np.expand_dims(np.asarray(image_obj, dtype="float32"), axis=0)
        variants = {
            "raw_0_255": raw,
            "div_255": raw / 255.0,
        }
        try:
            import tensorflow as tf

            processed = tf.keras.applications.efficientnet_v2.preprocess_input(raw.copy())
            variants["efficientnetv2_preprocess_input"] = (
                processed.numpy() if hasattr(processed, "numpy") else processed
            )
        except Exception:
            LOGGER.exception("Unable to compute EfficientNetV2 preprocess_input diagnostic.")
        return variants

    def _normalize_probs(self, values: np.ndarray) -> np.ndarray:
        probs = np.asarray(values, dtype="float32").reshape(-1)
        if self._model_outputs_logits:
            exps = np.exp(probs - np.max(probs))
            return exps / np.sum(exps)

        total = float(np.sum(probs))
        if 0.99 <= total <= 1.01 and np.all(probs >= 0):
            return probs
        LOGGER.warning(
            "Model output does not look like probabilities despite softmax metadata; applying softmax. "
            "sum=%.6f min=%.6f max=%.6f",
            total,
            float(np.min(probs)),
            float(np.max(probs)),
        )
        exps = np.exp(probs - np.max(probs))
        return exps / np.sum(exps)

    def _log_top_predictions(
        self,
        raw_output: np.ndarray,
        probs: np.ndarray,
        status: str,
        prefix: str = "Prediction",
    ) -> None:
        raw = np.asarray(raw_output, dtype="float32").reshape(-1)
        top_raw = np.argsort(raw)[::-1][:5]
        top_probs = np.argsort(probs)[::-1][:5]
        LOGGER.info(
            "%s top5 raw outputs: %s",
            prefix,
            [
                {
                    "idx": int(i),
                    "label": LABELS[int(i)],
                    "value": round(float(raw[int(i)]), 6),
                }
                for i in top_raw
            ],
        )
        LOGGER.info(
            "%s top5 labels confidence: %s",
            prefix,
            [
                {
                    "idx": int(i),
                    "label": LABELS[int(i)],
                    "confidence": round(float(probs[int(i)]), 6),
                }
                for i in top_probs
            ],
        )
        LOGGER.info("%s status returned to frontend: %s", prefix, status)

    def _compare_preprocessing_modes(self, image_bytes: bytes) -> None:
        if not ML_COMPARE_PREPROCESSING:
            return
        LOGGER.info("Preprocessing diagnostic enabled: testing raw, /255, EfficientNetV2 preprocess_input.")
        for mode, x in self._preprocess_variants(image_bytes).items():
            raw_output = np.asarray(self._model.predict(x, verbose=0)[0], dtype="float32")
            probs = self._normalize_probs(raw_output)
            LOGGER.info(
                "Preprocess mode=%s input_min=%.4f input_max=%.4f output_sum=%.6f",
                mode,
                float(np.min(x)),
                float(np.max(x)),
                float(np.sum(raw_output)),
            )
            self._log_top_predictions(raw_output, probs, "diagnostic", prefix=f"Preprocess {mode}")

    def _normalize_plant_hint(self, plant_hint: Optional[str]) -> Optional[str]:
        if not plant_hint:
            return None
        wanted = plant_hint.strip().lower()
        if wanted in PLANT_ALIASES:
            return PLANT_ALIASES[wanted]
        for plant in PLANT_KEYS:
            if plant.lower() == wanted:
                return plant
        return None

    def _plant_probability(self, probs: np.ndarray, plant: Optional[str]) -> float:
        if not plant:
            return 0.0
        indices = [i for i, label in enumerate(LABELS) if label.startswith(f"{plant}___")]
        return float(np.sum(probs[indices])) if indices else 0.0

    def _top_predictions(
        self,
        probs: np.ndarray,
        top_k: int,
        lang: Optional[str],
        plant_hint: Optional[str],
    ) -> list[dict]:
        allowed_plant = self._normalize_plant_hint(plant_hint)
        if allowed_plant:
            indices = [i for i, label in enumerate(LABELS) if label.startswith(f"{allowed_plant}___")]
            plant_total = float(np.sum(probs[indices]))
            if plant_total > 0:
                order = sorted(indices, key=lambda i: float(probs[i] / plant_total), reverse=True)[:top_k]
                return [
                    {
                        **class_info(LABELS[int(i)], lang),
                        "confidence": round(float(probs[int(i)] / plant_total), 4),
                    }
                    for i in order
                ]

        order = np.argsort(probs)[::-1][:top_k]
        return [
            {
                **class_info(LABELS[int(i)], lang),
                "confidence": round(float(probs[int(i)]), 4),
            }
            for i in order
        ]

    def predict(
        self,
        image_bytes: bytes,
        top_k: int = 3,
        lang: Optional[str] = "fr",
        plant_hint: Optional[str] = None,
    ) -> dict:
        self._try_load()
        if self._model is None or self._plant_detector is None:
            raise RuntimeError(self._model_error or "ML pipeline is not loaded.")

        leaf_signal = self._leaf_signal(image_bytes)
        plant_detection = self._detect_plant(image_bytes)

        if not self._has_enough_leaf_signal(leaf_signal):
            LOGGER.info(
                "Leaf signal rejected image: whole=%.4f center=%.4f yolo_label=%s yolo_confidence=%.4f",
                leaf_signal["whole"],
                leaf_signal["center"],
                plant_detection["label"],
                plant_detection["confidence"],
            )
            result = self._invalid_leaf_result(plant_detection["confidence"], lang)
            result["leaf_signal"] = leaf_signal
            result["plant_detection"] = plant_detection
            LOGGER.info("Prediction status returned to frontend: %s", result["status"])
            return result

        if not plant_detection["has_plant"]:
            LOGGER.info(
                "YOLO missed leaf, accepting via leaf signal: whole=%.4f center=%.4f",
                leaf_signal["whole"],
                leaf_signal["center"],
            )

        x = self._preprocess(image_bytes)
        LOGGER.info(
            "Prediction input preprocessing=%s min=%.4f max=%.4f mean=%.4f",
            self._preprocess_mode,
            float(np.min(x)),
            float(np.max(x)),
            float(np.mean(x)),
        )
        self._compare_preprocessing_modes(image_bytes)
        raw_output = np.asarray(self._model.predict(x, verbose=0)[0], dtype="float32")
        probs = self._normalize_probs(raw_output)

        allowed_plant = self._normalize_plant_hint(plant_hint)
        plant_confidence = self._plant_probability(probs, allowed_plant)
        if allowed_plant and plant_confidence < MIN_PLANT_HINT_CONFIDENCE:
            LOGGER.info(
                "Plant hint mismatch hint=%s plant_confidence=%.4f",
                plant_hint,
                plant_confidence,
            )
            result = self._invalid_leaf_result(plant_confidence, lang)
            LOGGER.info("Prediction status returned to frontend: %s", result["status"])
            return result

        top = self._top_predictions(probs, top_k, lang, plant_hint)
        best = dict(top[0])
        LOGGER.info(
            "Prediction label=%s confidence=%.4f yolo_label=%s yolo_confidence=%.4f backend=%s",
            best["label"],
            best["confidence"],
            plant_detection["label"],
            plant_detection["confidence"],
            self.backend,
        )

        best["top_k"] = top
        best["backend"] = self.backend
        best["plant_detection"] = plant_detection
        best["leaf_signal"] = leaf_signal
        if best["confidence"] < MIN_DISEASE_CONFIDENCE:
            result = self._low_confidence_result(best)
            self._log_top_predictions(raw_output, probs, result["status"])
            return result

        best["status"] = "ok"
        self._log_top_predictions(raw_output, probs, best["status"])
        return best


predictor = _Predictor()
