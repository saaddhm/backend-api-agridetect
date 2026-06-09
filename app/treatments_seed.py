"""
Données de traitement (module Treatment Recommendations).

- RICH : fiches détaillées rédigées à la main pour les maladies les plus courantes.
- Les autres classes sont initialisées automatiquement à partir des métadonnées du
  modèle (app/ml.py) : l'administrateur peut ensuite les compléter via le CRUD.
"""
from sqlmodel import Session, select

from .ml import CLASSES_INFO
from .models import DiseaseTreatment

# Noms anglais lisibles + traductions, par label de classe
NAMES = {
    "Tomato___Early_blight": ("Tomato Early Blight", "Alternariose précoce de la tomate", "لفحة الطماطم المبكرة"),
    "Tomato___Late_blight": ("Tomato Late Blight", "Mildiou de la tomate", "اللفحة المتأخرة للطماطم"),
    "Potato___Late_blight": ("Potato Late Blight", "Mildiou de la pomme de terre", "اللفحة المتأخرة للبطاطس"),
    "Apple___Apple_scab": ("Apple Scab", "Tavelure du pommier", "جرب التفاح"),
    "Corn___Common_rust": ("Corn Common Rust", "Rouille commune du maïs", "صدأ الذرة الشائع"),
}

RICH = {
    "Tomato___Early_blight": {
        "description": "Maladie fongique (Alternaria solani) qui attaque les feuilles, tiges et fruits de la tomate, surtout par temps chaud et humide.",
        "symptoms": ["Taches brunes concentriques (en cibles)", "Jaunissement des feuilles basses", "Chute des feuilles", "Lésions sur les fruits près du pédoncule"],
        "causes": ["Forte humidité et chaleur", "Mauvaise circulation de l'air", "Feuillage mouillé prolongé", "Carence ou stress de la plante"],
        "prevention": ["Rotation des cultures (3 ans)", "Éviter l'arrosage par aspersion", "Espacer les plants pour aérer", "Pailler le sol", "Retirer les débris végétaux"],
        "organic_treatment": ["Pulvérisation d'huile de neem", "Fongicide bio à base de cuivre", "Décoction de prêle"],
        "chemical_treatment": ["Mancozèbe", "Chlorothalonil", "Azoxystrobine"],
        "recommended_products": [{"name": "Bouillie bordelaise", "type": "Organic"}, {"name": "Mancozèbe 80 WP", "type": "Chemical"}],
        "recovery_time": "2-4 semaines",
        "expert_advice": "Retirez et détruisez les feuilles atteintes, traitez tous les 7-10 jours et surveillez chaque semaine.",
    },
    "Tomato___Late_blight": {
        "description": "Mildiou (Phytophthora infestans) : maladie très destructrice se propageant rapidement en climat frais et humide.",
        "symptoms": ["Taches brunes huileuses sur les feuilles", "Feutrage blanc au revers", "Brunissement des tiges", "Pourriture brune des fruits"],
        "causes": ["Humidité élevée et temps frais", "Pluies fréquentes", "Feuillage mouillé", "Proximité de plants infectés"],
        "prevention": ["Variétés résistantes", "Éviter l'arrosage du feuillage", "Bonne aération", "Surveillance après les pluies", "Élimination rapide des foyers"],
        "organic_treatment": ["Bouillie bordelaise en préventif (10-20 g/L)", "Huile de neem", "Élimination des parties atteintes"],
        "chemical_treatment": ["Métalaxyl + mancozèbe", "Cymoxanil", "Chlorothalonil"],
        "recommended_products": [{"name": "Bouillie bordelaise", "type": "Organic"}, {"name": "Métalaxyl-M", "type": "Chemical"}],
        "recovery_time": "3-5 semaines (selon précocité)",
        "expert_advice": "Agissez dès l'alerte mildiou (climat humide). Détruisez les plants gravement atteints pour limiter la propagation.",
    },
    "Potato___Late_blight": {
        "description": "Mildiou de la pomme de terre (Phytophthora infestans), maladie critique pouvant détruire une culture en quelques jours.",
        "symptoms": ["Lésions brun-noir sur feuilles et tiges", "Feutrage blanchâtre au revers", "Flétrissement rapide", "Pourriture brune des tubercules"],
        "causes": ["Climat frais et humide", "Forte hygrométrie", "Pluies répétées"],
        "prevention": ["Plants certifiés sains", "Buttage des tubercules", "Rotation des cultures", "Destruction des fanes avant récolte"],
        "organic_treatment": ["Bouillie bordelaise", "Cuivre en préventif"],
        "chemical_treatment": ["Métalaxyl + mancozèbe", "Fluazinam", "Mandipropamide"],
        "recommended_products": [{"name": "Bouillie bordelaise", "type": "Organic"}, {"name": "Fluazinam SC", "type": "Chemical"}],
        "recovery_time": "Variable — traitement préventif essentiel",
        "expert_advice": "Alternez les familles de fongicides pour éviter les résistances ; traitez avant les périodes humides.",
    },
    "Apple___Apple_scab": {
        "description": "Tavelure (Venturia inaequalis), principale maladie du pommier en climat humide et frais au printemps.",
        "symptoms": ["Taches olive à brun-noir veloutées sur feuilles", "Croûtes liégeuses sur fruits", "Déformation des fruits", "Chute prématurée des feuilles"],
        "causes": ["Printemps humide et frais", "Feuilles mortes infectées au sol", "Longues périodes d'humidité"],
        "prevention": ["Ramassage et destruction des feuilles mortes", "Variétés résistantes", "Taille pour aérer la frondaison"],
        "organic_treatment": ["Soufre mouillable", "Cuivre", "Bicarbonate de potassium"],
        "chemical_treatment": ["Difénoconazole", "Captane", "Myclobutanil"],
        "recommended_products": [{"name": "Soufre mouillable", "type": "Organic"}, {"name": "Captane 80 WG", "type": "Chemical"}],
        "recovery_time": "Gestion saisonnière (du débourrement à l'été)",
        "expert_advice": "Traitez préventivement selon le modèle de risque (température + humectation foliaire).",
    },
    "Corn___Common_rust": {
        "description": "Rouille commune (Puccinia sorghi) du maïs, favorisée par un temps doux et humide.",
        "symptoms": ["Pustules brun-rouge poudreuses sur les deux faces", "Jaunissement des feuilles", "Dessèchement en cas d'attaque forte"],
        "causes": ["Temps doux et humide", "Forte rosée", "Variétés sensibles"],
        "prevention": ["Hybrides résistants", "Semis précoce", "Surveillance en saison humide"],
        "organic_treatment": ["Soufre", "Pratiques culturales (aération, rotation)"],
        "chemical_treatment": ["Azoxystrobine", "Propiconazole", "Tébuconazole"],
        "recommended_products": [{"name": "Soufre", "type": "Organic"}, {"name": "Azoxystrobine SC", "type": "Chemical"}],
        "recovery_time": "1-3 semaines",
        "expert_advice": "Intervenez surtout si l'attaque est précoce et la variété sensible ; sinon l'impact reste limité.",
    },
}


def _humanize(label: str):
    plant, _, disease = label.partition("___")
    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ").strip().title()
    return plant.title(), f"{plant.title()} {disease}".strip()


def seed_treatments(session: Session) -> None:
    """Initialise une fiche par maladie si la table est vide (idempotent)."""
    for info in CLASSES_INFO:
        if info["is_healthy"]:
            continue
        label = info["label"]
        exists = session.exec(select(DiseaseTreatment).where(DiseaseTreatment.label == label)).first()
        if exists:
            continue
        plant_en, disease_en = _humanize(label)
        en, fr, ar = NAMES.get(label, (disease_en, info["disease"], info["disease"]))
        rich = RICH.get(label)
        if rich:
            session.add(DiseaseTreatment(
                label=label, disease_name=en, disease_name_fr=fr, disease_name_ar=ar,
                plant_name=info["plant"], severity=info["severity"], **rich))
        else:
            # Fiche de base dérivée des métadonnées du modèle
            session.add(DiseaseTreatment(
                label=label, disease_name=en, disease_name_fr=info["disease"], disease_name_ar=info["disease"],
                plant_name=info["plant"], severity=info["severity"],
                description=info.get("cause") or "",
                causes=[info["cause"]] if info.get("cause") else [],
                organic_treatment=[info["treatment"]] if info.get("treatment") else [],
                recommended_products=[],
                recovery_time="À préciser",
                expert_advice=info.get("treatment") or "Consultez un agronome local."))
    session.commit()
