import os
import re
import json
import time
import unicodedata

from google import genai
from google.genai import types
from dotenv import load_dotenv


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")

FALLBACKS = [
    m.strip()
    for m in os.getenv("GEMMA_FALLBACKS", "gemma-3-27b-it").split(",")
    if m.strip()
]

MAX_RETRIES = 3
RETRY_DELAY = 3

if not API_KEY:
    raise ValueError("GEMINI_API_KEY est absente du fichier .env")

client = genai.Client(api_key=API_KEY)


# ==================================================
# PROMPT COMMUN + REGLES PAR TYPE DE DOCUMENT
# ==================================================

BASE_PROMPT = """Tu es un expert en vérification de documents import-export.

Ta mission : analyser le texte issu d'un OCR et déterminer si le document
est bien de type "__LABEL__" et si ses informations obligatoires sont
présentes et exploitables.

Informations obligatoires :
__RULES__

INTERPRÉTATION :

Le texte OCR peut contenir des erreurs de reconnaissance, des fautes
d'orthographe, des caractères incorrects ou des mots déformés.
Interprète le texte dans son contexte, sans exiger une correspondance
exacte des mots. Ne considère pas une erreur de reconnaissance comme une
donnée manquante si l'information reste identifiable avec certitude.
Ne jamais inventer une information absente.

DÉCISION :

"OK" : le document est bien de ce type et toutes les informations
obligatoires sont présentes, compréhensibles et cohérentes.

"À VÉRIFIER" : le document n'est pas de ce type, ou une information
obligatoire est absente, illisible, ambiguë ou incohérente.

SORTIE :

Réponds uniquement avec une seule ligne JSON, sans balise markdown,
sans commentaire, sans autre clé.

Exemples valides :
{"__KEY__": "OK"}
{"__KEY__": "À VÉRIFIER"}

TEXTE OCR :
"""

DOC_TYPES = {
    "facture_commerciale": {
        "label": "Facture commerciale",
        "rules": """- Nom de l'entreprise
- Numéro de facture
- Date d'émission
- Nom de l'expéditeur
- Nom du destinataire
- Description des marchandises
- Valeur totale à payer
- Poids des marchandises
- Nombre de colis
- Pays d'expédition et pays de destination""",
    },
    "packing_list": {
        "label": "Packing list",
        "rules": """- Date d'émission
- Numéro de la facture commerciale associée
- Nom de l'expéditeur
- Nombre total de colis (cartons)""",
    },
    "document_transit": {
        "label": "Titre de transport (LTA aérien, B/L maritime ou CMR routier)",
        "rules": """- Numéro du document
- Type de document (LTA pour l'aérien, B/L pour le maritime, CMR pour le routier)
- Nom de la compagnie de transport
- Numéro de vol, nom du navire, plaque du camion ou de la moto (selon le mode)
- Lieu de départ (aéroport, port ou ville)
- Lieu d'arrivée (aéroport, port ou ville)
- Date de départ prévue
- Date d'arrivée prévue (ETA)
- Nom de l'expéditeur (prestataire)""",
    },
}


def build_prompt(doc_type):
    cfg = DOC_TYPES[doc_type]
    return (
        BASE_PROMPT
        .replace("__LABEL__", cfg["label"])
        .replace("__RULES__", cfg["rules"])
        .replace("__KEY__", doc_type)
    )


# ==================================================
# OUTILS
# ==================================================

def _normalize(text):
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.strip().upper()


def _status_from_value(value):
    norm = _normalize(value)
    if norm == "OK":
        return "OK"
    if norm in ("A VERIFIER", "A VERIFIER."):
        return "À VÉRIFIER"
    return None


def _parse_status(content, key):
    match = re.search(r"\{.*?\}", content, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))
            status = _status_from_value(data.get(key))
            if status:
                return status
        except json.JSONDecodeError:
            pass

    norm = _normalize(content)

    if "A VERIFIER" in norm:
        return "À VÉRIFIER"
    if re.search(r"\bOK\b", norm):
        return "OK"

    raise ValueError(f"Réponse inexploitable : {content!r}")


def _call_model(model, prompt):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2048,
        ),
    )

    content = response.text

    if not content or not content.strip():
        cand = response.candidates[0] if response.candidates else None
        raise ValueError(
            "Réponse vide. "
            f"finish_reason={getattr(cand, 'finish_reason', None)} | "
            f"prompt_feedback={response.prompt_feedback}"
        )

    return content


# ==================================================
# VERIFICATION D'UN DOCUMENT
# ==================================================

def understand_document(doc_type, ocr_text):

    if doc_type not in DOC_TYPES:
        raise ValueError(f"Type de document inconnu : {doc_type}")

    if not ocr_text or not ocr_text.strip():
        raise ValueError("Le texte OCR est vide.")

    prompt = build_prompt(doc_type) + ocr_text

    models = [MODEL] + [m for m in FALLBACKS if m != MODEL]
    last_error = None

    for model in models:
        print(f"    Modèle : {model}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                content = _call_model(model, prompt)
                return _parse_status(content, doc_type)

            except Exception as e:
                last_error = e
                print(f"      Tentative {attempt}/{MAX_RETRIES} échouée : {e}")

                # Modèle inexistant : inutile de réessayer
                if "404" in str(e) or "NOT_FOUND" in str(e):
                    break

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(
        f"Échec sur tous les modèles ({', '.join(models)}). "
        f"Dernière erreur : {last_error}"
    )