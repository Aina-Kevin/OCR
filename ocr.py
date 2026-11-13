import os
from pathlib import Path

import cv2
import pytesseract


# ==================================================
# CONFIGURATION TESSERACT
# ==================================================

DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Surcharge possible via .env : TESSERACT_CMD=...
tesseract_cmd = os.getenv("TESSERACT_CMD", DEFAULT_TESSERACT_PATH)

if Path(tesseract_cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


# ==================================================
# 1. CHARGEMENT DE L'IMAGE
# ==================================================

def load_image(path):

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible : {path}")

    return image


# ==================================================
# 2. REDIMENSIONNEMENT
# ==================================================

def resize_image(image, scale=2):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


# ==================================================
# 3. NIVEAUX DE GRIS
# ==================================================

def grayscale(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==================================================
# 4. REDUCTION DU BRUIT
# ==================================================

def denoise(image):

    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )


# ==================================================
# 5. CONTRASTE
# ==================================================

def improve_contrast(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return clahe.apply(image)


# ==================================================
# 6. BINARISATION
# ==================================================

def binarize(image):

    return cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]


# ==================================================
# 7. PRETRAITEMENT
# ==================================================

def preprocess(path):

    image = load_image(path)
    resized = resize_image(image)
    gray = grayscale(resized)
    clean = denoise(gray)
    contrast = improve_contrast(clean)
    binary = binarize(contrast)

    return {
        "original": image,
        "resized": resized,
        "grayscale": gray,
        "denoised": clean,
        "contrast": contrast,
        "binary": binary
    }


# ==================================================
# 8. EXTRACTION OCR
# ==================================================

def extract_text(image, lang="eng+fra", psm=6):

    try:
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=f"--oem 3 --psm {psm}"
        )
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract est introuvable. Installe-le ou définis TESSERACT_CMD dans .env."
        )
    except pytesseract.TesseractError as e:
        raise RuntimeError(
            f"Erreur Tesseract (langue '{lang}' installée ?) : {e}"
        )

    return text.strip()