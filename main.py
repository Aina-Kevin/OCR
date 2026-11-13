import sys
import json
import argparse
from pathlib import Path

from ocr import preprocess, extract_text
from llm import understand_document, DOC_TYPES


# ==================================================
# DOSSIERS : un sous-dossier par type de document
# ==================================================

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

FOLDERS = {
    "facture_commerciale": "facture",
    "packing_list": "packing list",
    "document_transit": "titre transport",
}


def get_images(root, doc_type):
    """Retourne la liste des images contenues dans le sous-dossier du document."""
    folder = Path(root) / FOLDERS[doc_type]

    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )


def check_document(doc_type, paths, lang, psm):
    """OCR de toutes les images du dossier + vérification LLM."""
    print(f"\n--- {doc_type} : {len(paths)} image(s) ---")

    texts = []

    for path in paths:
        print(f"  [1/3] Prétraitement : {path.name}")
        images = preprocess(path)

        print("  [2/3] OCR...")
        text = extract_text(images["binary"], lang=lang, psm=psm)

        if text:
            texts.append(text)

    ocr_text = "\n\n".join(texts)

    if not ocr_text:
        print("  Aucun texte détecté -> À VÉRIFIER")
        return "À VÉRIFIER", ""

    print("  [3/3] Analyse LLM...")
    status = understand_document(doc_type, ocr_text)

    print(f"  Résultat : {status}")
    return status, ocr_text


def main():

    parser = argparse.ArgumentParser(
        description="Vérification de documents Import-Export"
    )

    parser.add_argument(
        "--images",
        default="images",
        help="Dossier principal contenant les sous-dossiers "
             "'facture', 'packing list' et 'titre transport'"
    )

    parser.add_argument(
        "--docs",
        help="Documents attendus selon la prestation, séparés par des virgules "
             "(facture_commerciale,packing_list,document_transit). "
             "Par défaut : les 3."
    )

    parser.add_argument("--lang", default="eng+fra", help="Langues OCR")
    parser.add_argument("--psm", type=int, default=6, help="Mode Tesseract")

    args = parser.parse_args()

    root = Path(args.images)

    if not root.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {root}")

    if args.docs:
        expected = [d.strip() for d in args.docs.split(",") if d.strip()]
        for d in expected:
            if d not in DOC_TYPES:
                raise ValueError(
                    f"Document attendu inconnu : {d} "
                    f"(valeurs possibles : {', '.join(DOC_TYPES)})"
                )
    else:
        expected = list(DOC_TYPES)

    results = {}
    ocr_texts = {}

    for doc_type in expected:
        paths = get_images(root, doc_type)

        if not paths:
            results[doc_type] = "none"
            print(f"\n--- {doc_type} : aucune image dans '{FOLDERS[doc_type]}' -> none ---")
            continue

        status, text = check_document(doc_type, paths, args.lang, args.psm)
        results[doc_type] = status
        ocr_texts[doc_type] = text

    result_json = json.dumps(results, indent=2, ensure_ascii=False)

    print("\n========== RESULTAT FINAL ==========\n")
    print(result_json)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    (output_dir / "result.json").write_text(result_json, encoding="utf-8")
    print(f"\nRésultat sauvegardé : {output_dir / 'result.json'}")
    (output_dir / "result.json").write_text(result_json, encoding="utf-8")
    print(f"\nRésultat sauvegardé : {output_dir / 'result.json'}")

    # ==================================================
    # SUPPRESSION DES IMAGES APRÈS TRAITEMENT
    # ==================================================

    for doc_type in expected:
        paths = get_images(root, doc_type)

        for path in paths:
            path.unlink()
            print(f"Image supprimée : {path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        sys.exit(1)