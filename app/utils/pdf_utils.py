# app/utils/pdf_utils.py
import pymupdf as fitz  # PyMuPDF
from pathlib import Path
from app.core.logger import logger


def analyze_pdf_images(file_path: str):
    """
    Analiza imágenes dentro de un PDF y devuelve:
      - numero_imagenes
      - lista con:
            page: número de página
            image_index: índice de imagen
            width, height: dimensiones reales
            bbox: caja delimitadora en la página
            caption: texto cercano a la imagen
            links: enlaces cercanos en la página
    """
    path = Path(file_path)

    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.error(f"❌ No se pudo abrir PDF para análisis de imágenes: {e}")
        return 0, []

    images_data = []

    for page_index, page in enumerate(doc):
        img_list = page.get_images(full=True)

        # Obtener todo el texto en bloques (para captions)
        text_blocks = page.get_text("blocks")

        # Obtener todos los links del PDF
        page_links = page.get_links()

        for img_index, img in enumerate(img_list):
            xref = img[0]
            base = doc.extract_image(xref)

            try:
                bbox = page.get_image_bbox(img)
            except Exception:
                bbox = (0, 0, 0, 0)

            width, height = base.get("width"), base.get("height")

            caption = _extract_caption_near_bbox(bbox, text_blocks)
            related_links = _find_links_near_bbox(bbox, page_links)

            images_data.append({
                "page": page_index + 1,
                "image_index": img_index,
                "width": width,
                "height": height,
                "bbox": tuple(map(float, bbox)),
                "caption": caption,
                "links": related_links,
                "xref": xref,
            })

    logger.info(f"PDF analizado: {len(images_data)} imágenes detectadas.")
    return len(images_data), images_data


# ===============================================================
# 🔍 FUNCIONES DE SOPORTE
# ===============================================================

def _extract_caption_near_bbox(bbox, text_blocks, threshold=30):
    """
    Busca texto cerca (debajo o encima) del área de la imagen.
    threshold = distancia máxima en pixeles para considerar texto como caption.
    """

    x0, y0, x1, y1 = bbox
    candidates = []

    for block in text_blocks:
        bx0, by0, bx1, by1, text, *_ = block

        # Texto debajo de la imagen
        if abs(by0 - y1) <= threshold and _horiz_overlap((x0, x1), (bx0, bx1)):
            candidates.append(text)

        # Texto encima de la imagen
        if abs(y0 - by1) <= threshold and _horiz_overlap((x0, x1), (bx0, bx1)):
            candidates.append(text)

    return " ".join(candidates).strip() if candidates else None


def _find_links_near_bbox(bbox, links, threshold=20):
    """
    Encuentra hipervínculos cerca del área de la imagen.
    """
    x0, y0, x1, y1 = bbox
    related = []

    for link in links:
        lx0, ly0, lx1, ly1 = link.get("from", (0, 0, 0, 0))

        # Consideramos un link "cercano" si su área toca o está cerca del bbox
        if (
            abs(ly0 - y1) <= threshold or abs(y0 - ly1) <= threshold
            or abs(lx0 - x1) <= threshold or abs(x0 - lx1) <= threshold
        ):
            related.append(link)

    return related if related else None


def _horiz_overlap(a, b):
    """Evalúa si dos rangos horizontales se solapan."""
    a0, a1 = a
    b0, b1 = b
    return max(a0, b0) <= min(a1, b1)
