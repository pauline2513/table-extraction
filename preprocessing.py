from typing import List

import cv2
from PIL import Image, ImageEnhance
import numpy as np
import pypdfium2 as pdfium

# конвертация картинки в numpy в бгр и обратно

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

# конвертация загруженного пдф в картинку
def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 3.0) -> List[np.ndarray]:
    doc = pdfium.PdfDocument(pdf_bytes)
    pages: List[np.ndarray] = []
    for idx in range(len(doc)):
        page = doc[idx]
        pages.append(pil_to_bgr(page.render(scale=scale).to_pil()))
    # logger.info(f"Rendered PDF into {len(pages)} image page(s)")
    return pages

# препроцессинг страницы для детекции таблиц, убрали шум и через clahe повысили контрастность
def preprocess_page_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(
        image_bgr, None, h=7, hColor=7, templateWindowSize=7, searchWindowSize=21
    )
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# предобработка уже для распознавания после детекции
def preprocess_crop_for_paddle(crop_bgr: np.ndarray) -> np.ndarray:
    pil = bgr_to_pil(crop_bgr)
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    pil = ImageEnhance.Sharpness(pil).enhance(1.35)
    enhanced = pil_to_bgr(pil)
    # upscaled = cv2.resize(
    #     enhanced,
    #     None,
    #     fx=1.4,
    #     fy=1.4,
    #     interpolation=cv2.INTER_CUBIC,
    # )
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # binary = cv2.adaptiveThreshold(
    #     norm,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     31,
    #     10,
    # )
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)