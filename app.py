import io
import os
import sys

# ========== Critical environment variables ==========
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"  # This fixes the ArrayAttribute error
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_USE_GPU"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ===================================================

import json
# Now import the rest of your libraries
import streamlit as st
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import cv2
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PIL import Image
from typing import List, Optional, Tuple
from dataclasses import dataclass

from custom_dataclasses import ExtractedTable, ExtractionResult
from preprocessing import preprocess_crop_for_paddle, preprocess_page_for_detection, bgr_to_pil, pil_to_bgr, pdf_bytes_to_images
from triplet_extractor import extract_triplets_by_llm
from preprocessing import preprocess_crop_for_paddle, preprocess_page_for_detection, bgr_to_pil, pil_to_bgr, \
    pdf_bytes_to_images
# Import paddle and set device
import paddle

# Force CPU mode
paddle.set_device('cpu')

TABLE_MODEL_ID = "microsoft/table-transformer-detection"
TABLE_SCORE_THRESHOLD = 0.7


@dataclass
class DetectedBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    label: str


def log_processing(message: str) -> None:
    print(f"[processing] {message}")


# убираем NaN и лишние пробелы
def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.fillna("").astype(str)
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].str.strip()
    cleaned = cleaned.loc[(cleaned != "").any(axis=1)]
    cleaned = cleaned.loc[:, (cleaned != "").any(axis=0)]
    return cleaned.reset_index(drop=True)

# преобразуем строки в датафрейм (для случая pdfplumber)
def rows_to_dataframe(rows: List[List[str]]) -> Optional[pd.DataFrame]:
    cleaned_rows: List[List[str]] = []
    for row in rows:
        normalized = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(normalized):
            cleaned_rows.append(normalized)
    if not cleaned_rows:
        return None
    width = max(len(r) for r in cleaned_rows)
    padded = [r + [""] * (width - len(r)) for r in cleaned_rows]
    return sanitize_dataframe(pd.DataFrame(padded))

# --------------------------------- PADDLE OCR ДЛЯ ТАБЛИЦ---------------------------------
@st.cache_resource(show_spinner=False)
def load_paddle_table_engine():
    from paddleocr import PPStructureV3
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        return PPStructureV3(
            lang="ru",
            device="cpu",
            enable_mkldnn=False,
            enable_hpi=False,
            cpu_threads=4,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
        )
        print("Loaded Paddle PPStructureV3 table engine")
        return predictor
    except Exception as exc_v3:
        raise


@st.cache_resource(show_spinner=False)
def load_paddle_text_ocr():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="ru",
        device="cpu",
        enable_mkldnn=False,
        enable_hpi=False,
        cpu_threads=4,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


@st.cache_resource(show_spinner=False)
def load_table_transformer():
    try:
        import torch
        from transformers.models.auto.image_processing_auto import AutoImageProcessor
        from transformers.models.auto.modeling_auto import (
            AutoModelForObjectDetection as TableTransformerForObjectDetection,
        )
    except ImportError as exc:
        return {
            "processor": None,
            "model": None,
            "device": None,
            "torch": None,
            "error": str(exc),
        }

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        processor = AutoImageProcessor.from_pretrained(TABLE_MODEL_ID)
        model = TableTransformerForObjectDetection.from_pretrained(TABLE_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Loaded Table Transformer on {device}")
        return {
            "processor": processor,
            "model": model,
            "device": device,
            "torch": torch,
            "error": None,
        }
    except Exception as exc:
        return {
            "processor": None,
            "model": None,
            "device": None,
            "torch": None,
            "error": str(exc),
        }


# --------------------------------- НЕПОСРЕДСТВЕННО ДЕТЕКЦИЯ ТАБЛИЦ ---------------------------------

# вытаскиваем таблизу из результата paddle через html
def collect_table_html_v3(prediction) -> List[str]:
    html_list: List[str] = []
    html_map = getattr(prediction, "html", None)
    if isinstance(html_map, dict):
        html_list.extend(value for value in html_map.values() if isinstance(value, str))

    payload = getattr(prediction, "json", None)
    if isinstance(payload, dict):
        res = payload.get("res")
        if isinstance(res, dict):
            for item in res.get("table_res_list", []):
                if isinstance(item, dict):
                    html = item.get("pred_html")
                    if isinstance(html, str):
                        html_list.append(html)

    uniq: List[str] = []
    seen: set[str] = set()
    for html in html_list:
        key = html.strip()
        if key and key not in seen:
            uniq.append(key)
            seen.add(key)
    return uniq


def html_to_dataframes(table_html: str) -> List[pd.DataFrame]:
    try:
        candidates = pd.read_html(
            io.StringIO(table_html),
            decimal=",",
            thousands=" ",
            keep_default_na=False
        ) # разделитель для нецелых чисел это запятая. Если не указать явно, не будет считываться 0,67, результат будет 067. 
        # Но будет плохо работать в случае если разделителем будет точка.
    except ValueError:
        return []
    frames: List[pd.DataFrame] = []
    for frame in candidates:
        cleaned = sanitize_dataframe(frame)
        if not cleaned.empty:
            frames.append(cleaned)
    return frames

# --------------------------------- НЕПОСРЕДСТВЕННО РАСПОЗНАВАНИЕ ТАБЛИЦ ---------------------------------
def extract_tables_with_paddle(crop_bgr: np.ndarray) -> List[pd.DataFrame]:
    engine = load_paddle_table_engine()
    html_list: List[str] = []
    predictions = engine.predict(
        crop_bgr,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_table_recognition=True,
    )
    for pred in predictions:
        html_list.extend(collect_table_html_v3(pred))

    frames: List[pd.DataFrame] = []
    for html in html_list:
        frames.extend(html_to_dataframes(html))
    print(f"Paddle OCR extracted {len(frames)} table frame(s) from crop")
    return frames

# --------------------------------- ОБРАБОТКА КАРТИНОК  ---------------------------------
def process_images_with_detector(images_bgr: List[np.ndarray], source_name: str) -> ExtractionResult:
    all_tables: List[ExtractedTable] = []
    debug_images: List[np.ndarray] = []
    log_processing(f"Method=PP Structure V3, source={source_name}, pages={len(images_bgr)}")

    for page_idx, raw_page in enumerate(images_bgr, start=1):
        log_processing(f"PP Structure V3: processing page {page_idx}")
        page = preprocess_page_for_detection(raw_page)
        prepared_crop = preprocess_crop_for_paddle(page)  # обрежем файл по найденному ббоксу
        tables = extract_tables_with_paddle(prepared_crop)  # уже из обрезанного варианта извлечем таблицы
        if not tables:
            log_processing(f"PP Structure V3: no tables found on page {page_idx}")
            continue
        for table_idx, df in enumerate(tables, start=1):
            all_tables.append(
                ExtractedTable(
                    source=(
                        f"Page {page_idx}, table {table_idx}"
                    ),
                    dataframe=df,
                )
            )
        log_processing(f"PP Structure V3: found {len(tables)} table(s) on page {page_idx}")
    method = f"PP Structure V3"
    return ExtractionResult(method=method, tables=all_tables, debug_images=debug_images)
# --------------------------------- ИЗВЛЕЧЕНИЕ TEXT LIKE ТАБЛИЦ (или таблиц с объединенными вертикально клетками) ---------------------------------
def process_images_with_ocr(image_bgr: np.ndarray, source_name: str):
    log_processing(f"PaddleOCR: running OCR for source={source_name}")
    ocr = load_paddle_text_ocr()
    result = ocr.predict(image_bgr)
    json_results = []
    for res in result:
        res.print()
        res.save_to_img("output")
        json_results.append(res.json) 
    return json_results


def smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def preprocess_for_projection(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    return cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )


def preprocess_page_for_detection_adaptive(
        image_bgr: np.ndarray,
        table_type: str = "auto",
) -> np.ndarray:
    if table_type == "auto":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            100,
            minLineLength=100,
            maxLineGap=10,
        )
        table_type = "bordered" if lines is not None and len(lines) > 20 else "borderless"

    if table_type == "bordered":
        return preprocess_page_for_detection(image_bgr)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)


def iou(box1: DetectedBox, box2: DetectedBox) -> float:
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.w, box2.x + box2.w)
    y2 = min(box1.y + box1.h, box2.y + box2.h)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1.w * box1.h
    area2 = box2.w * box2.h
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(
        boxes: List[DetectedBox],
        iou_threshold: float = 0.5,
) -> List[DetectedBox]:
    if not boxes:
        return []

    pending = sorted(boxes, key=lambda item: item.score, reverse=True)
    result: List[DetectedBox] = []
    while pending:
        best = pending.pop(0)
        result.append(best)
        pending = [box for box in pending if iou(best, box) < iou_threshold]
    return result


def detect_table_regions_transformer_improved(
        image_bgr: np.ndarray,
        score_threshold: float = 0.5,
        use_multiscale: bool = True,
) -> Tuple[List[DetectedBox], Optional[str]]:
    bundle = load_table_transformer()
    if not bundle or bundle.get("error") or bundle.get("processor") is None or bundle.get("model") is None:
        log_processing(f"Table Transformer unavailable: {(bundle or {}).get('error') or 'unknown error'}")
        return [], (bundle or {}).get("error") or "Table detector is unavailable"

    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]
    torch = bundle["torch"]

    height, width = image_bgr.shape[:2]
    pil_image = bgr_to_pil(image_bgr)
    all_boxes: List[DetectedBox] = []

    scales = [1.0] if not use_multiscale else [0.5, 0.75, 1.0, 1.25, 1.5]
    for scale in scales:
        if scale != 1.0:
            scaled_pil = pil_image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.LANCZOS,
            )
        else:
            scaled_pil = pil_image

        inputs = processor(images=scaled_pil, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([scaled_pil.size[::-1]], device=device)
        detections = processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]
        id2label = model.config.id2label

        for score, label_id, box in zip(
                detections["scores"],
                detections["labels"],
                detections["boxes"],
        ):
            label_name = str(id2label.get(int(label_id), int(label_id))).lower()
            if "table" not in label_name:
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            all_boxes.append(
                DetectedBox(
                    x=x1,
                    y=y1,
                    w=max(1, x2 - x1),
                    h=max(1, y2 - y1),
                    score=float(score.item()),
                    label=label_name,
                )
            )

    if len(all_boxes) > 1:
        all_boxes = non_max_suppression(all_boxes, iou_threshold=0.5)

    all_boxes.sort(key=lambda box: (box.y, box.x))
    log_processing(f"Table Transformer: detected {len(all_boxes)} region(s)")
    return all_boxes, None


def merge_textlike_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        has_merged = False
        for idx, (mx1, my1, mx2, my2) in enumerate(merged):
            overlaps = not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2)
            close_y = abs(y1 - my2) <= 20 or abs(y2 - my1) <= 20
            if overlaps or close_y:
                merged[idx] = (
                    min(x1, mx1),
                    min(y1, my1),
                    max(x2, mx2),
                    max(y2, my2),
                )
                has_merged = True
                break
        if not has_merged:
            merged.append((x1, y1, x2, y2))
    return merged


def detect_textlike_table_regions(
        image_bgr: np.ndarray,
        min_table_height: int = 60,
        min_table_width: int = 220,
) -> List[DetectedBox]:
    bw = preprocess_for_projection(image_bgr)
    height, width = bw.shape[:2]

    h_proj = (bw > 0).sum(axis=1).astype(np.float32)
    h_proj_smooth = smooth_signal(h_proj, window=max(7, height // 100))
    text_threshold = np.percentile(h_proj_smooth, 35)
    text_regions = h_proj_smooth > text_threshold

    row_spans: List[Tuple[int, int]] = []
    start = None
    for y, has_text in enumerate(text_regions):
        if has_text and start is None:
            start = y
        elif not has_text and start is not None:
            if y - start >= min_table_height:
                row_spans.append((start, y))
            start = None
    if start is not None and height - start >= min_table_height:
        row_spans.append((start, height))

    candidate_boxes: List[Tuple[int, int, int, int]] = []
    for y1, y2 in row_spans:
        row_crop = bw[y1:y2, :]
        v_proj = (row_crop > 0).sum(axis=0).astype(np.float32)
        v_proj_smooth = smooth_signal(v_proj, window=max(5, width // 120))
        empty_threshold = np.percentile(v_proj_smooth, 22)
        empty_regions = v_proj_smooth <= empty_threshold

        gaps: List[Tuple[int, int]] = []
        gap_start = None
        for x, is_empty in enumerate(empty_regions):
            if is_empty and gap_start is None:
                gap_start = x
            elif not is_empty and gap_start is not None:
                if x - gap_start >= 10:
                    gaps.append((gap_start, x))
                gap_start = None
        if gap_start is not None and width - gap_start >= 10:
            gaps.append((gap_start, width))

        if len(gaps) >= 2:
            candidate_boxes.append((gaps[0][0], y1, gaps[-1][1], y2))

    merged_boxes = merge_textlike_boxes(candidate_boxes)
    return [
        DetectedBox(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            score=0.7,
            label="table_projection",
        )
        for x1, y1, x2, y2 in merged_boxes
        if (x2 - x1) >= min_table_width and (y2 - y1) >= min_table_height
    ]


def detect_table_regions_hybrid(
        image_bgr: np.ndarray,
) -> Tuple[List[DetectedBox], Optional[str]]:
    log_processing("Hybrid detector: trying Table Transformer first")
    transformer_boxes, error = detect_table_regions_transformer_improved(
        image_bgr,
        score_threshold=0.4,
    )

    if not transformer_boxes:
        log_processing("Hybrid detector: transformer returned no boxes, switching to projection detector")
        projection_boxes = detect_textlike_table_regions(image_bgr)
        if projection_boxes:
            log_processing(f"Hybrid detector: projection detector found {len(projection_boxes)} region(s)")
            return projection_boxes, None
    elif len(transformer_boxes) == 1 and transformer_boxes[0].score < 0.6:
        log_processing("Hybrid detector: low-confidence transformer result, adding projection detector")
        projection_boxes = detect_textlike_table_regions(image_bgr)
        if projection_boxes:
            log_processing(
                f"Hybrid detector: merged transformer + projection boxes, total before NMS={len(transformer_boxes) + len(projection_boxes)}"
            )
            return non_max_suppression(
                transformer_boxes + projection_boxes,
                iou_threshold=0.3,
            ), None

    log_processing(f"Hybrid detector: using transformer result, boxes={len(transformer_boxes)}")
    return transformer_boxes, error


def draw_boxes(image_bgr: np.ndarray, boxes: List[DetectedBox]) -> np.ndarray:
    debug = image_bgr.copy()
    for box in boxes:
        cv2.rectangle(
            debug,
            (box.x, box.y),
            (box.x + box.w, box.y + box.h),
            (0, 180, 0),
            2,
        )
        text = f"{box.label} {box.score:.2f}" if box.score else box.label
        cv2.putText(
            debug,
            text,
            (box.x, max(15, box.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    return debug


def expand_box(box: DetectedBox, img_shape, pad_x: int = 12, pad_y: int = 12) -> DetectedBox:
    h, w = img_shape[:2]
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.x + box.w + pad_x)
    y2 = min(h, box.y + box.h + pad_y)
    return DetectedBox(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        score=box.score,
        label=box.label,
    )


def extract_textlike_table_from_image(
        image_bgr: np.ndarray,
        source: str,
) -> Optional[ExtractedTable]:
    ocr_json = process_images_with_ocr(image_bgr, source_name=source)
    result = build_table_from_ocr_json(ocr_json)
    if not result.tables:
        return None

    dataframe = sanitize_dataframe(result.tables[0].dataframe)
    if dataframe.empty:
        return None

    return ExtractedTable(source=source, dataframe=dataframe)


def process_textlike_document(uploaded_name: str, raw_bytes: bytes) -> ExtractionResult:
    if uploaded_name.lower().endswith(".pdf"):
        pages = pdf_bytes_to_images(raw_bytes, scale=3.2)
        log_processing(f"Text-like mode: input=PDF, pages={len(pages)}")
    else:
        pages = [pil_to_bgr(Image.open(io.BytesIO(raw_bytes)))]
        log_processing("Text-like mode: input=image, pages=1")

    all_tables: List[ExtractedTable] = []
    debug_images: List[np.ndarray] = []
    log_processing("Method=Hybrid detector (Transformer + Projections) + PaddleOCR")

    for page_idx, page in enumerate(pages, start=1):
        log_processing(f"Text-like mode: processing page {page_idx}")
        page_processed = preprocess_page_for_detection_adaptive(page, table_type="auto")
        boxes, detector_error = detect_table_regions_hybrid(page_processed)
        log_processing(f"Text-like mode: page {page_idx}, detector boxes={len(boxes)}")

        if boxes:
            debug_images.append(draw_boxes(page_processed, boxes))

        extracted_on_page = False
        for box_idx, box in enumerate(boxes, start=1):
            if box.label == "table_projection":
                pad_x, pad_y = 50, 20
            else:
                pad_x, pad_y = 300, 30

            expanded_box = expand_box(box, page_processed.shape, pad_x=pad_x, pad_y=pad_y)
            crop = page_processed[
                expanded_box.y: expanded_box.y + expanded_box.h,
                expanded_box.x: expanded_box.x + expanded_box.w,
            ]
            try:
                log_processing(
                    f"Text-like mode: page {page_idx}, crop {box_idx}, detector={box.label}, score={box.score:.2f}"
                )
                table = extract_textlike_table_from_image(
                    crop,
                    source=(
                        f"Page {page_idx}, table {box_idx} "
                        f"({box.label}, score={box.score:.2f})"
                    ),
                )
            except Exception as exc:
                log_processing(
                    f"Text-like mode: page {page_idx}, crop {box_idx}, OCR extraction failed: {exc}"
                )
                table = None

            if table is not None:
                all_tables.append(table)
                extracted_on_page = True
                log_processing(f"Text-like mode: page {page_idx}, crop {box_idx}, table extracted successfully")

        if extracted_on_page:
            continue

        log_processing(f"Text-like mode: page {page_idx}, falling back to full-page OCR")
        try:
            table = extract_textlike_table_from_image(
                page_processed,
                source=f"Page {page_idx} (full page OCR fallback)",
            )
        except Exception as exc:
            log_processing(f"Text-like mode: page {page_idx}, full-page OCR fallback failed: {exc}")
            table = None

        if table is not None:
            all_tables.append(table)
            log_processing(f"Text-like mode: page {page_idx}, full-page OCR fallback extracted a table")

        if detector_error:
            log_processing(f"Text-like mode: detector error on page {page_idx}: {detector_error}")

    return ExtractionResult(
        method="Hybrid detector (Transformer + Projections) + PaddleOCR",
        tables=all_tables,
        debug_images=debug_images,
    )


def calculate_overlapping(a, b):
    top = max(a["y1"], b["y1"])
    bottom = min(a["y2"], b["y2"])
    return max(0, bottom - top)

def estimate_y_threshold(items, user_threshold=None):
    if user_threshold is not None:
        return user_threshold

    heights = [it["y2"] - it["y1"] for it in items]
    if not heights:
        return 18

    median_h = float(np.median(heights))
    return max(8, int(round(median_h * 0.6)))


def estimate_x_tolerance(rows):
    widths = []
    gaps = []

    for row in rows:
        row = sorted(row, key=lambda x: x["cx"])
        widths.extend([(it["x2"] - it["x1"]) for it in row])

        if len(row) >= 2:
            row_gaps = [row[i + 1]["cx"] - row[i]["cx"] for i in range(len(row) - 1)]
            gaps.extend([g for g in row_gaps if g > 0])

    median_w = float(np.median(widths)) if widths else 20.0
    median_gap = float(np.median(gaps)) if gaps else median_w * 2.5
    x_tol = min(median_gap * 0.42, median_w * 1.25)
    return max(12, int(round(x_tol)))


def cluster_column_centers(rows, x_tolerance):
    all_centers = sorted(it["cx"] for row in rows for it in row)
    if not all_centers:
        return []

    clusters = [{"values": [all_centers[0]], "center": all_centers[0]}]

    for cx in all_centers[1:]:
        last = clusters[-1]
        if abs(cx - last["center"]) <= x_tolerance:
            last["values"].append(cx)
            last["center"] = float(np.mean(last["values"]))
        else:
            clusters.append({"values": [cx], "center": cx})

    return [cl["center"] for cl in clusters]


def build_table_from_ocr_json(ocr_result, y_threshold=None, x_tolerance=None) -> ExtractionResult:
    if not ocr_result:
        raise ValueError("OCR result is empty")
    
    page = ocr_result[0].get("res", ocr_result[0])

    rec_texts = page.get("rec_texts", [])
    rec_boxes = page.get("rec_boxes", [])
    rec_scores = page.get("rec_scores", [])

    items = []
    for text, box, score in zip(rec_texts, rec_boxes, rec_scores):
        if not str(text).strip():
            continue

        x1, y1, x2, y2 = box
        items.append({
            "text": str(text).strip(),
            "score": score,
            "x1": float(x1), "y1": float(y1),
            "x2": float(x2), "y2": float(y2),
            "cx": (float(x1) + float(x2)) / 2.0,
            "cy": (float(y1) + float(y2)) / 2.0,
        })

    if not items:
        raise ValueError("OCR contains no non-empty text items")

    # ---------- группировка по строкам ----------
    items.sort(key=lambda x: (x["cy"], x["x1"]))
    y_threshold = estimate_y_threshold(items, y_threshold)

    rows = []
    for item in items:
        if not rows:
            rows.append([item])
            continue

        last_row = rows[-1]
        row_center = float(np.mean([x["cy"] for x in last_row]))
        row_proto = {
            "y1": min(x["y1"] for x in last_row),
            "y2": max(x["y2"] for x in last_row),
        }
        row_height = row_proto["y2"] - row_proto["y1"]

        same_row = (
            abs(item["cy"] - row_center) <= max(y_threshold, row_height * 0.35)
            or calculate_overlapping(item, row_proto) >= min((item["y2"] - item["y1"]), row_height) * 0.25
        )

        if same_row:
            last_row.append(item)
        else:
            rows.append([item])

    rows = [sorted(row, key=lambda x: x["x1"]) for row in rows]

    # ---------- построение колонок ----------
    if x_tolerance is None:
        x_tolerance = estimate_x_tolerance(rows)

    col_centers = cluster_column_centers(rows, x_tolerance)

    if not col_centers:
        raise ValueError("Failed to estimate column centers")

    # ---------- выравнивание строк по колонкам ----------
    table = []
    for row in rows:
        aligned = [None] * len(col_centers)

        for cell in row:
            distances = [abs(cell["cx"] - cc) for cc in col_centers]
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]

            if best_dist > x_tolerance:
                continue

            if aligned[best_idx] is None:
                aligned[best_idx] = cell["text"]
            else:
                aligned[best_idx] = f"{aligned[best_idx]} {cell['text']}".strip()

        table.append(aligned)

    # ---------- удаление полностью пустых колонок ----------
    non_empty_cols = [
        j for j in range(len(col_centers))
        if any(row[j] is not None and str(row[j]).strip() for row in table)
    ]

    if non_empty_cols:
        table = [[row[j] for j in non_empty_cols] for row in table]

    df = pd.DataFrame(table)

    extracted_table = ExtractedTable(
        source="ocr for text",
        dataframe=df,
    )

    return ExtractionResult(
        method="PaddleOCR",
        tables=[extracted_table],
        debug_images=[],
    )


# --------------------------------- ИЗВЛЕЧЕНИЕ ТАБЛИЦ ИЗ ЦИФРОВЫХ PDF ---------------------------------
def extract_tables_born_digital(pdf_bytes: bytes) -> List[ExtractedTable]:
    found: List[ExtractedTable] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            for table_idx, rows in enumerate(page.extract_tables() or [], start=1):
                df = rows_to_dataframe(rows)
                if df is None or df.empty:
                    continue
                found.append(
                    ExtractedTable(
                        source=f"Page {page_idx}, table {table_idx} (pdfplumber)",
                        dataframe=df,
                    )
                )
    print(f"pdfplumber extracted {len(found)} table(s)")
    return found


def process_born_digital_pdf(pdf_bytes: bytes) -> ExtractionResult:
    log_processing("Born-digital mode: trying pdfplumber first")
    direct_tables = extract_tables_born_digital(pdf_bytes)
    if direct_tables:
        log_processing(f"Born-digital mode: pdfplumber extracted {len(direct_tables)} table(s)")
        return ExtractionResult(
            method="pdfplumber",
            tables=direct_tables,
            debug_images=[],
        )

    log_processing("Born-digital mode: pdfplumber found no tables, fallback to detector branch")
    pages = pdf_bytes_to_images(pdf_bytes, scale=3.5)
    return process_images_with_detector(pages, source_name="born_digital_fallback")


# функции для вызова
def process_scan(uploaded_name: str, raw_bytes: bytes) -> ExtractionResult:
    if uploaded_name.lower().endswith(".pdf"):
        pages = pdf_bytes_to_images(raw_bytes, scale=3.5)
        log_processing(f"Scan mode: input=PDF, pages={len(pages)}")
    else:
        pages = [pil_to_bgr(Image.open(io.BytesIO(raw_bytes)))]
        log_processing("Scan mode: input=image, pages=1")
    return process_images_with_detector(pages, source_name="scan")

def process_screenshot(raw_bytes: bytes) -> ExtractionResult:
    image = pil_to_bgr(Image.open(io.BytesIO(raw_bytes)))
    return process_images_with_detector([image], source_name="screenshot")

def process_textlike_tables(raw_bytes: bytes):
    image = pil_to_bgr(Image.open(io.BytesIO(raw_bytes)))
    return process_images_with_ocr(image, source_name="textlike")

def build_excel_bytes(tables: List[ExtractedTable]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for idx, table in enumerate(tables, start=1):
            sheet_name = f"Table_{idx}"
            table.dataframe.to_excel(writer, sheet_name=sheet_name[:31], index=True)
    output.seek(0)
    return output.read()

# для вывода на экран пдф документа
def show_file_preview(uploaded_file, file_type: str):
    if uploaded_file is None:
        return
    raw = uploaded_file.getvalue()
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    if is_pdf:
        try:
            pages = pdf_bytes_to_images(raw, scale=1.5)
            if pages:
                st.image(
                    cv2.cvtColor(pages[0], cv2.COLOR_BGR2RGB),
                    caption="Превью первой страницы",
                    width='stretch',
                )
        except Exception as exc:
            st.info("Не удалось отобразить превью, но файл можно обработать.")
        return
    st.image(raw, caption="Превью скана", width='stretch')

# вывод таблицы и добавления сохранения файла в эксель
# def render_result(result: ExtractionResult, prefix: str):
#     st.write(f"Метод обработки: `{result.method}`")
#     if not result.tables:
#         st.warning("Таблицы не найдены.")
#         return

#     for idx, table in enumerate(result.tables, start=1):
#         st.markdown(f"**Таблица {idx}**")
#         st.caption(table.source)
#         st.dataframe(table.dataframe, width='stretch')

#     try:
#         excel_bytes = build_excel_bytes(result.tables)
#         st.download_button(
#             "Скачать Excel",
#             data=excel_bytes,
#             file_name=f"{prefix}_tables.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             key=f"download_{prefix}",
#         )
#     except Exception as exc:
#         st.error("Не удалось собрать Excel-файл.")

def render_result(result: ExtractionResult, prefix: str):
    st.write(f"Метод обработки: `{result.method}`")
    if not result.tables:
        st.warning("Таблицы не найдены.")
        return

    for idx, table in enumerate(result.tables, start=1):
        st.markdown(f"**Таблица {idx}**")
        st.caption(table.source)
        st.dataframe(table.dataframe, width="stretch")

    try:
        excel_bytes = build_excel_bytes(result.tables)
        st.download_button(
            "Скачать Excel",
            data=excel_bytes,
            file_name=f"{prefix}_tables.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_{prefix}",
        )
    except Exception as exc:
        st.error(f"Не удалось собрать Excel-файл: {exc}")

    st.markdown("---")
    st.subheader("Извлечение триплетов")

    triplets_key = f"{prefix}_triplets"
    triplets_error_key = f"{prefix}_triplets_error"

    if st.button("Извлечь триплеты", key=f"extract_triplets_{prefix}"):
        try:
            with st.spinner("Извлекаю триплеты..."):
                triplets_result = extract_triplets_by_llm(result)
                st.session_state[triplets_key] = triplets_result
                st.session_state[triplets_error_key] = ""
        except Exception as exc:
            st.session_state[triplets_key] = None
            st.session_state[triplets_error_key] = str(exc)

    triplets_error = st.session_state.get(triplets_error_key, "")
    triplets_result = st.session_state.get(triplets_key)

    if triplets_error:
        st.error(f"Ошибка извлечения триплетов: {triplets_error}")
    elif triplets_result is not None:
        st.write("Результат извлечения:")
        st.json(triplets_result)

        json_bytes = json.dumps(
            triplets_result,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        st.download_button(
            "Скачать JSON с триплетами",
            data=json_bytes,
            file_name=f"{prefix}_triplets.json",
            mime="application/json",
            key=f"download_triplets_{prefix}",
        )
# просто логика работы и отрисовка страницы
def main():
    st.set_page_config(page_title="Table Extraction", layout="wide")
    st.title("Извлечение таблиц из PDF и сканов")

    left, right = st.columns([1, 2], gap="large")
    # загрузка файла на левой части экрана
    with left:
        source_type = st.radio(
            "Тип документа",
            options=["Чистый структурированный PDF", "Скан", "Text-like или почти без границ (PDF или изображение)"],
            # добавление типовых сценариев обработки
            index=0,
        )
        if source_type == "Чистый структурированный PDF":
            file_type = "born_digital"
            allowed_types = ["pdf"]
        elif source_type == "Скан":
            file_type = "scan"
            allowed_types = ["pdf"]
        else:
            file_type = "text"
            allowed_types = ["png", "jpeg", "jpg"]

        uploaded = st.file_uploader("Загрузите файл", type=allowed_types, key="main_upload")  # загрузка файла
        show_file_preview(uploaded, file_type)

        run = st.button(
            "Извлечь таблицы",
            type="primary",
            disabled=uploaded is None,
        )
        if run and uploaded is not None: # запуск распознавания если файл загружен и нажата кнопка извлечения таблиц
            try:
                with st.spinner("Идет обработка..."):
                    raw = uploaded.getvalue()
                    if file_type == "born_digital":
                        result = process_born_digital_pdf(raw) # если пдф цифровой а не скан, то просто достаем таблицу через pdfplumber
                    elif file_type == "scan":
                        result = process_scan(uploaded.name, raw) # иначе запускаем OCR
                    else:
                        result = build_table_from_ocr_json(process_textlike_tables(raw))
                    st.session_state["main_result"] = result
                    st.session_state["main_error"] = ""
                    st.session_state["main_triplets"] = None
                    st.session_state["main_triplets_error"] = ""
            except Exception as exc:
                st.session_state["main_result"] = None
                st.session_state["main_error"] = str(exc)
    # отображение результата на правой части экрана
    with right:
        error = st.session_state.get("main_error", "")
        result = st.session_state.get("main_result")
        if error:
            st.error(f"Ошибка обработки: {error}")
        elif result is not None:
            render_result(result, prefix="main")
        else:
            st.info("Выберите тип документа и загрузите файл")

        st.markdown("---")
        st.subheader("Если результат неверный")
        st.write("Можно попробовать загрузить скриншот таблицы и обработать его повторно")

        screenshot = st.file_uploader(
            "Скриншот таблицы (PNG/JPG/JPEG)",
            type=["png", "jpg", "jpeg"],
            key="shot_upload",
        )
        rerun = st.button("Обработать скриншот", disabled=screenshot is None, key="shot_button")
        if rerun and screenshot is not None: # если не распозналось или распозналось плохо, можно попробовать загрузить просто скриншот таблицы с обработкой через детектор + OCR, иногда работает лучше
            try:
                with st.spinner("Обрабатка..."):
                    shot_result = process_screenshot(screenshot.getvalue())
                    st.session_state["shot_result"] = shot_result
                    st.session_state["shot_error"] = ""
                    st.session_state["screenshot_triplets"] = None
                    st.session_state["screenshot_triplets_error"] = ""
            except Exception as exc:
                st.session_state["shot_result"] = None
                st.session_state["shot_error"] = str(exc)

        shot_error = st.session_state.get("shot_error", "")
        shot_result = st.session_state.get("shot_result")
        if shot_error:
            st.error(f"Ошибка обработки: {shot_error}")
        elif shot_result is not None:
            st.markdown("Результат")
            render_result(shot_result, prefix="screenshot")


if __name__ == "__main__":
    main()