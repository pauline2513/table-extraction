import io
import logging
import os

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from PIL import Image

from custom_dataclasses import DetectedBox, ExtractedTable, ExtractionResult
from preprocessing import preprocess_crop_for_paddle, preprocess_page_for_detection, bgr_to_pil, pil_to_bgr, pdf_bytes_to_images

TABLE_MODEL_ID = "microsoft/table-transformer-detection"
TABLE_SCORE_THRESHOLD = 0.7


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("app_paddle")


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

# --------------------------------- ЗАГРУЗКА ДЕТЕКТОРА ТАБЛИЦ ---------------------------------
@st.cache_resource(show_spinner=False)
def load_table_transformer():
    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        processor = AutoImageProcessor.from_pretrained(TABLE_MODEL_ID)
        model = TableTransformerForObjectDetection.from_pretrained(TABLE_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info("Loaded table detector model on %s", device)
        return {"processor": processor, "model": model, "device": device, "error": None}
    except Exception as exc:
        logger.exception("Failed to load table detector")
        return {"processor": None, "model": None, "device": None, "error": str(exc)}



# --------------------------------- PADDLE OCR ДЛЯ ТАБЛИЦ---------------------------------
@st.cache_resource(show_spinner=False)
def load_paddle_table_engine():
    from paddleocr import PPStructureV3
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        predictor = PPStructureV3(
            lang="ru",
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
        )
        logger.info("Loaded Paddle PPStructureV3 table engine")
        return predictor
    except Exception as exc_v3:
        logger.warning("PPStructureV3 unavailable: %s", exc_v3)
        return None

# --------------------------------- НЕПОСРЕДСТВЕННО ДЕТЕКЦИЯ ТАБЛИЦ ---------------------------------
def detect_table_regions_transformer(image_bgr: np.ndarray) -> Tuple[List[DetectedBox], Optional[str]]:
    import torch
    bundle = load_table_transformer()
    if bundle is None:
        return [], "Something went wrong with Paddle OCR"

    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    pil_image = bgr_to_pil(image_bgr)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
    detections = processor.post_process_object_detection(
        outputs=outputs,
        threshold=TABLE_SCORE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]
    id2label = model.config.id2label
    height, width = image_bgr.shape[:2]

    boxes: List[DetectedBox] = []
    for score, label_id, box in zip(
        detections["scores"], detections["labels"], detections["boxes"]
    ):
        label_name = str(id2label.get(int(label_id), int(label_id))).lower()
        if "table" not in label_name:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        boxes.append(
            DetectedBox(
                x=x1,
                y=y1,
                w=max(1, x2 - x1),
                h=max(1, y2 - y1),
                score=float(score.item()),
                label=label_name,
            )
        )
    boxes.sort(key=lambda b: (b.y, b.x))
    logger.info("Transformer detector found %s region(s)", len(boxes))
    return boxes, None

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
    logger.info("Paddle OCR extracted %s table frame(s) from crop", len(frames))
    return frames

# для отрисовки ббоксов на документах
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

# функция для паддинга найденного ббокса таблицы
def expand_box(box, img_shape, pad_x=12, pad_y=12): 
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

# --------------------------------- ОБРАБОТКА КАРТИНОК С ДЕТЕКТОРОМ ---------------------------------
def process_images_with_detector(images_bgr: List[np.ndarray], source_name: str) -> ExtractionResult:
    all_tables: List[ExtractedTable] = []
    debug_images: List[np.ndarray] = []

    for page_idx, raw_page in enumerate(images_bgr, start=1):
        logger.info("Processing %s page %s", source_name, page_idx)
        page = preprocess_page_for_detection(raw_page)
        boxes, detector_error = detect_table_regions_transformer(page)
        if detector_error:
            logger.warning("Transformer detector error on page %s: %s", page_idx, detector_error)
        if not boxes:
            boxes = [DetectedBox(0, 0, page.shape[1], page.shape[0], 0.0, "full_page")]
            logger.info("No table boxes found, fallback to full-page OCR on page %s", page_idx)

        debug_images.append(draw_boxes(page, boxes))
        # print(boxes)
        for box_idx, box in enumerate(boxes, start=1):
            box = expand_box(box, page.shape, pad_x=300, pad_y=30) # чуть расширим ббокс найденной таблицы, 
                                                                # в особенности по оси x из-за того что может быть смещенный результат детекции и влияет на распознавания текста, выяавлено и настроено эмпирически
            crop = page[box.y : box.y + box.h, box.x : box.x + box.w]
            prepared_crop = preprocess_crop_for_paddle(crop) # обрежем файл по найденному ббоксу
            tables = extract_tables_with_paddle(prepared_crop) # уже из обрезанного варианта извлечем таблицы
            if not tables:
                logger.info("No tables in crop %s on page %s", box_idx, page_idx)
                continue
            for table_idx, df in enumerate(tables, start=1):
                all_tables.append(
                    ExtractedTable(
                        source=(
                            f"Page {page_idx}, box {box_idx}, table {table_idx} "
                            f"({box.label})"
                        ),
                        dataframe=df,
                    )
                )
    method = f"{TABLE_MODEL_ID} + PaddleOCR"
    return ExtractionResult(method=method, tables=all_tables, debug_images=debug_images)
# --------------------------------- ИЗВЛЕЧЕНИЕ TEXT LIKE ТАБЛИЦ ---------------------------------
def process_images_with_ocr(image_bgr: np.ndarray, source_name: str):
    logger.info("Processing %s page %s", source_name)
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        lang="ru",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    result = ocr.predict(image_bgr)
    json_results = []
    for res in result:
        res.print()
        res.save_to_img("output")
        json_results.append(res.json) 
    return json_results

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
        logger.info("pdfplumber opened PDF with %s page(s)", len(pdf.pages))
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
    logger.info("pdfplumber extracted %s table(s)", len(found))
    return found


def process_born_digital_pdf(pdf_bytes: bytes) -> ExtractionResult:
    direct_tables = extract_tables_born_digital(pdf_bytes)
    if direct_tables:
        return ExtractionResult(
            method="pdfplumber",
            tables=direct_tables,
            debug_images=[],
        )

    logger.info("pdfplumber found no tables, fallback to detector branch")
    pages = pdf_bytes_to_images(pdf_bytes, scale=3.5)
    return process_images_with_detector(pages, source_name="born_digital_fallback")


# функции для вызова
def process_scan(uploaded_name: str, raw_bytes: bytes) -> ExtractionResult:
    if uploaded_name.lower().endswith(".pdf"):
        pages = pdf_bytes_to_images(raw_bytes, scale=3.5)
    else:
        pages = [pil_to_bgr(Image.open(io.BytesIO(raw_bytes)))]
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
            logger.exception("Failed to render PDF preview: %s", exc)
            st.info("Не удалось отобразить превью, но файл можно обработать.")
        return
    st.image(raw, caption="Превью скана", width='stretch')

# вывод таблицы и добавления сохранения файла в эксель
def render_result(result: ExtractionResult, prefix: str):
    st.write(f"Метод обработки: `{result.method}`")
    if not result.tables:
        st.warning("Таблицы не найдены.")
        return

    for idx, table in enumerate(result.tables, start=1):
        st.markdown(f"**Таблица {idx}**")
        st.caption(table.source)
        st.dataframe(table.dataframe, width='stretch')

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
        logger.exception(f"Failed to build xlsx: {exc}")
        st.error("Не удалось собрать Excel-файл.")

    if result.debug_images:
        with st.expander("Найденные области таблиц", expanded=False):
            for page_idx, debug in enumerate(result.debug_images, start=1):
                st.image(
                    cv2.cvtColor(debug, cv2.COLOR_BGR2RGB),
                    caption=f"Страница {page_idx}",
                    width='stretch',
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
            options=["Чистый структурированный PDF", "Скан", "Text-like или почти без границ (надо загрузить скриншот)"], # добавление типовых сценариев обработки
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

        uploaded = st.file_uploader("Загрузите файл", type=allowed_types, key="main_upload") # загрузка файла
        show_file_preview(uploaded, file_type)

        run = st.button(
            "Извлечь таблицы",
            type="primary",
            disabled=uploaded is None,
        )
        if run and uploaded is not None: # запуск распознавания если файл загружен и нажата кнопка извлечения таблиц
            try:
                logger.info(f"Started main processing for file {uploaded.name}, type={file_type}")
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
                # logger.info("Main processing finished. Tables found: %s", len(result.tables))
            except Exception as exc:
                logger.exception("Main processing failed: %s", exc)
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
                logger.info("Started screenshot processing: %s", screenshot.name)
                with st.spinner("Обрабатка..."):
                    shot_result = process_screenshot(screenshot.getvalue())
                    st.session_state["shot_result"] = shot_result
                    st.session_state["shot_error"] = ""
                logger.info("Screenshot processing finished. Tables found: %s", len(shot_result.tables))
            except Exception as exc:
                logger.exception(f"Screenshot processing failed: {exc}")
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