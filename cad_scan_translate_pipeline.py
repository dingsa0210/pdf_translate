"""
Unified CAD scanned-PDF Chinese -> English translation pipeline.

Design goals:
  PDF render -> overlap tiled OCR -> OCR fusion/dedup -> short translation
  -> hard-clipped refill -> output PDF.

This script is intentionally separate from the historical
scan_translate_pipeline.py and ppstructure_translate_pipeline.py so their
existing experiment outputs remain comparable.
"""

import gc
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import fitz
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from config import (
    ENGINEERING_DICT,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_BATCH_SIZE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RENDER_DPI,
    TRANSLATE_ENGINE,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.environ.get(
    "PDF_PATH",
    os.path.join(BASE_DIR, "pdfs", "地脚螺栓预埋铁分布图-Model_1.pdf"),
)
WORK_DIR = os.environ.get("WORK_DIR", os.path.join(BASE_DIR, "cad_scan_work"))
OUTPUT_PDF = os.environ.get("OUTPUT_PDF", os.path.join(WORK_DIR, "output_cad_scan_translated.pdf"))

OCR_ENGINE = os.environ.get("OCR_ENGINE", "both").lower()  # both | ppstructure | rapid
OCR_TILE_SIZE = int(os.environ.get("OCR_TILE_SIZE", "2200"))
OCR_TILE_OVERLAP = int(os.environ.get("OCR_TILE_OVERLAP", "220"))
PP_MAX_SIDE_LEN = int(os.environ.get("PP_MAX_SIDE_LEN", str(OCR_TILE_SIZE)))
ANGLE_NEAR_HORIZONTAL = float(os.environ.get("ANGLE_NEAR_HORIZONTAL", "5.0"))
RIGHT_MARGIN = int(os.environ.get("RIGHT_MARGIN", "8"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.15"))
ENABLE_LINE_MERGE = os.environ.get("ENABLE_LINE_MERGE", "1") == "1"

FONT_PATH = os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Arial.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = ""

os.makedirs(WORK_DIR, exist_ok=True)
LOG_DIR = os.path.join(WORK_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"cad_scan_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logger.remove()
logger.add(
    sys.stdout,
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add(
    LOG_FILE,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="20 MB",
    retention="14 days",
)

_ppstructure = None
_rapid = None


def pdf_to_image(pdf_path, dpi):
    out = os.path.join(WORK_DIR, f"rendered_{dpi}dpi.png")
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(out)
    meta = {
        "dpi": int(dpi),
        "page_width_pt": float(page.rect.width),
        "page_height_pt": float(page.rect.height),
        "pixel_width": int(pix.width),
        "pixel_height": int(pix.height),
    }
    doc.close()
    logger.info(f"PDF rendered: {pix.width}x{pix.height}px @ {dpi}dpi")
    return out, meta


def _get_ppstructure():
    global _ppstructure
    if _ppstructure is not None:
        return _ppstructure
    from paddleocr import PPStructureV3

    _ppstructure = PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        text_det_limit_side_len=PP_MAX_SIDE_LEN,
    )
    logger.info(f"PP-StructureV3 ready (text_det_limit_side_len={PP_MAX_SIDE_LEN})")
    return _ppstructure


def _get_rapid():
    global _rapid
    if _rapid is not None:
        return _rapid
    from rapidocr_onnxruntime import RapidOCR

    _rapid = RapidOCR()
    logger.info("RapidOCR ready")
    return _rapid


def _safe_json(res_obj):
    if isinstance(res_obj, dict):
        return res_obj
    if hasattr(res_obj, "json"):
        j = res_obj.json
        return j() if callable(j) else j
    return {}


def _poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _has_chinese(text):
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _clamp_item(item, img_w, img_h):
    x1, y1, x2, y2 = item["bbox"]
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))
    item["bbox"] = [x1, y1, x2, y2]
    item["width"] = x2 - x1
    item["height"] = y2 - y1
    if item.get("box") and len(item["box"]) == 4:
        item["box"] = [
            [max(0, min(int(p[0]), img_w - 1)), max(0, min(int(p[1]), img_h - 1))]
            for p in item["box"]
        ]
    return item


def _parse_ppstructure(output, offset_x, offset_y, img_w, img_h):
    items = []
    for res in output:
        data = _safe_json(res)
        if not isinstance(data, dict):
            continue
        res_data = data.get("res", data)
        overall = res_data.get("overall_ocr_res", {})
        texts = overall.get("rec_texts", [])
        scores = overall.get("rec_scores", [])
        boxes = overall.get("rec_boxes", [])
        polys = overall.get("rec_polys", [])
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text or not _has_chinese(text):
                continue
            conf = float(scores[i]) if i < len(scores) else 0.0
            if conf < MIN_CONFIDENCE:
                continue
            poly = None
            if i < len(polys) and polys[i] is not None:
                arr = np.array(polys[i], dtype=np.float64)
                if arr.shape == (4, 2):
                    arr[:, 0] += offset_x
                    arr[:, 1] += offset_y
                    poly = [[int(p[0]), int(p[1])] for p in arr]
            if poly:
                bbox = _poly_to_bbox(poly)
            elif i < len(boxes) and boxes[i] is not None:
                vals = np.array(boxes[i]).flatten()
                if len(vals) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in vals]
                bbox = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
                poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
            else:
                continue
            dx = poly[1][0] - poly[0][0]
            dy = poly[1][1] - poly[0][1]
            item = {
                "text": text,
                "bbox": bbox,
                "box": poly,
                "angle": round(float(np.degrees(np.arctan2(dy, dx))), 1),
                "confidence": round(conf, 3),
                "source": "ppstructure",
                "sub_bboxes": [],
            }
            items.append(_clamp_item(item, img_w, img_h))
    return items


def _parse_rapid(result, offset_x, offset_y, img_w, img_h):
    items = []
    if not result:
        return items
    for row in result:
        box, text, conf = row[0], (row[1] or "").strip(), float(row[2])
        if not text or not _has_chinese(text) or conf < MIN_CONFIDENCE:
            continue
        poly = [[int(p[0] + offset_x), int(p[1] + offset_y)] for p in box]
        bbox = _poly_to_bbox(poly)
        dx = poly[1][0] - poly[0][0]
        dy = poly[1][1] - poly[0][1]
        item = {
            "text": text,
            "bbox": bbox,
            "box": poly,
            "angle": round(float(np.degrees(np.arctan2(dy, dx))), 1),
            "confidence": round(conf, 3),
            "source": "rapid",
            "sub_bboxes": [],
        }
        items.append(_clamp_item(item, img_w, img_h))
    return items


def _tile_regions(img_w, img_h, tile_size, overlap):
    step = max(1, tile_size - overlap)
    xs = list(range(0, max(1, img_w), step))
    ys = list(range(0, max(1, img_h), step))
    regions = []
    for y in ys:
        for x in xs:
            x2 = min(img_w, x + tile_size)
            y2 = min(img_h, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            region = (x1, y1, x2 - x1, y2 - y1)
            if region not in regions:
                regions.append(region)
    return regions


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _same_textish(a, b):
    if a == b:
        return True
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.68


def dedup_items(items):
    items = sorted(items, key=lambda it: (-(it.get("confidence", 0.0)), it["bbox"][1], it["bbox"][0]))
    kept = []
    for item in items:
        duplicate_idx = None
        for idx, old in enumerate(kept):
            if _bbox_iou(item["bbox"], old["bbox"]) >= 0.45 and _same_textish(item["text"], old["text"]):
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            kept.append(item)
            continue
        old = kept[duplicate_idx]
        better = (
            item.get("confidence", 0.0) > old.get("confidence", 0.0) + 0.03
            or len(item["text"]) > len(old["text"]) + 2
        )
        if better:
            kept[duplicate_idx] = item
    kept.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    return kept


def ocr_overlap_tiled(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img_h, img_w = img.shape[:2]
    regions = _tile_regions(img_w, img_h, OCR_TILE_SIZE, OCR_TILE_OVERLAP)
    logger.info(
        f"OCR tiled: {len(regions)} regions, tile={OCR_TILE_SIZE}, overlap={OCR_TILE_OVERLAP}, engine={OCR_ENGINE}"
    )
    tmp_dir = os.path.join(WORK_DIR, "_ocr_tiles")
    os.makedirs(tmp_dir, exist_ok=True)
    all_items = []

    pp = rapid = None
    if OCR_ENGINE in ("both", "ppstructure"):
        try:
            pp = _get_ppstructure()
        except Exception as exc:
            logger.warning(f"PP-Structure unavailable: {exc}")
    if OCR_ENGINE in ("both", "rapid"):
        try:
            rapid = _get_rapid()
        except Exception as exc:
            logger.warning(f"RapidOCR unavailable: {exc}")
    if pp is None and rapid is None:
        raise RuntimeError("No OCR engine available. Install paddleocr or rapidocr_onnxruntime.")

    for idx, (x, y, tw, th) in enumerate(regions, start=1):
        tile = img[y : y + th, x : x + tw]
        tile_path = os.path.join(tmp_dir, f"tile_{idx:04d}_{y}_{x}.png")
        cv2.imwrite(tile_path, tile)
        t0 = time.time()
        before = len(all_items)
        if pp is not None:
            try:
                all_items.extend(_parse_ppstructure(pp.predict(tile_path), x, y, img_w, img_h))
            except Exception as exc:
                logger.warning(f"PP tile {idx} failed: {exc}")
        if rapid is not None:
            try:
                result, _ = rapid(tile)
                all_items.extend(_parse_rapid(result, x, y, img_w, img_h))
            except Exception as exc:
                logger.warning(f"Rapid tile {idx} failed: {exc}")
        logger.debug(f"  tile {idx}/{len(regions)} ({x},{y}) {tw}x{th}: +{len(all_items)-before} in {time.time()-t0:.1f}s")
        try:
            os.remove(tile_path)
        except Exception:
            pass
        gc.collect()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    items = dedup_items(all_items)
    logger.info(f"OCR done: raw={len(all_items)}, dedup={len(items)}")
    return items


def _dark_horizontal_bar_between(img_gray, a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]
    y1, y2 = ay2, by1
    if y2 <= y1 or y2 - y1 > max(a["height"], b["height"]) * 1.8:
        return False
    x1, x2 = min(ax1, bx1), max(ax2, bx2)
    roi = img_gray[max(0, y1 - 2) : min(img_gray.shape[0], y2 + 3), max(0, x1) : min(img_gray.shape[1], x2)]
    if roi.size == 0:
        return False
    dark = roi < 120
    row_ratio = np.mean(dark, axis=1)
    return bool(np.max(row_ratio) > 0.28)


def merge_text_lines(items, img_path):
    if not ENABLE_LINE_MERGE or not items:
        return items
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    horizontal = [it for it in items if abs(float(it.get("angle", 0.0))) < ANGLE_NEAR_HORIZONTAL]
    rotated = [it for it in items if abs(float(it.get("angle", 0.0))) >= ANGLE_NEAR_HORIZONTAL]
    horizontal.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    merged = []
    used = set()
    for i, item in enumerate(horizontal):
        if i in used:
            continue
        group = [item]
        used.add(i)
        while True:
            last = group[-1]
            lx1, ly1, lx2, ly2 = last["bbox"]
            lh = max(1, ly2 - ly1)
            best = None
            best_gap = 10**9
            for j, cand in enumerate(horizontal):
                if j in used:
                    continue
                cx1, cy1, cx2, cy2 = cand["bbox"]
                ch = max(1, cy2 - cy1)
                gap = cy1 - ly2
                if gap < 0 or gap > 1.2 * ((lh + ch) / 2):
                    continue
                left_ok = abs(lx1 - cx1) <= 24
                center_ok = abs((lx1 + lx2) / 2 - (cx1 + cx2) / 2) <= 36
                overlap = min(lx2, cx2) - max(lx1, cx1)
                overlap_ok = overlap > 0 and overlap / max(lx2 - lx1, cx2 - cx1, 1) > 0.55
                if not (left_ok or center_ok or overlap_ok):
                    continue
                if img_gray is not None and _dark_horizontal_bar_between(img_gray, last, cand):
                    continue
                if gap < best_gap:
                    best = j
                    best_gap = gap
            if best is None:
                break
            group.append(horizontal[best])
            used.add(best)
        if len(group) == 1:
            merged.append(group[0])
            continue
        group.sort(key=lambda it: it["bbox"][1])
        x1 = min(it["bbox"][0] for it in group)
        y1 = min(it["bbox"][1] for it in group)
        x2 = max(it["bbox"][2] for it in group)
        y2 = max(it["bbox"][3] for it in group)
        merged.append(
            {
                "text": "\n".join(it["text"] for it in group),
                "bbox": [x1, y1, x2, y2],
                "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "angle": 0.0,
                "confidence": round(sum(it.get("confidence", 0.0) for it in group) / len(group), 3),
                "source": "+".join(sorted(set(it.get("source", "") for it in group))),
                "sub_bboxes": [it["bbox"] for it in group],
                "sub_texts": [it["text"] for it in group],
                "width": x2 - x1,
                "height": y2 - y1,
            }
        )
    out = merged + rotated
    out.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    logger.info(f"Line merge: {len(items)} -> {len(out)}")
    return out


def translate_with_dictionary(items):
    for item in items:
        text = item["text"]
        item["translated"] = ENGINEERING_DICT.get(text, item.get("translated", text))
    return items


def translate_with_llm(items):
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package unavailable; dictionary translation only")
        return translate_with_dictionary(items)
    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        logger.warning("LLM config incomplete; dictionary translation only")
        return translate_with_dictionary(items)

    items = translate_with_dictionary(items)
    cache_path = os.path.join(WORK_DIR, "translation_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    pending = []
    for idx, item in enumerate(items):
        text = item["text"]
        if item.get("translated", text) != text:
            continue
        if text in cache:
            item["translated"] = cache[text]
        else:
            pending.append((idx, item))

    if not pending:
        return items

    dict_sample = "\n".join(f"{k} -> {v}" for k, v in list(ENGINEERING_DICT.items())[:25])
    system_prompt = f"""You translate Chinese CAD drawing labels into very short professional English.
Rules:
1. Output only tagged records.
2. Preserve numbers, symbols, model codes, and line breaks.
3. Be extremely short; use standard engineering abbreviations.
4. If a source has multiple lines, return the same number of lines when possible.

Terminology:
{dict_sample}
"""
    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    new_cache = 0
    for start in range(0, len(pending), LLM_BATCH_SIZE):
        batch = pending[start : start + LLM_BATCH_SIZE]
        prompt = []
        for idx, item in batch:
            prompt.append(f"[ITEM_START]\nID: {idx}\nSRC: {item['text']}\n[ITEM_END]")
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n".join(prompt)}],
                temperature=LLM_TEMPERATURE,
                max_tokens=2500,
            )
            content = resp.choices[0].message.content or ""
            trans = {}
            for block in re.findall(r"\[ITEM_START\](.*?)\[ITEM_END\]", content, re.S):
                id_match = re.search(r"ID:\s*(\d+)", block)
                trn_match = re.search(r"TRN:\s*(.*)", block, re.S)
                if id_match and trn_match:
                    trans[int(id_match.group(1))] = trn_match.group(1).strip()
            for idx, item in batch:
                item["translated"] = trans.get(idx, item["text"])
                if item["translated"] != item["text"]:
                    cache[item["text"]] = item["translated"]
                    new_cache += 1
            logger.info(f"LLM batch {start // LLM_BATCH_SIZE + 1}: {len(trans)}/{len(batch)} returned")
            time.sleep(0.25)
        except Exception as exc:
            logger.warning(f"LLM batch failed: {exc}")
            for _, item in batch:
                item.setdefault("translated", item["text"])

    if new_cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    return items


def translate_items(items):
    if TRANSLATE_ENGINE == "llm":
        return translate_with_llm(items)
    return translate_with_dictionary(items)


def _load_font(font_path, size):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()


def _wrap_text(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        words = para.split(" ")
        current = words[0]
        for word in words[1:]:
            cand = current + " " + word
            bb = draw.textbbox((0, 0), cand, font=font)
            if bb[2] - bb[0] <= max_w:
                current = cand
            else:
                _append_char_wrapped(draw, current, font, max_w, lines)
                current = word
        _append_char_wrapped(draw, current, font, max_w, lines)
    return lines


def _append_char_wrapped(draw, text, font, max_w, out):
    bb = draw.textbbox((0, 0), text or " ", font=font)
    if bb[2] - bb[0] <= max_w:
        out.append(text)
        return
    cur = ""
    for ch in text:
        cand = cur + ch
        cb = draw.textbbox((0, 0), cand, font=font)
        if cb[2] - cb[0] <= max_w:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)


def _fit_text(draw, text, font_path, box_w, box_h, max_size):
    min_size = int(os.environ.get("MIN_FONT_SIZE", "4"))
    for size in range(max(min_size, int(max_size)), min_size - 1, -1):
        font = _load_font(font_path, size)
        spacing = max(0, int(size * 0.10))
        lines = _wrap_text(draw, text, font, max(1, box_w))
        total_h = 0
        max_lw = 0
        for line in lines:
            bb = draw.textbbox((0, 0), line or " ", font=font)
            total_h += bb[3] - bb[1]
            max_lw = max(max_lw, bb[2] - bb[0])
        if len(lines) > 1:
            total_h += spacing * (len(lines) - 1)
        if max_lw <= box_w and total_h <= box_h:
            return font, lines, spacing, total_h, False

    font = _load_font(font_path, min_size)
    spacing = 0
    lines = _wrap_text(draw, text, font, max(1, box_w))
    total_h = 0
    for line in lines:
        bb = draw.textbbox((0, 0), line or " ", font=font)
        total_h += bb[3] - bb[1]
    logger.debug(f"hard clip overflow: box={box_w}x{box_h}, text={text[:50]}")
    return font, lines, spacing, total_h, True


def _sample_text_color(original_bgr, x1, y1, x2, y2):
    h, w = original_bgr.shape[:2]
    roi = original_bgr[max(0, y1 - 2) : min(h, y2 + 2), max(0, x1 - 2) : min(w, x2 + 2)]
    if roi.size == 0:
        return (0, 0, 0)
    return (0, 0, 0) if float(np.mean(roi)) > 128 else (255, 255, 255)


def _alpha_at(base, layer, left, top):
    bw, bh = base.size
    lw, lh = layer.size
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(bw, left + lw), min(bh, top + lh)
    if x2 <= x1 or y2 <= y1:
        return
    crop = layer.crop((x1 - left, y1 - top, x2 - left, y2 - top))
    base.alpha_composite(crop, dest=(x1, y1))


def _draw_text_layer(text, box_w, box_h, font_path, fill):
    layer = Image.new("RGBA", (max(1, box_w), max(1, box_h)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font, lines, spacing, text_h, overflow = _fit_text(d, text, font_path, box_w, box_h, max(6, int(box_h * 0.92)))
    y = max(0, int((box_h - text_h) / 2)) if len(lines) == 1 and not overflow else 0
    for line in lines:
        d.text((0, y), line, fill=fill + (255,), font=font)
        bb = d.textbbox((0, 0), line or " ", font=font)
        y += (bb[3] - bb[1]) + spacing
        if y > box_h:
            break
    return layer


def _render_rotated_item(item, font_path, fill):
    x1, y1, x2, y2 = item["bbox"]
    bbox_w, bbox_h = max(1, x2 - x1), max(1, y2 - y1)
    box = item.get("box") or [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    (p0, p1, _p2, p3) = box[:4]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    base_len = max(1, int(round(np.hypot(dx, dy))))
    side_len = max(1, int(round(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))))
    local_w, local_h = max(base_len, side_len), min(base_len, side_len)
    local_w = max(1, min(local_w, max(bbox_w, bbox_h)))
    local_h = max(1, min(local_h, max(1, min(bbox_w, bbox_h))))

    local = _draw_text_layer(item.get("translated", item["text"]), local_w, local_h, font_path, fill)
    rotated = local.rotate(-angle, expand=True, resample=Image.BICUBIC)

    clipped = Image.new("RGBA", (bbox_w, bbox_h), (0, 0, 0, 0))
    left = int((bbox_w - rotated.size[0]) / 2)
    top = int((bbox_h - rotated.size[1]) / 2)
    _alpha_at(clipped, rotated, left, top)

    if box and len(box) == 4:
        mask = Image.new("L", (bbox_w, bbox_h), 0)
        md = ImageDraw.Draw(mask)
        pts = [(int(px - x1), int(py - y1)) for px, py in box]
        md.polygon(pts, fill=255)
        alpha = clipped.getchannel("A")
        clipped.putalpha(Image.fromarray(np.minimum(np.array(alpha), np.array(mask)).astype(np.uint8)))
    return clipped


def _erase_region(img_bgr, item):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = item["bbox"]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False
    inset = 1
    ex1, ey1 = min(x2, x1 + inset), min(y2, y1 + inset)
    ex2, ey2 = max(ex1, x2 - inset), max(ey1, y2 - inset)
    cv2.rectangle(img_bgr, (ex1, ey1), (ex2, ey2), (255, 255, 255), -1)
    return True


def inpaint_and_overlay(img_path, items, output_img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    original = img_bgr.copy()
    erased = 0
    for item in items:
        if item.get("translated", item["text"]) != item["text"] and _erase_region(img_bgr, item):
            erased += 1
    logger.info(f"Erased regions: {erased}")

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    rendered = skipped = 0
    font_path = FONT_PATH if FONT_PATH and os.path.exists(FONT_PATH) else None

    for item in items:
        translated = item.get("translated", item["text"])
        if translated == item["text"]:
            skipped += 1
            continue
        x1, y1, x2, y2 = item["bbox"]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        fill = _sample_text_color(original, x1, y1, x2, y2)
        angle = float(item.get("angle", 0.0))
        try:
            if abs(angle) >= ANGLE_NEAR_HORIZONTAL:
                layer = _render_rotated_item(item, font_path, fill)
                _alpha_at(pil, layer, x1, y1)
            elif item.get("sub_bboxes") and len(item["sub_bboxes"]) == len(translated.split("\n")):
                for text, sb in zip(translated.split("\n"), item["sub_bboxes"]):
                    sx1, sy1, sx2, sy2 = sb
                    sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)
                    layer = _draw_text_layer(text, sw, sh, font_path, fill)
                    _alpha_at(pil, layer, sx1, sy1)
            else:
                layer = _draw_text_layer(translated, box_w, box_h, font_path, fill)
                _alpha_at(pil, layer, x1, y1)
            rendered += 1
        except Exception as exc:
            logger.warning(f"Render failed for '{item['text'][:30]}': {exc}")
            skipped += 1

    pil.convert("RGB").save(output_img_path)
    logger.info(f"Refill done: rendered={rendered}, skipped={skipped}")


def image_to_pdf(img_path, output_pdf, dpi):
    img = Image.open(img_path)
    w, h = img.size
    doc = fitz.open()
    page = doc.new_page(width=w * 72.0 / dpi, height=h * 72.0 / dpi)
    page.insert_image(page.rect, filename=img_path, keep_proportion=False)
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    logger.info(f"Output PDF: {output_pdf}")


def generate_debug_pdf(img_path, items, output_pdf, dpi):
    pil = Image.open(img_path).convert("RGBA")
    d = ImageDraw.Draw(pil)
    font = _load_font(FONT_PATH, 12)
    for idx, item in enumerate(items):
        x1, y1, x2, y2 = item["bbox"]
        color = (0, 180, 0, 220) if item.get("confidence", 0) >= 0.8 else (220, 130, 0, 220)
        if item.get("box"):
            d.polygon([tuple(p) for p in item["box"]], outline=color)
        else:
            d.rectangle([x1, y1, x2, y2], outline=color, width=1)
        d.text((x1 + 2, max(0, y1 - 14)), str(idx), fill=(255, 0, 0, 255), font=font)
    png = output_pdf.replace(".pdf", ".png")
    pil.convert("RGB").save(png)
    image_to_pdf(png, output_pdf, dpi)


def main():
    logger.info("=" * 60)
    logger.info("Unified CAD scan translation pipeline")
    logger.info(f"Input: {PDF_PATH}")
    logger.info(f"Work dir: {WORK_DIR}")
    logger.info(f"Log: {LOG_FILE}")
    logger.info("=" * 60)
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(PDF_PATH)

    img_path, meta = pdf_to_image(PDF_PATH, RENDER_DPI)
    raw_items = ocr_overlap_tiled(img_path)
    items = merge_text_lines(raw_items, img_path)
    ocr_json = os.path.join(WORK_DIR, "ocr_result.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    generate_debug_pdf(img_path, items, os.path.join(WORK_DIR, "ocr_debug.pdf"), meta["dpi"])

    translated = translate_items(items)
    trans_json = os.path.join(WORK_DIR, "translation_mapping.json")
    with open(trans_json, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)

    output_img = os.path.join(WORK_DIR, "translated_page.png")
    inpaint_and_overlay(img_path, translated, output_img)
    image_to_pdf(output_img, OUTPUT_PDF, meta["dpi"])

    translated_count = sum(1 for it in translated if it.get("translated", it["text"]) != it["text"])
    logger.info("=" * 60)
    logger.info(f"OCR items: raw={len(raw_items)}, final={len(items)}")
    logger.info(f"Translated items: {translated_count}/{len(translated)}")
    logger.info(f"OCR JSON: {ocr_json}")
    logger.info(f"Translation JSON: {trans_json}")
    logger.info(f"Output: {OUTPUT_PDF}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
