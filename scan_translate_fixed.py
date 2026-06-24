"""
扫描型PDF中文→英文翻译 - RapidOCR + 分块OCR(避免内存溢出)
流程: 扫描PDF→渲染→分块OCR→合并结果→翻译→白底擦除→左对齐覆盖→重构PDF
"""
import os, json, sys, gc
import re
import time
import numpy as np
import cv2
import fitz
from PIL import Image, ImageDraw, ImageFont

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_BATCH_SIZE, LLM_TEMPERATURE,
    TRANSLATE_ENGINE, ENGINEERING_DICT, RENDER_DPI, CHUNK_SIZE, FONT_PATH,
)

PDF_PATH = r"d:\AIGC\projects\pdf_translate\pdfs\252206.10.5-1 A 一中间辊-A3.pdf"
WORK_DIR = r"d:\AIGC\projects\pdf_translate\scan_work"
OUTPUT_PDF = r"d:\AIGC\projects\pdf_translate\pdfs\252206.10.5-1 A 一中间辊-A3_translated_fixed.pdf"


def translate_with_dictionary(text_items: list) -> list:
    """使用离线术语字典翻译 - 仅精确匹配，不做模糊替换"""
    for item in text_items:
        text = item["text"]
        if text in ENGINEERING_DICT:
            item["translated"] = ENGINEERING_DICT[text]
        # Don't do fuzzy matching - let LLM handle longer text properly
    return text_items


def translate_with_llm(text_items: list) -> list:
    """使用 OpenAI 兼容 API 进行上下文感知翻译"""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai 未安装，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        print("LLM API 配置不完整 (LLM_API_BASE/LLM_API_KEY/LLM_MODEL)，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    # First, apply dictionary translations
    text_items = translate_with_dictionary(text_items)

    # Filter items that need LLM translation (dictionary didn't translate or translated = original)
    items_for_llm = []
    for i, item in enumerate(text_items):
        translated = item.get("translated", item["text"])
        if translated == item["text"]:  # Not translated by dictionary
            items_for_llm.append((i, item))

    if not items_for_llm:
        print("  All items translated by dictionary, skipping LLM")
        return text_items

    print(f"  Dictionary translated {len(text_items) - len(items_for_llm)} items, sending {len(items_for_llm)} to LLM")

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    # 构建术语字典片段注入 prompt
    dict_sample = "\n".join(
        [f'  "{cn}" → "{en}"' for cn, en in list(ENGINEERING_DICT.items())[:30]]
    )

    system_prompt = f"""你是一名专业的机械/电气工程翻译专家，精通ISO/DIN/GB标准术语。
请将CAD图纸中的中文文本翻译为英文。要求：
1. 使用国际通用的工程标准术语（ISO/DIN标准）
2. 必须简短！使用缩写（如Int.代替Intermediate，Mat'l代替Material，Req.代替Requirements）
3. 保持专业性和准确性
4. 如果文本包含数字、代号、字母等非中文字符，保留原样只翻译中文部分
5. 仅输出翻译结果，每行格式: 序号. 译文
6. 短语翻译控制在3个单词以内

常用术语参考:
{dict_sample}"""

    # 按Y坐标排序，空间邻近的文本在同一批
    sorted_items = sorted(items_for_llm, key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))

    batch_size = LLM_BATCH_SIZE
    total_batches = (len(sorted_items) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(sorted_items), batch_size):
        batch = sorted_items[batch_idx: batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        items_str = "\n".join(
            [f'{j + 1}. "{item[1]["text"]}"'
             for j, item in enumerate(batch)]
        )

        user_prompt = f"待翻译文本：\n{items_str}"

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                temperature=LLM_TEMPERATURE,
            )

            result_text = response.choices[0].message.content
            # 解析翻译结果
            lines = result_text.strip().split("\n")
            trans_map = {}
            for line in lines:
                match = re.match(r'(\d+)[.、．]\s*["""]?(.+?)["""]?\s*$', line.strip())
                if match:
                    idx = int(match.group(1))
                    trans = match.group(2).strip()
                    trans_map[idx] = trans

            for j, (orig_idx, item) in enumerate(batch):
                idx = j + 1
                item["translated"] = trans_map.get(idx, item["text"])

            matched = len(trans_map)
            print(f"  批次 {batch_num}/{total_batches}: 成功翻译 {matched}/{len(batch)} 条")

            # 宽松解析: 如果匹配率低，按行顺序对应
            if matched < len(batch) * 0.5:
                print(f"    匹配率较低，尝试宽松解析...")
                valid_lines = [
                    re.sub(r'^\d+[.、．]\s*["""]?', "", line.strip()).rstrip('"""')
                    for line in lines
                    if re.match(r'^\d+[.、．]', line.strip())
                ]
                for j, (orig_idx, item) in enumerate(batch):
                    if j + 1 not in trans_map and j < len(valid_lines):
                        item["translated"] = valid_lines[j]

            time.sleep(0.3)

        except Exception as e:
            print(f"  批次 {batch_num}/{total_batches}: 异常 - {e}，回退到字典翻译")
            # 该批次降级到字典翻译
            for orig_idx, item in batch:
                if "translated" not in item:
                    text = item["text"]
                    if text in ENGINEERING_DICT:
                        item["translated"] = ENGINEERING_DICT[text]
                    else:
                        result = text
                        changed = False
                        for cn, en in sorted(ENGINEERING_DICT.items(), key=lambda x: -len(x[0])):
                            if cn in result:
                                result = result.replace(cn, en)
                                changed = True
                        item["translated"] = result if changed else text

    return text_items



def pdf_to_image(pdf_path, dpi=200):
    """Render the scanned PDF page to an image and return image path + page metadata."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_path = os.path.join(WORK_DIR, f"rendered_{dpi}dpi.png")
    pix.save(img_path)

    page_meta = {
        "page_width_pt": float(page.rect.width),
        "page_height_pt": float(page.rect.height),
        "rotation": int(page.rotation),
        "dpi": int(dpi),
        "pixel_width": int(pix.width),
        "pixel_height": int(pix.height),
    }

    doc.close()
    print(f"  Rendered: {pix.width}x{pix.height} px @ {dpi}DPI")
    return img_path, page_meta

def ocr_with_rapid_chunked(img_path, chunk_size=4000):
    """分块OCR，避免大图内存溢出"""
    from rapidocr_onnxruntime import RapidOCR
    print("  Initializing RapidOCR...")
    engine = RapidOCR()
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}, chunk_size={chunk_size}")
    
    all_items = []
    chunk_idx = 0
    
    # 按行分块
    for y_start in range(0, h, chunk_size):
        for x_start in range(0, w, chunk_size):
            y_end = min(y_start + chunk_size, h)
            x_end = min(x_start + chunk_size, w)
            
            chunk = img[y_start:y_end, x_start:x_end]
            ch_h, ch_w = chunk.shape[:2]
            
            if ch_h < 50 or ch_w < 50:
                continue
            
            chunk_idx += 1
            try:
                result, elapse = engine(chunk)
            except Exception as e:
                print(f"  Chunk [{x_start}:{x_end}, {y_start}:{y_end}] error: {e}")
                continue
            
            if result:
                for item in result:
                    box = item[0]
                    text = item[1]
                    confidence = float(item[2])
                    
                    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
                        # 偏移坐标回全局
                        offset_box = [[p[0] + x_start, p[1] + y_start] for p in box]
                        xs = [p[0] for p in offset_box]
                        ys = [p[1] for p in offset_box]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        dx = offset_box[1][0] - offset_box[0][0]
                        dy = offset_box[1][1] - offset_box[0][1]
                        angle = np.degrees(np.arctan2(dy, dx))
                        all_items.append({
                            "text": text,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "box": [[int(p[0]), int(p[1])] for p in offset_box],
                            "angle": round(angle, 1),
                            "confidence": round(confidence, 3),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                        })
            
            del chunk, result
            gc.collect()
    
    # 去重（分块边界可能重复识别）
    seen = set()
    unique_items = []
    for item in all_items:
        key = (item["text"], tuple(item["bbox"]))
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    
    print(f"  Found {len(unique_items)} unique Chinese text regions (from {len(all_items)} raw)")
    return unique_items



def _load_font(font_path, font_size):
    """Load a font safely."""
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()


def _wrap_text_to_width(draw, text, font, max_width):
    """Wrap English text to fit into a given width."""
    if not text:
        return [""]

    paragraphs = text.split("\n")
    wrapped_lines = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            wrapped_lines.append("")
            continue

        words = para.split()
        if not words:
            wrapped_lines.append("")
            continue

        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = word

        # Handle a single long token that still does not fit
        if draw.textbbox((0, 0), current, font=font)[2] - draw.textbbox((0, 0), current, font=font)[0] > max_width and len(current) > 1:
            token = ""
            for ch in current:
                test = token + ch
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] <= max_width or not token:
                    token = test
                else:
                    wrapped_lines.append(token)
                    token = ch
            if token:
                wrapped_lines.append(token)
        else:
            wrapped_lines.append(current)

    return wrapped_lines


def _fit_text_to_box(draw, text, font_path, box_w, box_h, max_font_size):
    """Shrink and wrap text so it fits the target box."""
    min_font_size = 6
    max_font_size = max(min_font_size, max_font_size)

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(font_path, font_size)
        lines = _wrap_text_to_width(draw, text, font, max(1, box_w))
        spacing = max(0, int(font_size * 0.15))
        multiline = "\n".join(lines)

        bbox = draw.multiline_textbbox((0, 0), multiline, font=font, spacing=spacing, align="left")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if text_w <= box_w and text_h <= box_h:
            return font, lines, spacing, text_w, text_h

    # Fallback: use the minimum size even if it slightly overflows
    font = _load_font(font_path, min_font_size)
    lines = _wrap_text_to_width(draw, text, font, max(1, box_w))
    spacing = max(0, int(min_font_size * 0.15))
    multiline = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), multiline, font=font, spacing=spacing, align="left")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    return font, lines, spacing, text_w, text_h


def _sample_text_color(img_bgr, x1, y1, x2, y2):
    """Estimate text color from pixels around the original text region."""
    h, w = img_bgr.shape[:2]
    pad = 2
    regions = []

    top1, top2 = max(0, y1 - pad), max(0, y1)
    bottom1, bottom2 = min(h, y2), min(h, y2 + pad)
    left1, left2 = max(0, x1 - pad), max(0, x1)
    right1, right2 = min(w, x2), min(w, x2 + pad)

    if top2 > top1 and x2 > x1:
        regions.append(img_bgr[top1:top2, x1:x2])
    if bottom2 > bottom1 and x2 > x1:
        regions.append(img_bgr[bottom1:bottom2, x1:x2])
    if left2 > left1 and y2 > y1:
        regions.append(img_bgr[y1:y2, left1:left2])
    if right2 > right1 and y2 > y1:
        regions.append(img_bgr[y1:y2, right1:right2])

    if not regions:
        return (0, 0, 0)

    pixels = np.concatenate([r.reshape(-1, 3) for r in regions if r.size], axis=0)
    if pixels.size == 0:
        return (0, 0, 0)

    brightness = float(np.mean(pixels))
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)


def inpaint_and_overlay(img_path, translated_items, output_img_path):
    """For scanned CAD PDFs only:
    1) wipe OCR text regions with a flat background color
    2) draw translated text left-aligned
    3) keep the rest of the raster page unchanged
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {img_path}")

    h, w = img_bgr.shape[:2]
    original_bgr = img_bgr.copy()

    # 1) Wipe text regions with a flat fill to avoid inpaint shadows and halos.
    #    We keep the rectangle as tight as possible to avoid touching CAD lines.
    for item in translated_items:
        x1, y1, x2, y2 = item["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    font_path = FONT_PATH if os.path.exists(FONT_PATH) else None

    success = skip = 0
    for item in translated_items:
        bbox = item["bbox"]
        translated = item.get("translated", item["text"])
        original = item["text"]

        if translated == original:
            skip += 1
            continue

        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            skip += 1
            continue

        # Use a slightly smaller starting size to reduce overflow.
        max_font_size = max(8, int(box_h * 0.82))
        font, lines, spacing, text_w, text_h = _fit_text_to_box(
            draw, translated, font_path, box_w - 2, box_h - 2, max_font_size
        )

        # Left-aligned, vertical center-ish placement.
        tx = x1 + 1
        ty = y1 + max(0, int((box_h - text_h) / 2))

        text_color = _sample_text_color(original_bgr, x1, y1, x2, y2)
        draw.multiline_text(
            (tx, ty),
            "\n".join(lines),
            fill=text_color,
            font=font,
            spacing=spacing,
            align="left",
        )
        success += 1

    pil_img.save(output_img_path)
    print(f"  Overlay: success={success}, skip={skip}")
    return translated_items


def image_to_pdf(img_path, output_pdf, dpi=200):
    """Rebuild a one-page PDF from the translated raster image.

    This workflow is intentionally raster-only and is designed for scanned CAD PDFs.
    The page size is reconstructed from the rendered image size and the render DPI,
    so the output keeps the original page proportions.
    """
    img = Image.open(img_path)
    img_w, img_h = img.size

    page_w_pt = img_w * 72.0 / dpi
    page_h_pt = img_h * 72.0 / dpi

    doc = fitz.open()
    page = doc.new_page(width=page_w_pt, height=page_h_pt)

    # Insert the raster image to cover the whole page exactly.
    page.insert_image(page.rect, filename=img_path, keep_proportion=False)

    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    print(f"  Output: {output_pdf}")


def main():
    print("=" * 60)
    print("Scan-type PDF CN->EN Translation (RapidOCR + LLM)")
    print("=" * 60)
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found")
        return
    os.makedirs(WORK_DIR, exist_ok=True)

    print("\n[Step 1] Render PDF -> Image...")
    img_path, page_meta = pdf_to_image(PDF_PATH, dpi=RENDER_DPI)

    print("\n[Step 2] RapidOCR (chunked) recognition...")
    ocr_items = ocr_with_rapid_chunked(img_path, chunk_size=CHUNK_SIZE)
    ocr_json = os.path.join(WORK_DIR, "ocr_result.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(ocr_items, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {ocr_json}")
    for i, item in enumerate(ocr_items[:30]):
        print(f"    [{i+1}] {item['text']} (conf={item['confidence']}, angle={item['angle']}°)")

    print(f"\n[Step 3] Translate ({TRANSLATE_ENGINE} engine)...")
    if TRANSLATE_ENGINE == "llm":
        translated_items = translate_with_llm(ocr_items)
    else:
        translated_items = translate_with_dictionary(ocr_items)

    # Save translation mapping
    trans_json = os.path.join(WORK_DIR, "translation_mapping.json")
    with open(trans_json, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {trans_json}")

    print("\n[Step 4] Wipe text regions + overlay translated text...")
    output_img = os.path.join(WORK_DIR, "translated_page.png")
    inpaint_and_overlay(img_path, translated_items, output_img)

    print("\n[Step 5] Rebuild PDF...")
    image_to_pdf(output_img, OUTPUT_PDF, dpi=page_meta["dpi"])

    total = len(translated_items)
    translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Total Chinese regions: {total}")
    print(f"  Translated: {translated_count}")
    print(f"  Untranslated: {total - translated_count}")
    print(f"  Output: {OUTPUT_PDF}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()