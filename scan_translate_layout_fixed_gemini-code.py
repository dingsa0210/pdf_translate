"""
扫描型PDF中文→英文翻译 - RapidOCR + 分块OCR + 智能行合并排版优化版
流程: 扫描PDF→渲染→分块OCR→智能文本行合并→隔离标签翻译→白底擦除→绝对左对齐回填→重构PDF
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

PDF_PATH = r"d:\AIGC\projects\pdf_translate\pdfs\20260523-Rolling Mill Foundation Plan GZL24.7-17.8基础平面图-V2.0_1.pdf"
WORK_DIR = r"d:\AIGC\projects\pdf_translate\scan_work"
OUTPUT_PDF = r"d:\AIGC\projects\pdf_translate\pdfs\20260523-Rolling Mill Foundation Plan GZL24.7-17.8基础平面图-V2.0_1_translated.pdf"

def translate_with_dictionary(text_items: list) -> list:
    """使用离线术语字典翻译 - 仅精确匹配"""
    for item in text_items:
        text = item["text"]
        if text in ENGINEERING_DICT:
            item["translated"] = ENGINEERING_DICT[text]
    return text_items

def translate_with_llm(text_items: list) -> list:
    """使用独立结构化标签组进行LLM翻译，杜绝数字编号错位干扰"""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai 未安装，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        print("LLM API 配置不完整，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    text_items = translate_with_dictionary(text_items)

    items_for_llm = []
    for i, item in enumerate(text_items):
        if "translated" not in item or item["translated"] == item["text"]:
            items_for_llm.append((i, item))

    if not items_for_llm:
        print("  All items translated by dictionary, skipping LLM")
        return text_items

    print(f"  Dictionary translated {len(text_items) - len(items_for_llm)} items, sending {len(items_for_llm)} to LLM")

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    dict_sample = "\n".join([f'  "{cn}" → "{en}"' for cn, en in list(ENGINEERING_DICT.items())[:20]])

    system_prompt = f"""You are an expert CAD drawing translation assistant specializing in mechanical engineering.
Translate Chinese technical descriptions into professional English.
Guidelines:
1. Use standard international engineering terminology (ISO/DIN/GB standard).
2. Extremely short! Use standard abbreviations (e.g., Int. for Intermediate, Mat'l for Material, Req. for Requirements, Thk. for Thickness, DWG for Drawing, Qty. for Quantity).
3. If the source text contains numbers, symbols, or multi-line enumerations, preserve their internal structures EXACTLY.
4. For multi-line inputs, translate line-by-line. Keep the exact line breaks, ordering, and bullet numbers. Never merge lines or collapse lists.
5. Strict Format: You MUST output using the structured tags. Respond ONLY with [ITEM_START], ID, TRN, and [ITEM_END]. No extra prose.

Example Input:
[ITEM_START]
ID: 99
SRC: 4. 技术要求
图纸中材料为参考
[ITEM_END]

Example Output:
[ITEM_START]
ID: 99
TRN: 4. Tech. Requirements
Mat'l ref. only
[ITEM_END]

Terminology references:
{dict_sample}"""

    batch_size = LLM_BATCH_SIZE
    total_batches = (len(items_for_llm) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(items_for_llm), batch_size):
        batch = items_for_llm[batch_idx: batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        # 构建高隔离度的结构化输入报文
        user_prompt = "待翻译文本块列表如下：\n\n"
        for orig_idx, item in batch:
            user_prompt += f"[ITEM_START]\nID: {orig_idx}\nSRC: {item['text']}\n[ITEM_END]\n"

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2500,
                temperature=LLM_TEMPERATURE,
            )

            result_text = response.choices[0].message.content
            
            # 使用高鲁棒性的正则块提取解析器
            trans_map = {}
            blocks = re.findall(r"\[ITEM_START\](.*?)\[ITEM_END\]", result_text, re.DOTALL)
            
            for block in blocks:
                id_match = re.search(r"ID:\s*(\d+)", block)
                # TODO 这里指令遵循不好，应该用TRN的，解析结果还是用的SRC
                trn_match = re.search(r"SRC:\s*(.*)", block, re.DOTALL)
                if id_match and trn_match:
                    idx = int(id_match.group(1))
                    trans_map[idx] = trn_match.group(1).strip()

            # 回填翻译结果
            success_this_batch = 0
            for orig_idx, item in batch:
                if orig_idx in trans_map:
                    item["translated"] = trans_map[orig_idx]
                    success_this_batch += 1
                else:
                    # 未匹配时的兜底策略
                    item["translated"] = item["text"]

            print(f"  批次 {batch_num}/{total_batches}: 标签解析成功 {success_this_batch}/{len(batch)} 条")
            time.sleep(0.3)

        except Exception as e:
            print(f"  批次 {batch_num}/{total_batches} 异常: {e}，启用字典兜底替换")
            for orig_idx, item in batch:
                if "translated" not in item:
                    item["translated"] = item["text"]

    return text_items

def pdf_to_image(pdf_path, dpi=200):
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

def merge_ocr_items(items: list) -> list:
    """
    智能行合并算法：自动将CAD图纸中垂直层叠、严格左对齐的多行文本（如技术要求、列表说明）
    合并为一个大的多行逻辑文本块，从根本上杜绝回填时的分行重叠与混乱折行。
    """
    if not items:
        return []

    # 拆分水平文本和带有大角度的倾斜/垂直文本
    horizontal_items = [item for item in items if abs(item.get("angle", 0.0)) < 5.0]
    rotated_items = [item for item in items if abs(item.get("angle", 0.0)) >= 5.0]

    # 按纵向坐标顶部自上而下排序
    horizontal_items.sort(key=lambda x: x["bbox"][1])
    
    merged_items = []
    visited = set()

    for i, item in enumerate(horizontal_items):
        if i in visited:
            continue

        current_block = [item]
        visited.add(i)

        while True:
            added = False
            curr_x1 = current_block[-1]["bbox"][0]
            curr_x2 = current_block[-1]["bbox"][2]
            curr_y2 = current_block[-1]["bbox"][3]
            curr_h = curr_y2 - current_block[-1]["bbox"][1]

            best_next_idx = None
            best_next_gap = float('inf')

            for j, next_item in enumerate(horizontal_items):
                if j in visited:
                    continue

                nx1, ny1, nx2, ny2 = next_item["bbox"]
                nh = ny2 - ny1

                # 判定条件：1. 纵向相邻间距在一行高度的1.5倍以内
                v_gap = ny1 - curr_y2
                if 0 <= v_gap <= max(curr_h, nh) * 1.5:
                    # 2. 严格的左边界对齐（CAD文本列表的典型特征，容差50像素内）
                    if abs(curr_x1 - nx1) < 50:
                        if v_gap < best_next_gap:
                            best_next_gap = v_gap
                            best_next_idx = j

            if best_next_idx is not None:
                current_block.append(horizontal_items[best_next_idx])
                visited.add(best_next_idx)
                added = True
            else:
                break

        if len(current_block) == 1:
            merged_items.append(current_block[0])
        else:
            # 融合多个文本行的坐标框
            current_block.sort(key=lambda x: x["bbox"][1])
            bx1 = min(b["bbox"][0] for b in current_block)
            by1 = min(b["bbox"][1] for b in current_block)
            bx2 = max(b["bbox"][2] for b in current_block)
            by2 = max(b["bbox"][3] for b in current_block)
            
            combined_text = "\n".join(b["text"] for b in current_block)
            avg_conf = sum(b["confidence"] for b in current_block) / len(current_block)
            
            merged_items.append({
                "text": combined_text,
                "bbox": [bx1, by1, bx2, by2],
                "box": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                "angle": 0.0,
                "confidence": round(avg_conf, 3),
                "width": int(bx2 - bx1),
                "height": int(by2 - by1),
                "is_structured": True
            })

    print(f"  [智能合并] 原始OCR块数量: {len(items)} -> 合并后结构化块数量: {len(merged_items) + len(rotated_items)}")
    return merged_items + rotated_items

def ocr_with_rapid_chunked(img_path, chunk_size=4000):
    from rapidocr_onnxruntime import RapidOCR
    print("  Initializing RapidOCR...")
    engine = RapidOCR()
    
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    all_items = []
    
    for y_start in range(0, h, chunk_size):
        for x_start in range(0, w, chunk_size):
            y_end = min(y_start + chunk_size, h)
            x_end = min(x_start + chunk_size, w)
            
            chunk = img[y_start:y_end, x_start:x_end]
            ch_h, ch_w = chunk.shape[:2]
            if ch_h < 50 or ch_w < 50:
                continue
            
            try:
                result, _ = engine(chunk)
            except Exception:
                continue
            
            if result:
                for item in result:
                    box, text, confidence = item[0], item[1], float(item[2])
                    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
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
            gc.collect()
            
    seen = set()
    unique_items = []
    for item in all_items:
        key = (item["text"], tuple(item["bbox"]))
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
    return unique_items

def _load_font(font_path, font_size):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()

def _wrap_structured_text(draw, text, font, max_width):
    """保持段落原有换行标志的前提下，对超长行单行切分折行"""
    paragraphs = text.split("\n")
    wrapped_lines = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            wrapped_lines.append("")
            continue
        words = para.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                wrapped_lines.append(current)
                current = word
        wrapped_lines.append(current)
    return wrapped_lines

def _fit_text_to_box(draw, text, font_path, box_w, box_h, max_font_size):
    """计算英文缩放比例，给予一定的横向延展缓冲防止过紧挤压"""
    min_font_size = 5
    max_font_size = max(min_font_size, max_font_size)
    # CAD 图纸横向往往有空白，英文字符自然膨胀，允许横向适当延展15%减小字号缩减压力
    allowed_w = max(1, int(box_w * 1.15))

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(font_path, font_size)
        spacing = max(1, int(font_size * 0.15))
        lines = _wrap_structured_text(draw, text, font, allowed_w)

        max_line_w = 0
        total_h = 0
        for line in lines:
            sample = line if line else " "
            bbox = draw.textbbox((0, 0), sample, font=font)
            max_line_w = max(max_line_w, bbox[2] - bbox[0])
            total_h += (bbox[3] - bbox[1])
        if len(lines) > 1:
            total_h += spacing * (len(lines) - 1)

        if max_line_w <= allowed_w and total_h <= box_h:
            return font, lines, spacing, total_h

    # 兜底尺寸
    font = _load_font(font_path, min_font_size)
    spacing = max(1, int(min_font_size * 0.15))
    lines = _wrap_structured_text(draw, text, font, allowed_w)
    return font, lines, spacing, box_h

def _sample_text_color(img_bgr, x1, y1, x2, y2):
    h, w = img_bgr.shape[:2]
    pad = 2
    regions = []
    y1_p, y2_p = max(0, y1 - pad), min(h, y2 + pad)
    x1_p, x2_p = max(0, x1 - pad), min(w, x2 + pad)
    if y2_p > y1_p and x2_p > x1_p:
        regions.append(img_bgr[y1_p:y2_p, x1_p:x2_p])
    if not regions or regions[0].size == 0:
        return (0, 0, 0)
    return (0, 0, 0) if float(np.mean(regions[0])) > 128 else (255, 255, 255)

def _alpha_composite_at(base_rgba, layer_rgba, left, top):
    base_w, base_h = base_rgba.size
    layer_w, layer_h = layer_rgba.size
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(base_w, left + layer_w), min(base_h, top + layer_h)
    if x2 <= x1 or y2 <= y1:
        return
    crop = layer_rgba.crop((x1 - left, y1 - top, x2 - left, y2 - top))
    base_rgba.alpha_composite(crop, dest=(x1, y1))

def _render_text_item_layer(item, font_path, text_color):
    """旋转文本单独图层渲染（主要用于单行倾斜标注）"""
    x1, y1, x2, y2 = item["bbox"]
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    translated = item.get("translated", item["text"])
    angle = float(item.get("angle", 0.0))

    pad = max(4, int(min(box_w, box_h) * 0.12))
    local_w, local_h = box_w + pad * 2, box_h + pad * 2
    local = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(local)

    max_font_size = max(8, int(box_h * 0.85))
    font, lines, spacing, text_h = _fit_text_to_box(d, translated, font_path, box_w, box_h, max_font_size)

    ty = pad + max(0, int(((local_h - 2 * pad) - text_h) / 2))
    y_cursor = ty
    for line in lines:
        d.text((pad, y_cursor), line, fill=text_color + (255,), font=font)
        bbox = d.textbbox((0, 0), line if line else " ", font=font)
        y_cursor += (bbox[3] - bbox[1]) + spacing

    rotated = local.rotate(angle, expand=True, resample=Image.BICUBIC)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    left = int(cx - rotated.size[0] / 2)
    top = int(cy - rotated.size[1] / 2)
    return rotated, left, top

def inpaint_and_overlay(img_path, translated_items, output_img_path):
    """
    智能文本擦除与高保真对齐回填引擎
    1. 水平多行块：强行在绝对真实原x1坐标点写入，完美恢复多行列表的左对齐，断绝飘移。
    2. 倾斜多行块：采用中心变换贴回法。
    """
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    original_bgr = img_bgr.copy()

    # 先行统一精准擦除文本区域
    for item in translated_items:
        if item.get("translated", item["text"]) == item["text"]:
            continue
        x1, y1, x2, y2 = item["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
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

        text_color = _sample_text_color(original_bgr, x1, y1, x2, y2)
        angle = float(item.get("angle", 0.0))

        # 核心版面控制：水平长文本/合并块(技术要求等) 直接在主画布以绝对左边界绘制，绝不采用图层中心对齐
        if abs(angle) < 0.5:
            max_font_size = max(8, int(box_h * 0.85))
            if item.get("is_structured", False):
                # 如果是多行大并块，初始参考行高需要按行均分
                line_count = max(1, len(original.split("\n")))
                max_font_size = max(8, int((box_h / line_count) * 0.85))

            font, lines, spacing, text_h = _fit_text_to_box(draw, translated, font_path, box_w, box_h, max_font_size)
            
            tx = x1 + 2
            # 单行垂直居中，多行直接从原区域顶部往下排布防止空间不足
            ty = y1 + max(0, int((box_h - text_h) / 2)) if len(lines) == 1 else y1 + 2
            
            y_cursor = ty
            for line in lines:
                draw.text((tx, y_cursor), line, fill=text_color + (255,), font=font)
                l_bbox = draw.textbbox((0, 0), line if line else " ", font=font)
                y_cursor += (l_bbox[3] - l_bbox[1]) + spacing
            success += 1
        else:
            # 倾斜复杂文本维持旋转图层法
            try:
                layer, left, top = _render_text_item_layer(item, font_path, text_color)
                _alpha_composite_at(pil_img, layer, left, top)
                success += 1
            except Exception:
                skip += 1

    pil_img.convert("RGB").save(output_img_path)
    print(f"  版面回填完成: 成功渲染 {success} 块, 忽略/跳过 {skip} 块")
    return translated_items

def image_to_pdf(img_path, output_pdf, dpi=200):
    img = Image.open(img_path)
    img_w, img_h = img.size
    page_w_pt = img_w * 72.0 / dpi
    page_h_pt = img_h * 72.0 / dpi
    doc = fitz.open()
    page = doc.new_page(width=page_w_pt, height=page_h_pt)
    page.insert_image(page.rect, filename=img_path, keep_proportion=False)
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()
    print(f"  Output PDF: {output_pdf}")

def main():
    print("=" * 60)
    print("Scan-type PDF CN->EN Translation (Optimized Structural Layout)")
    print("=" * 60)
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found")
        return
    os.makedirs(WORK_DIR, exist_ok=True)

    print("\n[Step 1] Render PDF -> Image...")
    img_path, page_meta = pdf_to_image(PDF_PATH, dpi=RENDER_DPI)

    print("\n[Step 2] RapidOCR (chunked) recognition...")
    raw_ocr_items = ocr_with_rapid_chunked(img_path, chunk_size=CHUNK_SIZE)
    
    print("\n[Step 2.5] Executing Intelligent Block Merging...")
    ocr_items = merge_ocr_items(raw_ocr_items)
    
    ocr_json = os.path.join(WORK_DIR, "ocr_result.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(ocr_items, f, ensure_ascii=False, indent=2)

    print(f"\n[Step 3] Translate ({TRANSLATE_ENGINE} engine via Custom Tagged Protocol)...")
    if TRANSLATE_ENGINE == "llm":
        translated_items = translate_with_llm(ocr_items)
    else:
        translated_items = translate_with_dictionary(ocr_items)

    trans_json = os.path.join(WORK_DIR, "translation_mapping.json")
    with open(trans_json, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)

    print("\n[Step 4] Wipe text regions + strict left-aligned overlay...")
    output_img = os.path.join(WORK_DIR, "translated_page.png")
    inpaint_and_overlay(img_path, translated_items, output_img)

    print("\n[Step 5] Rebuild PDF...")
    image_to_pdf(output_img, OUTPUT_PDF, dpi=page_meta["dpi"])

    total = len(translated_items)
    translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Total Layout blocks: {total}")
    print(f"  Translated blocks: {translated_count}")
    print(f"  Output: {OUTPUT_PDF}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()