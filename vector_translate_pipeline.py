"""
Vector PDF CAD Drawing CN→EN Translation Pipeline

用于处理矢量型 CAD PDF：
1. 使用 PyMuPDF 提取可选择/可复制的中文矢量文本
2. 调用 LLM 或工程术语字典翻译
3. 原位白底擦除中文文本
4. 按原 bbox、字号、颜色、方向回填英文
5. 输出新的矢量 PDF

依赖:
    pip install pymupdf openai loguru
"""

import os
import re
import json
import math
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import fitz
from loguru import logger

from config import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_BATCH_SIZE,
    LLM_TEMPERATURE,
    TRANSLATE_ENGINE,
    ENGINEERING_DICT,
    FONT_PATH,
)


def has_chinese(text: str) -> bool:
    """判断文本是否包含中文。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def detect_text_orientation(chars: list) -> dict:
    """
    通过字符 origin 坐标分析文本方向。

    Returns:
        {
            "is_vertical": bool,
            "rotation_deg": float,
            "direction": "horizontal" | "vertical" | "rotated"
        }
    """
    if len(chars) < 2:
        return {
            "is_vertical": False,
            "rotation_deg": 0.0,
            "direction": "horizontal",
        }

    origins = [c.get("origin", (0, 0)) for c in chars]

    dx_total = 0.0
    dy_total = 0.0

    for i in range(1, len(origins)):
        dx_total += abs(origins[i][0] - origins[i - 1][0])
        dy_total += abs(origins[i][1] - origins[i - 1][1])

    is_vertical = False
    if dy_total > 0 and dx_total > 0:
        is_vertical = dy_total > dx_total * 1.5
    elif dy_total > 0 and dx_total == 0:
        is_vertical = True

    dx = origins[-1][0] - origins[0][0]
    dy = origins[-1][1] - origins[0][1]
    rotation_deg = math.degrees(math.atan2(dy, dx))

    if is_vertical:
        direction = "vertical"
    elif abs(rotation_deg) > 5:
        direction = "rotated"
    else:
        direction = "horizontal"

    return {
        "is_vertical": is_vertical,
        "rotation_deg": round(rotation_deg, 2),
        "direction": direction,
    }


def extract_text_info(pdf_path: str) -> list:
    """
    提取矢量 PDF 中所有中文文本及其坐标、字体、字号、颜色和方向信息。
    """
    doc = fitz.open(pdf_path)
    all_text_items = []

    logger.info(f"[Vector] PDF页数: {doc.page_count}")

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        page_text_count = 0
        page_vertical_count = 0

        for block in blocks.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    chars = span.get("chars", [])
                    text = "".join(c.get("c", "") for c in chars).strip()

                    if not text or not has_chinese(text):
                        continue

                    bbox = span.get("bbox", [0, 0, 0, 0])
                    bbox = [round(float(v), 2) for v in bbox]

                    orientation = detect_text_orientation(chars)
                    wmode = span.get("wmode", 0)
                    is_vertical = wmode == 1 or orientation["is_vertical"]

                    item = {
                        "page": page_idx + 1,
                        "text": text,
                        "bbox": bbox,
                        "font": span.get("font", ""),
                        "size": round(float(span.get("size", 12)), 2),
                        "color": span.get("color", 0),
                        "origin": [round(float(v), 2) for v in span.get("origin", (0, 0))],
                        "wmode": wmode,
                        "is_vertical": is_vertical,
                        "rotation_deg": orientation["rotation_deg"],
                        "direction": orientation["direction"],
                        "has_chinese": True,
                    }

                    all_text_items.append(item)
                    page_text_count += 1

                    if is_vertical:
                        page_vertical_count += 1

        logger.info(
            f"[Vector] 第 {page_idx + 1} 页: 中文文本 {page_text_count} 条, "
            f"竖排 {page_vertical_count} 条"
        )

    doc.close()
    return all_text_items


def translate_with_dictionary(text_items: list) -> list:
    """
    使用工程术语字典翻译。
    对矢量 PDF 可保留 step2 原有策略：精确匹配 + 长词优先模糊替换。
    """
    for item in text_items:
        text = item["text"]

        if text in ENGINEERING_DICT:
            item["translated"] = ENGINEERING_DICT[text]
            continue

        result = text
        changed = False

        for cn, en in sorted(ENGINEERING_DICT.items(), key=lambda x: -len(x[0])):
            if cn in result:
                result = result.replace(cn, en)
                changed = True

        item["translated"] = result if changed else text

    return text_items


def translate_with_llm(text_items: list) -> list:
    """
    使用 OpenAI 兼容 API 翻译矢量文本。
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("[Vector] openai 未安装，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        logger.warning("[Vector] LLM API 配置不完整，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    dict_sample = "\n".join(
        [f' "{cn}" → "{en}"' for cn, en in list(ENGINEERING_DICT.items())[:30]]
    )

    system_prompt = f"""You are a professional mechanical/electrical CAD drawing translation expert.

Translate Chinese text in CAD drawings into concise professional English.

Rules:
1. Use standard engineering terminology.
2. Keep translations short because CAD drawing space is limited.
3. Preserve numbers, codes, symbols, Latin letters and line structures.
4. For labels and table cells, prefer abbreviations.
5. Output only numbered translations, one line per item.
6. Items marked [VERTICAL] must be extremely short, usually 1-3 words.

Terminology references:
{dict_sample}
"""

    sorted_items = sorted(
        text_items,
        key=lambda x: (x["page"], x["bbox"][1], x["bbox"][0]),
    )

    batch_size = LLM_BATCH_SIZE
    total_batches = (len(sorted_items) + batch_size - 1) // batch_size
    all_translated = []

    for batch_idx in range(0, len(sorted_items), batch_size):
        batch = sorted_items[batch_idx: batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        items_str = "\n".join(
            [
                f'{j + 1}. "{item["text"]}" '
                f'{"[VERTICAL]" if item.get("is_vertical") else ""}'
                for j, item in enumerate(batch)
            ]
        )

        user_prompt = f"Translate the following CAD drawing text:\n{items_str}"

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

            result_text = response.choices[0].message.content or ""
            lines = result_text.strip().split("\n")

            trans_map = {}
            for line in lines:
                match = re.match(r'(\d+)[.、．]\s*["“”]?(.+?)["“”]?\s*$', line.strip())
                if match:
                    idx = int(match.group(1))
                    trans = match.group(2).strip()
                    trans_map[idx] = trans

            for j, item in enumerate(batch):
                idx = j + 1
                item["translated"] = trans_map.get(idx, item["text"])
                all_translated.append(item)

            matched = len(trans_map)
            logger.info(
                f"[Vector] 翻译批次 {batch_num}/{total_batches}: "
                f"成功解析 {matched}/{len(batch)} 条"
            )

            if matched < len(batch) * 0.5:
                logger.warning("[Vector] 翻译解析率较低，尝试宽松解析")
                valid_lines = [
                    re.sub(r'^\d+[.、．]\s*["“”]?', "", line.strip()).rstrip('"“”')
                    for line in lines
                    if re.match(r'^\d+[.、．]', line.strip())
                ]

                for j, item in enumerate(batch):
                    if j < len(valid_lines) and item.get("translated") == item["text"]:
                        item["translated"] = valid_lines[j]

            time.sleep(0.3)

        except Exception as e:
            logger.exception(f"[Vector] 翻译批次 {batch_num}/{total_batches} 失败: {e}")
            for item in batch:
                item["translated"] = item["text"]
                all_translated.append(item)

    return all_translated


def trim_vertical_translations(text_items: list) -> list:
    """
    对竖排文本译文做长度检查。
    若译文过长且字典里有更短译文，则回退到字典短译文。
    """
    font = fitz.Font("helv")
    trimmed_count = 0

    for item in text_items:
        if not item.get("is_vertical"):
            continue

        bbox = item["bbox"]
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]

        translated = item.get("translated", item["text"])
        original_text = item["text"]

        if translated == original_text:
            continue

        font_size = item["size"]
        char_height = (font.ascender - font.descender) * font_size

        if char_height > bbox_w:
            font_size = font_size * (bbox_w / char_height) * 0.85

        try:
            max_char_w = max(font.text_length(ch, fontsize=font_size) for ch in translated)
        except Exception:
            continue

        needed_h = max_char_w * 1.05 * len(translated)

        if needed_h > bbox_h * 1.2:
            dict_tr = ENGINEERING_DICT.get(original_text, "")
            if dict_tr and len(dict_tr) < len(translated):
                item["translated"] = dict_tr
                trimmed_count += 1

    if trimmed_count:
        logger.info(f"[Vector] 竖排文本缩写: {trimmed_count} 条")

    return text_items


def get_font_path() -> str:
    """
    获取可用英文无衬线字体。
    """
    if FONT_PATH and os.path.exists(FONT_PATH):
        return FONT_PATH

    font_dirs = [
        r"C:\Windows\Fonts",
        "/System/Library/Fonts",
        "/Library/Fonts",
        "/usr/share/fonts",
        "/usr/local/share/fonts",
    ]

    preferred = [
        "arial.ttf",
        "Arial.ttf",
        "calibri.ttf",
        "Helvetica.ttf",
        "helvetica.ttf",
        "DejaVuSans.ttf",
    ]

    for d in font_dirs:
        if not os.path.exists(d):
            continue

        for root, _, files in os.walk(d):
            for fname in preferred:
                if fname in files:
                    return os.path.join(root, fname)

    return "helv"


def calculate_fitted_text(text: str, bbox: list, original_size: float, font: fitz.Font) -> dict:
    """
    计算水平文本在 bbox 内的适配字号、位置和折行方式。
    """
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    if bbox_w <= 0 or bbox_h <= 0:
        return {
            "fontsize": original_size,
            "x": bbox[0],
            "y": bbox[1],
            "needs_split": False,
            "lines": [text],
        }

    text_width = font.text_length(text, fontsize=original_size)

    if text_width <= bbox_w * 1.05:
        return {
            "fontsize": original_size,
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + original_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    scale = bbox_w / text_width if text_width > 0 else 1.0
    fitted_size = original_size * max(scale, 0.4)
    fitted_width = font.text_length(text, fontsize=fitted_size)

    if fitted_width <= bbox_w * 1.1:
        return {
            "fontsize": fitted_size,
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + fitted_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    words = text.split()

    if len(words) <= 1:
        min_size = original_size * 0.3
        fitted_size = max(min_size, fitted_size)

        return {
            "fontsize": fitted_size,
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + fitted_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.text_length(test_line, fontsize=fitted_size) <= bbox_w:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    line_height = fitted_size * 1.2
    total_height = line_height * len(lines)

    while total_height > bbox_h and fitted_size > original_size * 0.25:
        fitted_size *= 0.9
        line_height = fitted_size * 1.2

        lines = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            if font.text_length(test_line, fontsize=fitted_size) <= bbox_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        total_height = line_height * len(lines)

    start_y = bbox[1] + (bbox_h - total_height) / 2 + fitted_size * 0.8

    return {
        "fontsize": fitted_size,
        "x": bbox[0],
        "y": start_y,
        "needs_split": len(lines) > 1,
        "lines": lines,
        "line_height": line_height,
    }


def _int_color_to_rgb(color) -> tuple:
    """
    将 PyMuPDF 的整数颜色转为 0~1 RGB。
    """
    if isinstance(color, int):
        r = ((color >> 16) & 0xFF) / 255.0
        g = ((color >> 8) & 0xFF) / 255.0
        b = (color & 0xFF) / 255.0
        return (r, g, b)

    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return tuple(float(x) for x in color[:3])

    return (0, 0, 0)


def insert_vertical_text(page: fitz.Page, text: str, bbox: list, font_size: float, color):
    """
    在竖排文本框内插入竖排英文。
    策略：单字符旋转 90 度后自下向上排列。
    """
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    n = len(text)

    if n == 0 or bbox_w <= 0 or bbox_h <= 0:
        return

    font = fitz.Font("helv")
    char_height = (font.ascender - font.descender) * font_size

    if char_height > bbox_w:
        font_size = font_size * (bbox_w / char_height) * 0.85
        char_height = (font.ascender - font.descender) * font_size

    actual_spacing = bbox_h / n

    try:
        max_char_rotated_h = max(font.text_length(ch, fontsize=font_size) for ch in text)
    except Exception:
        max_char_rotated_h = font_size

    if actual_spacing < max_char_rotated_h * 1.05:
        actual_spacing = max_char_rotated_h * 1.05

    for i, ch in enumerate(text):
        center_y = bbox[3] - actual_spacing * (i + 0.5)
        center_x = bbox[0] + bbox_w / 2

        tmp_doc = fitz.open()

        tmp_w = max(font.text_length(ch, fontsize=font_size) + 4, 4)
        tmp_h = max(char_height + 4, 4)

        tmp_doc.new_page(width=tmp_w, height=tmp_h)
        tmp_doc[0].insert_text(
            (2, tmp_h + font.descender * font_size - 2),
            ch,
            fontname="helv",
            fontsize=font_size,
            color=color,
        )

        placed_w = char_height
        placed_h = font.text_length(ch, fontsize=font_size)

        rect = fitz.Rect(
            center_x - placed_w / 2,
            center_y - placed_h / 2,
            center_x + placed_w / 2,
            center_y + placed_h / 2,
        )

        page.show_pdf_page(
            rect,
            tmp_doc,
            0,
            clip=fitz.Rect(0, 0, tmp_w, tmp_h),
            rotate=90,
        )

        tmp_doc.close()


def insert_rotated_text(page: fitz.Page, text: str, bbox: list, font_size: float, color, rotation_deg: float):
    """
    插入倾斜/旋转文本。
    注意：这个方法保留 step2 原思路，但对于复杂旋转文字仍建议后续继续优化。
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

    # PyMuPDF 的 insert_textbox 对任意角度支持有限。
    # 这里先使用简单策略：按矩形回填，必要时可后续改成临时 PDF 图层旋转。
    try:
        page.insert_textbox(
            rect,
            text,
            fontname="helv",
            fontsize=font_size,
            color=color,
            rotate=0,
            align=fitz.TEXT_ALIGN_LEFT,
        )
    except Exception:
        page.insert_text(
            (center_x, center_y),
            text,
            fontname="helv",
            fontsize=font_size,
            color=color,
        )


def redact_and_refill(pdf_path: str, output_path: str, text_items: list) -> dict:
    """
    原位擦除中文文本并回填英文。
    """
    doc = fitz.open(pdf_path)
    font_obj = fitz.Font("helv")

    stats = {
        "horizontal": 0,
        "vertical": 0,
        "rotated": 0,
        "split": 0,
        "skipped": 0,
        "errors": 0,
    }

    logger.info("[Vector] Phase 1: 擦除原中文文本")

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_items = [item for item in text_items if item["page"] == page_idx + 1]

        for item in page_items:
            bbox = item["bbox"]
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

            # 轻微外扩，避免中文残影；但 CAD 图纸中要避免遮挡线条，所以只扩 0.3 pt
            rect = rect + (-0.3, -0.3, 0.3, 0.3)

            page.add_redact_annot(rect, fill=(1, 1, 1))

        if page_items:
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    logger.info("[Vector] Phase 2: 回填英文译文")

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_items = [item for item in text_items if item["page"] == page_idx + 1]

        for item in page_items:
            bbox = item["bbox"]
            translated = item.get("translated", item["text"])
            original_text = item["text"]

            if translated == original_text and has_chinese(original_text):
                stats["skipped"] += 1
                continue

            font_size = float(item.get("size", 12))
            color = _int_color_to_rgb(item.get("color", 0))

            try:
                is_vertical = item.get("is_vertical", False)
                rotation_deg = float(item.get("rotation_deg", 0))
                direction = item.get("direction", "horizontal")

                if is_vertical:
                    insert_vertical_text(
                        page=page,
                        text=translated,
                        bbox=bbox,
                        font_size=font_size,
                        color=color,
                    )
                    stats["vertical"] += 1

                elif direction == "rotated" and abs(rotation_deg) > 5:
                    insert_rotated_text(
                        page=page,
                        text=translated,
                        bbox=bbox,
                        font_size=font_size,
                        color=color,
                        rotation_deg=rotation_deg,
                    )
                    stats["rotated"] += 1

                else:
                    fitted = calculate_fitted_text(
                        translated,
                        bbox,
                        font_size,
                        font_obj,
                    )

                    if fitted.get("needs_split"):
                        line_h = fitted.get("line_height", fitted["fontsize"] * 1.2)

                        for li, line in enumerate(fitted["lines"]):
                            y = fitted["y"] + li * line_h
                            page.insert_text(
                                (fitted["x"], y),
                                line,
                                fontname="helv",
                                fontsize=fitted["fontsize"],
                                color=color,
                            )

                        stats["split"] += 1

                    else:
                        page.insert_text(
                            (fitted["x"], fitted["y"]),
                            translated,
                            fontname="helv",
                            fontsize=fitted["fontsize"],
                            color=color,
                        )
                        stats["horizontal"] += 1

            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 10:
                    logger.warning(
                        f"[Vector] 回填失败: '{original_text}' -> '{translated}', 错误: {e}"
                    )

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    logger.info(f"[Vector] 输出完成: {output_path}")
    logger.info(f"[Vector] 回填统计: {stats}")

    return stats


def generate_vector_report(report_path: str, task_info: dict):
    """
    生成矢量 PDF 处理报告。
    """
    lines = [
        "# Vector PDF Translate Report",
        "",
        f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**PDF类型**: vector",
        f"**原文件**: {task_info.get('input_pdf', '')}",
        "",
        "## 统计",
        "",
        f"- 提取中文文本数: {task_info.get('text_count', 0)}",
        f"- 已翻译文本数: {task_info.get('translated_count', 0)}",
        f"- 水平回填: {task_info.get('stats', {}).get('horizontal', 0)}",
        f"- 竖排回填: {task_info.get('stats', {}).get('vertical', 0)}",
        f"- 旋转回填: {task_info.get('stats', {}).get('rotated', 0)}",
        f"- 多行拆分: {task_info.get('stats', {}).get('split', 0)}",
        f"- 跳过: {task_info.get('stats', {}).get('skipped', 0)}",
        f"- 错误: {task_info.get('stats', {}).get('errors', 0)}",
        "",
        "## 输出文件",
        "",
        "- `output_vector.pdf`: 矢量 PDF 翻译主输出",
        "- `extracted_text.json`: 提取到的中文矢量文本",
        "- `translation_mapping.json`: 翻译映射结果",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_vector_pipeline(pdf_path: str, work_dir: str, output_filename: str = "output_vector.pdf") -> dict:
    """
    矢量 PDF 一站式处理入口，供 API 调用。

    Returns:
        {
            "output_pdf": "...",
            "extracted_json": "...",
            "translation_json": "...",
            "report": "...",
            "text_count": int,
            "translated_count": int,
            "stats": dict
        }
    """
    os.makedirs(work_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Vector] CAD Drawing Vector PDF Translation Pipeline")
    logger.info("=" * 60)

    extracted_json = os.path.join(work_dir, "extracted_text.json")
    translation_json = os.path.join(work_dir, "translation_mapping.json")
    output_pdf = os.path.join(work_dir, output_filename)
    report_path = os.path.join(work_dir, "vector_report.md")

    text_items = extract_text_info(pdf_path)

    with open(extracted_json, "w", encoding="utf-8") as f:
        json.dump(text_items, f, ensure_ascii=False, indent=2)

    if TRANSLATE_ENGINE == "llm":
        translated_items = translate_with_llm(text_items)
    else:
        translated_items = translate_with_dictionary(text_items)

    translated_items = trim_vertical_translations(translated_items)

    with open(translation_json, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)

    translated_count = sum(
        1 for item in translated_items
        if item.get("translated", item["text"]) != item["text"]
    )

    stats = redact_and_refill(
        pdf_path=pdf_path,
        output_path=output_pdf,
        text_items=translated_items,
    )

    task_info = {
        "input_pdf": pdf_path,
        "text_count": len(text_items),
        "translated_count": translated_count,
        "stats": stats,
    }

    generate_vector_report(report_path, task_info)

    result = {
        "output_pdf": output_pdf,
        "extracted_json": extracted_json,
        "translation_json": translation_json,
        "report": report_path,
        "text_count": len(text_items),
        "translated_count": translated_count,
        "stats": stats,
    }

    logger.info(f"[Vector] 完成: {result}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vector PDF CAD Drawing Translator")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("-o", "--output", default=None, help="Output PDF path")
    parser.add_argument("-w", "--work-dir", default="vector_work", help="Work directory")

    args = parser.parse_args()

    result = run_vector_pipeline(args.pdf, args.work_dir)

    if args.output:
        import shutil
        shutil.copyfile(result["output_pdf"], args.output)
        logger.info(f"[Vector] 已复制输出到: {args.output}")


if __name__ == "__main__":
    main()