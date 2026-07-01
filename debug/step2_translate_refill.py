"""
Step 2: PDF CAD 图纸中文→英文无损翻译（优化版）
流程: 提取矢量文本 → LLM翻译(OpenAI兼容API) → 原位擦除+回填(含竖排) → 生成新PDF

依赖: pip install PyMuPDF openai
"""
import fitz
import json
import math
import os
import re
import time
from typing import Optional

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_BATCH_SIZE, LLM_TEMPERATURE,
    TRANSLATE_ENGINE, ENGINEERING_DICT, FONT_PATH,
)

# ============================================================
# 配置 (脚本专用路径，可通过命令行参数覆盖)
# ============================================================
PDF_PATH = r"D:\AIGC\projects\pdf_translate\pdfs\260400JRS--P5 (地坑、预埋件布置图) 3×A0.pdf"
OUTPUT_PATH = r"D:\AIGC\projects\pdf_translate\pdfs\260400JRS--P5 (地坑、预埋件布置图) 3×A0--translated_v2.pdf"
EXTRACTED_JSON = r"D:\AIGC\projects\pdf_translate\extracted_text.json"
TRANSLATION_JSON = r"D:\AIGC\projects\pdf_translate\translation_mapping_v2.json"


# ============================================================
# 翻译函数
# ============================================================

def translate_with_dictionary(text_items: list) -> list:
    """使用离线术语字典翻译 - 精确匹配 + 模糊匹配"""
    for item in text_items:
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

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    # 构建术语字典片段注入 prompt
    dict_sample = "\n".join(
        [f'  "{cn}" → "{en}"' for cn, en in list(ENGINEERING_DICT.items())[:30]]
    )

    system_prompt = f"""你是一名专业的机械/电气工程翻译专家，精通ISO/DIN/GB标准术语。
请将CAD图纸中的中文文本翻译为英文。要求：
1. 使用国际通用的工程标准术语（ISO/DIN标准）
2. 尽量简短，英文通常比中文更长，请适当缩写以适配图纸空间
3. 保持专业性和准确性
4. 如果文本包含数字、代号、字母等非中文字符，保留原样只翻译中文部分
5. 仅输出翻译结果，每行格式: 序号. 译文
6. [竖排文本] 标记了 [竖排] 的条目必须用最简短的译文（1-3个单词），因为它们会垂直排列在很窄的空间中

常用术语参考:
{dict_sample}"""

    # 按页码+Y坐标排序，空间邻近的文本在同一批
    sorted_items = sorted(text_items, key=lambda x: (x["page"], x["bbox"][1], x["bbox"][0]))

    batch_size = LLM_BATCH_SIZE
    all_translated = []
    total_batches = (len(sorted_items) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(sorted_items), batch_size):
        batch = sorted_items[batch_idx: batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        items_str = "\n".join(
            [f'{j + 1}. "{item["text"]}" {"[竖排]" if item.get("is_vertical") else ""}'
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

            for j, item in enumerate(batch):
                idx = j + 1
                item["translated"] = trans_map.get(idx, item["text"])
                all_translated.append(item)

            matched = len(trans_map)
            print(f"  批次 {batch_num}/{total_batches}: 成功翻译 {matched}/{len(batch)} 条")

            # 额外检查: 如果匹配率太低，可能是输出格式不一致，尝试宽松解析
            if matched < len(batch) * 0.5:
                print(f"    匹配率较低，尝试宽松解析...")
                # 宽松解析: 按行顺序对应
                valid_lines = [
                    re.sub(r'^\d+[.、．]\s*["""]?', "", line.strip()).rstrip('"""”')
                    for line in lines
                    if re.match(r'^\d+[.、．]', line.strip())
                ]
                for j, item in enumerate(batch):
                    if j + 1 not in trans_map and j < len(valid_lines):
                        item["translated"] = valid_lines[j]
                        # 更新 all_translated 中对应的 item
                        for k, at in enumerate(all_translated):
                            if at is item:
                                all_translated[k] = item
                                break

            time.sleep(0.3)

        except Exception as e:
            print(f"  批次 {batch_num}/{total_batches}: 异常 - {e}")
            for item in batch:
                item["translated"] = item["text"]
                all_translated.append(item)

    return all_translated


def trim_vertical_translations(text_items: list) -> list:
    """对竖排文本的译文进行长度检查，过长时回退到字典短译文

    竖排文本框通常很窄（~15pt宽），译文字符数受限于 bbox 高度。
    如果译文太长，旋转后每个字符会小到看不见。
    """
    font = fitz.Font("helv")
    trimmed_count = 0

    for item in text_items:
        if not item.get("is_vertical"):
            continue

        bbox = item["bbox"]
        bbox_h = bbox[3] - bbox[1]
        translated = item.get("translated", item["text"])
        original_text = item["text"]

        # 跳过未翻译的
        if translated == original_text:
            continue

        # 计算当前字号下每个字符旋转后的垂直占用
        font_size = item["size"]
        char_height = (font.ascender - font.descender) * font_size
        if char_height > (bbox[2] - bbox[0]):
            font_size = font_size * ((bbox[2] - bbox[0]) / char_height) * 0.85

        # 每个字符旋转后的垂直占用 = 其宽度
        max_char_w = max(font.text_length(ch, fontsize=font_size) for ch in translated)
        needed_h = max_char_w * 1.05 * len(translated)

        # 如果译文放不下，尝试字典短译文
        if needed_h > bbox_h * 1.2:  # 允许20%溢出
            dict_tr = ENGINEERING_DICT.get(original_text, "")
            if dict_tr and len(dict_tr) < len(translated):
                item["translated"] = dict_tr
                trimmed_count += 1

    if trimmed_count > 0:
        print(f"  竖排文本缩写: {trimmed_count} 条译文已替换为字典短译文")
    return text_items


# ============================================================
# 原位擦除与矢量回填（含竖排文本支持）
# ============================================================

def get_font_path() -> str:
    """获取可用的英文无衬线字体路径"""
    if FONT_PATH and os.path.exists(FONT_PATH):
        return FONT_PATH
    font_dirs = [r"C:\Windows\Fonts"]
    preferred = ["arial.ttf", "Arial.ttf", "calibri.ttf", "helvetica.ttf"]
    for d in font_dirs:
        if os.path.exists(d):
            for fname in preferred:
                fpath = os.path.join(d, fname)
                if os.path.exists(fpath):
                    return fpath
    return "helv"


def calculate_fitted_text(
    text: str, bbox: list, original_size: float, font: fitz.Font
) -> dict:
    """计算适配文本框的字号和位置

    Returns:
        dict with fontsize, x, y, needs_split, lines
    """
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]

    # 尝试单行放置
    text_width = font.text_length(text, fontsize=original_size)
    if text_width <= bbox_w * 1.05:  # 允许5%溢出
        return {
            "fontsize": original_size,
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + original_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    # 需要缩小字号
    scale = bbox_w / text_width if text_width > 0 else 1.0
    fitted_size = original_size * max(scale, 0.4)  # 最小字号为原字号的40%

    # 再次检查是否能放下
    fitted_width = font.text_length(text, fontsize=fitted_size)
    if fitted_width <= bbox_w * 1.1:
        return {
            "fontsize": fitted_size,
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + fitted_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    # 缩小后仍放不下，尝试多行拆分
    words = text.split()
    if len(words) <= 1:
        # 无空格可拆，强制缩小
        min_size = original_size * 0.3
        return {
            "fontsize": max(min_size, fitted_size),
            "x": bbox[0],
            "y": bbox[1] + (bbox_h + fitted_size * 0.8) / 2,
            "needs_split": False,
            "lines": [text],
        }

    # 贪心拆行
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

    # 计算行高
    line_height = fitted_size * 1.2
    total_height = line_height * len(lines)

    # 如果总高度超出bbox，继续缩小
    while total_height > bbox_h and fitted_size > original_size * 0.25:
        fitted_size *= 0.9
        line_height = fitted_size * 1.2
        # 重新拆行
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

    # 垂直居中起始Y
    start_y = bbox[1] + (bbox_h - total_height) / 2 + fitted_size * 0.8

    return {
        "fontsize": fitted_size,
        "x": bbox[0],
        "y": start_y,
        "needs_split": len(lines) > 1,
        "lines": lines,
        "line_height": line_height,
    }


def insert_vertical_text(page: fitz.Page, text: str, bbox: list, font_size: float, font_path: str, color):
    """在竖排文本框中插入竖排英文文本

    策略: 每个字符通过临时页面+show_pdf_page旋转90度CCW放置
    文本从下到上排列，每个字符逆时针旋转90度

    字号尽量保持原始大小不缩小。如果译文放不下，允许溢出bbox。
    文本长度的控制由 trim_vertical_translations 在翻译阶段处理。
    """
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    n = len(text)
    if n == 0:
        return

    font = fitz.Font("helv")

    # 仅当旋转后字符高度超出bbox宽度时才缩小（确保水平方向放得下）
    char_height = (font.ascender - font.descender) * font_size
    if char_height > bbox_w:
        font_size = font_size * (bbox_w / char_height) * 0.85
        char_height = (font.ascender - font.descender) * font_size

    # 保持原始字号，不因垂直空间不足而缩小
    # 间距基于bbox均匀分布（即使溢出也保持字号）
    actual_spacing = bbox_h / n
    # 确保间距至少能容纳最宽字符，防止重叠
    max_char_rotated_h = max(font.text_length(ch, fontsize=font_size) for ch in text)
    if actual_spacing < max_char_rotated_h * 1.05:
        actual_spacing = max_char_rotated_h * 1.05

    for i, ch in enumerate(text):
        # 从下到上: 第一个字符在底部，最后一个在顶部
        center_y = bbox[3] - actual_spacing * (i + 0.5)
        center_x = bbox[0] + bbox_w / 2

        # 创建临时页面，写入单个字符
        tmp_doc = fitz.open()
        tmp_w = font.text_length(ch, fontsize=font_size) + 4
        tmp_h = char_height + 4
        tmp_doc.new_page(width=tmp_w, height=tmp_h)
        tmp_doc[0].insert_text(
            (2, tmp_h + font.descender * font_size - 2),
            ch,
            fontname="helv",
            fontsize=font_size,
            color=color,
        )

        # 旋转后: 临时页面高度 -> 主页面宽度, 临时页面宽度 -> 主页面高度
        placed_w = char_height  # 旋转后水平占用
        placed_h = font.text_length(ch, fontsize=font_size)  # 旋转后垂直占用
        rect = fitz.Rect(
            center_x - placed_w / 2,
            center_y - placed_h / 2,
            center_x + placed_w / 2,
            center_y + placed_h / 2,
        )

        page.show_pdf_page(
            rect, tmp_doc, 0,
            clip=fitz.Rect(0, 0, tmp_w, tmp_h),
            rotate=90,
        )
        tmp_doc.close()


def insert_rotated_text(page: fitz.Page, text: str, bbox: list, font_size: float, color, rotation_deg: float):
    """插入旋转文本（适用于非0度/90度的倾斜文本）"""
    # 使用 TextWriter + Matrix 实现旋转
    tw = fitz.TextWriter(page.rect)
    font = fitz.Font("helv")
    tw.append(fitz.Point(0, 0), text, font=font, fontsize=font_size)

    # 创建旋转矩阵
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    mat = fitz.Matrix(1, 1).prerotate(rotation_deg)
    mat = mat.pretranslate(center_x, center_y)

    tw.write_text(page, color=color, matrix=mat)


def redact_and_refill(pdf_path: str, output_path: str, text_items: list):
    """原位擦除中文文本，回填英文翻译（支持竖排文本）"""
    doc = fitz.open(pdf_path)

    font_path = get_font_path()
    print(f"使用字体: {font_path}")

    # Phase 1: 擦除所有原文本
    print("Phase 1: 擦除原文本...")
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_items = [item for item in text_items if item["page"] == page_idx + 1]

        for item in page_items:
            bbox = item["bbox"]
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            page.add_redact_annot(rect, fill=(1, 1, 1))

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

    # Phase 2: 回填翻译文本
    print("Phase 2: 回填翻译文本...")
    success_count = 0
    vertical_count = 0
    rotated_count = 0
    split_count = 0
    skip_count = 0
    error_count = 0

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        page_items = [item for item in text_items if item["page"] == page_idx + 1]

        for item in page_items:
            bbox = item["bbox"]
            translated = item.get("translated", item["text"])
            original_text = item["text"]

            # 跳过未翻译的纯中文文本
            if translated == original_text and any(
                "一" <= ch <= "鿿" for ch in original_text
            ):
                skip_count += 1
                continue

            font_size = item["size"]
            color = item.get("color", 0)
            # 将整数颜色转为RGB元组
            if isinstance(color, int):
                r = ((color >> 16) & 0xFF) / 255.0
                g = ((color >> 8) & 0xFF) / 255.0
                b = (color & 0xFF) / 255.0
                color = (r, g, b)

            try:
                is_vertical = item.get("is_vertical", False)
                rotation_deg = item.get("rotation_deg", 0)
                direction = item.get("direction", "horizontal")

                if is_vertical:
                    # 竖排文本：逐字符垂直排列
                    insert_vertical_text(
                        page, translated, bbox, font_size, font_path, color
                    )
                    vertical_count += 1

                elif direction == "rotated" and abs(rotation_deg) > 5:
                    # 旋转文本：使用矩阵旋转
                    insert_rotated_text(
                        page, translated, bbox, font_size, color, rotation_deg
                    )
                    rotated_count += 1

                else:
                    # 水平文本：智能适配bbox
                    font_obj = fitz.Font("helv")
                    fitted = calculate_fitted_text(
                        translated, bbox, font_size, font_obj
                    )

                    if fitted["needs_split"]:
                        # 多行文本
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
                        split_count += 1
                    else:
                        # 单行文本
                        page.insert_text(
                            (fitted["x"], fitted["y"]),
                            translated,
                            fontname="helv",
                            fontsize=fitted["fontsize"],
                            color=color,
                        )

                    success_count += 1

            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"  回填失败: '{original_text}' -> '{translated}' | 错误: {e}")

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    print(f"\n回填统计:")
    print(f"  水平文本: {success_count}")
    print(f"  竖排文本: {vertical_count}")
    print(f"  旋转文本: {rotated_count}")
    print(f"  多行拆分: {split_count}")
    print(f"  跳过(未翻译): {skip_count}")
    print(f"  失败: {error_count}")
    print(f"  输出: {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    if not os.path.exists(PDF_PATH):
        print(f"错误: PDF文件不存在: {PDF_PATH}")
        return

    # 1. 加载提取结果
    if not os.path.exists(EXTRACTED_JSON):
        print("错误: 未找到 extracted_text.json，请先运行 step1_extract_text.py")
        return

    with open(EXTRACTED_JSON, "r", encoding="utf-8") as f:
        text_items = json.load(f)
    print(f"从 {EXTRACTED_JSON} 加载了 {len(text_items)} 条文本")

    vertical_items = sum(1 for item in text_items if item.get("is_vertical"))
    print(f"  其中竖排文本: {vertical_items}")

    # 2. 翻译
    print(f"\n使用翻译引擎: {TRANSLATE_ENGINE}")
    if TRANSLATE_ENGINE == "llm":
        translated_items = translate_with_llm(text_items)
    else:
        translated_items = translate_with_dictionary(text_items)

    # 对竖排文本进行译文长度检查，过长时回退到字典短译文
    translated_items = trim_vertical_translations(translated_items)

    # 保存翻译对照表
    with open(TRANSLATION_JSON, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)
    print(f"翻译对照表已保存: {TRANSLATION_JSON}")

    # 翻译统计
    translated_count = sum(
        1 for item in translated_items if item.get("translated", item["text"]) != item["text"]
    )
    print(f"翻译完成: {translated_count}/{len(translated_items)} 条文本已翻译")

    # 3. 原位擦除与回填
    print(f"\n开始原位擦除与回填...")
    redact_and_refill(PDF_PATH, OUTPUT_PATH, translated_items)


if __name__ == "__main__":
    main()
