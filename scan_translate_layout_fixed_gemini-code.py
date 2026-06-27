"""
扫描型PDF中文→英文翻译 - RapidOCR + 分块OCR + 智能行合并排版优化版
流程: 扫描PDF→渲染→分块OCR→单元格优先检测→智能文本行合并→隔离标签翻译→白底擦除→绝对左对齐回填→重构PDF

v2.0 优化（按扫描PDF图纸处理规则）:
  - 单元格优先检测：先找格网，再分配OCR块，处理同格多box
  - 单元格最小化擦除，2px内缩保护格线
  - 增强CAD线 vs 表格线判别（矩形闭合验证、相邻格一致性）
  - 修复_fit_text_to_box回退高度造假 → 文本重叠
  - 修复_wrap_structured_text长单词不折行 → 宽度溢出
  - 单元格内按内容比例分配行高，硬裁剪防溢出
  - 检测原文对齐方式（左/中/右）并复刻
  - 旋转文本图层溢出保护
  - 翻译阶段保留单元格原始换行结构
"""
import os, json, sys, gc
import re
import time
import numpy as np
import cv2
import fitz
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_BATCH_SIZE, LLM_TEMPERATURE,
    TRANSLATE_ENGINE, ENGINEERING_DICT, RENDER_DPI, CHUNK_SIZE, FONT_PATH,
)

PDF_PATH = r"d:\AIGC\projects\pdf_translate\pdfs\20260523-Rolling Mill Foundation Plan GZL24.7-17.8基础平面图-V2.0_1.pdf"
WORK_DIR = r"d:\AIGC\projects\pdf_translate\scan_work"
OUTPUT_PDF = r"d:\AIGC\projects\pdf_translate\pdfs\20260523-Rolling Mill Foundation Plan GZL24.7-17.8基础平面图-V2.0_1_translated.pdf"

# ---- 版面渲染控制常量（可由环境变量覆盖）----
# 近水平判定阈值：合并分类与回填走水平/旋转分支必须用同一阈值，
# 否则会出现 2.9°/3.7° 这类近水平文本被误判进旋转图层路径
ANGLE_NEAR_HORIZONTAL = float(os.environ.get("ANGLE_NEAR_HORIZONTAL", "5.0"))
# 右页边留白（像素），水平绘制时绝不允许文字越过 page_w - RIGHT_MARGIN
RIGHT_MARGIN = int(os.environ.get("RIGHT_MARGIN", "8"))

# ---- 单元格检测常量 ----
CELL_MIN_W = int(os.environ.get("CELL_MIN_W", "20"))
CELL_MIN_H = int(os.environ.get("CELL_MIN_H", "20"))
CELL_MAX_W = int(os.environ.get("CELL_MAX_W", "1200"))
CELL_MAX_H = int(os.environ.get("CELL_MAX_H", "400"))  # 放宽到400px，容纳多行单元格（如"借（通）用\n件登记"）
CELL_ERASE_INSET = int(os.environ.get("CELL_ERASE_INSET", "2"))  # 格内擦除缩进，保护格线
CELL_WHITE_THRESHOLD = float(os.environ.get("CELL_WHITE_THRESHOLD", "0.80"))  # 格内白底比例

# ---- loguru 日志配置 ----
LOG_DIR = os.path.join(WORK_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
from datetime import datetime
LOG_FILE = os.path.join(LOG_DIR, f"cell_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# 移除默认 handler，添加控制台 + 文件双输出
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add(LOG_FILE, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}", rotation="10 MB", retention="7 days")

# ---- 单元格全局注册表 ----
# cell_registry: {(cl, ct, cr, cb) -> cell_id}
# cell_texts: {cell_id -> [(item_idx, original_text, translated_text), ...]}
_cell_registry = {}
_cell_texts = {}
_cell_counter = [0]  # 可变计数器


def _cell_id(cell_key: tuple) -> str:
    """获取或分配单元格唯一编号。"""
    if cell_key not in _cell_registry:
        _cell_counter[0] += 1
        cid = f"Cell_{_cell_counter[0]:03d}"
        _cell_registry[cell_key] = cid
        _cell_texts[cid] = []
        logger.debug(f"  [注册] {cid} @ ({cell_key[0]},{cell_key[1]})-({cell_key[2]},{cell_key[3]}) "
                     f"宽={cell_key[2]-cell_key[0]}px 高={cell_key[3]-cell_key[1]}px")
    return _cell_registry[cell_key]


def _register_text_in_cell(cell_key: tuple, item_idx: int, original: str, translated: str):
    """记录某个OCR项的文本被分配到哪个单元格。"""
    cid = _cell_id(cell_key)
    _cell_texts[cid].append((item_idx, original, translated))
    logger.info(f"  [分配] {cid} ← OCR#{item_idx} 原文='{original[:40]}{'...' if len(original)>40 else ''}'")


def _merge_nested_cells():
    """合并嵌套/重叠的单元格（同一真实格被_find_table_cell多次检测为不同格）。

    策略：如果格A被格B完全包含（或>80%面积重叠），保留较大的格B，将A的OCR文本迁移到B。
    同时合并共享3条边界的格（如顶部对齐、左右对齐仅底部不同）。

    返回: {removed_cell_key: keeper_cell_key} 映射表，供后续更新 item.cell 引用。
    """
    cell_remap = {}  # removed_key -> keeper_key
    if len(_cell_registry) < 2:
        return cell_remap

    # cell_registry: {(cl,ct,cr,cb) -> cid}
    cells = list(_cell_registry.keys())
    merged_count = 0

    for i in range(len(cells)):
        ci = cells[i]
        if ci not in _cell_registry:
            continue
        cl_i, ct_i, cr_i, cb_i = ci
        area_i = (cr_i - cl_i) * (cb_i - ct_i)

        for j in range(i + 1, len(cells)):
            cj = cells[j]
            if cj not in _cell_registry:
                continue
            cl_j, ct_j, cr_j, cb_j = cj
            area_j = (cr_j - cl_j) * (cb_j - ct_j)

            # 重叠面积
            ox1, oy1 = max(cl_i, cl_j), max(ct_i, ct_j)
            ox2, oy2 = min(cr_i, cr_j), min(cb_i, cb_j)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            overlap = (ox2 - ox1) * (oy2 - oy1)

            # 共享3条边界（水平坐标相同且垂直坐标恰好拼接 / 或相反）
            same_left = abs(cl_i - cl_j) <= 3
            same_right = abs(cr_i - cr_j) <= 3
            same_top = abs(ct_i - ct_j) <= 3
            same_bottom = abs(cb_i - cb_j) <= 3
            share_3_edges = (same_left + same_right + same_top + same_bottom) >= 3

            # 合并条件：>70%重叠 或 共享3条边
            min_area = min(area_i, area_j)
            should_merge = (min_area > 0 and overlap > min_area * 0.70) or share_3_edges

            if should_merge:
                # 保留较大的格
                if area_i >= area_j:
                    keeper, removed = ci, cj
                    keeper_cid = _cell_registry[ci]
                    removed_cid = _cell_registry[cj]
                else:
                    keeper, removed = cj, ci
                    keeper_cid = _cell_registry[cj]
                    removed_cid = _cell_registry[ci]

                # 迁移OCR文本
                if removed_cid in _cell_texts:
                    _cell_texts.setdefault(keeper_cid, []).extend(_cell_texts[removed_cid])
                    del _cell_texts[removed_cid]

                # 更新注册表
                del _cell_registry[removed]
                cell_remap[removed] = keeper  # 记录映射
                merged_count += 1
                logger.info(f"  [格合并] {removed_cid} → {keeper_cid}: "
                            f"({removed[0]},{removed[1]})-({removed[2]},{removed[3]}) 合并入 "
                            f"({keeper[0]},{keeper[1]})-({keeper[2]},{keeper[3]}) "
                            f"(重叠={overlap/min_area*100:.0f}%)")

    if merged_count:
        logger.info(f"  [格合并] 共合并 {merged_count} 对嵌套/重叠单元格")
    return cell_remap


def _generate_cell_report(report_path: str):
    """生成单元格→OCR文本映射报告（Markdown表格格式）。

    输出清晰的表格展示:
      - 哪些OCR文本在表格单元格内
      - 哪些OCR原文本在同一单元格
      - 每个单元格的坐标、尺寸、翻译状态
    """
    if not _cell_texts:
        logger.info("  [报告] 无单元格数据，跳过报告生成")
        return

    lines = []
    lines.append("# 单元格→OCR文本映射报告")
    lines.append("")
    lines.append(f"**注册单元格总数**: {_cell_counter[0]}  ")
    lines.append(f"**含文本单元格数**: {sum(1 for v in _cell_texts.values() if v)}  ")
    lines.append(f"**总OCR项数**: {sum(len(v) for v in _cell_texts.values())}  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 构建逆向索引: (cl,ct,cr,cb) -> cell_id
    coord_to_id = {v: k for k, v in _cell_registry.items()}

    # 按cell_id排序
    for cid in sorted(_cell_texts.keys(), key=lambda x: int(x.split("_")[1])):
        texts = _cell_texts[cid]
        # 找坐标
        cell_key = None
        for ck, cname in _cell_registry.items():
            if cname == cid:
                cell_key = ck
                break

        if cell_key:
            cl, ct, cr, cb = cell_key
            cw, ch = cr - cl, cb - ct
            coord_str = f"({cl},{ct})→({cr},{cb}) {cw}×{ch}px"
        else:
            coord_str = "坐标未知"

        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"- **坐标**: {coord_str}")
        lines.append(f"- **OCR项数**: {len(texts)}")
        lines.append("")

        if len(texts) > 1:
            lines.append(f"> ⚠️ **同一单元格包含 {len(texts)} 个OCR文本块**")
            lines.append("")

        lines.append("| # | OCR原文 | 翻译结果 | 状态 |")
        lines.append("|---|---------|----------|------|")

        for idx, orig, trans in texts:
            is_trans = "✓ 已翻译" if orig != trans else "✗ 未翻译"
            # 截断过长文本
            orig_short = orig[:60] + "..." if len(orig) > 60 else orig
            trans_short = trans[:60] + "..." if len(trans) > 60 else trans
            lines.append(f"| OCR#{idx} | `{orig_short}` | `{trans_short}` | {is_trans} |")

        lines.append("")

    # 未含文本的格
    empty = [cid for cid in sorted(_cell_texts.keys()) if not _cell_texts[cid]]
    if empty:
        lines.append("---")
        lines.append("")
        lines.append("## ⚠️ 空单元格（注册但无OCR文本）")
        lines.append("")
        for cid in empty:
            cell_key = None
            for ck, cname in _cell_registry.items():
                if cname == cid:
                    cell_key = ck
                    break
            if cell_key:
                lines.append(f"- **{cid}**: ({cell_key[0]},{cell_key[1]})→({cell_key[2]},{cell_key[3]})")
            else:
                lines.append(f"- **{cid}**")
        lines.append("")

    report = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"  [报告] 单元格映射报告已保存: {report_path}")

    # 同时输出到控制台（精简版）
    print("\n" + "=" * 70)
    print("  单元格 → OCR文本 映射总览")
    print("=" * 70)
    for cid in sorted(_cell_texts.keys(), key=lambda x: int(x.split("_")[1])):
        texts = _cell_texts[cid]
        cell_key = None
        for ck, cname in _cell_registry.items():
            if cname == cid:
                cell_key = ck
                break
        if cell_key:
            cw, ch = cell_key[2] - cell_key[0], cell_key[3] - cell_key[1]
            flag = " ⚠️多文本" if len(texts) > 1 else ""
            print(f"  {cid} | {cw}×{ch}px | {len(texts)}项{flag}")
            for idx, orig, trans in texts:
                status = "✓" if orig != trans else "✗"
                print(f"    {status} OCR#{idx}: {orig[:45]} → {trans[:45]}")
    print("=" * 70)


def _clear_cell_registry():
    """清空注册表（每次运行前重置）。"""
    _cell_registry.clear()
    _cell_texts.clear()
    _cell_counter[0] = 0


def translate_with_dictionary(text_items: list) -> list:
    """使用离线术语字典翻译 - 仅精确匹配"""
    for item in text_items:
        text = item["text"]
        if text in ENGINEERING_DICT:
            item["translated"] = ENGINEERING_DICT[text]
    return text_items


def translate_with_llm(text_items: list) -> list:
    """使用独立结构化标签组进行LLM翻译，杜绝数字编号错位干扰。

    v2.0: 单元格内保留原文换行结构（不再合并），让模型按行翻译，
          回填时可按原文行结构还原排版。
    """
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
2. Extremely short! Use standard abbreviations (e.g., Int. for Intermediate, Mat'l for Material, Req. for Requirements, Thk. for Thickness, DWG for Drawing, Qty. for Quantity, Dia. for Diameter, Lgth. for Length, No. for Number, Grd. for Grade).
3. If the source text contains numbers, symbols, or multi-line enumerations, preserve their internal structures EXACTLY.
4. For multi-line inputs, translate line-by-line. Keep the exact line breaks, ordering, and bullet numbers. Never merge lines or collapse lists.
5. Strict Format: You MUST output using the structured tags. Respond ONLY with [ITEM_START], ID, TRN, and [ITEM_END]. No extra prose.
6. CRITICAL for table cells, annotations, labels: output ≤3 words. Abbreviate aggressively! e.g. "Anchor Bolt Dia.", "Nut Exposed Lgth.", "Sec. Grouting Thk.", "Reserved Hole Size"

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

        # 构建高隔离度的结构化输入报文。
        # v2.0: 保留单元格原文换行结构，让模型按行翻译
        user_prompt = "待翻译文本块列表如下：\n\n"
        for orig_idx, item in batch:
            src_text = item['text']
            user_prompt += f"[ITEM_START]\nID: {orig_idx}\nSRC: {src_text}\n[ITEM_END]\n"

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
                # 优先取 TRN 字段（模型实际输出的译文）；
                # 仅当模型漏掉 TRN 标签时才回退到 SRC，避免译文被原文覆盖
                trn_match = re.search(r"TRN:\s*(.*?)(?:\n\[ITEM_END\]|\Z)", block, re.DOTALL)
                if not trn_match:
                    trn_match = re.search(r"SRC:\s*(.*)", block, re.DOTALL)
                if id_match and trn_match:
                    idx = int(id_match.group(1))
                    trans_map[idx] = trn_match.group(1).strip()

            # 回填翻译结果
            success_this_batch = 0
            for orig_idx, item in batch:
                if orig_idx in trans_map:
                    translated = trans_map[orig_idx]
                    item["translated"] = translated
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


def _order_quad(box):
    """把 OCR 4点 box 规范成 [tl, tr, br, bl] 顺序（顺时针）。"""
    pts = np.array(box[:4], dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _group_coords(coords, gap=4):
    """把连续相邻的坐标合并成一条线的代表坐标（线条通常 1~3px 厚）。"""
    if len(coords) == 0:
        return []
    out = []
    start = prev = coords[0]
    for c in coords[1:]:
        if c - prev <= gap:
            prev = c
        else:
            out.append((start + prev) // 2)
            start = prev = c
    out.append((start + prev) // 2)
    return out


def _detect_table_lines(img_gray):
    """全图检测长直线，返回 (h_lines, v_lines, hmask, vmask)。

    h_lines/v_lines: [(坐标, 起点, 终点), ...]
    hmask/vmask: bool 掩码
    """
    h_img, w_img = img_gray.shape
    bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h_len = max(30, 120)
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))) > 0
    v_len = max(30, 120)
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))) > 0
    h_cand = _group_coords(np.where(np.sum(hmask, axis=1) > 5)[0])
    v_cand = _group_coords(np.where(np.sum(vmask, axis=0) > 5)[0])
    h_out = []
    for y in h_cand:
        dark_cols = np.where(hmask[y])[0]
        if len(dark_cols) and (dark_cols[-1] - dark_cols[0]) > 200:
            h_out.append((int(y), int(dark_cols[0]), int(dark_cols[-1])))
    v_out = []
    for x in v_cand:
        dark_rows = np.where(vmask[:, x])[0]
        if len(dark_rows) and (dark_rows[-1] - dark_rows[0]) > 200:
            v_out.append((int(x), int(dark_rows[0]), int(dark_rows[-1])))
    return h_out, v_out, hmask, vmask


def _detect_all_table_cells(h_lines, v_lines, hmask, vmask, img_gray, ocr_items=None):
    """单元格优先检测：对每个OCR项运行_find_table_cell，去重后返回所有格。

    v2.3: 放弃全局线枚举（CAD中线与表格线交错导致漏检），
          改为对每个OCR块执行目标搜索，再合并去重。

    返回: [(left, top, right, bottom), ...]
    """
    if not ocr_items:
        logger.info("  [格网检测] 无OCR项，跳过格网检测")
        return [], [], []

    logger.info(f"  [格网检测] 对{len(ocr_items)}个OCR项执行目标格搜索...")

    cells_set = set()  # 用 set 自动去重
    found_count = 0
    for idx, item in enumerate(ocr_items):
        cell = _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray, margin=50)
        if cell is not None:
            # 放宽margin重试
            cl, ct, cr, cb = cell
            # 微调: 用_register_text_in_cell中的逻辑检查重叠
            cells_set.add(cell)
            found_count += 1

    cells = sorted(cells_set, key=lambda c: (c[1], c[0]))

    # 相邻格一致性过滤
    if len(cells) > 1:
        filtered = []
        isolated = []
        for i, (cl, ct, cr, cb) in enumerate(cells):
            has_neighbor = False
            for j, (nl, nt, nr, nb) in enumerate(cells):
                if i == j:
                    continue
                if (abs(ct - nb) <= 5 or abs(cb - nt) <= 5) and (max(cl, nl) < min(cr, nr)):
                    has_neighbor = True
                    break
                if (abs(cr - nl) <= 5 or abs(cl - nr) <= 5) and (max(ct, nt) < min(cb, nb)):
                    has_neighbor = True
                    break
            if has_neighbor or len(cells) <= 3:
                filtered.append((cl, ct, cr, cb))
            else:
                isolated.append((cl, ct, cr, cb))
        if isolated:
            for iso in isolated:
                logger.info(f"  [格网检测] 排除孤立格 ({iso[0]},{iso[1]})-({iso[2]},{iso[3]})")
        cells = filtered

    for cell in cells:
        _cell_id(cell)

    logger.info(f"  [格网检测] 最终确认 {len(cells)} 个单元格 (从{found_count}次命中去重)")
    for cell in cells:
        cid = _cell_registry[cell]
        logger.info(f"    {cid}: ({cell[0]},{cell[1]})-({cell[2]},{cell[3]}) {cell[2]-cell[0]}x{cell[3]-cell[1]}px")
    return cells, [], []


def _line_crosses_region(mask, coord, r1, r2):
    """验证线条在区域 [r1, r2] 内是否有像素存在。"""
    h, w = mask.shape
    r1, r2 = max(0, int(r1)), min(w - 1, int(r2))
    if r2 <= r1:
        return False
    if coord < 0 or coord >= h:
        return False
    return bool(np.any(mask[int(coord), r1:r2 + 1]))


def _find_table_cell(bbox, h_lines, v_lines, hmask, vmask, img_gray, margin=50):
    """用已检测出的表格线，求出包围 bbox 的单元格四边。

    通过 hmask/vmask 精确判断每条候选线在 text bbox 区域内是否真实存在，
    杜绝右下标题栏的线被误用到左下表格（即使线跨全图长，中间如果被空白截断也通不过）。

    v2.2: 如果标准搜索失败，自动扩大margin重试；仍失败则用图像边界兜底。
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = img_gray.shape

    def _search(m):
        """在给定margin下搜索四边。返回 (gl,gt,gr,gb) 或 None。"""
        lx, rx = max(0, x1 - m), min(w_img - 1, x2 + m)
        ty_roi, by_roi = max(0, y1 - m), min(h_img - 1, y2 + m)
        top_cands, bot_cands = [], []
        for y, _, _ in h_lines:
            if y < 0 or y >= h_img:
                continue
            if not np.any(hmask[y, lx:rx + 1]):
                continue
            if y <= y1:
                top_cands.append(y)
            if y >= y2:
                bot_cands.append(y)
        top_cands.sort()
        bot_cands.sort()
        lft_cands, rgt_cands = [], []
        for x, _, _ in v_lines:
            if x < 0 or x >= w_img:
                continue
            if not np.any(vmask[ty_roi:by_roi + 1, x]):
                continue
            if x <= x1:
                lft_cands.append(x)
            if x >= x2:
                rgt_cands.append(x)
        lft_cands.sort()
        rgt_cands.sort()
        if not (top_cands and bot_cands and lft_cands and rgt_cands):
            return None
        gl, gt, gr, gb = lft_cands[-1], top_cands[-1], rgt_cands[0], bot_cands[0]
        cw, ch = gr - gl, gb - gt
        if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
            logger.debug(f"  [_find_table_cell] 尺寸不符: ({gl},{gt})-({gr},{gb}) {cw}x{ch}px "
                         f"(允许{CELL_MIN_W}-{CELL_MAX_W} x {CELL_MIN_H}-{CELL_MAX_H})")
            return None
        # 白底验证
        if gt + 2 < gb - 2 and gl + 2 < gr - 2:
            roi = img_gray[gt + 2:gb - 2, gl + 2:gr - 2]
            if roi.size == 0 or float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
                return None
        return gl, gt, gr, gb

    # 尝试标准margin
    result = _search(margin)
    if result is not None:
        return result

    # 扩大搜索范围重试（处理靠近页面边缘的单元格）
    result = _search(max(margin, 200))
    if result is not None:
        logger.debug(f"  [_find_table_cell] 扩大margin=200成功: bbox({x1},{y1})-({x2},{y2})")
        return result

    # 最后兜底：用图像边界替代缺失的格线
    top_cands, bot_cands = [], []
    for y, _, _ in h_lines:
        if y < 0 or y >= h_img:
            continue
        lx, rx = max(0, x1 - 200), min(w_img - 1, x2 + 200)
        if not np.any(hmask[y, lx:rx + 1]):
            continue
        if y <= y1:
            top_cands.append(y)
        if y >= y2:
            bot_cands.append(y)
    top_cands.sort()
    bot_cands.sort()
    lft_cands, rgt_cands = [], []
    for x, _, _ in v_lines:
        if x < 0 or x >= w_img:
            continue
        ty_roi, by_roi = max(0, y1 - 200), min(h_img - 1, y2 + 200)
        if not np.any(vmask[ty_roi:by_roi + 1, x]):
            continue
        if x <= x1:
            lft_cands.append(x)
        if x >= x2:
            rgt_cands.append(x)
    lft_cands.sort()
    rgt_cands.sort()

    # 用图像边界补全缺失的边
    gl = lft_cands[-1] if lft_cands else 0
    gt = top_cands[-1] if top_cands else 0
    gr = rgt_cands[0] if rgt_cands else w_img - 1
    gb = bot_cands[0] if bot_cands else h_img - 1
    cw, ch = gr - gl, gb - gt
    if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
        return None
    if gt + 2 < gb - 2 and gl + 2 < gr - 2:
        roi = img_gray[gt + 2:gb - 2, gl + 2:gr - 2]
        if roi.size > 0 and float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
            return None
    logger.info(f"  [_find_table_cell] 边界兜底: bbox({x1},{y1})-({x2},{y2}) → ({gl},{gt})-({gr},{gb})")
    return gl, gt, gr, gb


def _is_in_table(item, h_lines, v_lines, hmask, vmask, img_gray):
    """判断 OCR 块是否落在表格单元格内。"""
    return _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray) is not None


def merge_ocr_items(items: list, img_bgr=None) -> list:
    """
    单元格优先 + 智能行合并算法。

    v2.0 三阶段流程：
      1. 全图检测所有表格格网（独立于OCR块，无位置偏见）
      2. 分配OCR到格（同格合并）+ 未分配的走智能行合并
      3. 返回合并结果
    """
    if not items:
        return []

    img_gray = None
    h_lines, v_lines = [], []
    hmask, vmask = None, None
    all_cells = []

    if img_bgr is not None:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h_lines, v_lines, hmask, vmask = _detect_table_lines(img_gray)
        print(f"  [表格线检测] 局部水平线 {len(h_lines)} 条, 局部竖线 {len(v_lines)} 条")

        # === 阶段1: 全图格网检测 (v2.1: 传入OCR项用于过滤CAD假格) ===
        all_cells, tbl_h, tbl_v = _detect_all_table_cells(h_lines, v_lines, hmask, vmask, img_gray, ocr_items=items)

        # === 阶段2: 分配OCR到格 ===
        cell_map = {}  # cell_key -> [items]
        unassigned = []
        for idx, item in enumerate(items):
            x1, y1, x2, y2 = item["bbox"]
            best_cell = None
            best_overlap = 0
            best_overlap_ratio = 0
            for (cl, ct, cr, cb) in all_cells:
                # 计算OCR bbox与格的重叠面积
                ox1, oy1 = max(x1, cl), max(y1, ct)
                ox2, oy2 = min(x2, cr), min(y2, cb)
                if ox2 > ox1 and oy2 > oy1:
                    overlap = (ox2 - ox1) * (oy2 - oy1)
                    # OCR bbox 至少 50% 在格内才算归属
                    item_area = (x2 - x1) * (y2 - y1)
                    if item_area > 0:
                        ratio = overlap / item_area
                        if ratio > 0.5 and overlap > best_overlap:
                            best_overlap = overlap
                            best_overlap_ratio = ratio
                            best_cell = (cl, ct, cr, cb)
            if best_cell is not None:
                cell_map.setdefault(best_cell, []).append(item)
                _register_text_in_cell(best_cell, idx, item["text"], item.get("translated", item["text"]))
                logger.debug(f"    OCR#{idx} 重叠率={best_overlap_ratio:.1%} → {_cell_id(best_cell)}")
            else:
                unassigned.append(item)
                if all_cells:
                    # 找最近格的距离，帮助诊断为何未分配
                    min_dist = min(
                        max(0, max(cl - x2, x1 - cr, ct - y2, y1 - cb))
                        for (cl, ct, cr, cb) in all_cells
                    )
                    logger.debug(f"    OCR#{idx} 未分配 (最近格距离={min_dist}px) '{item['text'][:30]}'")
                else:
                    logger.debug(f"    OCR#{idx} 未分配 (无候选格) '{item['text'][:30]}'")

        # 同格合并
        merged_cell = []
        merge_cnt = 0
        for cell_key, grp in cell_map.items():
            cl, ct, cr, cb = cell_key
            cid = _cell_id(cell_key)
            if len(grp) == 1:
                it = grp[0]
                it["cell"] = cell_key
                it["in_table"] = True
                merged_cell.append(it)
                logger.info(f"  [同格] {cid}: 1个OCR块 '{it['text'][:40]}'")
            else:
                # 按原文纵坐标排序
                grp.sort(key=lambda x: x["bbox"][1])
                bx1 = min(g["bbox"][0] for g in grp)
                by1 = min(g["bbox"][1] for g in grp)
                bx2 = max(g["bbox"][2] for g in grp)
                by2 = max(g["bbox"][3] for g in grp)
                # 保存每个子块的原始bbox，用于渲染时按比例分配行高
                sub_bboxes = [g["bbox"] for g in grp]
                sub_texts = [g["text"] for g in grp]
                merged_cell.append({
                    "text": "\n".join(sub_texts),
                    "bbox": [bx1, by1, bx2, by2],
                    "box": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                    "angle": 0.0,
                    "confidence": round(sum(g.get("confidence", 1.0) for g in grp) / len(grp), 3),
                    "width": int(bx2 - bx1),
                    "height": int(by2 - by1),
                    "in_table": True,
                    "is_structured": False,
                    "cell": cell_key,
                    "sub_bboxes": sub_bboxes,  # v2.0: 保留原始子块bbox用于行高分配
                })
                merge_cnt += 1
                logger.info(f"  [同格合并] {cid}: {len(grp)}个OCR块合并, 子文本: {sub_texts}")
        logger.info(f"  [格网分配] 总计: {len(all_cells)} 格, 含OCR格 {len(cell_map)} 个, 同格合并 {merge_cnt} 组, 未分配 {len(unassigned)} 项")

        # 对未分配的做 in_table 标记（回退到传统方法）
        in_table_fallback = 0
        for idx, item in enumerate(unassigned):
            try:
                if _is_in_table(item, h_lines, v_lines, hmask, vmask, img_gray):
                    item["in_table"] = True
                    in_table_fallback += 1
                    logger.debug(f"    OCR未分配项 回退in_table=True: '{item['text'][:30]}'")
                else:
                    item["in_table"] = False
            except Exception:
                item["in_table"] = False
        if in_table_fallback > 0:
            logger.info(f"  [回退检测] {in_table_fallback}个未分配项通过_find_table_cell标记为in_table")
    else:
        unassigned = list(items)

    # === 阶段3: 未分配项的智能行合并（非表格文本） ===
    # 拆分水平文本和带有大角度的倾斜/垂直文本（阈值与回填阶段统一）
    horizontal_items = [item for item in unassigned if abs(item.get("angle", 0.0)) < ANGLE_NEAR_HORIZONTAL]
    rotated_items = [item for item in unassigned if abs(item.get("angle", 0.0)) >= ANGLE_NEAR_HORIZONTAL]

    # 按纵向坐标顶部自上而下排序
    horizontal_items.sort(key=lambda x: x["bbox"][1])

    merged_text = []
    visited = set()

    for i, item in enumerate(horizontal_items):
        if i in visited:
            continue

        current_block = [item]
        visited.add(i)
        # 起点：若当前块在表格网格内，则不允许合并（单元格各自独立）
        block_locked = bool(item.get("in_table", False))

        while not block_locked:
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
                # 候选落在表格单元格内则跳过，不合并
                if next_item.get("in_table", False):
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

            if not added:
                break

        if len(current_block) == 1:
            merged_text.append(current_block[0])
        else:
            # 融合多个文本行的坐标框
            current_block.sort(key=lambda x: x["bbox"][1])
            bx1 = min(b["bbox"][0] for b in current_block)
            by1 = min(b["bbox"][1] for b in current_block)
            bx2 = max(b["bbox"][2] for b in current_block)
            by2 = max(b["bbox"][3] for b in current_block)

            combined_text = "\n".join(b["text"] for b in current_block)
            avg_conf = sum(b["confidence"] for b in current_block) / len(current_block)

            merged_text.append({
                "text": combined_text,
                "bbox": [bx1, by1, bx2, by2],
                "box": [[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]],
                "angle": 0.0,
                "confidence": round(avg_conf, 3),
                "width": int(bx2 - bx1),
                "height": int(by2 - by1),
                "is_structured": True
            })

    final_items = merged_text + rotated_items
    if img_bgr is not None:
        final_items = merged_text + rotated_items + merged_cell
    print(f"  [智能合并] 原始OCR块数量: {len(items)} -> 合并后结构化块数量: {len(final_items)}")

    return final_items


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
    """保持段落原有换行标志的前提下，对超长行单行切分折行。

    v2.0: 增加字符级切分——当单个词（无空格）超过max_width时，按字符强制折行，
          防止长代号/数字串永不折行导致宽度溢出。
    """
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
        # 处理最后一个词（可能超长）
        _append_with_char_break(draw, current, font, max_width, wrapped_lines)
    return wrapped_lines


def _append_with_char_break(draw, text, font, max_width, out_lines):
    """将文本追加到 out_lines，若超过max_width则按字符切分。"""
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        out_lines.append(text)
        return
    # 字符级切分
    current_chars = ""
    for ch in text:
        candidate = current_chars + ch
        cb = draw.textbbox((0, 0), candidate, font=font)
        if cb[2] - cb[0] <= max_width:
            current_chars = candidate
        else:
            if current_chars:
                out_lines.append(current_chars)
            current_chars = ch
    if current_chars:
        out_lines.append(current_chars)


def _fit_text_to_box(draw, text, font_path, box_w, box_h, max_font_size, max_width=None):
    """计算英文缩放比例，给予一定的横向延展缓冲防止过紧挤压。

    v2.0 修复: 回退时返回真实文本高度（而非伪造box_h），并增加 overflow 标志。

    参数:
        box_w: 原文 bbox 宽度（用于估算默认允许宽度 box_w*1.15）
        box_h: 原文 bbox 高度
        max_width: 允许绘制的硬上限（如受右页边约束）。若给出，则取
                   min(box_w*1.15, max_width)，确保文字绝不越过该边界。

    返回: (font, lines, spacing, total_h, overflow)
          overflow=True 表示即使最小字号也放不下（调用方可据此缩小容器或裁剪）
    """
    min_font_size = 5
    max_font_size = max(min_font_size, max_font_size)
    # CAD 图纸横向往往有空白，英文字符自然膨胀，允许横向适当延展15%减小字号缩减压力
    allowed_w = max(1, int(box_w * 1.15))
    if max_width is not None:
        allowed_w = max(1, min(allowed_w, int(max_width)))

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
            return font, lines, spacing, total_h, False

    # v2.0: 兜底——计算真实高度而非伪造 box_h
    font = _load_font(font_path, min_font_size)
    spacing = max(1, int(min_font_size * 0.15))
    lines = _wrap_structured_text(draw, text, font, allowed_w)
    real_h = 0
    for line in lines:
        sample = line if line else " "
        bbox = draw.textbbox((0, 0), sample, font=font)
        real_h += (bbox[3] - bbox[1])
    if len(lines) > 1:
        real_h += spacing * (len(lines) - 1)
    logger.debug(f"  [文本溢出] min_font={min_font_size}px 仍无法适配: "
                 f"box={box_w}x{box_h}px allowed_w={allowed_w}px real_h={real_h:.0f}px "
                 f"text='{text[:40]}{'...' if len(text)>40 else ''}'")
    return font, lines, spacing, real_h, True  # overflow=True


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


def _detect_text_alignment(original_bgr, bbox, draw, original_text, font_path):
    """检测原文在bbox内的水平对齐方式（左/中/右）。

    通过比较原文本bbox与容器bbox的左右边距来判断。
    返回: "left" | "center" | "right"
    """
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    if box_w <= 0:
        return "left"

    # 使用一个中等字号估算原文本宽度
    test_font_size = max(8, int((y2 - y1) * 0.7))
    try:
        font = _load_font(font_path, test_font_size)
    except Exception:
        return "left"

    # 用翻译后文本近似估算宽度
    text_lines = original_text.split("\n")
    if not text_lines:
        return "left"
    max_text_w = 0
    for line in text_lines:
        b = draw.textbbox((0, 0), line if line else " ", font=font)
        max_text_w = max(max_text_w, b[2] - b[0])

    if max_text_w >= box_w * 0.9:
        return "left"  # 文本已填满，对齐无意义

    # 估算原文文本起始位置：检查原图中文本区域左侧是否有空白
    # 简单策略：检查bbox左边缘一小段竖条的亮度分布
    h, w = original_bgr.shape[:2]
    margin_check_w = max(3, int(box_w * 0.08))
    lx1, lx2 = max(0, x1), min(w - 1, x1 + margin_check_w)
    rx1, rx2 = max(0, x2 - margin_check_w), min(w - 1, x2)
    ly1, ly2 = max(0, y1), min(h - 1, y2)

    if lx2 > lx1 and ly2 > ly1:
        left_region = original_bgr[ly1:ly2, lx1:lx2]
        left_dark_ratio = float(np.mean(left_region < 128)) if left_region.size > 0 else 0
    else:
        left_dark_ratio = 0

    if rx2 > rx1 and ly2 > ly1:
        right_region = original_bgr[ly1:ly2, rx1:rx2]
        right_dark_ratio = float(np.mean(right_region < 128)) if right_region.size > 0 else 0
    else:
        right_dark_ratio = 0

    # 两边都有暗像素 → 左对齐（右边可能是网格线）；仅右边有 → 右对齐；都不多 → 居中
    if right_dark_ratio > 0.05 and left_dark_ratio < 0.02:
        return "right"
    if abs(left_dark_ratio - right_dark_ratio) < 0.03:
        return "center"
    return "left"


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
    """旋转文本单独图层渲染（主要用于单行倾斜/垂直标注）。

    关键：英文沿 OCR 4点 box 的【真实基线（长边）】排版，而不是按 AABB
    的 box_w×box_h 排版。否则近垂直标注的 AABB 窄列会把英文折成竖块，
    整体旋转后变成垂直于标注箭头（问题3）。
    做法：取 box[0]->box[1] 基线向量求角度 θ 与长度 L（可用宽度），
    取与基线垂直方向的短边 H（行高）；在一张 L×H 的水平画布上单行排版，
    再整体绕中心旋转后按原 box 中心贴回。
    注意：PIL rotate(θ) 的正方向是数学逆时针，与图像坐标 atan2(dy,dx)
    （y 轴向下）方向相反；故此处用 rotate(-θ) 才能让文字落到原标注方向。

    v2.0: 渲染后检查溢出，若文本层超出原bbox则等比缩小到fit。
    """
    box = item.get("box")
    if not box or len(box) < 4:
        # 兜底：退化到 AABB
        x1, y1, x2, y2 = item["bbox"]
        box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    (bx0, by0), (bx1, by1), (bx2, by2), (bx3, by3) = box[:4]
    # 基线向量 box[0]->box[1]
    dx, dy = bx1 - bx0, by1 - by0
    angle = float(np.degrees(np.arctan2(dy, dx)))
    base_len = float(np.hypot(dx, dy))
    # 高度方向取 box[0]->box[3]
    side_len = float(np.hypot(bx3 - bx0, by3 - by0))
    # 长边=可用宽度，短边=行高；近垂直标注时长边在垂直方向，这里自动取 max/min
    L = max(base_len, side_len)
    H = min(base_len, side_len)
    L = max(1, int(round(L)))
    H = max(1, int(round(H)))

    translated = item.get("translated", item["text"])

    # v2.0: 尝试多次缩小直到文本fit
    for attempt in range(3):
        pad = max(4, int(H * 0.12))
        local_w, local_h = L + pad * 2, H + pad * 2
        local = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(local)

        max_font_size = max(8, int(H * 0.82))
        font, lines, spacing, text_h, overflow = _fit_text_to_box(d, translated, font_path, L, H, max_font_size)

        ty = pad + max(0, int((H - text_h) / 2))
        y_cursor = ty
        for line in lines:
            d.text((pad, y_cursor), line, fill=text_color + (255,), font=font)
            bbox = d.textbbox((0, 0), line if line else " ", font=font)
            y_cursor += (bbox[3] - bbox[1]) + spacing

        rotated = local.rotate(-angle, expand=True, resample=Image.BICUBIC)
        # 原文本中心（AABB 中心，与历史行为一致）
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = int(cx - rotated.size[0] / 2)
        top = int(cy - rotated.size[1] / 2)

        # v2.0: 溢出检查——如果旋转后图层超标，缩小H和L重试
        bbox_w, bbox_h = x2 - x1, y2 - y1
        if rotated.size[0] <= bbox_w * 1.3 and rotated.size[1] <= bbox_h * 1.3:
            return rotated, left, top
        L = max(1, int(L * 0.85))
        H = max(1, int(H * 0.85))

    return rotated, left, top


def inpaint_and_overlay(img_path, translated_items, output_img_path):
    """
    智能文本擦除与高保真对齐回填引擎

    v2.0 优化:
      - 单元格擦除2px内缩保护格线
      - 格内全部文本（含未翻译）统一擦除，用原图底色
      - 单元格多行按原文比例分配行高，硬裁剪防溢出
      - 检测原文对齐方式并复刻
      - success 计数后置
    """
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    original_bgr = img_bgr.copy()
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines, hmask, vmask = _detect_table_lines(img_gray)

    # 收集所有确实被翻译的文本项
    translated_set = set()
    for item in translated_items:
        if item.get("translated", item["text"]) != item["text"]:
            translated_set.add(id(item))

    # === 擦除阶段 ===
    # 策略：
    #  - 表格单元格内：擦除【整个单元格内部】（框线内侧留 CELL_ERASE_INSET px），
    #    白底填充。擦除格内所有文本（含未翻译的），但保留表格框线。
    #  - 非表格普通文本：直接白色矩形填充 bbox，干净彻底、不留中文残影。
    cell_cnt = fill_cnt = 0
    cells_erased = set()  # 避免同一格重复擦除
    cells_not_erased = {}  # 跟踪为何某些格未被擦除

    logger.info("=" * 50)
    logger.info("[擦除阶段] 开始扫描...")

    for item_idx, item in enumerate(translated_items):
        x1, y1, x2, y2 = item["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        # 优先使用预建的 cell，否则回退到 _find_table_cell
        cell = item.get("cell")
        if cell is None and item.get("in_table", False):
            cell = _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray)
            if cell:
                logger.debug(f"  [擦除] OCR#{item_idx} 回退_find_table_cell成功 → {_cell_id(cell)}")
            else:
                logger.debug(f"  [擦除] OCR#{item_idx} in_table=True但_find_table_cell返回None，bbox=({x1},{y1})-({x2},{y2})")

        if cell is not None:
            if cell in cells_erased:
                continue  # 已擦除
            cl, ct, cr, cb = cell
            cid = _cell_id(cell)
            inset = CELL_ERASE_INSET
            ecl = max(0, min(cl + inset, w - 1))
            ecr = max(0, min(cr - inset, w - 1))
            ect = max(0, min(ct + inset, h - 1))
            ecb = max(0, min(cb - inset, h - 1))
            if ecr > ecl and ecb > ect:
                if ect < ecb and ecl < ecr:
                    cell_region = original_bgr[ect:ecb, ecl:ecr]
                    fill_color = (255, 255, 255) if float(np.mean(cell_region)) > 200 else (240, 240, 240)
                else:
                    fill_color = (255, 255, 255)
                cv2.rectangle(img_bgr, (ecl, ect), (ecr, ecb), fill_color, -1)
                cells_erased.add(cell)
                cell_cnt += 1
                logger.info(f"  [擦除] {cid}: 格内({ecl},{ect})-({ecr},{ecb}) {ecr-ecl}x{ecb-ect}px 填充色={fill_color}")
            else:
                logger.warning(f"  [擦除跳过] {cid}: 内缩后尺寸无效 ecl={ecl} ecr={ecr} ect={ect} ecb={ecb}")
            continue

        # 非单元格项：擦除时向外扩展3px，覆盖OCR漏识别的残余笔画
        if id(item) in translated_set:
            ex1, ey1 = max(0, x1 - 3), max(0, y1 - 3)
            ex2, ey2 = min(w - 1, x2 + 3), min(h - 1, y2 + 3)
            cv2.rectangle(img_bgr, (ex1, ey1), (ex2, ey2), (255, 255, 255), -1)
            fill_cnt += 1
            logger.debug(f"  [擦除-普通] OCR#{item_idx} bbox({x1},{y1})-({x2},{y2}) → 扩展({ex1},{ey1})-({ex2},{ey2})")

    # 报告：哪些已注册的格完全没有被擦除（可能不含OCR文本，或cell未正确传递）
    all_registered = set(_cell_registry.keys())
    untouched = all_registered - cells_erased
    if untouched:
        for cell_key in untouched:
            cid = _cell_registry[cell_key]
            logger.warning(f"  [擦除遗漏] {cid}: 注册但未被擦除！坐标({cell_key[0]},{cell_key[1]})-({cell_key[2]},{cell_key[3]})")
    logger.info(f"  [擦除汇总] 单元格擦除: {cell_cnt}/{len(all_registered)} 格 | 普通填充: {fill_cnt} 块 | 遗漏: {len(untouched)} 格")

    # === v2.2: 合并嵌套/重叠单元格（修复同格多检，如Cell_003+004） ===
    cell_remap = _merge_nested_cells()

    # v2.2: 更新所有item.cell引用（被合并的旧格→新keeper格）
    if cell_remap:
        for item in translated_items:
            old_cell = item.get("cell")
            if old_cell is not None and old_cell in cell_remap:
                item["cell"] = cell_remap[old_cell]
                logger.debug(f"  [cell更新] item '{item['text'][:30]}' cell {_cell_registry.get(old_cell, '?')} → {_cell_registry.get(cell_remap[old_cell], '?')}")

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    font_path = FONT_PATH if os.path.exists(FONT_PATH) else None

    # === v2.2: 先构建 cell→items 映射，再按格分组渲染（解决同格多OCR重叠） ===
    cell_items_map = {}  # cell_key -> [item, ...]
    non_cell_items = []
    success = skip = 0

    for item in translated_items:
        translated = item.get("translated", item["text"])
        original = item["text"]
        if translated == original:
            skip += 1
            continue

        x1, y1, x2, y2 = item["bbox"]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            skip += 1
            continue

        angle = float(item.get("angle", 0.0))
        if abs(angle) < ANGLE_NEAR_HORIZONTAL and bool(item.get("in_table", False)):
            cell = item.get("cell")
            if cell is None:
                cell = _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray)
                item["cell"] = cell
                if cell is not None:
                    _register_text_in_cell(cell, -1, original, translated)
            if cell is not None:
                cell_items_map.setdefault(cell, []).append(item)
                continue
        non_cell_items.append(item)

    # 渲染非单元格项（原有逻辑）
    for item in non_cell_items:
        _render_single_item(item, draw, pil_img, original_bgr, font_path, w, h, success, skip)

    # 渲染单元格项：按格分组，多行统一渲染
    for cell_key, items in cell_items_map.items():
        _render_cell_group(cell_key, items, draw, pil_img, original_bgr, font_path)
        success += 1


    pil_img.convert("RGB").save(output_img_path)
    print(f"  版面回填完成: 成功渲染 {success} 块, 忽略/跳过 {skip} 块")
    return translated_items


def _render_single_item(item, draw, pil_img, original_bgr, font_path, w, h, success, skip):
    """渲染单个非单元格项（结构化文本、旋转文本等）。"""
    bbox = item["bbox"]
    translated = item.get("translated", item["text"])
    original = item["text"]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return success, skip + 1

    text_color = _sample_text_color(original_bgr, x1, y1, x2, y2)
    angle = float(item.get("angle", 0.0))
    max_width_edge = max(1, (w - RIGHT_MARGIN) - (x1 + 2))

    if abs(angle) < ANGLE_NEAR_HORIZONTAL:
        tx = x1 + 2
        if item.get("is_structured", False):
            src_lines = [ln for ln in translated.split("\n")]
            n = max(1, len(src_lines))
            row_h = box_h / n
            y_cursor = y1 + 1
            min_fs = 5
            max_fs = max(8, int(row_h * 0.85))
            common_font = None
            common_wrap = []
            for fs in range(max_fs, min_fs - 1, -1):
                try:
                    test_font = _load_font(font_path, fs)
                except Exception:
                    continue
                ok = True
                wraps = []
                for ln in src_lines:
                    wlines = _wrap_structured_text(draw, ln, test_font, max_width_edge)
                    max_lw = 0
                    for wl in wlines:
                        wb = draw.textbbox((0, 0), wl if wl else " ", font=test_font)
                        max_lw = max(max_lw, wb[2] - wb[0])
                    if max_lw > max_width_edge:
                        ok = False
                        break
                    spacing = max(1, int(fs * 0.15))
                    lh_total = sum((draw.textbbox((0, 0), wl if wl else " ", font=test_font)[3] -
                                    draw.textbbox((0, 0), wl if wl else " ", font=test_font)[1] + spacing)
                                   for wl in wlines)
                    if lh_total > row_h:
                        ok = False
                        break
                    wraps.append((test_font, wlines, spacing, lh_total))
                if ok:
                    common_font = test_font
                    common_wrap = wraps
                    break
            if common_wrap:
                for (fnt, flines, fsp, fth) in common_wrap:
                    ly = y_cursor + max(0, int((row_h - fth) / 2))
                    for sub in flines:
                        draw.text((tx, ly), sub, fill=text_color + (255,), font=fnt)
                        sb = draw.textbbox((0, 0), sub if sub else " ", font=fnt)
                        ly += (sb[3] - sb[1]) + fsp
                    y_cursor += row_h
                return success + 1, skip
            else:
                for ln in src_lines:
                    fnt, flines, fsp, fth, _ = _fit_text_to_box(
                        draw, ln, font_path, box_w, row_h, max(8, int(row_h * 0.85)), max_width=max_width_edge)
                    ly = y_cursor + max(0, int((row_h - fth) / 2))
                    for sub in flines:
                        draw.text((tx, ly), sub, fill=text_color + (255,), font=fnt)
                        sb = draw.textbbox((0, 0), sub if sub else " ", font=fnt)
                        ly += (sb[3] - sb[1]) + fsp
                    y_cursor += row_h
                return success + 1, skip
        else:
            max_font_size = max(8, int(box_h * 0.85))
            font, lines, spacing, text_h, _ = _fit_text_to_box(
                draw, translated, font_path, box_w, box_h, max_font_size, max_width=max_width_edge)
            ty = y1 + max(0, int((box_h - text_h) / 2)) if len(lines) == 1 else y1 + 2
            y_cursor = ty
            for line in lines:
                draw.text((tx, y_cursor), line, fill=text_color + (255,), font=font)
                l_bbox = draw.textbbox((0, 0), line if line else " ", font=font)
                y_cursor += (l_bbox[3] - l_bbox[1]) + spacing
            return success + 1, skip
    else:
        try:
            layer, left, top = _render_text_item_layer(item, font_path, text_color)
            _alpha_composite_at(pil_img, layer, left, top)
            return success + 1, skip
        except Exception:
            return success, skip + 1


def _render_cell_group(cell_key, items, draw, pil_img, original_bgr, font_path):
    """按单元格分组渲染：将一个格内的所有OCR项作为多行文本统一渲染到临时图层。"""
    cl, ct, cr, cb = cell_key
    cid = _cell_id(cell_key)
    inner_cl = cl + CELL_ERASE_INSET
    inner_ct = ct + CELL_ERASE_INSET
    inner_cr = cr - CELL_ERASE_INSET
    inner_cb = cb - CELL_ERASE_INSET
    layer_w = max(1, inner_cr - inner_cl)
    layer_h = max(1, inner_cb - inner_ct)

    # 收集所有文本行（按纵坐标排序，每个item=一行）
    items_sorted = sorted(items, key=lambda it: it["bbox"][1])
    src_lines = []
    orig_heights = []
    for it in items_sorted:
        src_lines.append(it.get("translated", it["text"]))
        orig_heights.append(it["bbox"][3] - it["bbox"][1])

    n = len(src_lines)
    total_orig_h = sum(orig_heights)
    if total_orig_h > 0:
        row_heights = [max(1, layer_h * oh / total_orig_h) for oh in orig_heights]
    else:
        row_heights = [layer_h / n] * n

    # 取第一个item的X起始位置
    tx = items_sorted[0]["bbox"][0] + 2
    local_tx = max(0, tx - inner_cl)
    local_draw_w = max(1, layer_w - local_tx)

    # 检测对齐方式（用第一个item）
    alignment = _detect_text_alignment(original_bgr, items_sorted[0]["bbox"], draw, items_sorted[0]["text"], font_path)

    logger.info(f"  [渲染组] {cid}: {n}个OCR项, 格内={layer_w}x{layer_h}px, 对齐={alignment}, "
                f"行高={[round(rh,1) for rh in row_heights]}px")

    # 统一字号搜索
    row_fonts = []
    unified_fs = None
    min_fs, max_fs = 5, max(8, int(min(row_heights) * 0.85))
    for fs in range(int(max_fs), min_fs - 1, -1):
        try_font = _load_font(font_path, fs)
        all_fit = True
        temp_wraps = []
        for k, ln in enumerate(src_lines):
            rh = row_heights[k]
            try_spacing = max(1, int(fs * 0.15))
            try_lines = _wrap_structured_text(draw, ln, try_font, local_draw_w)
            try_h = 0
            for tl in try_lines:
                tb = draw.textbbox((0, 0), tl if tl else " ", font=try_font)
                try_h += (tb[3] - tb[1])
            if len(try_lines) > 1:
                try_h += try_spacing * (len(try_lines) - 1)
            if try_h > rh * 1.05:
                all_fit = False
                logger.debug(f"    {cid} 行{k} 字号{fs} 溢出: {try_h:.1f}>{rh*1.05:.1f} 文本='{ln[:30]}'")
                break
            temp_wraps.append((try_font, try_lines, try_spacing, try_h))
        if all_fit:
            unified_fs = fs
            row_fonts = temp_wraps
            break

    if not row_fonts:
        logger.warning(f"  [渲染组回退] {cid}: 无统一字号, 改用逐行独立字号")
        for k, ln in enumerate(src_lines):
            rh = row_heights[k]
            fnt, flines, fsp, fth, overflow = _fit_text_to_box(
                draw, ln, font_path, local_draw_w, rh, max(8, int(rh * 0.82)), max_width=local_draw_w)
            row_fonts.append((fnt, flines, fsp, fth))
            if overflow:
                logger.warning(f"    {cid} 行{k} 最小字号仍溢出! rh={rh:.0f}px h={fth:.0f}px '{ln[:40]}'")
    else:
        logger.info(f"  [渲染组适配] {cid}: 统一字号={unified_fs}px")

    # 渲染到临时图层
    cell_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(cell_layer)

    y_cursor = 0
    for k, (fnt, flines, fsp, fth) in enumerate(row_fonts):
        rh = row_heights[k]
        ly = y_cursor + max(0, int((rh - fth) / 2))
        it_color = _sample_text_color(original_bgr, *items_sorted[k]["bbox"])
        for sub in flines:
            sub_bbox = layer_draw.textbbox((0, 0), sub if sub else " ", font=fnt)
            sub_w = sub_bbox[2] - sub_bbox[0]
            if alignment == "center":
                lx = local_tx + max(0, int((local_draw_w - sub_w) / 2))
            elif alignment == "right":
                lx = local_tx + max(0, local_draw_w - sub_w)
            else:
                lx = local_tx
            layer_draw.text((lx, ly), sub, fill=it_color + (255,), font=fnt)
            ly += (layer_draw.textbbox((0, 0), sub if sub else " ", font=fnt)[3] -
                   layer_draw.textbbox((0, 0), sub if sub else " ", font=fnt)[1]) + fsp
        y_cursor += rh
        if y_cursor > layer_h + 2:
            logger.warning(f"    {cid} 行{k} 渲染溢出: y={y_cursor:.0f} > layer_h={layer_h}")

    pil_img.alpha_composite(cell_layer, dest=(inner_cl, inner_ct))
    logger.info(f"  [渲染组完成] {cid}: {n}项, 图层{layer_w}x{layer_h}px @({inner_cl},{inner_ct})")


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


def _generate_debug_cell_pdf(img_path: str, debug_pdf_path: str, dpi: int = 200):
    """生成单元格调试PDF：将所有注册的单元格用红框标出，格内填写Cell_XXX编号。

    用于目视对比检查：验证哪些格被检测到、格边界是否正确、是否有遗漏。
    """
    if not _cell_registry:
        logger.warning("  [调试PDF] 无注册单元格，跳过生成")
        return

    from PIL import ImageDraw as IDraw
    pil_img = Image.open(img_path).convert("RGBA")
    draw = IDraw.Draw(pil_img)

    # 尝试加载小号字体（用于格内编号）
    try:
        debug_font = _load_font(FONT_PATH, 12)
    except Exception:
        debug_font = ImageFont.load_default()

    for cell_key, cid in _cell_registry.items():
        cl, ct, cr, cb = cell_key
        cw, ch = cr - cl, cb - ct

        # 红框：格边界（外框）
        draw.rectangle([cl, ct, cr, cb], outline=(255, 0, 0, 255), width=2)
        # 黄框：擦除内缩区域
        inset = CELL_ERASE_INSET
        draw.rectangle([cl + inset, ct + inset, cr - inset, cb - inset],
                       outline=(255, 255, 0, 128), width=1)

        # 格内填写编号
        label = cid  # e.g. "Cell_001"
        # 选字号——格小用小字
        fs = max(6, min(int(ch * 0.45), int(cw / len(label) * 1.6), 24))
        try:
            cell_font = _load_font(FONT_PATH, fs)
        except Exception:
            cell_font = debug_font

        tb = draw.textbbox((0, 0), label, font=cell_font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = cl + max(2, (cw - tw) // 2)
        ty = ct + max(2, (ch - th) // 2)

        # 白底衬底（提高可读性）
        draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], fill=(255, 255, 255, 200))
        draw.text((tx, ty), label, fill=(255, 0, 0, 255), font=cell_font)

    # 也标记未入格的OCR项（灰色虚线框）
    logger.info(f"  [调试PDF] 已绘制 {len(_cell_registry)} 个单元格的红框+编号")

    # 保存为PNG再转PDF
    debug_png = debug_pdf_path.replace(".pdf", ".png")
    pil_img.convert("RGB").save(debug_png)
    image_to_pdf(debug_png, debug_pdf_path, dpi=dpi)
    logger.info(f"  [调试PDF] 已保存: {debug_pdf_path}")


def main():
    _clear_cell_registry()  # 每次运行前重置注册表
    logger.info("=" * 60)
    logger.info("Scan-type PDF CN->EN Translation (v2.0 Cell-First + Diagnostic Logging)")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    print("=" * 60)
    print("Scan-type PDF CN->EN Translation (v2.0 Cell-First with Diagnostics)")
    print(f"  Log: {LOG_FILE}")
    print("=" * 60)
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found")
        logger.error(f"PDF not found: {PDF_PATH}")
        return
    os.makedirs(WORK_DIR, exist_ok=True)

    print("\n[Step 1] Render PDF -> Image...")
    img_path, page_meta = pdf_to_image(PDF_PATH, dpi=RENDER_DPI)

    print("\n[Step 2] RapidOCR (chunked) recognition...")
    raw_ocr_items = ocr_with_rapid_chunked(img_path, chunk_size=CHUNK_SIZE)
    logger.info(f"OCR完成: 识别到 {len(raw_ocr_items)} 个中文文本块")

    print("\n[Step 2.5] Cell-First Intelligent Block Merging...")
    # 传入渲染图像，用于检测表格网格并抑制跨单元格合并（保护表格边框/单元格独立）
    merge_img = cv2.imread(img_path)
    ocr_items = merge_ocr_items(raw_ocr_items, img_bgr=merge_img)
    
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

    # === 生成单元格调试PDF（红框+编号） ===
    debug_pdf = os.path.join(WORK_DIR, "cell_debug.pdf")
    print("\n[Step 5.5] Generate Cell Debug PDF...")
    _generate_debug_cell_pdf(output_img, debug_pdf, dpi=page_meta["dpi"])

    total = len(translated_items)
    translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])

    # === 诊断总结报告 ===
    logger.info("=" * 60)
    logger.info("诊断总结报告")
    logger.info("=" * 60)
    logger.info(f"  总OCR块: {len(raw_ocr_items)}")
    logger.info(f"  合并后块: {total}")
    logger.info(f"  已翻译块: {translated_count}")
    logger.info(f"  注册单元格总数: {_cell_counter[0]}")
    
    # 每个单元格的文本汇总
    for cid in sorted(_cell_texts.keys()):
        texts = _cell_texts[cid]
        logger.info(f"  {cid}: {len(texts)}个OCR项")
        for idx, orig, trans in texts:
            is_translated = "✓" if orig != trans else "✗(未翻译)"
            logger.info(f"    OCR#{idx} {is_translated} 原文='{orig[:50]}' → 译文='{trans[:50]}'")

    # 检查是否有空文件（注册但无文本的单元格）
    empty_cells = [cid for cid in sorted(_cell_texts.keys()) if not _cell_texts[cid]]
    if empty_cells:
        logger.warning(f"  空单元格（注册但无OCR文本）: {empty_cells}")

    logger.info(f"  详细日志已保存至: {LOG_FILE}")
    logger.info("=" * 60)

    # === 生成单元格→OCR映射报告（Markdown表格） ===
    report_path = os.path.join(WORK_DIR, "cell_report.md")
    _generate_cell_report(report_path)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Total Layout blocks: {total}")
    print(f"  Translated blocks: {translated_count}")
    print(f"  Registered cells: {_cell_counter[0]}")
    print(f"  Cell report: {report_path}")
    print(f"  Cell debug PDF: {debug_pdf}")
    print(f"  Diagnostic log: {LOG_FILE}")
    print(f"  Output: {OUTPUT_PDF}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
