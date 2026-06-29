"""
扫描型PDF中文→英文翻译 - RapidOCR + 分块OCR + 智能行合并排版优化版
流程: 扫描PDF→渲染→分块OCR→单元格优先检测→智能文本行合并→隔离标签翻译→白底擦除→绝对左对齐回填→重构PDF

v3.1 优化（按扫描PDF图纸处理规则）:
  - 单元格优先检测：三阶段级联（轮廓矩形→线交叉网格→文本间隙推理）+ 回退
  - 多尺度线段检测（Canny + HoughP）：支持≥30px短线段，替代旧版≥200px长线过滤
  - 线交叉网格交点过滤：仅保留≥3交点线（排除CAD结构线），候选格数从O(H²×V²)→可控
  - CAD线 vs 表格线甄别：基于邻格一致性过滤孤立假格 + 文本邻近过滤
  - 文本间隙推理区域约束：仅在已检测表格密集区运行，避免全图假格
  - 文本邻近后过滤：移除不含OCR文本且远离文本的假格
  - 单元格最小化擦除，2px内缩保护格线
  - 增强CAD线 vs 表格线判别（矩形闭合验证、相邻格一致性）
  - _find_table_cell回退margin缩小至30px（原50px），减少跨格误匹配
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
    CELL_DETECT_ENGINE,
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
CELL_WHITE_THRESHOLD = float(os.environ.get("CELL_WHITE_THRESHOLD", "0.65"))  # 格内白底比例（原0.80→0.65，文本密集格容错）
# CELL_DETECT_ENGINE 已从 config.py 导入，通过 .env 文件配置

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


# ═══════════════════════════════════════════════════════════════
# v3.3 单元格检测核心（参考 识别单元格建议.md：形态学连通域法）
# 策略级联：形态学连通域(首选) → 轮廓矩形检测 → 线交叉网格 → 文本间隙(禁用) → 回退
# ═══════════════════════════════════════════════════════════════

# ---- 新增检测常量 ----
HOUGH_THRESH = int(os.environ.get("HOUGH_THRESH", "20"))        # Hough线检测投票阈值（降低，CAD细线像素少）
HOUGH_MIN_LEN = int(os.environ.get("HOUGH_MIN_LEN", "15"))       # 最小线段长（原30→15，短竖线也能捕获）
HOUGH_MAX_GAP = int(os.environ.get("HOUGH_MAX_GAP", "15"))       # 线段断裂容忍（原8→15，连接断线）
GRID_INTERSECT_DENSITY = int(os.environ.get("GRID_INTERSECT_DENSITY", "2"))  # 表格线至少与N条垂直线相交（原3→2，含边缘线）
CAD_LINE_MAX_INTERSECT = int(os.environ.get("CAD_LINE_MAX_INTERSECT", "1"))  # CAD线最多与N条线相交（原2→1，更激进排除）

# ---- v3.3 形态学连通域检测常量 ----
MORPH_H_KERNEL = int(os.environ.get("MORPH_H_KERNEL", "35"))   # 水平线提取核宽度
MORPH_V_KERNEL = int(os.environ.get("MORPH_V_KERNEL", "35"))   # 竖直线提取核高度
MORPH_DILATE_SIZE = int(os.environ.get("MORPH_DILATE_SIZE", "2"))  # 膨胀迭代次数（修复断线）
MORPH_MIN_CELL_AREA = int(os.environ.get("MORPH_MIN_CELL_AREA", "200"))  # 最小格面积（px²）


def _find_cells_by_morphological_components(img_gray):
    """v3.3 形态学连通域法：纯线框 → 膨胀闭合 → 反色 → 连通域 = 原子单元格。

    参考识别单元格建议.md：对于线条清晰、布局规整的工程图，
    形态学提取线条 + 找白色连通区域是最高效、位置最精准的方法。

    算法：
      1. 双通道二值化取反（线条变白，背景变黑）
      2. 形态学开操作提取横线（宽核）和竖线（高核）
      3. 合并横竖线 → 膨胀闭合断口
      4. 反色（格内变白）= 每个格子是独立白色连通域
      5. cv2.connectedComponentsWithStats → 每个连通域 = 一个原子格
      6. 过滤：面积、长宽比、边界排除

    返回: [(left, top, right, bottom), ...]
    """
    h_img, w_img = img_gray.shape

    # 1. 二值化取反（线条变白=255，背景变黑=0）
    bw_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 6)
    bw_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.bitwise_or(bw_adaptive, bw_otsu)

    # 2. 形态学提取横线和竖线
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_H_KERNEL, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MORPH_V_KERNEL))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # 3. 合并线条 + 膨胀闭合
    table_mask = cv2.add(h_lines, v_lines)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(table_mask, dilate_kernel, iterations=MORPH_DILATE_SIZE)

    # 4. 反色：现在格内是白色(255)，线条是黑色(0)
    inv_mask = cv2.bitwise_not(dilated)

    # 5. 连通域分析：每个白色连通区 = 一个原子格
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv_mask, connectivity=8)

    # 6. 过滤和提取
    cells = []
    img_area = h_img * w_img
    for label_id in range(1, num_labels):  # 跳过 label 0（背景=黑色区域）
        x_np, y_np, w_np, h_np, area_np = stats[label_id]
        x, y, w, h_rect, area = int(x_np), int(y_np), int(w_np), int(h_np), int(area_np)
        # 过滤：面积过小 → 噪点；面积过大 → 整图背景
        if area < MORPH_MIN_CELL_AREA:
            continue
        if area > img_area * 0.85:  # 不要整图背景
            continue
        # 过滤：尺寸范围
        if not (CELL_MIN_W <= w <= CELL_MAX_W and CELL_MIN_H <= h_rect <= CELL_MAX_H):
            continue
        # 过滤：极端长宽比（可能是线条残留）
        if w > 0 and h_rect > 0:
            ratio = max(w, h_rect) / min(w, h_rect)
            if ratio > 40:  # 极细长的不是格子
                continue
        # 过滤：边界上不完整的格子（触及图像边缘2px可能是残缺格）
        if x <= 1 or y <= 1 or x + w >= w_img - 1 or y + h_rect >= h_img - 1:
            # 允许图像边缘的完整格通过，但标记
            pass  # 不排除边缘格，CAD标题栏常在图像边界

        cells.append((x, y, x + w, y + h_rect))

    logger.info(f"  [形态学连通域] 检测到 {len(cells)} 个原子单元格 "
                f"(H核={MORPH_H_KERNEL} V核={MORPH_V_KERNEL} 膨胀={MORPH_DILATE_SIZE})")
    return cells


def _detect_all_line_segments(img_gray):
    """多尺度线检测：Canny边缘 + 概率Hough变换 → 全尺度线段（≥30px）。

    替代原来的单一形态学长线检测（≥200px），现可检测任意长度线段。
    返回: (h_segs, v_segs) 水平/竖直线段 [(x1,y1,x2,y2), ...]
    """
    h_img, w_img = img_gray.shape

    # 1. 双通道二值化：自适应（局部对比度）+ OTSU（全局阈值），互补捕捉细线
    bw_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 21, 6)  # 块21→更敏感
    bw_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 合并两种二值化结果（取OR，保留所有可能的线段像素）
    bw = cv2.bitwise_or(bw_adaptive, bw_otsu)

    # 2. Canny边缘检测（更低阈值捕捉CAD细线，1-2px边缘能量低）
    edges = cv2.Canny(bw, 20, 80, apertureSize=3)

    # 3. 概率Hough线段检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESH,
                             minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)

    h_segs, v_segs = [], []
    if lines is None:
        logger.info(f"  [线段检测] HoughP → 0条线段")
        return h_segs, v_segs

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        angle = float(np.degrees(np.arctan2(dy, dx))) if dx > 0 else 90.0

        if angle < 15:  # 近水平（±15°）
            h_segs.append(tuple(map(int, (x1, y1, x2, y2))))
        elif angle > 75:  # 近竖直（±15°）
            v_segs.append(tuple(map(int, (x1, y1, x2, y2))))
        # 中间角度的线丢弃（CAD图纸斜线非格网）

    logger.info(f"  [线段检测] HoughP → {len(h_segs)} 水平段 + {len(v_segs)} 竖直线段 "
                f"(minLen={HOUGH_MIN_LEN}px)")
    return h_segs, v_segs


def _detect_rectangular_contours(img_gray):
    """轮廓检测：找二值图中闭合矩形轮廓 → 直接单元格候选。

    CAD图纸中完整的表格单元格是封闭矩形，轮廓检测可精确捕获。
    返回: [(left, top, right, bottom), ...] 闭合矩形候选格列表
    """
    h_img, w_img = img_gray.shape

    # 自适应二值化
    bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 8)

    # 形态学闭运算：连接小断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 查找外轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        # 尺寸过滤
        if not (CELL_MIN_W <= w_rect <= CELL_MAX_W and CELL_MIN_H <= h_rect <= CELL_MAX_H):
            continue

        # 矩形度验证：轮廓面积 vs 外接矩形面积
        area = cv2.contourArea(cnt)
        rect_area = w_rect * h_rect
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < 0.55:  # v3.2: 原0.75→0.55，细线框轮廓面积小
            continue

        # 白底验证
        roi = img_gray[y + 2:y + h_rect - 2, x + 2:x + w_rect - 2] if h_rect > 4 and w_rect > 4 else None
        if roi is None or roi.size == 0:
            continue
        if float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
            continue

        cells.append((x, y, x + w_rect, y + h_rect))

    logger.info(f"  [轮廓检测] 找到 {len(cells)} 个闭合矩形候选格")
    return cells


def _build_line_intersection_grid(h_segs, v_segs, img_gray):
    """线交叉网格：从水平/竖直线段集合构建候选单元格。

    v3.1: 激进预过滤 — 只保留含≥3个垂直线交点的线（排除CAD结构线）。
    算法：
      1. 计算每条线的交点数量（交点=附近的垂直线段）
      2. 过滤掉<3交点的线（CAD结构线特征）
      3. 用剩余线生成候选格

    返回: [(left, top, right, bottom), ...]
    """
    if len(h_segs) < 2 or len(v_segs) < 2:
        return []

    h_img, w_img = img_gray.shape
    INTERSECT_WINDOW = 100  # 交点搜索窗口（px）
    MIN_INTERSECTIONS = GRID_INTERSECT_DENSITY  # 用常量，默认2（原硬编码3）

    # 归一化线段为坐标
    def _normalize(segs, is_horizontal, gap=5):
        coords = {}
        for (x1, y1, x2, y2) in segs:
            key = (y1 + y2) // 2 if is_horizontal else (x1 + x2) // 2
            rng = (min(x1, x2), max(x1, x2)) if is_horizontal else (min(y1, y2), max(y1, y2))
            if key not in coords:
                coords[key] = []
            coords[key].append(rng)
        merged = {}
        keys = sorted(coords.keys())
        for k in keys:
            merged_k = k
            for mk in sorted(merged.keys()):
                if abs(k - mk) <= gap:
                    merged_k = mk
                    break
            if merged_k not in merged:
                merged[merged_k] = []
            merged[merged_k].extend(coords[k])
        return {mk: merged[mk] for mk in sorted(merged.keys())}

    h_merged = _normalize(h_segs, is_horizontal=True, gap=5)
    v_merged = _normalize(v_segs, is_horizontal=False, gap=5)

    # 计算每条横线与竖线的交点数量
    h_intersect_counts = {}
    for y in h_merged:
        cnt = 0
        for x, ranges in v_merged.items():
            for (r1, r2) in ranges:
                if r1 - INTERSECT_WINDOW <= y <= r2 + INTERSECT_WINDOW:
                    cnt += 1
                    break
        h_intersect_counts[y] = cnt

    v_intersect_counts = {}
    for x in v_merged:
        cnt = 0
        for y, ranges in h_merged.items():
            for (r1, r2) in ranges:
                if r1 - INTERSECT_WINDOW <= x <= r2 + INTERSECT_WINDOW:
                    cnt += 1
                    break
        v_intersect_counts[x] = cnt

    # 过滤：只保留≥MIN_INTERSECTIONS交点的"表格线"（边缘线也可能只有2个交点，降低门槛）
    h_filtered = [y for y, c in h_intersect_counts.items() if c >= MIN_INTERSECTIONS]
    v_filtered = [x for x, c in v_intersect_counts.items() if c >= MIN_INTERSECTIONS]

    logger.info(f"  [线交叉网格] 交点过滤: {len(h_merged)}→{len(h_filtered)}条水平线, "
                f"{len(v_merged)}→{len(v_filtered)}条竖直线 (需≥{MIN_INTERSECTIONS}交点)")

    if len(h_filtered) < 2 or len(v_filtered) < 2:
        return []

    # v3.2: 四边封闭验证辅助函数
    # 检查在y坐标附近是否存在水平线段覆盖[x1, x2]（容差=tolerance）
    def _has_h_segment_near(y, x1, x2, tolerance=6):
        for seg_y, ranges in h_merged.items():
            if abs(seg_y - y) <= tolerance:
                for (r1, r2) in ranges:
                    if r1 <= x1 + tolerance and r2 >= x2 - tolerance:
                        return True
        # 放宽到原始线段级别再试
        tol2 = tolerance + 4
        for seg_y, ranges in h_merged.items():
            if abs(seg_y - y) <= tol2:
                for (r1, r2) in ranges:
                    if r1 <= x1 + tol2 and r2 >= x2 - tol2:
                        return True
        return False

    def _has_v_segment_near(x, y1, y2, tolerance=6):
        for seg_x, ranges in v_merged.items():
            if abs(seg_x - x) <= tolerance:
                for (r1, r2) in ranges:
                    if r1 <= y1 + tolerance and r2 >= y2 - tolerance:
                        return True
        tol2 = tolerance + 4
        for seg_x, ranges in v_merged.items():
            if abs(seg_x - x) <= tol2:
                for (r1, r2) in ranges:
                    if r1 <= y1 + tol2 and r2 >= y2 - tol2:
                        return True
        return False

    # 生成候选格（仅用过滤后的线）+ 四边封闭验证
    cells = []
    for i in range(len(h_filtered) - 1):
        for j in range(len(v_filtered) - 1):
            top, bot = h_filtered[i], h_filtered[i + 1]
            lft, rgt = v_filtered[j], v_filtered[j + 1]
            cw, ch = rgt - lft, bot - top
            if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
                continue

            # v3.2: 四边封闭验证（替代旧版仅检查3个角点）
            top_ok = _has_h_segment_near(top, lft, rgt, tolerance=5)
            bot_ok = _has_h_segment_near(bot, lft, rgt, tolerance=5)
            lft_ok = _has_v_segment_near(lft, top, bot, tolerance=5)
            rgt_ok = _has_v_segment_near(rgt, top, bot, tolerance=5)
            sides_ok = sum([top_ok, bot_ok, lft_ok, rgt_ok])

            # 至少三边有线段证据（有一条边可能是虚线或轻微断线）
            if sides_ok < 3:
                continue

            # 对于仅有3条边的，放宽角点检查
            if sides_ok == 3:
                check_margin = 10  # 放宽角点检查窗口
            else:
                check_margin = 6

            # 角点验证：至少2个角点有暗像素（确认真实闭合）
            corners_ok = 0
            corners = [(top, lft), (top, rgt), (bot, lft), (bot, rgt)]
            for cy_c, cx_c in corners:
                y1p = max(0, int(cy_c) - check_margin)
                y2p = min(h_img, int(cy_c) + check_margin)
                x1p = max(0, int(cx_c) - check_margin)
                x2p = min(w_img, int(cx_c) + check_margin)
                if y2p > y1p and x2p > x1p:
                    patch = img_gray[y1p:y2p, x1p:x2p]
                    if patch.size > 0 and np.any(patch < 128):
                        corners_ok += 1

            if corners_ok < 2:
                continue

            # 白底验证
            roi = img_gray[top + 2:bot - 2, lft + 2:rgt - 2] if bot - top > 4 and rgt - lft > 4 else None
            if roi is None or roi.size == 0:
                continue
            # v3.2: 降低白底阈值，文本密集的格白底比例可能不高
            if float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD * 0.85:
                continue

            cells.append((lft, top, rgt, bot))

    logger.info(f"  [线交叉网格] 生成 {len(cells)} 个候选格（含四边封闭验证）")
    return cells


def _classify_grid_region(cells, img_gray):
    """表格区域甄别：过滤可能是CAD结构线形成的假格。

    判定逻辑:
      1. 真正的表格通常有多个相邻单元格（形成网格）
      2. 相邻格尺寸高度一致（同一行）或宽度一致（同一列）
      3. 孤立格（无同行/同列邻居）且尺寸不规则的 → CAD假格

    返回: 过滤后的格列表
    """
    if len(cells) <= 1:
        return cells

    validated = []
    for i, (cl, ct, cr, cb) in enumerate(cells):
        cw, ch = cr - cl, cb - ct
        has_row_neighbor = False
        has_col_neighbor = False
        row_size_match = 0
        col_size_match = 0

        for j, (nl, nt, nr, nb) in enumerate(cells):
            if i == j:
                continue
            nw, nh = nr - nl, nb - nt

            # 同行邻居：顶部对齐±8px 且水平相邻
            if abs(ct - nt) <= 8 and abs(cb - nb) <= 8:
                if abs(cr - nl) <= 10 or abs(cl - nr) <= 10:
                    has_row_neighbor = True
                    if abs(ch - nh) <= 6:
                        row_size_match += 1

            # 同列邻居：左侧对齐±8px 且垂直相邻
            if abs(cl - nl) <= 8 and abs(cr - nr) <= 8:
                if abs(cb - nt) <= 10 or abs(ct - nb) <= 10:
                    has_col_neighbor = True
                    if abs(cw - nw) <= 6:
                        col_size_match += 1

        # 判定：有行/列邻居 + 尺寸一致的 → 真实表格格
        if (has_row_neighbor and row_size_match >= 1) or (has_col_neighbor and col_size_match >= 1):
            validated.append((cl, ct, cr, cb))
        elif has_row_neighbor or has_col_neighbor:
            validated.append((cl, ct, cr, cb))
        else:
            logger.debug(f"  [CAD过滤] 排除孤立假格 ({cl},{ct})-({cr},{cb}) {cw}×{ch}px")

    removed = len(cells) - len(validated)
    if removed > 0:
        logger.info(f"  [CAD过滤] 排除 {removed} 个孤立假格")
    return validated


def _find_cells_by_text_gaps(ocr_items, img_gray, existing_cells=None):
    """v3.1 文本间隙推理：仅在已有表格格附近区域使用OCR文本反推单元格边界。

    限制：只处理被>=2个已有单元格覆盖的"表格密集区"，
          避免在全图非表格区域制造假格。

    返回: [(left, top, right, bottom), ...]
    """
    if len(ocr_items) < 4:  # 至少4个OCR项才可能存在表格
        return []

    h_img, w_img = img_gray.shape

    # 如果有已有格，仅在其邻域内搜索
    if existing_cells and len(existing_cells) >= 3:
        # 计算表格密集区 = 已检测格的包围盒
        all_cl = min(c[0] for c in existing_cells)
        all_ct = min(c[1] for c in existing_cells)
        all_cr = max(c[2] for c in existing_cells)
        all_cb = max(c[3] for c in existing_cells)
        # 扩展20%缓冲
        bw = int((all_cr - all_cl) * 0.2)
        bh = int((all_cb - all_ct) * 0.2)
        region = (max(0, all_cl - bw), max(0, all_ct - bh),
                  min(w_img, all_cr + bw), min(h_img, all_cb + bh))
        logger.info(f"  [文本间隙] 限定区域: ({region[0]},{region[1]})-({region[2]},{region[3]})")
    else:
        # 没有已有格 → 很可能没有表格 → 不运行
        return []

    # 收集区域内的OCR坐标
    rl, rt, rr, rb = region
    lefts, rights, tops, bots = [], [], [], []
    for item in ocr_items:
        x1, y1, x2, y2 = item["bbox"]
        if x1 >= rl - 10 and x2 <= rr + 10 and y1 >= rt - 10 and y2 <= rb + 10:
            lefts.append(x1)
            rights.append(x2)
            tops.append(y1)
            bots.append(y2)

    if len(lefts) < 4:
        return []

    def _cluster_coords(coords, tolerance=20):
        if not coords:
            return []
        sorted_c = sorted(coords)
        clusters = []
        current = [sorted_c[0]]
        for c in sorted_c[1:]:
            if c - current[-1] <= tolerance:
                current.append(c)
            else:
                clusters.append(int(np.median(current)))
                current = [c]
        clusters.append(int(np.median(current)))
        return clusters

    left_clusters = _cluster_coords(lefts, tolerance=25)
    right_clusters = _cluster_coords(rights, tolerance=25)
    top_clusters = _cluster_coords(tops, tolerance=20)
    bot_clusters = _cluster_coords(bots, tolerance=20)

    if len(left_clusters) < 2 or len(top_clusters) < 2:
        return []

    col_edges = sorted(set(left_clusters + right_clusters))
    row_edges = sorted(set(top_clusters + bot_clusters))

    cells = []
    if len(col_edges) >= 2 and len(row_edges) >= 2:
        for yi in range(len(row_edges) - 1):
            for xi in range(len(col_edges) - 1):
                cl2, cr2 = col_edges[xi], col_edges[xi + 1]
                ct2, cb2 = row_edges[yi], row_edges[yi + 1]
                cw, ch = cr2 - cl2, cb2 - ct2
                if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
                    continue
                # 区域边界检查
                if cl2 < rl - 10 or cr2 > rr + 10 or ct2 < rt - 10 or cb2 > rb + 10:
                    continue
                roi = img_gray[ct2 + 2:cb2 - 2, cl2 + 2:cr2 - 2] if ch > 4 and cw > 4 else None
                if roi is None or roi.size == 0:
                    continue
                if float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
                    continue
                cells.append((cl2, ct2, cr2, cb2))

    logger.info(f"  [文本间隙] 区域内聚类:{len(left_clusters)}L/{len(top_clusters)}T → {len(cells)}个候选格")
    return cells


def _remove_contained_cells(cells):
    """移除被其他格完全包含的嵌套格。"""
    if len(cells) <= 1:
        return cells
    keep = []
    for i, (cl, ct, cr, cb) in enumerate(cells):
        contained = False
        for j, (nl, nt, nr, nb) in enumerate(cells):
            if i == j:
                continue
            if nl <= cl and nt <= ct and nr >= cr and nb >= cb:
                if (nr - nl) * (nb - nt) > (cr - cl) * (cb - ct):
                    contained = True
                    logger.debug(f"  [去重] 移除被包含格 ({cl},{ct})-({cr},{cb}) "
                                 f"被 ({nl},{nt})-({nr},{nb}) 包含")
                    break
        if not contained:
            keep.append((cl, ct, cr, cb))
    return keep


def _detect_table_lines(img_gray):
    """v3.0: 保留旧版接口兼容性，但内部使用新的多尺度线段检测。

    返回: (h_lines, v_lines, hmask, vmask)
    注: h_segs/v_segs 额外作为全局变量传递（通过闭包替代方案）
    """
    h_img, w_img = img_gray.shape
    # 保留旧版形态学掩码（供 _find_table_cell 回退使用）
    bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h_len = max(20, 80)  # 原(30,120)→(20,80)，匹配新阈值
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))) > 0
    v_len = max(20, 80)
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))) > 0

    # v3.0 新增：多尺度线段检测（替代旧的 >200px 长线过滤）
    h_segs, _v_segs = _detect_all_line_segments(img_gray)
    # 将线段转换为旧版 h_lines/v_lines 格式：[(坐标, 起点, 终点), ...]
    h_out = []
    h_coord_map = {}
    for (x1, y1, x2, y2) in h_segs:
        y = (y1 + y2) // 2
        if y not in h_coord_map:
            h_coord_map[y] = []
        h_coord_map[y].extend([x1, x2])
    for y in sorted(h_coord_map.keys()):
        xs = h_coord_map[y]
        h_out.append((int(y), int(min(xs)), int(max(xs))))

    v_out = []
    v_coord_map = {}
    for (x1, y1, x2, y2) in _v_segs:
        x = (x1 + x2) // 2
        if x not in v_coord_map:
            v_coord_map[x] = []
        v_coord_map[x].extend([y1, y2])
    for x in sorted(v_coord_map.keys()):
        ys = v_coord_map[x]
        v_out.append((int(x), int(min(ys)), int(max(ys))))

    return h_out, v_out, hmask, vmask


# 全图线段缓存（v3.0 新增，供 _detect_all_table_cells_v3 使用）
_global_h_segs = []
_global_v_segs = []


# ═══════════════════════════════════════════════════════════════
# PP-Structure 引擎（PaddleOCR 表格识别）
# ═══════════════════════════════════════════════════════════════

# 全局缓存的 PaddleOCR 实例（惰性初始化，仅导入一次）
_ppocr_instance = None
_ppocr_available = None  # None=未检测, True=可用, False=不可用


def _get_ppocr():
    """惰性初始化 PaddleOCR 实例（PP-Structure 表格识别引擎）。

    首次调用时导入 paddleocr，最多等待 120 秒；超时或失败则标记为不可用。
    注意：PaddlePaddle 在 CPU-only 环境下导入可能需要 5+ 分钟，本函数默认超时 120s。
         如需调整超时，设置环境变量 PPOCR_IMPORT_TIMEOUT（秒）。

    返回: PaddleOCR 实例 或 None（不可用时）
    """
    global _ppocr_instance, _ppocr_available
    if _ppocr_available is False:
        return None
    if _ppocr_instance is not None:
        return _ppocr_instance

    import_timeout = int(os.environ.get("PPOCR_IMPORT_TIMEOUT", "120"))
    try:
        import time as _time
        import sys as _sys
        import threading as _threading

        _t0 = _time.time()
        logger.info("  [PP-Structure] 首次加载 PPStructureV3 引擎（超时={import_timeout}s）...")

        # 在后台线程中导入，主线程等待超时
        result = {"ocr": None, "error": None, "done": False}

        def _import_ppocr():
            try:
                # TableRecognitionPipelineV2 = PP-Structure V2 表格识别管道
                # 比 PPStructureV3 更轻量，专注于表格单元格检测
                from paddleocr import TableRecognitionPipelineV2
                result["ocr"] = TableRecognitionPipelineV2(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
            except Exception as e:
                result["error"] = str(e)
            finally:
                result["done"] = True

        thread = _threading.Thread(target=_import_ppocr, daemon=True)
        thread.start()
        thread.join(timeout=import_timeout)

        if not result["done"]:
            _ppocr_available = False
            logger.warning(f"  [PP-Structure] PPStructureV3 导入超时 ({import_timeout}s)。"
                           f"CPU-only 环境建议使用 opencv_v3 引擎。"
                           f"可设置 PPOCR_IMPORT_TIMEOUT=600 延长等待。")
            return None

        if result["error"]:
            _ppocr_available = False
            logger.warning(f"  [PP-Structure] PPStructureV3 导入失败: {result['error']}")
            return None

        _ppocr_instance = result["ocr"]
        _ppocr_available = True
        logger.info(f"  [PP-Structure] PPStructureV3 引擎就绪 (耗时 {_time.time() - _t0:.1f}s)")
        return _ppocr_instance

    except Exception as e:
        _ppocr_available = False
        logger.warning(f"  [PP-Structure] PaddleOCR 引擎不可用: {e}")
        return None


def _detect_all_table_cells_ppstructure(h_lines, v_lines, hmask, vmask,
                                         img_gray, img_bgr, ocr_items=None):
    """PPStructureV3 表格单元格检测：使用 PaddleOCR 的 PPStructureV3 管道。

    流程：
      1. 用 PPStructureV3 对全图做表格结构识别
      2. 解析返回的 cell_boxes 坐标
      3. 将 cell 坐标注入 _cell_registry

    不可用时自动回退到 OpenCV v3 引擎。
    返回: [(left, top, right, bottom), ...], [], []
    """
    ocr = _get_ppocr()
    if ocr is None:
        logger.warning("  [PP-Structure] 引擎不可用，回退到 OpenCV v3")
        return _detect_all_table_cells_opencv(
            h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)

    logger.info("  [PP-Structure] 执行 PaddleOCR 表格识别...")
    try:
        # PaddleOCR 3.x API: recognize_table 或 predict
        # 保存临时图像供 PaddleOCR 读取
        import tempfile
        import cv2 as _cv2
        import os as _os

        tmp_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)) if '__file__' in dir() else _os.getcwd(),
            'scan_work', '_ppstructure_input.png')
        _cv2.imwrite(tmp_path, img_bgr)

        # use_ocr_model=True: 管道自己做文字检测+识别，确保单元格有OCR锚定
        result = ocr.predict(tmp_path, use_layout_detection=False, use_ocr_model=True)

        # 尝试清理临时文件
        try:
            _os.remove(tmp_path)
        except Exception:
            pass

        # 解析 PaddleOCR 返回结果
        cells = _parse_paddleocr_table_result(result, img_gray)
        logger.info(f"  [PP-Structure] 解析到 {len(cells)} 个单元格")

        # 如果 PP-Structure 结果太少（<3个格），回退到 OpenCV
        if len(cells) < 3:
            logger.warning(f"  [PP-Structure] 检测到的单元格太少({len(cells)}个)，回退到 OpenCV v3")
            return _detect_all_table_cells_opencv(
                h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)

        # 注册单元格
        for cell in cells:
            _cell_id(cell)

        logger.info(f"  [PP-Structure检测汇总] 最终确认 {len(cells)} 个单元格")
        for cell in cells:
            cid = _cell_registry[cell]
            logger.info(f"    {cid}: ({cell[0]},{cell[1]})-({cell[2]},{cell[3]}) "
                        f"{cell[2]-cell[0]}x{cell[3]-cell[1]}px")

        return cells, [], []

    except Exception as e:
        logger.warning(f"  [PP-Structure] 表格识别失败: {e}，回退到 OpenCV v3")
        import traceback
        logger.debug(traceback.format_exc())
        return _detect_all_table_cells_opencv(
            h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)


def _parse_paddleocr_table_result(result, img_gray):
    """从 TableRecognitionPipelineV2 的表格识别结果中提取单元格坐标。

    输出格式（list of TableRecognitionResult-like dict）:
      [{
        'input_path': '...',
        'table_res_list': [
          {'cell_box_list': [[x1,y1,x2,y2], ...], 'html': '...'},
          ...
        ]
      }]

    返回: [(left, top, right, bottom), ...]
    """
    cells = []

    if not result or not isinstance(result, list):
        return cells

    for res_item in result:
        if not isinstance(res_item, dict):
            continue

        # TableRecognitionPipelineV2: table_res_list[].cell_box_list
        table_res_list = res_item.get('table_res_list', [])
        if isinstance(table_res_list, list):
            for tbl in table_res_list:
                if isinstance(tbl, dict):
                    # 优先: cell_box_list (PaddleX 原生格式)
                    cell_boxes = tbl.get('cell_box_list', [])
                    if cell_boxes:
                        cells.extend(_validate_and_filter_cells(cell_boxes, img_gray))
                    # 备选: cell_boxes
                    cell_boxes2 = tbl.get('cell_boxes', [])
                    if cell_boxes2:
                        cells.extend(_validate_and_filter_cells(cell_boxes2, img_gray))

        # 兼容 PPStructureV3 格式: res.tables[].cell_boxes
        res = res_item.get('res', {})
        if isinstance(res, dict):
            tables = res.get('tables', [])
            if isinstance(tables, list):
                for tbl in tables:
                    if isinstance(tbl, dict):
                        cell_boxes = tbl.get('cell_boxes', [])
                        cells.extend(_validate_and_filter_cells(cell_boxes, img_gray))

    return cells


def _validate_and_filter_cells(cell_boxes, img_gray):
    """验证和过滤 PaddleOCR 返回的单元格坐标。"""
    cells = []
    if not cell_boxes:
        return cells

    h_img, w_img = img_gray.shape

    for box in cell_boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = map(int, box[:4])
        # 确保坐标顺序正确
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        cw, ch = x2 - x1, y2 - y1
        if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
            continue

        # 白底验证
        roi = img_gray[y1 + 2:y2 - 2, x1 + 2:x2 - 2] if ch > 4 and cw > 4 else None
        if roi is None or roi.size == 0:
            continue
        if float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
            continue

        cells.append((x1, y1, x2, y2))

    return cells


# ═══════════════════════════════════════════════════════════════
# 单元格检测调度器（引擎切换入口）
# ═══════════════════════════════════════════════════════════════

def _detect_all_table_cells_dispatch(h_lines, v_lines, hmask, vmask,
                                      img_gray, img_bgr=None, ocr_items=None):
    """根据 CELL_DETECT_ENGINE 环境变量调度单元格检测引擎。

    "opencv_v3"  → 纯 OpenCV 多策略检测 (v3.1)
    "ppstructure" → PaddleOCR PP-Structure 表格识别（不可用时自动回退）

    返回: [(left, top, right, bottom), ...], [], []
    """
    engine = CELL_DETECT_ENGINE.lower()
    logger.info(f"  [调度] 单元格检测引擎: {engine}")

    if engine == "ppstructure":
        if img_bgr is None:
            logger.warning("  [调度] PP-Structure 需要 BGR 图像，回退到 OpenCV v3")
            return _detect_all_table_cells_opencv(
                h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)
        return _detect_all_table_cells_ppstructure(
            h_lines, v_lines, hmask, vmask, img_gray, img_bgr, ocr_items=ocr_items)
    else:
        # 默认: opencv_v3
        return _detect_all_table_cells_opencv(
            h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)


def _detect_all_table_cells_opencv(h_lines, v_lines, hmask, vmask, img_gray, ocr_items=None):
    """OpenCV v3.3 多策略单元格检测级联：
    阶段M: 形态学连通域 — 首选：提取纯线框→膨胀闭合→反色连通域=原子单元格
    阶段A: 轮廓矩形检测 — 补充：精确捕获闭合单元格
    阶段B: 线交叉网格 — 补充：线段交叉构建候选格
    阶段C: 文本间隙推理 — 默认禁用 (OCR位置不可靠)
    阶段D: 回退 — 对未覆盖的OCR项，调用 _find_table_cell
    后过滤: 移除被形态学格包含的子格 + 无文本孤立格
    """
    cells_set = set()
    morph_cells = []  # 形态学格作为"权威格"，后处理时保护它们

    # === 阶段M: 形态学连通域（v3.3 首选：纯线框→连通域=原子格）===
    morph_cells = _find_cells_by_morphological_components(img_gray)
    for c in morph_cells:
        cells_set.add(c)
    logger.info(f"  [阶段M] 形态学连通域: {len(morph_cells)} 个原子单元格")

    # === 阶段A: 轮廓矩形检测 ===
    contour_cells = _detect_rectangular_contours(img_gray)
    for c in contour_cells:
        cells_set.add(c)
    logger.info(f"  [阶段A] 轮廓检测: {len(contour_cells)} 个闭合矩形格")

    # === 阶段B: 线交叉网格（v3.1: 交点过滤） ===
    h_segs, v_segs = _detect_all_line_segments(img_gray)
    _global_h_segs[:] = h_segs
    _global_v_segs[:] = v_segs

    grid_cells = _build_line_intersection_grid(h_segs, v_segs, img_gray)
    for c in grid_cells:
        cells_set.add(c)
    logger.info(f"  [阶段B] 线交叉网格: {len(grid_cells)} 个候选格")

    # === 阶段C: 文本间隙推理（v3.2: 默认禁用，防止OCR字符间距误创假格拆分真实单元格） ===
    # 规则：封闭框内不应包含除文本外的其它线框。OCR字符间距不应被当作格线。
    # 若确实需要启用，设置环境变量 ENABLE_TEXT_GAP_CELLS=1
    gap_cells = []
    enable_gap = os.environ.get("ENABLE_TEXT_GAP_CELLS", "0") == "1"
    if enable_gap and ocr_items:
        existing_before_c = list(cells_set)
        gap_cells = _find_cells_by_text_gaps(ocr_items, img_gray, existing_cells=existing_before_c)
        # v3.2 安全过滤：移除被已有A+B格完全包含的gap格（防拆分）
        gap_filtered = []
        for gc in gap_cells:
            gl, gt, gr, gb = gc
            contained_in_existing = any(
                el <= gl and et <= gt and er >= gr and eb >= gb and (el != gl or et != gt or er != gr or eb != gb)
                for (el, et, er, eb) in existing_before_c
            )
            if not contained_in_existing:
                gap_filtered.append(gc)
            else:
                logger.debug(f"  [阶段C过滤] 移除被A+B格包含的gap格 ({gl},{gt})-({gr},{gb}) "
                             f"{gr-gl}x{gb-gt}px — 防止拆分真实单元格")
        gap_cells = gap_filtered
        for c in gap_cells:
            cells_set.add(c)
        logger.info(f"  [阶段C] 文本间隙推理: {len(gap_cells)} 个候选格（已过滤被包含格）")
    else:
        logger.info(f"  [阶段C] 文本间隙推理: 已禁用 (ENABLE_TEXT_GAP_CELLS={os.environ.get('ENABLE_TEXT_GAP_CELLS','0')})")

    # === 阶段D: 回退 _find_table_cell（覆盖前三个阶段遗漏的OCR文本） ===
    fallback_count = 0
    if ocr_items:
        for idx, item in enumerate(ocr_items):
            bx1, by1, bx2, by2 = item["bbox"]
            covered = any(
                cl <= bx1 and ct <= by1 and cr >= bx2 and cb >= by2
                for (cl, ct, cr, cb) in cells_set
            )
            if covered:
                continue
            cell = _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray, margin=30)
            if cell is not None:
                cells_set.add(cell)
                fallback_count += 1
        if fallback_count:
            logger.info(f"  [阶段D] 回退 _find_table_cell: {fallback_count} 个补充格")

    # === 后处理 ===
    cells = sorted(cells_set, key=lambda c: (c[1], c[0]))

    # v3.3: 先去除被形态学格包含的子格（形态学格是原子格，不应被其他阶段拆开）
    if morph_cells:
        cells_before_morph_filter = len(cells)
        cells = [
            c for c in cells
            if c in morph_cells or not any(
                ml <= c[0] and mt <= c[1] and mr >= c[2] and mb >= c[3]
                and (ml != c[0] or mt != c[1] or mr != c[2] or mb != c[3])
                for (ml, mt, mr, mb) in morph_cells
            )
        ]
        removed_by_morph = cells_before_morph_filter - len(cells)
        if removed_by_morph:
            logger.info(f"  [后处理] 移除被形态学原子格包含的子格: {removed_by_morph} 个")

    cells = _remove_contained_cells(cells)
    if ocr_items:
        cells = _filter_cells_by_text_proximity(cells, ocr_items)
    cells = _classify_grid_region(cells, img_gray)

    for cell in cells:
        _cell_id(cell)

    logger.info(f"  [检测汇总] 最终确认 {len(cells)} 个单元格 "
                f"(形态学{len(morph_cells)}+轮廓{len(contour_cells)}+网格{len(grid_cells)}+回退{fallback_count})")
    for cell in cells:
        cid = _cell_registry[cell]
        logger.info(f"    {cid}: ({cell[0]},{cell[1]})-({cell[2]},{cell[3]}) {cell[2]-cell[0]}x{cell[3]-cell[1]}px")
    return cells, [], []


def _detect_all_table_cells_v3(h_lines, v_lines, hmask, vmask, img_gray, ocr_items=None):
    """v3.1 向后兼容包装：默认走 OpenCV 引擎。
    可通过设置环境变量 CELL_DETECT_ENGINE=ppstructure 切换到 PP-Structure。
    """
    return _detect_all_table_cells_opencv(h_lines, v_lines, hmask, vmask, img_gray, ocr_items=ocr_items)



def _filter_cells_by_text_proximity(cells, ocr_items, max_gap=50):
    """v3.1 文本邻近过滤：移除不含OCR文本且离最近文本>max_gap的孤立格。

    这类格通常是CAD图纸中的空白装饰框、结构线交叉形成的假矩形等。
    """
    if not ocr_items:
        return cells

    ocr_bboxes = [(it["bbox"][0], it["bbox"][1], it["bbox"][2], it["bbox"][3]) for it in ocr_items]

    def _cell_contains_text(cl, ct, cr, cb):
        """格内是否包含OCR文本"""
        for ox1, oy1, ox2, oy2 in ocr_bboxes:
            # OCR中心在格内
            ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            if cl <= ocx <= cr and ct <= ocy <= cb:
                return True
            # 或OCR框与格>50%重叠
            ol = max(cl, ox1)
            ot = max(ct, oy1)
            or_ = min(cr, ox2)
            ob = min(cb, oy2)
            if ol < or_ and ot < ob:
                overlap = (or_ - ol) * (ob - ot)
                ocr_area = (ox2 - ox1) * (oy2 - oy1)
                if ocr_area > 0 and overlap > ocr_area * 0.3:
                    return True
        return False

    def _nearest_text_dist(cl, ct, cr, cb):
        """格中心到最近OCR中心的距离"""
        cx, cy = (cl + cr) // 2, (ct + cb) // 2
        min_d = float('inf')
        for ox1, oy1, ox2, oy2 in ocr_bboxes:
            ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
            d = np.hypot(cx - ocx, cy - ocy)
            if d < min_d:
                min_d = d
        return min_d

    keep = []
    removed = 0
    for cl, ct, cr, cb in cells:
        if _cell_contains_text(cl, ct, cr, cb):
            keep.append((cl, ct, cr, cb))
        elif _nearest_text_dist(cl, ct, cr, cb) <= max_gap:
            keep.append((cl, ct, cr, cb))
        else:
            removed += 1
            logger.debug(f"  [文本过滤] 移除无文本格 ({cl},{ct})-({cr},{cb})")

    if removed:
        logger.info(f"  [文本过滤] 移除 {removed} 个远离OCR文本的孤立格")
    return keep


def _find_table_cell(bbox, h_lines, v_lines, hmask, vmask, img_gray, margin=30):
    """v3.0 回退方案：用检测出的表格线围绕 bbox 中心点定位单元格。

    优化点：默认margin缩小为30px（原50→30），减少跨格误匹配；
           放宽重试从200→120，兜底从200→120；
           保留形态学掩码验证确保线条在区域内存在。
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = img_gray.shape
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    def _search(m):
        lx, rx = max(0, cx - m), min(w_img - 1, cx + m)
        ty_roi, by_roi = max(0, cy - m), min(h_img - 1, cy + m)

        top_cands, bot_cands = [], []
        for y, _, _ in h_lines:
            if y < 0 or y >= h_img:
                continue
            if not np.any(hmask[y, lx:rx + 1]):
                continue
            if y <= cy:
                top_cands.append(y)
            if y >= cy:
                bot_cands.append(y)
        top_cands.sort()
        bot_cands.sort()

        lft_cands, rgt_cands = [], []
        for x, _, _ in v_lines:
            if x < 0 or x >= w_img:
                continue
            if not np.any(vmask[ty_roi:by_roi + 1, x]):
                continue
            if x <= cx:
                lft_cands.append(x)
            if x >= cx:
                rgt_cands.append(x)
        lft_cands.sort()
        rgt_cands.sort()

        if not (top_cands and bot_cands and lft_cands and rgt_cands):
            return None

        gl, gt, gr, gb = lft_cands[-1], top_cands[-1], rgt_cands[0], bot_cands[0]
        cw, ch = gr - gl, gb - gt
        if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
            return None

        # 白底验证
        roi = img_gray[gt + 2:gb - 2, gl + 2:gr - 2] if gb - gt > 4 and gr - gl > 4 else None
        if roi is None or roi.size == 0:
            return None
        if float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
            return None

        return gl, gt, gr, gb

    # 尝试标准margin（v3.0: 30px，比原来的50更紧）
    result = _search(margin)
    if result is not None:
        return result

    # 放宽重试（v3.0: 120px，比原来的200更紧）
    result = _search(max(margin, 120))
    if result is not None:
        return result

    # 兜底：图像边界
    top_cands, bot_cands = [], []
    for y, _, _ in h_lines:
        if y < 0 or y >= h_img:
            continue
        lx, rx = max(0, cx - 120), min(w_img - 1, cx + 120)
        if not np.any(hmask[y, lx:rx + 1]):
            continue
        if y <= cy:
            top_cands.append(y)
        if y >= cy:
            bot_cands.append(y)
    top_cands.sort()
    bot_cands.sort()

    lft_cands, rgt_cands = [], []
    for x, _, _ in v_lines:
        if x < 0 or x >= w_img:
            continue
        ty_roi, by_roi = max(0, cy - 120), min(h_img - 1, cy + 120)
        if not np.any(vmask[ty_roi:by_roi + 1, x]):
            continue
        if x <= cx:
            lft_cands.append(x)
        if x >= cx:
            rgt_cands.append(x)
    lft_cands.sort()
    rgt_cands.sort()

    gl = lft_cands[-1] if lft_cands else 0
    gt = top_cands[-1] if top_cands else 0
    gr = rgt_cands[0] if rgt_cands else w_img - 1
    gb = bot_cands[0] if bot_cands else h_img - 1

    cw, ch = gr - gl, gb - gt
    if not (CELL_MIN_W <= cw <= CELL_MAX_W and CELL_MIN_H <= ch <= CELL_MAX_H):
        return None

    roi = img_gray[gt + 2:gb - 2, gl + 2:gr - 2] if gb - gt > 4 and gr - gl > 4 else None
    if roi is not None and roi.size > 0 and float(np.mean(roi > 180)) < CELL_WHITE_THRESHOLD:
        return None

    return gl, gt, gr, gb


def _is_in_table(item, h_lines, v_lines, hmask, vmask, img_gray):
    """判断 OCR 块是否落在表格单元格内。"""
    return _find_table_cell(item["bbox"], h_lines, v_lines, hmask, vmask, img_gray) is not None


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

        # === 阶段1: 全图格网检测 (v3.1: 调度引擎 — CELL_DETECT_ENGINE 控制) ===
        all_cells, tbl_h, tbl_v = _detect_all_table_cells_dispatch(
            h_lines, v_lines, hmask, vmask, img_gray, img_bgr=img_bgr, ocr_items=items)

        # === 阶段2: 分配OCR到格（先分割跨格文本） ===
        # v2.4: 如果OCR条目的bbox跨越多条格子水平线，按线位比例分割。
        # 这样"地脚\n螺栓直径\n螺纹长度"会被 y=8242 一线切分为两个格子各自的文本。
        if all_cells:
            h_lines_set = set()
            for (cl, ct, cr, cb) in all_cells:
                h_lines_set.add(ct)
                h_lines_set.add(cb)
            h_lines_sorted = sorted(h_lines_set)

            split_items = []
            for idx, item in enumerate(items):
                x1, y1, x2, y2 = item["bbox"]
                text = item["text"]
                # 找落在bbox内部的水平线
                crossing = [ly for ly in h_lines_sorted if y1 < ly < y2]
                if "\n" in text and crossing:
                    src_lines = text.split("\n")
                    total_h = y2 - y1
                    if len(src_lines) >= 2 and total_h > 0:
                        # 按线位比例分割：找到每条线在文本中的分割点
                        cut_points = sorted([(ly - y1) / total_h for ly in crossing])
                        # 估算每行文本的位置比例（等分）
                        line_ratio = 1.0 / len(src_lines)
                        segments = []  # [(start_line, end_line), ...]
                        cur = 0
                        for cp in cut_points:
                            split_line = int(cp / line_ratio)  # 哪一行被切
                            split_line = max(cur, min(split_line, len(src_lines)))
                            if split_line > cur:
                                segments.append((cur, split_line))
                            cur = split_line
                        if cur < len(src_lines):
                            segments.append((cur, len(src_lines)))

                        if len(segments) > 1:
                            logger.info(f"  [跨格分割] OCR#{idx} '{text[:40]}' 被线{[(round(ly,0)) for ly in crossing]}分割为{len(segments)}段")
                            for seg_start, seg_end in segments:
                                sub_text = "\n".join(src_lines[seg_start:seg_end])
                                sy1 = int(y1 + total_h * seg_start / len(src_lines))
                                sy2 = int(y1 + total_h * seg_end / len(src_lines))
                                sub_item = dict(item)
                                sub_item["text"] = sub_text
                                sub_item["bbox"] = [x1, sy1, x2, sy2]
                                split_items.append(sub_item)
                            continue
                split_items.append(item)
            items = split_items
            logger.info(f"  [跨格分割] 处理完成，{len(items)}个条目")

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

    drawn_count = 0
    for cell_key, cid in _cell_registry.items():
        # v2.4: 只绘制含文本的单元格（过滤空格）
        if cid not in _cell_texts or not _cell_texts[cid]:
            continue
        drawn_count += 1
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
    logger.info(f"  [调试PDF] 已绘制 {drawn_count}/{len(_cell_registry)} 个含文本单元格的红框+编号")

    # 保存为PNG再转PDF
    debug_png = debug_pdf_path.replace(".pdf", ".png")
    pil_img.convert("RGB").save(debug_png)
    image_to_pdf(debug_png, debug_pdf_path, dpi=dpi)
    logger.info(f"  [调试PDF] 已保存: {debug_pdf_path}")


def main():
    _clear_cell_registry()  # 每次运行前重置注册表
    logger.info("=" * 60)
    logger.info("Scan-type PDF CN->EN Translation (v3.0 Multi-Strategy Cell Detection + Diagnostic Logging)")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    print("=" * 60)
    print("Scan-type PDF CN->EN Translation (v3.0 Multi-Strategy Cell Detection)")
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
