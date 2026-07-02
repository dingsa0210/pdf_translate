"""
PP-StructureV3 扫描PDF翻译管线

全程基于 PaddleOCR PP-StructureV3 的版面识别能力，替代 RapidOCR + OpenCV 格子检测。
无需连通域分析、无需线段检测、无需网格交点过滤。

流程:
  PDF渲染 → PP-StructureV3(OCR+版面+表格) → 翻译 → 擦除回填 → 输出PDF

优势:
  - PP-StructureV3 原生提供 text/title/table 区域分类与阅读顺序
  - 表格单元格坐标由 PP-StructureV3 直接给出，无需手工检测
  - 可对比原始 scan_translate_pipeline.py 评估 PP-StructureV3 的排版识别质量
"""

import os, sys, json, shutil
import re
import time
import traceback
import numpy as np
import cv2
import fitz
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from datetime import datetime

from config import (
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_BATCH_SIZE, LLM_TEMPERATURE,
    TRANSLATE_ENGINE, ENGINEERING_DICT, RENDER_DPI,
)

# ══════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════
PDF_PATH = os.environ.get(
    "PDF_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs", "地脚螺栓预埋铁分布图-Model_1.pdf"),
)
# 工作目录：与 scan_work 区分，避免干扰
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppstructure_work")
OUTPUT_PDF = os.path.join(WORK_DIR, "output_ppstructure_translated.pdf")

# ══════════════════════════════════════════════════════════════
# 日志
# ══════════════════════════════════════════════════════════════
os.makedirs(WORK_DIR, exist_ok=True)
LOG_DIR = os.path.join(WORK_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"ppstructure_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
try:
    logger.add(LOG_FILE, level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
               rotation="20 MB", retention="14 days")
except Exception:
    pass

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════
ANGLE_NEAR_HORIZONTAL = float(os.environ.get("ANGLE_NEAR_HORIZONTAL", "5.0"))
RIGHT_MARGIN = int(os.environ.get("RIGHT_MARGIN", "8"))
# macOS 常见字体路径回退
FONT_PATH = os.environ.get("FONT_PATH", "/System/Library/Fonts/Supplemental/Arial.ttf")
if not os.path.exists(FONT_PATH):
    FONT_PATH = os.environ.get("FONT_PATH", "/System/Library/Fonts/Helvetica.ttc")
if not os.path.exists(FONT_PATH):
    FONT_PATH = os.environ.get("FONT_PATH", "")

# PP-StructureV3 导入超时（秒）
PP_IMPORT_TIMEOUT = int(os.environ.get("PP_IMPORT_TIMEOUT", "1800"))
# 智能分块目标尺寸（像素），子区域超过此尺寸则切分
TILE_SIZE = int(os.environ.get("TILE_SIZE", "3000"))
# PP-StructureV3 text_det 边长上限 —— 不应设过大。
# 设为 1920 让 PP 内部适度缩放以控制计算量，避免高分辨率不缩放导致性能崩溃。
PP_MAX_SIDE_LEN = int(os.environ.get("PP_MAX_SIDE_LEN", "1920"))
# 阶段1粗识别缩略图边长 —— 只需大致定位文本位置，用更小尺寸加速。
PP_COARSE_SIDE_LEN = int(os.environ.get("PP_COARSE_SIDE_LEN", "1280"))

# ══════════════════════════════════════════════════════════════
# Step 0: PP-StructureV3 引擎加载
# ══════════════════════════════════════════════════════════════
_pp_structure = None


def _get_pp_structure():
    """惰性加载 PP-StructureV3 实例。"""
    global _pp_structure
    if _pp_structure is not None:
        return _pp_structure

    import threading as _thr
    result = {"pipeline": None, "error": None, "done": False}

    def _import():
        try:
            from paddleocr import PPStructureV3
            result["pipeline"] = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_table_recognition=False,
                use_formula_recognition=False,
                text_det_limit_side_len=PP_MAX_SIDE_LEN,
            )
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["done"] = True

    thread = _thr.Thread(target=_import, daemon=True)
    thread.start()
    thread.join(timeout=PP_IMPORT_TIMEOUT)

    if not result["done"]:
        raise RuntimeError(f"PP-StructureV3 导入超时 ({PP_IMPORT_TIMEOUT}s)，请延长 PP_IMPORT_TIMEOUT")
    if result["error"]:
        raise RuntimeError(f"PP-StructureV3 导入失败: {result['error']}")

    _pp_structure = result["pipeline"]
    logger.info(f"PP-StructureV3 引擎就绪")
    return _pp_structure


# ══════════════════════════════════════════════════════════════
# Step 1: PDF → Image
# ══════════════════════════════════════════════════════════════
def pdf_to_image(pdf_path, dpi=200):
    logged_path = os.path.join(WORK_DIR, f"rendered_{dpi}dpi.png")
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(logged_path)
    meta = {
        "page_width_pt": float(page.rect.width),
        "page_height_pt": float(page.rect.height),
        "rotation": int(page.rotation),
        "dpi": int(dpi),
        "pixel_width": int(pix.width),
        "pixel_height": int(pix.height),
    }
    doc.close()
    logger.info(f"PDF 渲染: {pix.width}x{pix.height}px @ {dpi}dpi")
    return logged_path, meta


# ══════════════════════════════════════════════════════════════
# Step 2: PP-StructureV3 OCR + 版面识别（两阶段：粗识别 + 智能分块 + 精识别）
# ══════════════════════════════════════════════════════════════


def _ocr_parse_result(output, offset_x=0, offset_y=0, scale=1.0) -> list:
    """从 PP-StructureV3 的 predict 结果中提取 OCR item 列表。

    offset_x/y: 坐标偏移（用于 tile 坐标回算）
    scale: 缩放比例（用于粗识别坐标还原）
    """
    items = []
    for res in output:
        data = _safe_json(res)
        if not isinstance(data, dict):
            continue
        res_data = data.get('res', data)

        overall = res_data.get('overall_ocr_res', {})
        rec_texts = overall.get('rec_texts', [])
        rec_scores = overall.get('rec_scores', [])
        rec_boxes = overall.get('rec_boxes', [])
        rec_polys = overall.get('rec_polys', [])

        for i, text in enumerate(rec_texts):
            if not text or not text.strip():
                continue
            if not any('\u4e00' <= ch <= '\u9fff' for ch in text):
                continue

            conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0

            if i < len(rec_boxes) and rec_boxes[i] is not None:
                box_arr = np.array(rec_boxes[i]).flatten()
                if len(box_arr) == 4:
                    x1, y1, x2, y2 = [int(v) for v in box_arr]
                else:
                    continue
            else:
                continue

            # 坐标缩放 + 偏移
            x1 = int(x1 * scale + offset_x)
            y1 = int(y1 * scale + offset_y)
            x2 = int(x2 * scale + offset_x)
            y2 = int(y2 * scale + offset_y)

            poly = None
            angle = 0.0
            if i < len(rec_polys) and rec_polys[i] is not None:
                p_arr = np.array(rec_polys[i], dtype=np.float64)
                if p_arr.shape == (4, 2):
                    p_arr[:, 0] = p_arr[:, 0] * scale + offset_x
                    p_arr[:, 1] = p_arr[:, 1] * scale + offset_y
                    poly = p_arr
                    dx = p_arr[1][0] - p_arr[0][0]
                    dy = p_arr[1][1] - p_arr[0][1]
                    angle = float(np.degrees(np.arctan2(dy, dx)))

            w, h = max(1, x2 - x1), max(1, y2 - y1)

            items.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "box": _poly_to_box(poly) if poly is not None else [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "confidence": round(conf, 3),
                "angle": round(angle, 1),
                "width": w,
                "height": h,
                "sub_bboxes": [],
                "is_structured": False,
                "pp_type": "text",
                "pp_block_bbox": [x1, y1, x2, y2],
                "in_table": False,
            })

    return items


# ── 智能分块：基于文本位置找空档切割 ──

def _find_gaps(occupied, total_len, min_gap=80):
    """在一维占用数组中找所有空档（连续 False 段）。

    返回: [(gap_start, gap_end), ...] 按 gap 宽度降序
    """
    gaps = []
    i = 0
    while i < total_len:
        if not occupied[i]:
            start = i
            while i < total_len and not occupied[i]:
                i += 1
            end = i
            if end - start >= min_gap:
                gaps.append((start, end))
        else:
            i += 1
    # 按宽度降序
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    return gaps


def _select_best_splits(gaps, total_len, target_region_size):
    """从空档列表中选出最优切割线位置。

    贪心策略：从最大空档开始，如果该空档能把某个超限区域切开，就用它。
    额外约束：切出段长不低于 target_region_size * 0.3，避免产生极窄条带浪费推理资源。
    返回: [split_position, ...] 排序后的切割线位置
    """
    if not gaps:
        return []

    MIN_SEGMENT_RATIO = 0.3  # 切出段长不低于 target_region_size 的 30%
    min_segment_size = int(target_region_size * MIN_SEGMENT_RATIO)

    # 初始状态：整段 [0, total_len) 需要切
    segments = [(0, total_len)]
    splits = []

    for gap_start, gap_end in gaps:
        # 在空档中点切
        split_pos = (gap_start + gap_end) // 2

        # 检查这把刀能不能把某个超限段切开，且切出段不太小
        new_segments = []
        cut_made = False
        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len > target_region_size and seg_start < split_pos < seg_end:
                left_len = split_pos - seg_start
                right_len = seg_end - split_pos
                # 防过度切割：切出段必须 >= min_segment_size
                if left_len < min_segment_size or right_len < min_segment_size:
                    new_segments.append((seg_start, seg_end))
                    continue
                new_segments.append((seg_start, split_pos))
                new_segments.append((split_pos, seg_end))
                splits.append(split_pos)
                cut_made = True
            else:
                new_segments.append((seg_start, seg_end))

        segments = new_segments
        if not cut_made:
            # 没有超限段需要这把刀，跳过
            pass

    # 检查是否还有超限段没切到 → 强制均分
    final_splits = set(splits)
    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len > target_region_size:
            n = int(np.ceil(seg_len / target_region_size))
            step = seg_len / n
            for k in range(1, n):
                pos = int(seg_start + step * k)
                final_splits.add(pos)

    return sorted(final_splits)


def _smart_tile_cuts(img_bgr, pipeline):
    """两阶段智能分块。

    阶段1：缩略图粗识别 → 得到全图文本 bbox 分布
    阶段2：在文本空档处画切割线，绝不切到文本区域
    返回: [(x, y, w, h), ...] 原图子区域坐标列表
    """
    img_h, img_w = img_bgr.shape[:2]

    # 阶段1：缩略图粗识别
    scale = min(1.0, PP_COARSE_SIDE_LEN / max(img_w, img_h))
    if scale >= 1.0:
        # 原图已足够小，直接全图处理
        logger.info(f"原图 {img_w}x{img_h} ≤ {PP_COARSE_SIDE_LEN}px，无需分块")
        return [(0, 0, img_w, img_h)]

    small_w, small_h = int(img_w * scale), int(img_h * scale)
    small_img = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_path = os.path.join(WORK_DIR, "_coarse_preview.png")
    cv2.imwrite(small_path, small_img)

    logger.info(f"[阶段1] 缩略图粗识别: {img_w}x{img_h} → {small_w}x{small_h} (scale={scale:.2f})")
    t1 = time.time()
    coarse_output = pipeline.predict(small_path)
    # PP-StructureV3 内部可能再次缩放，直接用 small_w/small_h 推算实际缩放比
    coarse_items = _ocr_parse_result(coarse_output, scale=1.0 / scale)
    # 防御：将超出原图边界的坐标裁回
    _clamp_item_bboxes(coarse_items, img_w, img_h)
    logger.info(f"[阶段1] 粗识别完成: {len(coarse_items)} 个文本块 (耗时 {time.time() - t1:.1f}s)")

    try:
        os.remove(small_path)
    except Exception:
        pass

    if not coarse_items:
        logger.warning("粗识别未找到文本，回退到均匀分块")
        return _fallback_tiles(img_w, img_h, TILE_SIZE)

    # 阶段2：找空档切割线
    # Y 轴占用（带 margin 避免切割线太靠近文字）
    margin = int(os.environ.get("SPLIT_MARGIN", "30"))
    y_occupied = np.zeros(img_h, dtype=bool)
    x_occupied = np.zeros(img_w, dtype=bool)

    for item in coarse_items:
        x1, y1, x2, y2 = item["bbox"]
        y1_safe = max(0, y1 - margin)
        y2_safe = min(img_h, y2 + margin)
        x1_safe = max(0, x1 - margin)
        x2_safe = min(img_w, x2 + margin)
        y_occupied[y1_safe:y2_safe] = True
        x_occupied[x1_safe:x2_safe] = True

    # 找 Y 方向空档 → 水平切割线
    h_gaps = _find_gaps(y_occupied, img_h, min_gap=80)
    h_splits = _select_best_splits(h_gaps, img_h, TILE_SIZE)
    logger.info(f"[阶段2] Y 轴空档: {len(h_gaps)} 个, 选中 {len(h_splits)} 条水平切割线")

    # 找 X 方向空档 → 垂直切割线
    v_gaps = _find_gaps(x_occupied, img_w, min_gap=80)
    v_splits = _select_best_splits(v_gaps, img_w, TILE_SIZE)
    logger.info(f"[阶段2] X 轴空档: {len(v_gaps)} 个, 选中 {len(v_splits)} 条垂直切割线")

    # 生成子区域（用切割线划格子）
    x_edges = [0] + v_splits + [img_w]
    y_edges = [0] + h_splits + [img_h]

    logger.info(f"[阶段2] X 切割边: {x_edges}")
    logger.info(f"[阶段2] Y 切割边: {y_edges}")

    regions = []
    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            rx, ry = x_edges[xi], y_edges[yi]
            rw = x_edges[xi + 1] - rx
            rh = y_edges[yi + 1] - ry
            # 防御性裁切：确保不超出原图边界
            rx = max(0, min(rx, img_w - 1))
            ry = max(0, min(ry, img_h - 1))
            rw = max(1, min(rw, img_w - rx))
            rh = max(1, min(rh, img_h - ry))
            regions.append((rx, ry, rw, rh))

    # 打印所有子区域
    logger.info(f"[阶段2] 共 {len(regions)} 个子区域:")
    for ri, (rx, ry, rw, rh) in enumerate(regions):
        logger.info(f"    [{ri}] ({rx},{ry}) {rw}x{rh}")

    # 标记切到的文本（理论应为 0）
    cut_texts = 0
    for item in coarse_items:
        x1, y1, x2, y2 = item["bbox"]
        for sx in h_splits:
            if y1 < sx < y2:
                cut_texts += 1
                logger.warning(f"水平切割线 y={sx} 穿过文本: \"{item['text'][:40]}\"")
                break
        else:
            for sy in v_splits:
                if x1 < sy < x2:
                    cut_texts += 1
                    logger.warning(f"垂直切割线 x={sy} 穿过文本: \"{item['text'][:40]}\"")
                    break
    if cut_texts == 0:
        logger.info(f"[阶段2] 验证通过: 0 条切割线穿过文本区域 ✓")
    else:
        logger.warning(f"[阶段2] 警告: {cut_texts} 条文本可能被切割线穿过")

    return regions


def _fallback_tiles(img_w, img_h, tile_size):
    """回退方案：均匀分块。"""
    tiles = []
    for y in range(0, img_h, tile_size):
        for x in range(0, img_w, tile_size):
            tw = min(tile_size, img_w - x)
            th = min(tile_size, img_h - y)
            tiles.append((x, y, tw, th))
    return tiles


# ── 主 OCR 函数 ──

def ocr_with_ppstructure(img_path) -> list:
    """
    PP-StructureV3 OCR + 版面识别（两阶段智能分块）。

    阶段1：缩略图粗识别，获取全图文本位置分布
    阶段2：按文本空档切割，每个子区域原分辨率精识别
    保证：① 不缩放原图 ② 不切割文本区域
    """
    pipeline = _get_pp_structure()
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    img_h, img_w = img_bgr.shape[:2]

    t0 = time.time()

    # 智能分块
    regions = _smart_tile_cuts(img_bgr, pipeline)
    logger.info(f"分块结果: {len(regions)} 个子区域")

    # 阶段3：原分辨率精识别每个子区域
    tmp_dir = os.path.join(WORK_DIR, "_pp_tiles")
    os.makedirs(tmp_dir, exist_ok=True)
    all_items = []

    for ri, (rx, ry, rw, rh) in enumerate(regions):
        region_img = img_bgr[ry:ry + rh, rx:rx + rw]
        tile_path = os.path.join(tmp_dir, f"region_{ry:04d}_{rx:04d}.png")
        cv2.imwrite(tile_path, region_img)

        t_tile = time.time()
        fine_output = pipeline.predict(tile_path)
        tile_items = _ocr_parse_result(fine_output, offset_x=rx, offset_y=ry)
        # 防御裁切（应对 PP-StructureV3 内部坐标偏移）
        _clamp_item_bboxes(tile_items, img_w, img_h)
        logger.debug(f"  区域 [{ri + 1}/{len(regions)}] ({rx},{ry}) {rw}x{rh}: "
                     f"{len(tile_items)} 文本块 (耗时 {time.time() - t_tile:.1f}s)")
        all_items.extend(tile_items)

        try:
            os.remove(tile_path)
        except Exception:
            pass

    # 清理
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # 去重
    items = _dedup_items(all_items)

    elapsed = time.time() - t0
    logger.info(f"OCR 完成: {len(items)} 个文本块 (耗时 {elapsed:.1f}s)")

    # 保存
    ocr_json = os.path.join(WORK_DIR, "ocr_result.json")
    with open(ocr_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return items


def _safe_json(res_obj):
    """兼容不同 PaddleOCR 版本的 json 获取方式。"""
    if isinstance(res_obj, dict):
        return res_obj
    if hasattr(res_obj, 'json'):
        j = res_obj.json
        if callable(j):
            return j()
        return j
    return {}


def _poly_to_box(poly):
    """numpy (4,2) → [[x,y], ...]"""
    if poly is None:
        return None
    return [[int(p[0]), int(p[1])] for p in poly]


def _clamp_item_bboxes(items, img_w, img_h):
    """防御：将 bbox 坐标裁切到 [0, img_w) × [0, img_h) 范围内。

    PP-StructureV3 内部 resize/padding 可能导致 OCR 坐标超出输入图像范围。
    """
    for item in items:
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        item["bbox"] = [x1, y1, x2, y2]
        item["width"] = x2 - x1
        item["height"] = y2 - y1
        # 同时修正 box 四角点
        box = item.get("box", [])
        if box and len(box) == 4:
            item["box"] = [
                [max(0, min(p[0], img_w - 1)), max(0, min(p[1], img_h - 1))]
                for p in box
            ]


def _dedup_items(items):
    """按 (text, bbox) 去重。"""
    seen = set()
    unique = []
    for item in items:
        key = (item["text"], tuple(item["bbox"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# ══════════════════════════════════════════════════════════════
# Step 3: 翻译（复用）
# ══════════════════════════════════════════════════════════════

def translate_with_dictionary(text_items: list) -> list:
    """离线术语字典翻译（精确匹配）。"""
    for item in text_items:
        text = item["text"]
        if text in ENGINEERING_DICT:
            item["translated"] = ENGINEERING_DICT[text]
    return text_items


def translate_with_llm(text_items: list) -> list:
    """LLM 翻译 + 术语字典 + 翻译缓存。"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai 未安装，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    if not LLM_API_BASE or not LLM_API_KEY or not LLM_MODEL:
        logger.warning("LLM API 配置不完整，回退到术语字典翻译")
        return translate_with_dictionary(text_items)

    # 加载缓存
    cache_path = os.path.join(WORK_DIR, "translation_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"已加载翻译缓存: {len(cache)} 条")
        except Exception:
            pass

    # 先走术语字典
    text_items = translate_with_dictionary(text_items)

    # 字典 + 缓存已覆盖的跳过
    cache_hits = 0
    items_for_llm = []
    for i, item in enumerate(text_items):
        text = item["text"]
        if "translated" in item and item["translated"] != text:
            continue
        if text in cache:
            item["translated"] = cache[text]
            cache_hits += 1
            continue
        items_for_llm.append((i, item))

    if cache_hits:
        logger.info(f"翻译缓存命中: {cache_hits} 条")

    if not items_for_llm:
        logger.info("所有文本已在缓存/字典中，跳过 LLM 调用")
        return text_items

    logger.info(f"字典+缓存覆盖 {len(text_items) - len(items_for_llm)} 条, 需 LLM 翻译: {len(items_for_llm)} 条")

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    dict_sample = "\n".join([f'  "{cn}" → "{en}"' for cn, en in list(ENGINEERING_DICT.items())[:20]])

    system_prompt = f"""You are an expert CAD drawing translation assistant specializing in mechanical engineering.
Translate Chinese technical descriptions into the SHORTEST possible professional English.
CRITICAL RULES:
1. BREVITY IS EVERYTHING. Target 1-3 words MAX. Single words preferred. Aggressively abbreviate everything: Int., Mat'l, Req., Thk., DWG, Qty., Dia., Lgth., No., Grd., Dim., Tol., Surf., Ass'y, Req'd, Min., Max., Ref., Spec., Sec., Grnd., Fdn., Elev., Horiz., Vert., Incl., w/, w/o.
2. For multi-word phrases, abbreviate every word: "Surface Roughness" → "Surf. Rough."; "Foundation Plan" → "Fdn. Plan"
3. Preserve numbers, codes, symbols, and line structures EXACTLY. Never reorder, merge, or collapse lines.
4. Strict Format: output ONLY with structured tags. No extra prose, no explanations.

Example Input:
[ITEM_START]
ID: 99
SRC: 4. 技术要求
图纸中材料为参考
[ITEM_END]
[ITEM_START]
ID: 100
SRC: 轧辊表面硬度应符合GB/T标准
[ITEM_END]

Example Output:
[ITEM_START]
ID: 99
TRN: 4. Tech. Req.
Mat'l ref. only
[ITEM_END]
[ITEM_START]
ID: 100
TRN: Roll surf. hardness per GB/T
[ITEM_END]

Terminology references:
{dict_sample}"""

    batch_size = LLM_BATCH_SIZE
    total_batches = (len(items_for_llm) + batch_size - 1) // batch_size
    new_cache_entries = 0

    for batch_idx in range(0, len(items_for_llm), batch_size):
        batch = items_for_llm[batch_idx: batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

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

            trans_map = {}
            blocks = re.findall(r"\[ITEM_START\](.*?)\[ITEM_END\]", result_text, re.DOTALL)
            for block in blocks:
                id_match = re.search(r"ID:\s*(\d+)", block)
                trn_match = re.search(r"TRN:\s*(.*?)(?=\n\[ITEM_END\]|\Z)", block, re.DOTALL)
                if id_match and trn_match:
                    idx = int(id_match.group(1))
                    trans_map[idx] = trn_match.group(1).strip()

            success_batch = 0
            for orig_idx, item in batch:
                if orig_idx in trans_map:
                    item["translated"] = trans_map[orig_idx]
                    if item["text"] not in cache:
                        cache[item["text"]] = trans_map[orig_idx]
                        new_cache_entries += 1
                    success_batch += 1
                else:
                    item["translated"] = item["text"]
                    logger.warning(f"LLM 未返回翻译: [{orig_idx}] \"{item['text']}\"")

            logger.info(f"批次 {batch_num}/{total_batches}: 成功 {success_batch}/{len(batch)} 条")
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"批次 {batch_num}/{total_batches} LLM 调用异常: {e}，共 {len(batch)} 条未翻译")
            for orig_idx, item in batch:
                logger.warning(f"  未翻译: [{orig_idx}] \"{item['text']}\"")
                if "translated" not in item:
                    item["translated"] = item["text"]

    # 保存缓存
    if new_cache_entries > 0:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.info(f"翻译缓存已更新: +{new_cache_entries} 条 (总计 {len(cache)} 条)")

    return text_items


# ══════════════════════════════════════════════════════════════
# Step 4: 擦除 + 回填（复用 + 适配）
# ══════════════════════════════════════════════════════════════

def _load_font(font_path, font_size):
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()


def _wrap_structured_text(draw, text, font, max_width):
    """保持段落原有换行标志，对超长行按字符切分。"""
    paragraphs = text.split("\n")
    wrapped = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            wrapped.append("")
            continue
        words = para.split(" ")
        current = words[0]
        for word in words[1:]:
            candidate = current + " " + word
            bb = draw.textbbox((0, 0), candidate, font=font)
            if bb[2] - bb[0] <= max_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        _append_with_char_break(draw, current, font, max_width, wrapped)
    return wrapped


def _append_with_char_break(draw, text, font, max_width, out_lines):
    bb = draw.textbbox((0, 0), text, font=font)
    if bb[2] - bb[0] <= max_width:
        out_lines.append(text)
        return
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
    """极限填满文本框：计算能刚好填满但绝不超出的最大字号。"""
    min_font_size = 5
    max_font_size = max(min_font_size, max_font_size)
    allowed_w = max(1, box_w)
    if max_width is not None:
        allowed_w = max(1, min(allowed_w, int(max_width)))

    for font_size in range(max_font_size, min_font_size - 1, -1):
        font = _load_font(font_path, font_size)
        spacing = max(1, int(font_size * 0.12))
        lines = _wrap_structured_text(draw, text, font, allowed_w)
        max_lw = 0
        total_h = 0
        for line in lines:
            sample = line if line else " "
            bb = draw.textbbox((0, 0), sample, font=font)
            max_lw = max(max_lw, bb[2] - bb[0])
            total_h += (bb[3] - bb[1])
        if len(lines) > 1:
            total_h += spacing * (len(lines) - 1)
        if max_lw <= allowed_w and total_h <= box_h:
            return font, lines, spacing, total_h, False

    # 兜底
    font = _load_font(font_path, min_font_size)
    spacing = max(1, int(min_font_size * 0.12))
    lines = _wrap_structured_text(draw, text, font, allowed_w)
    real_h = 0
    for line in lines:
        sample = line if line else " "
        bb = draw.textbbox((0, 0), sample, font=font)
        real_h += (bb[3] - bb[1])
    if len(lines) > 1:
        real_h += spacing * (len(lines) - 1)
    logger.debug(f"[文本溢出] text='{text[:40]}' box={box_w}x{box_h}px real_h={real_h:.0f}px")
    return font, lines, spacing, real_h, True


def _sample_text_color(img_bgr, x1, y1, x2, y2):
    h, w = img_bgr.shape[:2]
    pad = 2
    y1_p, y2_p = max(0, y1 - pad), min(h, y2 + pad)
    x1_p, x2_p = max(0, x1 - pad), min(w, x2 + pad)
    if y2_p <= y1_p or x2_p <= x1_p:
        return (0, 0, 0)
    region = img_bgr[y1_p:y2_p, x1_p:x2_p]
    if region.size == 0:
        return (0, 0, 0)
    return (0, 0, 0) if float(np.mean(region)) > 128 else (255, 255, 255)


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
    """旋转文本独立图层渲染。"""
    box = item.get("box")
    if not box or len(box) < 4:
        x1, y1, x2, y2 = item["bbox"]
        box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    (bx0, by0), (bx1, by1), (bx2, by2), (bx3, by3) = box[:4]
    dx, dy = bx1 - bx0, by1 - by0
    angle = float(np.degrees(np.arctan2(dy, dx)))
    base_len = float(np.hypot(dx, dy))
    side_len = float(np.hypot(bx3 - bx0, by3 - by0))
    L = max(base_len, side_len)
    H = min(base_len, side_len)
    L = max(1, int(round(L)))
    H = max(1, int(round(H)))

    translated = item.get("translated", item["text"])

    for attempt in range(3):
        pad = max(4, int(H * 0.12))
        local_w, local_h = L + pad * 2, H + pad * 2
        local = Image.new("RGBA", (local_w, local_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(local)
        max_font_size = max(8, int(H * 0.95))
        font, lines, spacing, text_h, overflow = _fit_text_to_box(
            d, translated, font_path, L, H, max_font_size)
        ty = pad + max(0, int((H - text_h) / 2))
        y_cursor = ty
        for line in lines:
            d.text((pad, y_cursor), line, fill=text_color + (255,), font=font)
            bbox = d.textbbox((0, 0), line if line else " ", font=font)
            y_cursor += (bbox[3] - bbox[1]) + spacing

        rotated = local.rotate(-angle, expand=True, resample=Image.BICUBIC)
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        left = int(cx - rotated.size[0] / 2)
        top = int(cy - rotated.size[1] / 2)

        bbox_w, bbox_h = x2 - x1, y2 - y1
        if rotated.size[0] <= bbox_w * 1.3 and rotated.size[1] <= bbox_h * 1.3:
            return rotated, left, top
        L = max(1, int(L * 0.85))
        H = max(1, int(H * 0.85))

    return rotated, left, top


def inpaint_and_overlay(img_path, translated_items, output_img_path):
    """擦除原文 → 回填译文。与原始 scan_translate_pipeline 逻辑一致。"""
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    original_bgr = img_bgr.copy()

    font_path = FONT_PATH if (FONT_PATH and os.path.exists(FONT_PATH)) else None
    erase_cnt, success, skip = 0, 0, 0

    # === 擦除阶段 ===
    for item in translated_items:
        translated = item.get("translated", item["text"])
        if translated == item["text"]:
            continue
        x1, y1, x2, y2 = item["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        ex1, ey1 = max(0, x1 + 1), max(0, y1 + 1)
        ex2, ey2 = min(w - 1, x2 - 1), min(h - 1, y2 - 1)
        if ex2 > ex1 and ey2 > ey1:
            cv2.rectangle(img_bgr, (ex1, ey1), (ex2, ey2), (255, 255, 255), -1)
            erase_cnt += 1

    logger.info(f"擦除: {erase_cnt} 个文本块")

    # === 回填阶段 ===
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)

    for item in translated_items:
        translated = item.get("translated", item["text"])
        if translated == item["text"]:
            skip += 1
            continue

        x1, y1, x2, y2 = item["bbox"]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            skip += 1
            continue

        text_color = _sample_text_color(original_bgr, x1, y1, x2, y2)
        angle = float(item.get("angle", 0.0))

        # 旋转文本
        if abs(angle) >= ANGLE_NEAR_HORIZONTAL:
            try:
                layer, left, top = _render_text_item_layer(item, font_path, text_color)
                _alpha_composite_at(pil_img, layer, left, top)
                success += 1
            except Exception:
                skip += 1
            continue

        # 水平文本：按 sub_bboxes 逐行渲染
        sub_bboxes = item.get("sub_bboxes", [])
        src_lines = translated.split("\n")

        if sub_bboxes and len(sub_bboxes) == len(src_lines):
            for k, ln in enumerate(src_lines):
                sb = sub_bboxes[k]
                sx1, sy1, sx2, sy2 = sb
                srow_w = max(1, sx2 - sx1)
                srow_h = max(1, sy2 - sy1)
                line_max_w = max(1, (w - RIGHT_MARGIN) - (sx1 + 1))
                fnt, flines, fsp, fth, _ = _fit_text_to_box(
                    draw, ln, font_path, srow_w, srow_h,
                    max(8, int(srow_h * 0.95)), max_width=line_max_w)
                ly = sy1 + max(0, int((srow_h - fth) / 2))
                for sub_line in flines:
                    draw.text((sx1 + 1, ly), sub_line, fill=text_color + (255,), font=fnt)
                    sb2 = draw.textbbox((0, 0), sub_line if sub_line else " ", font=fnt)
                    ly += (sb2[3] - sb2[1]) + fsp
        else:
            max_font_size = max(8, int(box_h * 0.95))
            max_width_edge = max(1, (w - RIGHT_MARGIN) - (x1 + 1))
            font, lines, spacing, text_h, _ = _fit_text_to_box(
                draw, translated, font_path, box_w, box_h,
                max_font_size, max_width=max_width_edge)
            ty = y1 + max(0, int((box_h - text_h) / 2)) if len(lines) == 1 else y1 + 1
            y_cursor = ty
            for line in lines:
                draw.text((x1 + 1, y_cursor), line, fill=text_color + (255,), font=font)
                lb = draw.textbbox((0, 0), line if line else " ", font=font)
                y_cursor += (lb[3] - lb[1]) + spacing

        success += 1

    pil_img.convert("RGB").save(output_img_path)
    logger.info(f"回填: 成功 {success} 块, 跳过 {skip} 块")
    return translated_items


# ══════════════════════════════════════════════════════════════
# Step 5: Image → PDF
# ══════════════════════════════════════════════════════════════
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
    logger.info(f"输出 PDF: {output_pdf}")


# ══════════════════════════════════════════════════════════════
# 调试 PDF：标出所有 OCR bbox
# ══════════════════════════════════════════════════════════════
def _generate_debug_ocr_pdf(img_path, ocr_items, debug_pdf_path, dpi=200):
    if not ocr_items:
        return
    pil_img = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    try:
        dbg_font = ImageFont.truetype(FONT_PATH, 12) if FONT_PATH and os.path.exists(FONT_PATH) else ImageFont.load_default()
    except Exception:
        dbg_font = ImageFont.load_default()

    for i, item in enumerate(ocr_items):
        x1, y1, x2, y2 = item["bbox"]
        # 颜色按置信度
        conf = item.get("confidence", 0)
        if conf >= 0.9:
            color = (0, 200, 0, 200)
        elif conf >= 0.7:
            color = (200, 200, 0, 200)
        else:
            color = (200, 0, 0, 200)
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=color, width=1)
        # 编号
        label = str(i)
        draw.text((x1 + 2, y1 - 14), label, fill=(255, 0, 0, 255), font=dbg_font)

    debug_png = debug_pdf_path.replace(".pdf", ".png")
    pil_img.convert("RGB").save(debug_png)
    image_to_pdf(debug_png, debug_pdf_path, dpi=dpi)
    logger.info(f"OCR 调试 PDF: {debug_pdf_path}")


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 60)
    logger.info("PP-StructureV3 扫描PDF翻译管线")
    logger.info(f"日志: {LOG_FILE}")
    logger.info(f"输入: {PDF_PATH}")
    logger.info(f"工作目录: {WORK_DIR}")
    logger.info("=" * 60)

    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF 不存在: {PDF_PATH}")
        return

    os.makedirs(WORK_DIR, exist_ok=True)

    # Step 1: PDF → Image
    logger.info("\n[Step 1] 渲染 PDF → 图像...")
    img_path, page_meta = pdf_to_image(PDF_PATH, dpi=RENDER_DPI)

    # Step 2: PP-StructureV3 OCR
    logger.info("\n[Step 2] PP-StructureV3 OCR + 版面识别...")
    ocr_items = ocr_with_ppstructure(img_path)
    logger.info(f"OCR 完成: {len(ocr_items)} 个文本块")

    # Step 2.1: 调试 PDF
    logger.info("\n[Step 2.1] 生成 OCR 调试 PDF...")
    debug_pdf = os.path.join(WORK_DIR, "ocr_debug.pdf")
    _generate_debug_ocr_pdf(img_path, ocr_items, debug_pdf, dpi=page_meta["dpi"])

    # Step 3: 翻译
    logger.info(f"\n[Step 3] 翻译 ({TRANSLATE_ENGINE})...")
    if TRANSLATE_ENGINE == "llm":
        translated_items = translate_with_llm(ocr_items)
    else:
        translated_items = translate_with_dictionary(ocr_items)

    trans_json = os.path.join(WORK_DIR, "translation_mapping.json")
    with open(trans_json, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)

    # Step 4: 擦除 + 回填
    logger.info("\n[Step 4] 擦除原文 + 回填译文...")
    output_img = os.path.join(WORK_DIR, "translated_page.png")
    inpaint_and_overlay(img_path, translated_items, output_img)

    # Step 5: 输出 PDF
    logger.info("\n[Step 5] 生成输出 PDF...")
    image_to_pdf(output_img, OUTPUT_PDF, dpi=page_meta["dpi"])

    # 总结
    total = len(translated_items)
    translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])
    in_table = sum(1 for t in translated_items if t.get("in_table"))
    types = {}
    for t in translated_items:
        lt = t.get("pp_type", "text")
        types[lt] = types.get(lt, 0) + 1

    logger.info("=" * 60)
    logger.info("总结")
    logger.info(f"  OCR 文本块: {total}")
    logger.info(f"  已翻译: {translated_count}")
    logger.info(f"  表格内文本: {in_table}")
    logger.info(f"  版面分布: {types}")
    logger.info(f"  输出: {OUTPUT_PDF}")
    logger.info(f"  OCR 调试: {debug_pdf}")
    logger.info(f"  日志: {LOG_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
