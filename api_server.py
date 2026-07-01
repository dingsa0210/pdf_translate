"""
PDF Translate API Server - CAD 图纸中文→英文翻译

自动检测 PDF 类型（矢量型 / 扫描型），路由到对应管线完成翻译→回填全流程。

- 矢量型 PDF：PyMuPDF 提取文本 → LLM/字典翻译 → 原位白底擦除+回填英文
- 扫描型 PDF：OCR → 翻译 → OCR合并框 / OpenCV单元格 双方法回填对比

启动: conda run -n modelscope uvicorn api_server:app --host 0.0.0.0 --port 8000
访问: http://localhost:8000/docs
"""
import os, sys, json, shutil, secrets, time, threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
import uvicorn
import cv2
import fitz as fitz_standalone  # 用于 PDF 类型检测
import config

# 配置 logger（FastAPI 上下文，控制台输出）
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ── 导入扫描型 pipeline（OCR 路线）──────────────────────────
import importlib.util

_base = os.path.dirname(os.path.abspath(__file__))

# scan pipeline
_scan_path = os.path.join(_base, "scan_translate_pipeline.py")
_scan_spec = importlib.util.spec_from_file_location("scan_pipeline", _scan_path)
scan_pipeline = importlib.util.module_from_spec(_scan_spec)
sys.modules["scan_pipeline"] = scan_pipeline
_scan_spec.loader.exec_module(scan_pipeline)

# 扫描管线符号
_scan_pdf_to_image           = scan_pipeline.pdf_to_image
_scan_ocr_with_rapid_chunked = scan_pipeline.ocr_with_rapid_chunked
_scan_merge_ocr_items        = scan_pipeline.merge_ocr_items
_scan_translate_with_llm     = scan_pipeline.translate_with_llm
_scan_translate_with_dictionary = scan_pipeline.translate_with_dictionary
_scan_inpaint_and_overlay    = scan_pipeline.inpaint_and_overlay
_scan_inpaint_overlay_cell_based = scan_pipeline._inpaint_overlay_cell_based
_scan_generate_debug_ocr_pdf    = scan_pipeline._generate_debug_ocr_pdf
_scan_generate_debug_cell_pdf   = scan_pipeline._generate_debug_cell_pdf
_scan_image_to_pdf           = scan_pipeline.image_to_pdf
_scan_clear_cell_registry    = scan_pipeline._clear_cell_registry
SCAN_CHUNK_SIZE              = scan_pipeline.CHUNK_SIZE

# vector pipeline
_vector_path = os.path.join(_base, "vector_translate_pipeline.py")
_vector_spec = importlib.util.spec_from_file_location("vector_pipeline", _vector_path)
vector_pipeline = importlib.util.module_from_spec(_vector_spec)
sys.modules["vector_pipeline"] = vector_pipeline
_vector_spec.loader.exec_module(vector_pipeline)

# 矢量管线符号
_vector_extract_text_info     = vector_pipeline.extract_text_info
_vector_translate_with_llm    = vector_pipeline.translate_with_llm
_vector_translate_with_dictionary = vector_pipeline.translate_with_dictionary
_vector_trim_vertical         = vector_pipeline.trim_vertical_translations
_vector_redact_and_refill     = vector_pipeline.redact_and_refill
_vector_has_chinese           = vector_pipeline.has_chinese
_vector_generate_report       = vector_pipeline.generate_vector_report

TRANSLATE_ENGINE = config.TRANSLATE_ENGINE

# ── 应用初始化 ──────────────────────────────────────────────
app = FastAPI(
    title="PDF Translate API",
    description="""
## CAD 图纸 PDF 中文→英文翻译服务

### 功能
- 上传 PDF 文件，**自动检测类型**（矢量型 / 扫描型），择最优管线处理
- **矢量型 PDF**：直接提取矢量文本 → LLM/字典翻译 → 原位擦除+回填（保留字号/颜色/方向）
- **扫描型 PDF**：OCR 识别 → 翻译 → 双方法回填对比（OCR合并框 / OpenCV单元格）

### 使用方法
1. `POST /api/upload` — 上传 PDF（自动检测类型）
2. `POST /api/process/{task_id}` — 启动处理
3. `GET /api/status/{task_id}` — 查看进度
4. `GET /api/download/{task_id}/{filename}` — 下载结果
""",
    version="2.0.0",
)

# ── 启动事件：启动定期清理后台线程 ──────────────────────────────
@app.on_event("startup")
async def _startup_cleanup_daemon():
    thread = threading.Thread(target=_cleanup_daemon, daemon=True, name="cleanup-daemon")
    thread.start()

# 任务存储目录
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = BASE_DIR / "api_tasks"
TASKS_DIR.mkdir(exist_ok=True)

# 旧任务清理周期（天）
TASK_RETENTION_DAYS = 7

# 内存中的任务状态
tasks: dict = {}
_tasks_lock = threading.Lock()   # 保护 tasks 字典的并发读写

# 扫描管线串行锁：scan_translate_pipeline.py 内部使用模块级全局变量
# WORK_DIR / PDF_PATH / 翻译缓存 / 单元格注册表，并发会相互覆盖污染。
# 矢量管线无此问题，可并行执行。
_scan_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════
# 定期清理过期任务目录
# ══════════════════════════════════════════════════════════════

def _cleanup_old_tasks(retention_days: int = TASK_RETENTION_DAYS) -> int:
    """删除 api_tasks/ 下超过 retention_days 天的任务目录。返回删除数量。"""
    deleted = 0
    now = time.time()
    cutoff = now - retention_days * 86400
    if not TASKS_DIR.exists():
        return 0
    for entry in TASKS_DIR.iterdir():
        if entry.is_dir():
            try:
                mtime = entry.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(entry)
                    deleted += 1
                    logger.info(f"清理过期任务目录: {entry.name} (修改于 {datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M})")
            except Exception as e:
                logger.warning(f"清理任务目录 {entry.name} 失败: {e}")
    return deleted


def _cleanup_daemon(interval_seconds: int = 86400):
    """后台守护线程：定期清理过期任务目录（默认每24小时）。"""
    # 启动时立即执行一次
    deleted = _cleanup_old_tasks()
    if deleted:
        logger.info(f"启动时清理: 已删除 {deleted} 个过期任务目录")
    while True:
        time.sleep(interval_seconds)
        try:
            deleted = _cleanup_old_tasks()
            if deleted:
                logger.info(f"定期清理: 已删除 {deleted} 个过期任务目录")
        except Exception as e:
            logger.error(f"定期清理异常: {e}")


# ══════════════════════════════════════════════════════════════
# PDF 类型检测
# ══════════════════════════════════════════════════════════════

def detect_pdf_type(pdf_path: str) -> str:
    """
    检测 PDF 为矢量型（含可提取的中文文本）还是扫描型（图片）。
    抽样检查前 3 页，任一页提取到中文即判定为矢量型。

    Returns:
        "vector" — 矢量型，有可选择/可复制的矢量中文文本
        "scan"   — 扫描型，需要 OCR 处理
    """
    doc = fitz_standalone.open(pdf_path)
    try:
        pages_to_check = min(doc.page_count, 3)
        for page_idx in range(pages_to_check):
            page = doc[page_idx]
            text = page.get_text("text")
            if _vector_has_chinese(text):
                logger.info(f"[Detect] 第 {page_idx + 1} 页发现中文矢量文本 → 矢量型 PDF")
                return "vector"
        logger.info(f"[Detect] 前 {pages_to_check} 页无中文矢量文本 → 扫描型 PDF")
        return "scan"
    finally:
        doc.close()


# ══════════════════════════════════════════════════════════════
# 后台管线
# ══════════════════════════════════════════════════════════════

def _run_scan_pipeline(pdf_path: str, work_dir: str, task_id: str):
    """在后台运行扫描型 PDF 翻译管线（OCR 路线 + 双方法对比）。

    注意：scan_translate_pipeline.py 内部使用模块级全局变量
    WORK_DIR / PDF_PATH / 翻译缓存 / 单元格注册表，因此对所有扫描任务
    加串行锁 _scan_lock，防止并发时相互覆盖污染。"""
    task = tasks.get(task_id)
    if not task:
        return

    # 扫描管线串行执行（保护模块级全局变量 WORK_DIR / PDF_PATH / 翻译缓存）
    with _scan_lock:
        # 在锁内设置模块全局变量，确保并发时不会互相覆盖
        scan_pipeline.WORK_DIR = work_dir
        scan_pipeline.PDF_PATH = pdf_path
        _scan_clear_cell_registry()

        task["status"] = "processing"
        task["progress"] = "初始化扫描管线..."

        try:
            dpi = config.RENDER_DPI

            # Step 1: 渲染
            task["progress"] = "渲染PDF为图像..."
            img_path, page_meta = _scan_pdf_to_image(pdf_path, dpi=dpi)

            # Step 2: OCR
            task["progress"] = "RapidOCR 识别..."
            raw_ocr_items = _scan_ocr_with_rapid_chunked(img_path, chunk_size=SCAN_CHUNK_SIZE)
            task["ocr_count"] = len(raw_ocr_items)

            # Step 2.1: OCR调试PDF
            task["progress"] = "生成OCR调试PDF..."
            ocr_debug_pdf = os.path.join(work_dir, "ocr_debug.pdf")
            _scan_generate_debug_ocr_pdf(img_path, raw_ocr_items, ocr_debug_pdf, dpi=dpi)

            # Step 2.5: 合并
            task["progress"] = "智能文本块合并..."
            merge_img = cv2.imread(img_path)
            ocr_items = _scan_merge_ocr_items(raw_ocr_items, img_bgr=merge_img)

            ocr_json = os.path.join(work_dir, "ocr_result.json")
            with open(ocr_json, "w", encoding="utf-8") as f:
                json.dump(ocr_items, f, ensure_ascii=False, indent=2)

            # Step 2.6: 合并后OCR调试PDF
            ocr_merged_debug_pdf = os.path.join(work_dir, "ocr_debug_merged.pdf")
            _scan_generate_debug_ocr_pdf(img_path, ocr_items, ocr_merged_debug_pdf, dpi=dpi)

            # Step 3: 翻译
            task["progress"] = "翻译中..."
            if TRANSLATE_ENGINE == "llm":
                translated_items = _scan_translate_with_llm(ocr_items)
            else:
                translated_items = _scan_translate_with_dictionary(ocr_items)

            trans_json = os.path.join(work_dir, "translation_mapping.json")
            with open(trans_json, "w", encoding="utf-8") as f:
                json.dump(translated_items, f, ensure_ascii=False, indent=2)

            translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])
            task["translated_count"] = translated_count

            # Step 4: OCR Box 方法回填（主输出）
            task["progress"] = "OCR Box 方法回填..."
            output_img = os.path.join(work_dir, "translated_page.png")
            _scan_inpaint_and_overlay(img_path, translated_items, output_img)
            output_pdf = os.path.join(work_dir, "output_ocr_box.pdf")
            _scan_image_to_pdf(output_img, output_pdf, dpi=dpi)

            # Step 5: Cell-Based 方法回填（对比输出）
            task["progress"] = "Cell-Based 方法回填..."
            output_cell_img = os.path.join(work_dir, "translated_page_cell_based.png")
            _scan_inpaint_overlay_cell_based(img_path, translated_items, output_cell_img)
            output_cell_pdf = os.path.join(work_dir, "output_cell_based.pdf")
            _scan_image_to_pdf(output_cell_img, output_cell_pdf, dpi=dpi)

            # Cell debug PDF
            task["progress"] = "生成Cell调试PDF..."
            cell_debug_pdf = os.path.join(work_dir, "cell_debug.pdf")
            _scan_generate_debug_cell_pdf(output_img, cell_debug_pdf, dpi=dpi)

            # 收集结果文件
            task["result_files"] = {
                "output_ocr_box.pdf": {
                    "path": output_pdf,
                    "description": "主输出 - OCR合并框回填方法（推荐）",
                },
                "output_cell_based.pdf": {
                    "path": output_cell_pdf,
                    "description": "对照组 - OpenCV单元格回填方法",
                },
                "ocr_debug.pdf": {
                    "path": ocr_debug_pdf,
                    "description": "OCR原始识别框（合并前，颜色=置信度）",
                },
                "ocr_debug_merged.pdf": {
                    "path": ocr_merged_debug_pdf,
                    "description": "OCR合并后文本框",
                },
                "cell_debug.pdf": {
                    "path": cell_debug_pdf,
                    "description": "单元格检测框（红框+编号）",
                },
                "ocr_result.json": {
                    "path": ocr_json,
                    "description": "OCR识别+合并结果JSON",
                },
                "translation_mapping.json": {
                    "path": trans_json,
                    "description": "翻译映射JSON",
                },
                "rendered_image.png": {
                    "path": img_path,
                    "description": "PDF渲染原图",
                },
            }

            # 生成对比报告
            compare_report = os.path.join(work_dir, "comparison_report.md")
            _generate_scan_comparison_report(compare_report, task)
            task["result_files"]["comparison_report.md"] = {
                "path": compare_report,
                "description": "双方法对比报告",
            }

            task["status"] = "completed"
            task["progress"] = "完成"

        except Exception as e:
            import traceback
            task["status"] = "failed"
            task["error"] = str(e)
            task["traceback"] = traceback.format_exc()


def _run_vector_pipeline(pdf_path: str, work_dir: str, task_id: str):
    """在后台运行矢量型 PDF 翻译管线（PyMuPDF 提取 + 原位擦除回填）。"""
    task = tasks.get(task_id)
    if not task:
        return
    task["status"] = "processing"
    task["progress"] = "初始化矢量管线..."

    try:
        # Step 1: 提取矢量文本
        task["progress"] = "提取矢量中文文本..."
        text_items = _vector_extract_text_info(pdf_path)
        task["text_count"] = len(text_items)

        extracted_json = os.path.join(work_dir, "extracted_text.json")
        with open(extracted_json, "w", encoding="utf-8") as f:
            json.dump(text_items, f, ensure_ascii=False, indent=2)

        if not text_items:
            task["status"] = "completed"
            task["progress"] = "完成（无中文文本）"
            task["result_files"] = {
                "extracted_text.json": {
                    "path": extracted_json,
                    "description": "提取结果（无中文文本）",
                },
            }
            task["translated_count"] = 0
            return

        # Step 2: 翻译
        task["progress"] = "翻译中..."
        if TRANSLATE_ENGINE == "llm":
            translated_items = _vector_translate_with_llm(text_items)
        else:
            translated_items = _vector_translate_with_dictionary(text_items)

        translated_items = _vector_trim_vertical(translated_items)

        translation_json = os.path.join(work_dir, "translation_mapping.json")
        with open(translation_json, "w", encoding="utf-8") as f:
            json.dump(translated_items, f, ensure_ascii=False, indent=2)

        translated_count = sum(
            1 for item in translated_items
            if item.get("translated", item["text"]) != item["text"]
        )
        task["translated_count"] = translated_count

        # Step 3: 擦除 + 回填
        task["progress"] = "原位擦除+回填英文..."
        output_pdf = os.path.join(work_dir, "output_vector.pdf")
        stats = _vector_redact_and_refill(
            pdf_path=pdf_path,
            output_path=output_pdf,
            text_items=translated_items,
        )

        # 收集结果文件
        task["result_files"] = {
            "output_vector.pdf": {
                "path": output_pdf,
                "description": "主输出 - 矢量PDF翻译结果（推荐）",
            },
            "extracted_text.json": {
                "path": extracted_json,
                "description": "提取到的中文矢量文本+坐标",
            },
            "translation_mapping.json": {
                "path": translation_json,
                "description": "翻译映射JSON",
            },
        }

        # 生成报告
        report_path = os.path.join(work_dir, "vector_report.md")
        task_info = {
            "input_pdf": pdf_path,
            "text_count": len(text_items),
            "translated_count": translated_count,
            "stats": stats,
        }
        _vector_generate_report(report_path, task_info)
        task["result_files"]["vector_report.md"] = {
            "path": report_path,
            "description": "矢量管线处理报告",
        }

        task["status"] = "completed"
        task["progress"] = "完成"
        task["stats"] = stats

    except Exception as e:
        import traceback
        task["status"] = "failed"
        task["error"] = str(e)
        task["traceback"] = traceback.format_exc()


# ══════════════════════════════════════════════════════════════
# 报告生成
# ══════════════════════════════════════════════════════════════

def _generate_scan_comparison_report(report_path: str, task: dict):
    """生成扫描型管线双方法对比报告。"""
    lines = [
        "# PDF Translate - 扫描型双方法对比报告",
        "",
        f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**PDF类型**: scan",
        f"**OCR识别块数**: {task.get('ocr_count', 'N/A')}",
        f"**翻译块数**: {task.get('translated_count', 'N/A')}",
        "",
        "## 输出文件",
        "",
        "| 文件 | 方法 | 说明 |",
        "|------|------|------|",
        "| `output_ocr_box.pdf` | OCR合并框 | **推荐** - 擦除和回填均基于OCR合并文本框 |",
        "| `output_cell_based.pdf` | OpenCV单元格 | 对照组 - 擦除和回填基于连通域检测的表格单元格 |",
        "| `ocr_debug.pdf` | - | OCR原始识别框（合并前） |",
        "| `ocr_debug_merged.pdf` | - | OCR合并后文本框 |",
        "| `cell_debug.pdf` | - | OpenCV检测的单元格框 |",
        "",
        "## 方法差异",
        "",
        "| 维度 | OCR Box Method | Cell-Based Method |",
        "|------|---------------|-------------------|",
        "| 擦除范围 | OCR合并bbox内缩1px | 单元格整块 / bbox外扩3px |",
        "| 回填基准 | item.sub_bboxes 原文坐标 | cell边界 / bbox |",
        "| 多行处理 | 逐行恢复原始sub_bbox位置 | 格内按比例分配行高 |",
        "| 表格线保护 | 仅擦除文字区域 | 内缩保护格线 |",
        "| 适用场景 | 所有文本类型统一处理 | 表格+非表格分治 |",
        "",
        "## 建议",
        "",
        "- 对比两个输出，重点关注表格区域的翻译文本是否超出单元格、是否遮挡格线",
        "- OCR Box Method 通常能更精确地保留原文排版",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── API 端点 ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """交互式 Web 翻译界面"""
    return _WEB_APP_HTML


# ══════════════════════════════════════════════════════════════
# Web 应用前端（内联单页 HTML）
# ══════════════════════════════════════════════════════════════

_WEB_APP_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF CAD 图纸翻译</title>
<style>
  :root {
    --bg: #f0f2f5;
    --card-bg: #fff;
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #1f2937;
    --text-secondary: #6b7280;
    --border: #e5e7eb;
    --shadow: 0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
    --shadow-lg: 0 10px 25px rgba(0,0,0,.1);
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #fff;
    padding: 28px 32px;
    text-align: center;
  }
  .header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -.02em; }
  .header p { font-size: .9rem; opacity: .85; margin-top: 4px; }
  .header .api-link { margin-top: 10px; }
  .header .api-link a { color: #c7d2fe; font-size: .82rem; text-decoration: none; }
  .header .api-link a:hover { text-decoration: underline; }

  /* ── Main Container ── */
  .container { max-width: 900px; margin: 0 auto; padding: 24px 20px 60px; }

  /* ── Upload Zone ── */
  .upload-zone {
    background: var(--card-bg);
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    margin-bottom: 24px;
    position: relative;
  }
  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--primary);
    background: #f5f3ff;
  }
  .upload-zone.dragover { border-style: solid; }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 12px; }
  .upload-zone .title { font-size: 1.1rem; font-weight: 600; margin-bottom: 4px; }
  .upload-zone .hint { font-size: .85rem; color: var(--text-secondary); }
  .upload-zone input[type=file] { display: none; }
  .upload-zone .limits {
    margin-top: 16px;
    display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;
  }
  .upload-zone .limits span {
    font-size: .78rem; color: var(--text-secondary);
    background: #f9fafb; padding: 4px 12px; border-radius: 20px;
  }

  /* ── Batch Actions Bar ── */
  .batch-bar {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; flex-wrap: wrap;
  }
  .btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 10px 22px; border-radius: var(--radius-sm);
    font-size: .9rem; font-weight: 600; cursor: pointer;
    border: none; transition: all .15s;
  }
  .btn-primary { background: var(--primary); color: #fff; }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
  .btn-outline {
    background: #fff; color: var(--text); border: 1.5px solid var(--border);
  }
  .btn-outline:hover { border-color: #9ca3af; }
  .btn-danger { background: #fef2f2; color: var(--danger); border: 1px solid #fecaca; }
  .btn-danger:hover { background: #fee2e2; }
  .batch-bar .count { font-size: .85rem; color: var(--text-secondary); }

  /* ── Task Cards ── */
  .task-list { display: flex; flex-direction: column; gap: 12px; }
  .task-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    display: flex; align-items: center; gap: 16px;
    box-shadow: var(--shadow);
    transition: box-shadow .2s;
  }
  .task-card:hover { box-shadow: var(--shadow-lg); }

  .task-card .icon-col {
    width: 40px; height: 40px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
  }
  .status-icon-uploading { background: #eef2ff; color: var(--primary); }
  .status-icon-processing { background: #fef9c3; color: #ca8a04; }
  .status-icon-completed { background: #d1fae5; color: #059669; }
  .status-icon-failed { background: #fee2e2; color: #dc2626; }

  .task-card .info { flex: 1; min-width: 0; }
  .task-card .info .filename {
    font-weight: 600; font-size: .9rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .task-card .info .meta {
    font-size: .78rem; color: var(--text-secondary); margin-top: 2px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: .72rem; font-weight: 600;
  }
  .badge-vector { background: #dbeafe; color: #1d4ed8; }
  .badge-scan { background: #fce7f3; color: #be185d; }

  .task-card .progress-wrap { width: 140px; flex-shrink: 0; }
  .progress-bar-bg {
    width: 100%; height: 6px; background: #f3f4f6; border-radius: 3px;
    overflow: hidden;
  }
  .progress-bar-fill {
    height: 100%; background: var(--primary);
    border-radius: 3px; transition: width .5s ease;
    width: 0%;
  }
  .progress-bar-fill.done { background: var(--success); }
  .progress-bar-fill.failed { background: var(--danger); }
  .progress-text { font-size: .72rem; color: var(--text-secondary); margin-top: 4px; }

  .task-card .actions { flex-shrink: 0; display: flex; gap: 6px; }
  .download-btn {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 6px 14px; font-size: .8rem; font-weight: 600;
    border-radius: 6px; cursor: pointer; border: none;
    text-decoration: none; transition: all .15s;
  }
  .download-btn.primary {
    background: var(--primary); color: #fff;
  }
  .download-btn.primary:hover { background: var(--primary-hover); }
  .download-btn.secondary {
    background: #f3f4f6; color: var(--text);
  }
  .download-btn.secondary:hover { background: #e5e7eb; }

  /* ── Empty State ── */
  .empty-state {
    text-align: center; padding: 48px 20px; color: var(--text-secondary);
  }
  .empty-state .icon { font-size: 3rem; margin-bottom: 12px; opacity: .5; }

  /* ── Footer ── */
  .footer {
    text-align: center; padding: 24px; color: var(--text-secondary);
    font-size: .78rem;
  }
  .footer a { color: var(--primary); text-decoration: none; }

  /* ── Pulse Animation ── */
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.5 } }
  .pulse { animation: pulse 1.8s ease-in-out infinite; }
</style>
</head>
<body>

<div class="header">
  <h1>📐 CAD 图纸 PDF 翻译</h1>
  <p>中文 → 英文 · 自动识别矢量/扫描 · 原位回填不偏移</p>
  <div class="api-link">
    <a href="/docs" target="_blank">📖 Swagger API 文档</a>
  </div>
</div>

<div class="container">

  <!-- Upload Zone -->
  <div class="upload-zone" id="uploadZone">
    <div class="icon">📤</div>
    <div class="title">点击上传或拖拽 PDF 文件到此处</div>
    <div class="hint">支持矢量型 &amp; 扫描型 CAD 图纸，自动识别类型</div>
    <input type="file" id="fileInput" accept=".pdf" multiple>
    <div class="limits">
      <span>📄 每次最多 10 个文件</span>
      <span>⚡ 自动识别矢量/扫描</span>
      <span>🔤 LLM + 术语字典双语翻译</span>
    </div>
  </div>

  <!-- Batch Actions -->
  <div class="batch-bar" id="batchBar" style="display:none">
    <button class="btn btn-primary" id="btnProcessAll" onclick="processAll()">
      ▶ 全部翻译
    </button>
    <button class="btn btn-outline" id="btnDownloadAll" onclick="downloadAll()" style="display:none">
      ⬇ 下载全部主结果
    </button>
    <button class="btn btn-danger" onclick="clearAll()">
      ✕ 清空列表
    </button>
    <span class="count" id="fileCount"></span>
  </div>

  <!-- Task Cards -->
  <div class="task-list" id="taskList"></div>
  <div class="empty-state" id="emptyState">
    <div class="icon">📋</div>
    <p>上传 PDF 文件开始翻译</p>
  </div>

</div>

<div class="footer">
  PDF Translate v2 · Powered by PyMuPDF + OpenCV + LLM
</div>

<script>
  const API = '';
  const MAX_FILES = 10;
  let tasks = [];               // { id, filename, pdfType, status, progress, translatedCount, ... }
  let processingQueue = [];
  let isProcessing = false;

  // ── DOM refs ──
  const uploadZone = document.getElementById('uploadZone');
  const fileInput  = document.getElementById('fileInput');
  const taskList   = document.getElementById('taskList');
  const emptyState = document.getElementById('emptyState');
  const batchBar   = document.getElementById('batchBar');
  const fileCount  = document.getElementById('fileCount');
  const btnProcessAll = document.getElementById('btnProcessAll');
  const btnDownloadAll = document.getElementById('btnDownloadAll');

  // ── Upload Zone Events ──
  uploadZone.addEventListener('click', () => fileInput.click());
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
    fileInput.value = '';
  });

  async function handleFiles(fileList) {
    const files = Array.from(fileList).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (tasks.length + files.length > MAX_FILES) {
      alert(`最多 ${MAX_FILES} 个文件，当前已有 ${tasks.length} 个`);
      return;
    }
    for (const file of files) {
      await uploadFile(file);
    }
    render();
    updateBatchBar();
  }

  // ── Upload single file ──
  async function uploadFile(file) {
    const tempId = 'uploading_' + Date.now() + '_' + Math.random().toString(36).slice(2,6);
    tasks.push({ id: tempId, filename: file.name, status: 'uploading', progress: '上传中...' });
    render();

    try {
      const formData = new FormData();
      formData.append('file', file);

      const resp = await fetch(API + '/api/upload', { method: 'POST', body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Upload failed');
      }
      const data = await resp.json();

      // Replace temp entry
      const idx = tasks.findIndex(t => t.id === tempId);
      if (idx >= 0) {
        tasks[idx] = {
          id: data.task_id,
          filename: data.filename,
          pdfType: data.pdf_type,
          status: 'ready',
          progress: '等待处理',
        };
      }
    } catch (e) {
      const idx = tasks.findIndex(t => t.id === tempId);
      if (idx >= 0) {
        tasks[idx] = { ...tasks[idx], status: 'failed', progress: '上传失败', error: e.message };
      }
    }
    render();
  }

  // ── Process All ──
  async function processAll() {
    const ready = tasks.filter(t => t.status === 'ready');
    if (ready.length === 0) return;
    btnProcessAll.disabled = true;

    processingQueue = ready.map(t => t.id);
    isProcessing = true;
    processQueue();
  }

  async function processQueue() {
    if (!isProcessing || processingQueue.length === 0) {
      isProcessing = false;
      btnProcessAll.disabled = false;
      updateBatchBar();
      return;
    }

    const taskId = processingQueue[0];
    const task = tasks.find(t => t.id === taskId);
    if (!task || (task.status !== 'ready' && task.status !== 'processing')) {
      processingQueue.shift();
      processQueue();
      return;
    }

    try {
      task.status = 'processing';
      task.progress = '启动管线...';
      render();

      const resp = await fetch(API + '/api/process/' + taskId, { method: 'POST' });
      if (!resp.ok) {
        throw new Error('启动处理失败');
      }

      // Poll until done
      await pollTask(taskId);
    } catch (e) {
      task.status = 'failed';
      task.progress = '处理失败';
      task.error = e.message;
    }

    processingQueue.shift();
    render();
    updateBatchBar();
    processQueue();
  }

  async function pollTask(taskId) {
    let retries = 0;
    while (true) {
      await new Promise(r => setTimeout(r, 2000));
      const task = tasks.find(t => t.id === taskId);
      if (!task) break;

      try {
        const resp = await fetch(API + '/api/status/' + taskId);
        if (!resp.ok) {
          if (++retries > 5) throw new Error('状态查询失败');
          continue;
        }
        retries = 0;
        const data = await resp.json();

        task.status = data.status;
        task.progress = data.progress;
        if (data.translated_count !== undefined) task.translatedCount = data.translated_count;
        if (data.text_count !== undefined) task.textCount = data.text_count;
        if (data.files) task.files = data.files;

        render();
        if (data.status === 'completed' || data.status === 'failed') {
          if (data.error) task.error = data.error;
          // Fetch detailed results
          if (data.status === 'completed') await fetchResults(taskId);
          break;
        }
      } catch (e) {
        if (++retries > 5) {
          task.status = 'failed';
          task.error = e.message;
          break;
        }
      }
    }
  }

  async function fetchResults(taskId) {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;
    try {
      const resp = await fetch(API + '/api/results/' + taskId);
      if (resp.ok) {
        const data = await resp.json();
        task.resultFiles = data.files;
        render();
      }
    } catch (e) { /* ignore */ }
  }

  // ── Download helpers ──
  function downloadFile(taskId, filename) {
    const a = document.createElement('a');
    a.href = API + '/api/download/' + taskId + '/' + encodeURIComponent(filename);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function getMainFile(task) {
    if (!task.resultFiles) return null;
    // vector: output_vector.pdf; scan: output_ocr_box.pdf
    const names = task.resultFiles.map(f => f.filename);
    const candidates = ['output_vector.pdf', 'output_ocr_box.pdf'];
    for (const c of candidates) {
      if (names.includes(c)) return c;
    }
    return names[0] || null;
  }

  function getComparisonFile(task) {
    if (!task.resultFiles) return null;
    if (task.pdfType !== 'scan') return null;
    const names = task.resultFiles.map(f => f.filename);
    return names.includes('output_cell_based.pdf') ? 'output_cell_based.pdf' : null;
  }

  function downloadAll() {
    for (const task of tasks) {
      if (task.status !== 'completed') continue;
      const main = getMainFile(task);
      if (main) {
        setTimeout(() => downloadFile(task.id, main), 200 * tasks.indexOf(task));
      }
    }
  }

  // ── Clear All ──
  function clearAll() {
    isProcessing = false;
    processingQueue = [];
    for (const t of tasks) {
      if (t.status === 'completed' || t.status === 'failed') {
        fetch(API + '/api/tasks/' + t.id, { method: 'DELETE' }).catch(() => {});
      }
    }
    tasks = [];
    render();
    updateBatchBar();
  }

  // ── Update Batch Bar ──
  function updateBatchBar() {
    const ready = tasks.filter(t => t.status === 'ready').length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const processing = tasks.filter(t => t.status === 'processing').length;

    if (tasks.length === 0) {
      batchBar.style.display = 'none';
    } else {
      batchBar.style.display = 'flex';
    }

    fileCount.textContent = `${tasks.length} 个文件 · ${ready} 待处理 · ${processing} 处理中 · ${completed} 已完成`;

    btnProcessAll.disabled = ready === 0 || isProcessing;
    btnDownloadAll.style.display = completed > 0 ? '' : 'none';
    if (completed > 0) btnDownloadAll.textContent = `⬇ 下载全部主结果 (${completed})`;
  }

  // ── Render Task Cards ──
  function render() {
    taskList.innerHTML = tasks.map(t => taskCardHTML(t)).join('');
    emptyState.style.display = tasks.length === 0 ? '' : 'none';
    updateBatchBar();
  }

  function taskCardHTML(task) {
    const statusIcons = {
      'uploading':  { cls: 'status-icon-uploading',  icon: '⏳' },
      'ready':      { cls: 'status-icon-processing', icon: '⏸' },
      'processing': { cls: 'status-icon-processing', icon: '⚙' },
      'completed':  { cls: 'status-icon-completed',  icon: '✓' },
      'failed':     { cls: 'status-icon-failed',     icon: '✕' },
    };
    const si = statusIcons[task.status] || statusIcons['ready'];

    // Badge
    let badge = '';
    if (task.pdfType === 'vector') badge = '<span class="badge badge-vector">矢量型</span>';
    else if (task.pdfType === 'scan') badge = '<span class="badge badge-scan">扫描型</span>';

    // Progress bar
    let progressBar = '';
    let barClass = '';
    let progressText = task.progress || '';
    if (task.status === 'processing') {
      progressBar = `<div class="progress-bar-fill pulse" style="width:60%"></div>`;
    } else if (task.status === 'completed') {
      barClass = 'done';
      progressBar = `<div class="progress-bar-fill done" style="width:100%"></div>`;
      const info = [];
      if (task.translatedCount !== undefined) info.push(`${task.translatedCount} 项已翻译`);
      progressText = info.length ? info.join(' · ') : '完成';
    } else if (task.status === 'failed') {
      barClass = 'failed';
      progressBar = `<div class="progress-bar-fill failed" style="width:100%"></div>`;
      progressText = task.error || '失败';
    } else if (task.status === 'uploading') {
      progressBar = `<div class="progress-bar-fill pulse" style="width:40%"></div>`;
    }

    // Actions (download buttons)
    let actions = '';
    if (task.status === 'completed' && task.resultFiles) {
      const mainFile = getMainFile(task);
      const cmpFile  = getComparisonFile(task);

      if (mainFile) {
        actions += `<a class="download-btn primary" href="${API}/api/download/${task.id}/${encodeURIComponent(mainFile)}" download="${mainFile}" onclick="setTimeout(()=>updateBatchBar(),500)">⬇ 下载结果</a>`;
      }
      if (cmpFile) {
        actions += `<a class="download-btn secondary" href="${API}/api/download/${task.id}/${encodeURIComponent(cmpFile)}" download="${cmpFile}" title="基于连通域检测的对比结果">🔬 对比结果</a>`;
      }
    }

    return `
      <div class="task-card">
        <div class="icon-col ${si.cls}">${si.icon}</div>
        <div class="info">
          <div class="filename" title="${esc(task.filename)}">${esc(task.filename)}</div>
          <div class="meta">
            ${badge}
            ${task.textCount !== undefined ? `<span>${task.textCount} 文本</span>` : ''}
          </div>
        </div>
        <div class="progress-wrap">
          <div class="progress-bar-bg">${progressBar}</div>
          <div class="progress-text">${esc(progressText)}</div>
        </div>
        <div class="actions">${actions}</div>
      </div>`;
  }

  function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
</script>
</body>
</html>
"""


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(..., description="要翻译的 PDF 文件（CAD 图纸，自动识别矢量/扫描类型）")):
    """上传 PDF 文件，自动检测类型，返回任务 ID 和 PDF 类型。"""
    safe_filename = Path(file.filename).name  # 防路径遍历：仅取文件名
    if not safe_filename or not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 文件")

    task_id = secrets.token_urlsafe(16)  # 128-bit 随机令牌
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = task_dir / safe_filename
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 自动检测 PDF 类型
    pdf_type = detect_pdf_type(str(pdf_path))

    tasks[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "work_dir": str(task_dir),
        "pdf_type": pdf_type,
        "status": "uploaded",
        "progress": "等待处理",
        "created_at": datetime.now().isoformat(),
    }

    return {
        "task_id": task_id,
        "filename": file.filename,
        "pdf_type": pdf_type,
        "status": "uploaded",
        "message": f"文件已上传，检测为 **{pdf_type}** 型 PDF。请调用 /api/process/{task_id} 开始处理",
    }


@app.post("/api/process/{task_id}")
async def process_task(task_id: str, background_tasks: BackgroundTasks):
    """启动后台处理管线（根据 PDF 类型自动路由）。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在，请先上传文件")

    if task["status"] == "processing":
        return {"task_id": task_id, "status": "processing", "message": "正在处理中..."}

    pdf_path = task["pdf_path"]
    work_dir = task["work_dir"]
    pdf_type = task.get("pdf_type", "scan")  # 默认按扫描处理
    os.makedirs(work_dir, exist_ok=True)

    if pdf_type == "vector":
        # 矢量管线：设置 scan_pipeline.WORK_DIR 以避免 scan_translate_pipeline 导入时的日志报错
        scan_pipeline.WORK_DIR = work_dir
        background_tasks.add_task(_run_vector_pipeline, pdf_path, work_dir, task_id)
    else:
        # 扫描管线：WORK_DIR / PDF_PATH / cell registry 在 _run_scan_pipeline 的锁内设置
        background_tasks.add_task(_run_scan_pipeline, pdf_path, work_dir, task_id)

    return {
        "task_id": task_id,
        "pdf_type": pdf_type,
        "status": "processing",
        "message": f"{pdf_type} 型管线已启动，请用 /api/status/{task_id} 查看进度",
    }


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """查询任务状态和进度。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    result = {
        "task_id": task_id,
        "filename": task.get("filename"),
        "pdf_type": task.get("pdf_type"),
        "status": task.get("status"),
        "progress": task.get("progress"),
        "text_count": task.get("text_count", task.get("ocr_count")),
        "translated_count": task.get("translated_count"),
    }

    if task.get("pdf_type") == "vector" and task.get("stats"):
        result["stats"] = task["stats"]

    if task["status"] == "failed":
        result["error"] = task.get("error", "未知错误")

    if task["status"] == "completed":
        result["files"] = list(task.get("result_files", {}).keys())

    return result


@app.get("/api/results/{task_id}")
async def list_results(task_id: str):
    """列出任务的所有结果文件。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task["status"] != "completed":
        raise HTTPException(400, f"任务未完成，当前状态: {task['status']}")

    files = []
    for name, info in task.get("result_files", {}).items():
        files.append({
            "filename": name,
            "description": info["description"],
            "download_url": f"/api/download/{task_id}/{name}",
        })

    return {"task_id": task_id, "files": files}


@app.get("/api/download/{task_id}/{filename:path}")
async def download_file(task_id: str, filename: str):
    """下载指定的结果文件。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task["status"] != "completed":
        raise HTTPException(400, f"任务未完成，当前状态: {task['status']}")

    result_files = task.get("result_files", {})
    if filename not in result_files:
        available = list(result_files.keys())
        raise HTTPException(404, f"文件不存在。可用: {available}")

    file_path = result_files[filename]["path"]
    if not os.path.exists(file_path):
        logger.debug(f"文件缺失: {file_path}")
        raise HTTPException(404, "文件不存在或已被清理")

    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/octet-stream",
    )


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务及其所有文件。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    # 清理文件
    work_dir = Path(task["work_dir"])
    if work_dir.exists():
        shutil.rmtree(work_dir)

    del tasks[task_id]
    return {"message": "任务已删除"}


# ── 启动入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    logger.debug("Starting PDF Translate API Server...")
    logger.debug("Swagger UI: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
