"""
PDF Translate API Server - CAD 图纸中文→英文翻译 + 双方法对比

提供 Swagger UI 界面上传 PDF，自动完成 OCR→翻译→回填全流程，
同时产出 OCR合并框 和 OpenCV单元格 两种回填结果用于对比。

启动: conda run -n modelscope uvicorn api_server:app --host 0.0.0.0 --port 8000
访问: http://localhost:8000/docs
"""
import os, sys, json, shutil, secrets, time
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
import uvicorn
import cv2
import config

# 配置 logger（FastAPI 上下文，控制台输出）
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# ── 导入核心 pipeline 函数 ──────────────────────────────────
# 主脚本文件名含连字符，需要用 importlib 导入
import importlib.util

_base = os.path.dirname(os.path.abspath(__file__))
_main_path = os.path.join(_base, "scan_translate_layout_fixed_gemini-code.py")
_spec = importlib.util.spec_from_file_location("pipeline", _main_path)
pipeline = importlib.util.module_from_spec(_spec)
sys.modules["pipeline"] = pipeline
_spec.loader.exec_module(pipeline)

# 重新导出常用符号
pdf_to_image           = pipeline.pdf_to_image
ocr_with_rapid_chunked = pipeline.ocr_with_rapid_chunked
merge_ocr_items        = pipeline.merge_ocr_items
translate_with_llm     = pipeline.translate_with_llm
translate_with_dictionary = pipeline.translate_with_dictionary
inpaint_and_overlay    = pipeline.inpaint_and_overlay
_inpaint_overlay_cell_based = pipeline._inpaint_overlay_cell_based
_generate_debug_ocr_pdf    = pipeline._generate_debug_ocr_pdf
_generate_debug_cell_pdf   = pipeline._generate_debug_cell_pdf
image_to_pdf           = pipeline.image_to_pdf
_clear_cell_registry   = pipeline._clear_cell_registry
TRANSLATE_ENGINE       = pipeline.TRANSLATE_ENGINE
CHUNK_SIZE             = pipeline.CHUNK_SIZE

# ── 应用初始化 ──────────────────────────────────────────────
app = FastAPI(
    title="PDF Translate API",
    description="""
## CAD 图纸 PDF 中文→英文翻译服务

### 功能
- 上传 PDF 文件，自动完成 OCR → 翻译 → 回填全流程
- 同时生成两种回填结果用于对比：
  - **OCR Box Method**: 按 OCR 合并文本框回填（推荐）
  - **Cell-Based Method**: 按 OpenCV 单元格回填（对照组）

### 使用方法
1. `POST /api/upload` — 上传 PDF
2. `POST /api/process/{task_id}` — 启动处理
3. `GET /api/status/{task_id}` — 查看进度
4. `GET /api/download/{task_id}/{filename}` — 下载结果
""",
    version="1.0.0",
)

# 任务存储目录
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = BASE_DIR / "api_tasks"
TASKS_DIR.mkdir(exist_ok=True)

# 内存中的任务状态
tasks: dict = {}


# ── 辅助函数 ────────────────────────────────────────────────
def _run_pipeline(pdf_path: str, work_dir: str, task_id: str):
    """在后台运行完整翻译管线。"""
    task = tasks.get(task_id)
    if not task:
        return
    task["status"] = "processing"
    task["progress"] = "初始化..."

    try:
        dpi = config.RENDER_DPI

        # Step 1: 渲染
        task["progress"] = "渲染PDF为图像..."
        img_path, page_meta = pdf_to_image(pdf_path, dpi=dpi)

        # Step 2: OCR
        task["progress"] = "RapidOCR 识别..."
        raw_ocr_items = ocr_with_rapid_chunked(img_path, chunk_size=CHUNK_SIZE)
        task["ocr_count"] = len(raw_ocr_items)

        # Step 2.1: OCR调试PDF
        task["progress"] = "生成OCR调试PDF..."
        ocr_debug_pdf = os.path.join(work_dir, "ocr_debug.pdf")
        _generate_debug_ocr_pdf(img_path, raw_ocr_items, ocr_debug_pdf, dpi=dpi)

        # Step 2.5: 合并
        task["progress"] = "智能文本块合并..."
        merge_img = cv2.imread(img_path)
        ocr_items = merge_ocr_items(raw_ocr_items, img_bgr=merge_img)

        ocr_json = os.path.join(work_dir, "ocr_result.json")
        with open(ocr_json, "w", encoding="utf-8") as f:
            json.dump(ocr_items, f, ensure_ascii=False, indent=2)

        # Step 2.6: 合并后OCR调试PDF
        ocr_merged_debug_pdf = os.path.join(work_dir, "ocr_debug_merged.pdf")
        _generate_debug_ocr_pdf(img_path, ocr_items, ocr_merged_debug_pdf, dpi=dpi)

        # Step 3: 翻译
        task["progress"] = "翻译中..."
        if TRANSLATE_ENGINE == "llm":
            translated_items = translate_with_llm(ocr_items)
        else:
            translated_items = translate_with_dictionary(ocr_items)

        trans_json = os.path.join(work_dir, "translation_mapping.json")
        with open(trans_json, "w", encoding="utf-8") as f:
            json.dump(translated_items, f, ensure_ascii=False, indent=2)

        translated_count = sum(1 for t in translated_items if t.get("translated", t["text"]) != t["text"])
        task["translated_count"] = translated_count

        # Step 4: OCR Box 方法回填（主输出）
        task["progress"] = "OCR Box 方法回填..."
        output_img = os.path.join(work_dir, "translated_page.png")
        inpaint_and_overlay(img_path, translated_items, output_img)
        output_pdf = os.path.join(work_dir, "output_ocr_box.pdf")
        image_to_pdf(output_img, output_pdf, dpi=dpi)

        # Step 5: Cell-Based 方法回填（对比输出）
        task["progress"] = "Cell-Based 方法回填..."
        output_cell_img = os.path.join(work_dir, "translated_page_cell_based.png")
        _inpaint_overlay_cell_based(img_path, translated_items, output_cell_img)
        output_cell_pdf = os.path.join(work_dir, "output_cell_based.pdf")
        image_to_pdf(output_cell_img, output_cell_pdf, dpi=dpi)

        # Cell debug PDF
        task["progress"] = "生成Cell调试PDF..."
        cell_debug_pdf = os.path.join(work_dir, "cell_debug.pdf")
        _generate_debug_cell_pdf(output_img, cell_debug_pdf, dpi=dpi)

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
        _generate_comparison_report(compare_report, task)
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


def _generate_comparison_report(report_path: str, task: dict):
    """生成两种方法的对比报告。"""
    lines = [
        "# PDF Translate - 双方法对比报告",
        "",
        f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
    """根路径重定向到 Swagger UI"""
    return """
    <html>
    <head><title>PDF Translate API</title></head>
    <body>
    <h1>PDF Translate API</h1>
    <p>CAD 图纸中文→英文翻译服务</p>
    <ul>
      <li><a href="/docs">/docs</a> — Swagger UI</li>
      <li><a href="/redoc">/redoc</a> — ReDoc</li>
    </ul>
    </body>
    </html>
    """


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(..., description="要翻译的 PDF 文件（CAD 图纸）")):
    """上传 PDF 文件，返回任务 ID。"""
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

    tasks[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "work_dir": str(task_dir),
        "status": "uploaded",
        "progress": "等待处理",
        "created_at": datetime.now().isoformat(),
    }

    return {
        "task_id": task_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "文件已上传，请调用 /api/process/{task_id} 开始处理",
    }


@app.post("/api/process/{task_id}")
async def process_task(task_id: str, background_tasks: BackgroundTasks):
    """启动后台处理管线。"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在，请先上传文件")

    if task["status"] == "processing":
        return {"task_id": task_id, "status": "processing", "message": "正在处理中..."}

    # 设置动态路径
    pdf_path = task["pdf_path"]
    work_dir = task["work_dir"]

    # 临时覆盖全局 WORK_DIR（供子函数使用）
    pipeline.WORK_DIR = work_dir
    pipeline.PDF_PATH = pdf_path
    os.makedirs(work_dir, exist_ok=True)
    _clear_cell_registry()

    # 后台运行
    background_tasks.add_task(_run_pipeline, pdf_path, work_dir, task_id)

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "处理已启动，请用 /api/status/{task_id} 查看进度",
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
        "status": task.get("status"),
        "progress": task.get("progress"),
        "ocr_count": task.get("ocr_count"),
        "translated_count": task.get("translated_count"),
    }

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
