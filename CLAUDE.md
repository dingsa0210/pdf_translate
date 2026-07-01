# PDF Translate - CAD 图纸无损翻译工具

## 项目概述

将 CAD 导出的 PDF 图纸中的中文文本近乎无损地翻译为英文。支持两种 PDF 类型，自动检测并路由：

- **矢量型 PDF**：直接解析文本坐标 → LLM/字典翻译 → 原位擦除+回填（保留字号/颜色/方向）
- **扫描型/栅格化 PDF**：渲染为图像 → RapidOCR 识别 → 翻译 → 双方法回填对比

核心技术栈：PyMuPDF + OpenAI 兼容 LLM API + RapidOCR（扫描型）+ OpenCV（单元格检测）+ FastAPI

## 项目结构

```
config.py                        # 共享配置（环境变量加载，含 .env 支持）
vector_translate_pipeline.py     # 矢量型 PDF 一站式管线（提取→翻译→擦除→回填）
scan_translate_pipeline.py       # 扫描型 PDF 一站式管线（OCR→翻译→双方法回填对比，3000+ 行）
api_server.py                    # FastAPI HTTP 服务（Swagger UI，自动检测 PDF 类型并路由）
engineering_dict.json            # 工程术语字典（中→英）
扫描PDF图纸处理规则.txt            # 扫描型 PDF 处理的设计约束文档
debug/                           # 废弃/旧版脚本归档
  step1_extract_text.py
  step2_translate_refill.py
  step3_scan_translate.py
pdfs/                            # 测试用 PDF 文件
scan_work/                       # 扫描管线工作目录（中间产物）
vector_work/                     # 矢量管线工作目录（中间产物）
api_tasks/                       # API 任务隔离目录（按 task_id 分文件夹）
```

## 启动方式

### API 服务（推荐）
```bash
conda run -n modelscope uvicorn api_server:app --host 0.0.0.0 --port 8000
```
访问 `http://localhost:8000/docs` 进入 Swagger UI，上传 PDF 即可自动翻译。

### 引擎切换
通过环境变量 `TRANSLATE_ENGINE` 控制翻译引擎：
- `llm`（默认）— OpenAI 兼容 API（SiliconFlow Hunyuan-MT-7B 等）
- `dictionary` — 离线工程术语字典翻译

通过环境变量 `CELL_DETECT_ENGINE` 控制扫描管线的单元格检测：
- `opencv_v3`（默认）— 纯 OpenCV 三阶段级联检测
- `ppstructure` — PaddleOCR 表格识别

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 首页，导向 Swagger UI |
| `POST` | `/api/upload` | 上传 PDF，自动检测类型，返回 task_id 和 pdf_type |
| `POST` | `/api/process/{task_id}` | 启动后台处理（按类型自动路由管线） |
| `GET` | `/api/status/{task_id}` | 查询任务进度、统计和结果文件列表 |
| `GET` | `/api/results/{task_id}` | 列出所有结果文件及描述、下载链接 |
| `GET` | `/api/download/{task_id}/{filename}` | 下载指定结果文件 |
| `DELETE` | `/api/tasks/{task_id}` | 删除任务及所有文件 |

### 矢量型管线 API 输出
- `output_vector.pdf` — 主输出，矢量 PDF 翻译结果
- `extracted_text.json` — 提取到的中文文本 + 坐标 + 字号 + 颜色 + 方向
- `translation_mapping.json` — 翻译映射 JSON
- `vector_report.md` — 处理报告（含水平/竖排/旋转回填统计）

### 扫描型管线 API 输出
- `output_ocr_box.pdf` — 主输出，OCR 合并框回填方法（推荐）
- `output_cell_based.pdf` — 对照组，OpenCV 单元格回填方法
- `ocr_debug.pdf` — OCR 原始识别框（合并前，颜色=置信度）
- `ocr_debug_merged.pdf` — OCR 合并后文本框
- `cell_debug.pdf` — 单元格检测框（红框+编号）
- `ocr_result.json` — OCR 识别+合并结果 JSON
- `translation_mapping.json` — 翻译映射 JSON
- `rendered_image.png` — PDF 渲染原图
- `comparison_report.md` — 双方法对比报告

## PDF 类型检测逻辑

`api_server.py` 中的 `detect_pdf_type()` 函数：
- 用 PyMuPDF 抽样检查前 3 页
- 任一页用 `page.get_text("text")` 提取到包含中文（Unicode \u4e00-\u9fff）的文本 → 判定为 `vector`
- 否则 → 判定为 `scan`

## 矢量型管线设计 (vector_translate_pipeline.py)

### 核心函数
| 函数 | 说明 |
|------|------|
| `extract_text_info(pdf_path)` | PyMuPDF 提取所有含中文的 span，含 bbox/字号/颜色/方向 |
| `detect_text_orientation(chars)` | 通过字符 origin 坐标序列判断水平/竖排/旋转 |
| `translate_with_llm(items)` | 批量 LLM 翻译，含竖排标记 [VERTICAL] |
| `translate_with_dictionary(items)` | 术语字典翻译（精确匹配 + 长词优先） |
| `trim_vertical_translations(items)` | 竖排译文长度检查，过长则回退到字典短译文 |
| `calculate_fitted_text(...)` | 水平文字适配 bbox（缩放/折行） |
| `insert_vertical_text(...)` | 竖排英文回填（单字符旋转 90°，自下向上） |
| `insert_rotated_text(...)` | 倾斜/旋转文本回填 |
| `redact_and_refill(...)` | 两阶段：Phase 1 白底擦除 + Phase 2 按类型回填 |
| `run_vector_pipeline(pdf_path, work_dir)` | 一站式入口 |

### 回填类型分支
1. `is_vertical` → `insert_vertical_text()`
2. `direction == "rotated"` 且 `|rotation_deg| > 5` → `insert_rotated_text()`
3. 水平文本 → `calculate_fitted_text()` → 必要时多行拆分

## 扫描型管线设计 (scan_translate_pipeline.py)

### 核心流程
```
PDF → pdf_to_image() 渲染 → RapidOCR 分块识别 → merge_ocr_items() 智能合并
    → translate_with_llm()/dictionary() 翻译 → 双方法回填
        ├── inpaint_and_overlay()     → OCR 合并框方法（推荐，效果更好）
        └── _inpaint_overlay_cell_based() → OpenCV 单元格方法（对比用，连通域不准）
```

### merge_ocr_items() 三阶段合并详解

`merge_ocr_items()` 是扫描管线的核心函数，其内部三阶段流程决定了 OCR 文本的合并结果，进而影响最终两个回填输出的质量：

```
merge_ocr_items(items, img_bgr=merge_img)
  │
  ├─ 阶段1: 全图格网检测
  │   _detect_all_table_cells_dispatch() → opencv_v3 引擎
  │     → _find_cells_by_morphological_components()  ← 形态学连通域检测
  │     → 轮廓矩形 + 线交叉网格 + 文本间隙推理（补充阶段）
  │   ⚠️ 连通域检测结果在此被用作"权威格"，是后续合并的基础
  │
  ├─ 阶段2: OCR 分配到格 + 同格合并
  │   ① 跨格分割 (L1667-1712):
  │      用格网水平边界线切割多行 OCR 文本
  │      例: "地脚\n螺栓直径\n螺纹长度" 被格线切成两个格子各自文本
  │   ② 格网分配 (L1714-1749):
  │      OCR bbox 与连通域 cell 重叠率 > 50% → 标记 in_table=True
  │   ③ 同格合并 (L1751-1788):
  │      同一 cell 内的多个 OCR 块合并为一个（保留 sub_bboxes）
  │      → 产物 merged_cell，最终追加到 final_items
  │
  └─ 阶段3: 未分配项智能行合并
      对未落入任何 cell 的 OCR 项做基于空间邻近度的段落合并
      in_table=True 的项被锁定，不参与跨格合并
```

### ⚠️ 连通域检测对两个输出的影响分析

连通域检测（`_find_cells_by_morphological_components`）虽然是 `opencv_v3` 引擎的核心方法，**但在实际测试中其单元格检测并不准确**。关键问题是：即使在用户偏好的 OCR 框输出（`inpaint_and_overlay`）中，连通域检测也不可避免地参与其中。

**影响传导路径：**

| 影响点 | 位置 | 如何污染 OCR 框输出 |
|--------|------|---------------------|
| 跨格分割 | L1667-1712 | 若连通域格线位置不准，会把本属同一格的多行文本错误切开 |
| 格网分配 | L1714-1749 | 重叠率判断依赖 cell 坐标，不准的 cell 导致 OCR 块被错误分配或遗漏 |
| 同格合并 | L1751-1788 | 不合规的 cell 边界使其内 OCR 文本被强行合并为一个单元 |
| 合并阻断 | L1824-1843 | in_table=True 的项在阶段3被锁定，无法通过智能行合并修复 |

**结论：** `merged_cell` 产物会混入 `final_items`（L1915），作为两个回填函数的共享输入。连通域检测的不准确性通过以上四条路径传导，两个输出都会受到影响。当前 `output_ocr_box.pdf` 效果仍可接受是因为阶段3的智能行合并对未分配项做了较好的补偿，但格内合并质量仍受连通域精度制约。

**后续优化方向：** 如果要在 merge 阶段摆脱对连通域检测的依赖，可考虑：
- 将阶段2替换为纯基于 OCR 文本几何特征的格网推断（不依赖图像形态学）
- 或在阶段2中降低连通域结果的"权威"地位，允许阶段3的智能行合并覆盖格内项

### 单元格检测（opencv_v3 四阶段级联）
1. **形态学连通域**（首选/权威）— `connectedComponentsWithStats`，横竖线形态学提取后反色连通
2. **轮廓矩形检测**（补充）— 找闭合矩形轮廓
3. **线交叉网格**（补充）— Canny + HoughP 多尺度线段（≥30px）→ 交点过滤（≥3交点线）→ 网格推理
4. **文本间隙推理**（默认禁用）— 需 `ENABLE_TEXT_GAP_CELLS=1`

后处理：阶段1的连通域格被视为"原子格"，其他阶段子格若被其包含则移除。

### 关键函数
| 函数 | 说明 |
|------|------|
| `pdf_to_image(pdf_path, dpi)` | PyMuPDF 渲染 PDF 为图像 |
| `ocr_with_rapid_chunked(img_path, chunk_size)` | RapidOCR 分块识别（避免大图 OOM） |
| `merge_ocr_items(items, img_bgr)` | 三阶段智能合并：格网检测 → 格分配+同格合并 → 未分配智能行合并 |
| `_detect_all_table_cells_dispatch(...)` | 单元格检测引擎调度（opencv_v3 / ppstructure） |
| `_find_cells_by_morphological_components(img_gray)` | 形态学连通域单元格检测 |
| `inpaint_and_overlay(...)` | OCR Box 方法（推荐）：擦除 → 回填（保留 sub_bbox 原文坐标） |
| `_inpaint_overlay_cell_based(...)` | Cell-Based 方法（对比用）：格内擦除 2px 内缩保护格线 |
| `_generate_debug_ocr_pdf(...)` | 生成 OCR 识别框调试 PDF |
| `_generate_debug_cell_pdf(...)` | 生成单元格检测框调试 PDF |

### 扫描管线约束
- 单元格最小化擦除，2px 内缩保护格线
- 检测原文对齐方式（左/中/右）并复刻
- 翻译后回填不超出原 bbox
- 近水平文本判定阈值：`ANGLE_NEAR_HORIZONTAL = 5.0°`

## 翻译策略

两层策略，两者管线共用：
1. **术语字典精确匹配** — `engineering_dict.json` 包含机械/电气/轧钢等领域 ISO/DIN/GB 标准术语
2. **LLM 上下文翻译** — 对字典未覆盖的文本，按空间邻近度聚类后批量发送给 LLM

LLM 的 system prompt 中注入术语字典样本。矢量管线中竖排文本标记为 `[VERTICAL]`，要求极简译文（1-3 个单词）。

## 环境配置

使用 conda `modelscope` 环境。依赖：
```bash
pip install PyMuPDF openai python-dotenv rapidocr-onnxruntime opencv-python Pillow fastapi uvicorn loguru
```

环境变量通过 `.env` 文件配置（`python-dotenv` 自动加载）：
- `LLM_API_BASE` — OpenAI 兼容 API 地址（默认 SiliconFlow）
- `LLM_API_KEY` — API 密钥
- `LLM_MODEL` — 模型名称（默认 `tencent/Hunyuan-MT-7B`）
- `LLM_BATCH_SIZE` — LLM 批处理大小（默认 40）
- `TRANSLATE_ENGINE` — 翻译引擎：`llm` 或 `dictionary`
- `RENDER_DPI` — 扫描管线渲染 DPI（默认 200）
- `CHUNK_SIZE` — OCR 分块大小（默认 4000px）
- `FONT_PATH` — 英文字体路径（macOS 自动搜索 `/System/Library/Fonts`）
- `CELL_DETECT_ENGINE` — 单元格检测引擎（默认 `opencv_v3`）

所有配置集中在 `config.py`，从环境变量加载，带默认值回退。

## 关键设计约束

- **不可信外部数据**：PDF 文件是外部输入，所有解析结果需校验
- **文本溢出防护**：英文通常比中文长 1.5-2.5 倍，回填时需动态缩放字体
- **竖排文本**：CAD 图纸中存在大量竖排/旋转文本，需特殊处理坐标和渲染
- **CAD 线 vs 表格线甄别**：扫描管线中必须区分结构线和表格线，避免误检测
- **内存管理**：扫描型 PDF 渲染大图时需分块 OCR，避免 OOM
- **API 密钥安全**：`.env` 已在 `.gitignore` 中，切勿提交真实密钥

## 术语字典

`engineering_dict.json` 是项目核心资产，包含机械/电气/轧钢等领域标准术语映射。修改时保持 JSON 格式，键为中文原文，值为英文译文。

## 代码风格

- 函数和变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 所有配置集中在 `config.py`，通过环境变量可覆盖
- 管线脚本均支持命令行参数和 Python API 调用两种方式
- API 中 `_scan_*` / `_vector_*` 前缀区分不同管线的导入符号
