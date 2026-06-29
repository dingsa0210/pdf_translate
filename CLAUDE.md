# PDF Translate - CAD 图纸无损翻译工具

## 项目概述

将 CAD 导出的 PDF 图纸中的中文文本无损翻译为英文。支持两种 PDF 类型：
- **矢量型 PDF**：直接解析文本坐标，原位擦除+回填（step1 → step2 流程）
- **扫描型/栅格化 PDF**：渲染为图像，OCR 识别后覆盖翻译（step3 流程）

核心技术栈：PyMuPDF + OpenAI 兼容 LLM API + RapidOCR（扫描型）

## 项目结构

```
config.py                        # 共享配置（环境变量加载，含 .env 支持）
step1_extract_text.py            # 矢量PDF文本提取（含竖排检测）
step2_translate_refill.py        # 矢量PDF翻译+原位回填
step3_scan_translate.py          # 扫描型PDF翻译（RapidOCR + 图像覆盖）
scan_translate_fixed.py          # 扫描翻译变体（白底擦除+左对齐）
scan_translate_layout_fixed.py   # 扫描翻译变体（布局修复版）
engineering_dict.json            # 工程术语字典（中→英）
.env.example                     # 环境变量模板
pdfs/                            # 输入PDF文件目录
scan_work/                       # 扫描翻译工作目录（中间产物）
```

## 环境配置

使用 conda `modelscope` 环境（全局规则）：

```bash
# 运行脚本
conda run -n modelscope python step1_extract_text.py [pdf_path]

# 安装依赖
conda run -n modelscope pip install PyMuPDF openai python-dotenv rapidocr-onnxruntime opencv-python Pillow
```

环境变量通过 `.env` 文件配置（`python-dotenv` 自动加载）：
- `LLM_API_BASE` — OpenAI 兼容 API 地址（默认 SiliconFlow）
- `LLM_API_KEY` — API 密钥
- `LLM_MODEL` — 模型名称（默认 tencent/Hunyuan-MT-7B）
- `TRANSLATE_ENGINE` — 翻译引擎：`llm`（在线API）或 `dictionary`（离线术语库）
- `FONT_PATH` — 英文字体路径

## 工作流程

### 矢量型 PDF（推荐）
```bash
conda run -n modelscope python step1_extract_text.py <pdf_path>     # 1. 提取文本 → extracted_text.json
conda run -n modelscope python step2_translate_refill.py <pdf_path>  # 2. 翻译+回填 → 输出PDF
```

### 扫描型 PDF
```bash
conda run -n modelscope python step3_scan_translate.py <pdf_path>    # OCR+翻译+覆盖
```

## 翻译策略

翻译采用两层策略：
1. **术语字典精确匹配** — `engineering_dict.json` 包含工程标准术语（ISO/DIN/GB）
2. **LLM 上下文翻译** — 对字典未覆盖的文本，按空间邻近度聚类后批量发送给 LLM

LLM 的 system prompt 中注入术语字典样本，确保翻译符合工程惯例。竖排文本标记为 `[竖排]`，要求极简译文（1-3 个单词）。

## 关键设计约束

- **不可信外部数据**：PDF 文件是外部输入，所有解析结果需校验
- **文本溢出防护**：英文通常比中文长 1.5-2.5 倍，回填时需动态缩放字体
- **竖排文本**：CAD 图纸中存在大量竖排/旋转文本，需特殊处理坐标和渲染
- **内存管理**：扫描型 PDF 渲染大图时需分块处理，避免内存溢出
- **API 密钥安全**：`.env` 已在 `.gitignore` 中，切勿提交真实密钥

## 术语字典

`engineering_dict.json` 是项目核心资产，包含机械/电气/轧钢等领域标准术语映射。修改时保持 JSON 格式，键为中文原文，值为英文译文。

## 代码风格

- 函数和变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 所有配置集中在 `config.py`，通过环境变量可覆盖
- 脚本均支持命令行参数覆盖默认路径
