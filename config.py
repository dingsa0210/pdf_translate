"""
共享配置模块 - 从环境变量加载所有可配置项，带默认值回退
使用方式: 设置环境变量或创建 .env 文件
"""
import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv 未安装时回退到纯环境变量

# ============================================================
# LLM 翻译配置 (OpenAI 兼容接口)
# ============================================================
LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://api.siliconflow.cn/v1/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "tencent/Hunyuan-MT-7B")
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "40"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

# 翻译引擎选择: "llm" (OpenAI兼容API) 或 "dictionary" (离线术语库)
TRANSLATE_ENGINE = os.environ.get("TRANSLATE_ENGINE", "llm")

# ============================================================
# OCR / 渲染参数
# ============================================================
RENDER_DPI = int(os.environ.get("RENDER_DPI", "200"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "4000"))

# ============================================================
# 字体路径
# ============================================================
FONT_PATH = os.environ.get("FONT_PATH", r"C:\Windows\Fonts\arial.ttf")

# ============================================================
# 单元格检测引擎: "opencv_v3" (纯OpenCV) | "ppstructure" (PaddleOCR表格识别)
# ============================================================
CELL_DETECT_ENGINE = os.environ.get("CELL_DETECT_ENGINE", "opencv_v3")

# ============================================================
# 工程术语字典 (从外部 JSON 文件加载)
# ============================================================
_dict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engineering_dict.json")
if os.path.exists(_dict_path):
    with open(_dict_path, "r", encoding="utf-8") as _f:
        ENGINEERING_DICT = json.load(_f)
else:
    print(f"警告: 未找到工程术语字典文件: {_dict_path}")
    ENGINEERING_DICT = {}
