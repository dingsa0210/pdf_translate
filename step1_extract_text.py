"""
Step 1: 使用 PyMuPDF 解析目标 PDF，提取矢量文本信息（含竖排文本检测）
解析要素: 文本内容、bbox、origin、font、size、color、wmode(书写方向)、rotation(旋转角度)
"""
import fitz  # PyMuPDF
import json
import math
import os

# 默认PDF路径，可通过命令行参数覆盖
PDF_PATH = r"D:\AIGC\projects\pdf_translate\pdfs\260400JRS--P5 (地坑、预埋件布置图) 3×A0.pdf"


def detect_text_orientation(chars: list) -> dict:
    """通过字符origin坐标分析文本方向

    Returns:
        dict with keys:
            is_vertical: bool - 是否为竖排文本
            rotation_deg: float - 旋转角度（度）
            direction: str - "horizontal" | "vertical" | "rotated"
    """
    if len(chars) < 2:
        return {"is_vertical": False, "rotation_deg": 0, "direction": "horizontal"}

    origins = [c.get("origin", (0, 0)) for c in chars]
    # 计算相邻字符origin的变化量
    dx_total = 0.0
    dy_total = 0.0
    for i in range(1, len(origins)):
        dx_total += abs(origins[i][0] - origins[i - 1][0])
        dy_total += abs(origins[i][1] - origins[i - 1][1])

    # 竖排判断: Y方向变化显著大于X方向
    is_vertical = False
    if dy_total > 0 and dx_total > 0:
        is_vertical = dy_total > dx_total * 1.5
    elif dy_total > 0 and dx_total == 0:
        is_vertical = True

    # 计算整体旋转角度
    if len(origins) >= 2:
        dx = origins[-1][0] - origins[0][0]
        dy = origins[-1][1] - origins[0][1]
        rotation_deg = math.degrees(math.atan2(dy, dx))
    else:
        rotation_deg = 0

    if is_vertical:
        direction = "vertical"
    elif abs(rotation_deg) > 5:  # 允许5度误差
        direction = "rotated"
    else:
        direction = "horizontal"

    return {
        "is_vertical": is_vertical,
        "rotation_deg": round(rotation_deg, 2),
        "direction": direction,
    }


def extract_text_info(pdf_path: str) -> list:
    """提取PDF中所有中文文本及其坐标、字体属性、方向信息"""
    doc = fitz.open(pdf_path)
    print(f"PDF总页数: {doc.page_count}")

    all_text_items = []

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        print(f"\n{'=' * 60}")
        print(f"第 {page_idx + 1} 页 | 尺寸: {page.rect.width:.1f} x {page.rect.height:.1f} points")
        print(f"{'=' * 60}")

        # 使用 rawdict 获取字符级详细信息（含origin和matrix）
        blocks = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_text_count = 0
        page_vertical_count = 0

        for block in blocks["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    chars = span.get("chars", [])
                    text = "".join(c.get("c", "") for c in chars).strip()
                    if not text:
                        continue

                    # 检测是否包含中文
                    has_chinese = any("一" <= ch <= "鿿" for ch in text)
                    if not has_chinese:
                        continue

                    bbox = span.get("bbox", [0, 0, 0, 0])
                    bbox = [round(v, 2) for v in bbox]

                    # 分析文本方向
                    orientation = detect_text_orientation(chars)

                    # 判断wmode (0=水平, 1=垂直)
                    wmode = span.get("wmode", 0)

                    # 如果wmode==1或者字符分析认为是竖排，标记为竖排
                    is_vertical = wmode == 1 or orientation["is_vertical"]

                    item = {
                        "page": page_idx + 1,
                        "text": text,
                        "bbox": bbox,
                        "font": span.get("font", ""),
                        "size": round(span.get("size", 12), 2),
                        "color": span.get("color", 0),
                        "origin": [round(v, 2) for v in span.get("origin", (0, 0))],
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

        print(f"本页含中文文本片段数: {page_text_count}")
        if page_vertical_count > 0:
            print(f"  其中竖排文本: {page_vertical_count}")

    doc.close()
    return all_text_items


def main():
    import sys

    pdf_path = PDF_PATH
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"错误: PDF文件不存在: {pdf_path}")
        return

    print(f"正在解析: {pdf_path}")
    print(f"文件大小: {os.path.getsize(pdf_path) / 1024 / 1024:.2f} MB")

    text_items = extract_text_info(pdf_path)

    print(f"\n{'=' * 60}")
    print(f"提取结果汇总")
    print(f"{'=' * 60}")
    print(f"含中文的文本片段总数: {len(text_items)}")

    vertical_count = sum(1 for item in text_items if item["is_vertical"])
    horizontal_count = len(text_items) - vertical_count
    print(f"  水平文本: {horizontal_count}")
    print(f"  竖排文本: {vertical_count}")

    fonts_used = set(item["font"] for item in text_items)
    print(f"使用的字体: {fonts_used}")

    # 显示前30条中文文本
    print(f"\n前30条中文文本示例:")
    print(f"{'-' * 80}")
    for i, item in enumerate(text_items[:30]):
        vmark = " [竖排]" if item["is_vertical"] else ""
        print(
            f"[{i + 1}] text='{item['text']}' | font={item['font']} | "
            f"size={item['size']} | bbox={item['bbox']} | "
            f"dir={item['direction']}{vmark}"
        )

    # 保存完整结果到JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "extracted_text.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(text_items, f, ensure_ascii=False, indent=2)
    print(f"\n完整提取结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
