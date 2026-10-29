"""生成服务器 e2e 用样张（含 ground truth），输出到指定目录。

本地运行（Windows 字体），产物 scp 上服务器。用法：
    python make_samples.py <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ZH_FONT = "C:/Windows/Fonts/msyh.ttc"
EN_FONT = "C:/Windows/Fonts/arial.ttf"

SAMPLES: list[dict] = [
    {
        "name": "sample_zh",
        "lines": ["入口点文字识别模块", "多语言光学字符识别", "二零二六年十月二十八日"],
        "font": ZH_FONT,
        "size": 40,
    },
    {
        "name": "sample_en",
        "lines": ["EntryPoint OCR Module 2026", "The quick brown fox jumps", "GPU CUDA Pascal Benchmark"],
        "font": EN_FONT,
        "size": 36,
    },
    {
        "name": "sample_doc",
        "lines": [
            "PaddleOCR-VL 评测样张",
            "",
            "本文档用于验证端到端文档理解管线。",
            "标题应被解析为一级标题。",
            "正文段落应保持原始顺序。",
        ],
        "font": ZH_FONT,
        "size": 32,
    },
]


def render(sample: dict, out_dir: Path) -> Path:
    font = ImageFont.truetype(sample["font"], sample["size"])
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    widths = [draw.textlength(t, font=font) for t in sample["lines"]]
    w = int(max(widths)) + 80
    line_h = sample["size"] + 18
    h = line_h * len(sample["lines"]) + 80
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y = 40
    for text in sample["lines"]:
        if text:
            d.text((40, y), text, fill="black", font=font)
        y += line_h
    out = out_dir / f"{sample['name']}.png"
    img.save(out)
    return out


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SAMPLES:
        p = render(s, out_dir)
        gt = out_dir / f"{s['name']}.gt.txt"
        gt.write_text("\n".join(t for t in s["lines"] if t), encoding="utf-8")
        print(f"{p}  +  {gt}")


if __name__ == "__main__":
    main()
