"""paddleocr 测试夹具：注入 FakePaddleOCR / FakePaddleOCRVL（零真实推理、零网络）。

T-45：新增 v5 变体（EP_MODEL_ID → 显式模型名 pin）与 PaddleOCR-VL
（/predict/doc_vl，local_import 模型根探测）的契约测试。重型依赖
（paddleocr/paddle）经 sys.modules 注入替身模块，本机无 paddle 也可全绿。
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_DIR))

import adapter  # noqa: E402

# 1x1 白色 PNG（无外部依赖的合法图片字节）
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4//8/AwAI/AL+"
    "hc2rNAAAAABJRU5ErkJggg=="
)


class FakeOCRResult:
    """OCRResult 替身：rec_texts/rec_scores/rec_polys 的 dict 风格访问。"""

    def __init__(self, lines: list[tuple[str, float, list]]):
        self._d = {
            "rec_texts": [t for t, _, _ in lines],
            "rec_scores": [c for _, c, _ in lines],
            "rec_polys": [p for _, _, p in lines],
        }

    def __getitem__(self, k):
        return self._d[k]

    def get(self, k, default=None):
        return self._d.get(k, default)


class FakePaddleOCR:
    """PaddleOCR 替身：记录最近一次构造 kwargs 与 predict 调用。"""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs
        self.predict_calls: list[str] = []

    def predict(self, image_path):
        self.predict_calls.append(image_path)
        return [
            FakeOCRResult(
                [
                    ("入口点", 0.98, [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]),
                    ("EntryPoint", 0.91, [[0.0, 6.0], [12.0, 6.0], [12.0, 11.0], [0.0, 11.0]]),
                ]
            )
        ]


class FakeVLResult:
    """PaddleOCR-VL 单页结果替身：markdown dict（text/page_texts）+ to_json。"""

    def __init__(self, page_text: str):
        self.markdown = {"text": page_text, "page_texts": [page_text]}
        self._obj = {"parsing_res_list": [], "markdown": self.markdown}

    def to_json(self) -> str:
        import json

        return json.dumps(self._obj, ensure_ascii=False)


class FakePaddleOCRVL:
    """PaddleOCRVL 管线替身：记录构造 kwargs 与 predict 调用。"""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.kwargs = kwargs
        self.predict_calls: list[tuple[str, dict]] = []

    def predict(self, input, **kwargs):
        self.predict_calls.append((str(input), kwargs))
        return [FakeVLResult("# 标题\n\n正文段落")]


@pytest.fixture()
def adapter_env(monkeypatch, tmp_path):
    """隔离环境：CPU 后端 + 空变体 + tmp 模型根；重置引擎缓存与落位探测。"""
    monkeypatch.setattr(adapter, "EP_BACKEND", "cpu")
    monkeypatch.setattr(adapter, "EP_MODEL_DIR", "")
    monkeypatch.setattr(adapter, "EP_MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "")
    monkeypatch.setattr(adapter, "EP_DEVICE_INDEX", "0")
    monkeypatch.setattr(adapter, "_ocr_engine", None)
    monkeypatch.setattr(adapter, "_ocr_engine_lang", None)
    monkeypatch.setattr(adapter, "_ocr_engine_angle_cls", None)
    monkeypatch.setattr(adapter, "_ocr_engine_variant", None)
    monkeypatch.setattr(adapter, "_vl_engine", None)
    monkeypatch.setattr(adapter, "_pps3_engine", None)
    return tmp_path


@pytest.fixture()
def fake_paddleocr(adapter_env, monkeypatch):
    """sys.modules 注入 paddleocr 替身模块（PaddleOCR/PaddleOCRVL 均为 Fake）。"""
    FakePaddleOCR.last_kwargs = None
    FakePaddleOCRVL.last_kwargs = None
    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = FakePaddleOCR
    mod.PaddleOCRVL = FakePaddleOCRVL
    monkeypatch.setitem(sys.modules, "paddleocr", mod)
    return mod


def make_vl_root(root: Path | None = None, *, layout_name: str = "PP-DocLayoutV2",
                 vl_name: str | None = "PaddleOCR-VL-0.9B", flat: bool = False) -> Path:
    """按 README/CONTRACT 落位口径搭一个可探测的 VL 模型根。

    缺省落位在契约位 <EP_MODELS_ROOT>/paddleocr-vl-0.9b/（adapter_env 已注入）。
    """
    base = Path(adapter.EP_MODELS_ROOT) if adapter.EP_MODELS_ROOT else Path(root or ".")
    target = base / adapter.PVL_TARGET_DIR
    layout = target / layout_name
    layout.mkdir(parents=True, exist_ok=True)
    (layout / "inference.pdiparams").write_bytes(b"\x01" * 16)
    (layout / "inference.pdmodel").write_bytes(b"\x01" * 8)
    (layout / "config.json").write_text("{}", encoding="utf-8")
    vl_dir = target if flat or vl_name is None else target / vl_name
    vl_dir.mkdir(parents=True, exist_ok=True)
    (vl_dir / "model.safetensors").write_bytes(b"\x02" * 16)
    (vl_dir / "config.json").write_text("{}", encoding="utf-8")
    return target
