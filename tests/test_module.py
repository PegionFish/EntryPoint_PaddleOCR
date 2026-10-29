"""module.toml 结构契约测试（T-45 变体扩充 + v4/pp-structure-v3 不回归）。"""

from __future__ import annotations

import tomllib
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent


def _load() -> dict:
    with open(MODULE_DIR / "module.toml", "rb") as f:
        return tomllib.load(f)


def _models_by_id() -> dict[str, dict]:
    return {m["id"]: m for m in _load()["models"]}


def test_module_identity():
    manifest = _load()
    assert manifest["module"]["id"] == "paddleocr"
    assert manifest["module"]["category"] == "ocr"
    assert manifest["runtime"]["entrypoint"] == "adapter.py"
    # 服务器模块 venv python 3.12 口径（T-42 pin 约定）
    assert manifest["runtime"]["python_version"] == ">=3.10,<3.13"


def test_variant_count_and_ids():
    ids = set(_models_by_id())
    # 存量三变体不回归 + T-45 两新变体
    assert {"v4-chinese", "v4-multilingual", "pp-structure-v3", "v5-chinese", "paddleocr-vl"} <= ids


def test_v4_variants_unchanged():
    models = _models_by_id()
    assert models["v4-chinese"] == {
        "id": "v4-chinese",
        "name": "PP-OCRv4 中文 (推荐)",
        "source": "url",
        "url": "auto",
        "target_dir": "paddleocr-v4-chinese",
        "size_estimate_mb": 200,
        "default": True,
    }
    assert models["v4-multilingual"]["target_dir"] == "paddleocr-v4-multilingual"
    assert models["v4-multilingual"]["url"] == "auto"
    assert models["pp-structure-v3"]["source"] == "local_import"
    assert models["pp-structure-v3"]["target_dir"] == "paddleocr-pp-structure-v3"
    assert models["pp-structure-v3"]["size_estimate_mb"] == 3138


def test_default_variant_still_v4_chinese():
    defaults = [m for m in _load()["models"] if m.get("default")]
    assert [m["id"] for m in defaults] == ["v4-chinese"]


def test_v5_chinese_variant_declaration():
    m = _models_by_id()["v5-chinese"]
    # 与 v4 变体同口径：模块自管下载（url=auto），独立 target_dir
    assert m["source"] == "url"
    assert m["url"] == "auto"
    assert m["target_dir"] == "paddleocr-v5-chinese"
    assert not m.get("default")
    assert m["size_estimate_mb"] > 0


def test_paddleocr_vl_variant_declaration():
    m = _models_by_id()["paddleocr-vl"]
    # local_import 沿 pp-structure-v3 先例：存在即就绪，平台不下载
    assert m["source"] == "local_import"
    assert m["target_dir"] == "paddleocr-vl-0.9b"
    assert not m.get("default")
    assert m["size_estimate_mb"] >= 2000  # 0.9B safetensors ~1.9GB + 版面模型


def test_target_dirs_unique():
    target_dirs = [m["target_dir"] for m in _load()["models"]]
    assert len(target_dirs) == len(set(target_dirs))


def test_capabilities_contract():
    caps = {c["name"]: c for c in _load()["interface"]["capabilities"]}
    # 存量能力不回归 + doc_vl 新能力
    assert set(caps) == {"recognize", "doc_understand", "doc_vl"}
    # recognize 参数不动
    assert set(caps["recognize"]["params"]) == {"lang", "use_angle_cls", "det_db_thresh"}
    # doc_understand 参数不动（chart 默认 false 的 paddlex bug 口径保持）
    assert caps["doc_understand"]["params"]["chart"]["default"] is False
    assert caps["doc_understand"]["params"]["output_format"]["options"] == ["json", "md"]
    # doc_vl：VL 端到端解析，无 table/formula/chart 分开关
    assert set(caps["doc_vl"]["params"]) == {"doc_orientation", "doc_unwarping", "dpi", "output_format"}
    assert caps["doc_vl"]["max_file_size_mb"] == 100
    assert caps["doc_vl"]["input_type"] == "image"
