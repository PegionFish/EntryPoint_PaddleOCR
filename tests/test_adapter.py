"""paddleocr adapter 契约测试（Fake 引擎注入，零真实推理；T-45 变体分支）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import adapter
from conftest import FakePaddleOCR, FakePaddleOCRVL, PNG_1X1, make_vl_root


def _client() -> TestClient:
    return TestClient(adapter.app)


def _write_image(tmp_path, name="in.png"):
    p = tmp_path / name
    p.write_bytes(PNG_1X1)
    return p


# ─── health / info ───────────────────────────────────────────────────────────


def test_health_contract(adapter_env):
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["module"] == "paddleocr"
    assert body["version"] == adapter.MODULE_VERSION


def test_info_contract(adapter_env):
    with _client() as c:
        r = c.get("/info")
    body = r.json()
    assert body["module"] == "paddleocr"
    assert body["backend"] == "cpu"
    caps = [c["name"] for c in body["capabilities"]]
    assert caps == ["recognize", "doc_understand", "doc_vl"]
    vl = next(c for c in body["capabilities"] if c["name"] == "doc_vl")
    assert "doc_orientation" in vl["params"]
    assert "table" not in vl["params"]  # VL 端到端，无分开关


# ─── recognize：v4 默认行为不回归 ─────────────────────────────────────────────


def test_recognize_json_body_default_variant(fake_paddleocr, adapter_env):
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/recognize", json={"input_path": str(img)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["output_type"] == "json"
    assert body["result"]["text"] == "入口点\nEntryPoint"
    line = body["result"]["lines"][0]
    assert line["text"] == "入口点"
    assert line["confidence"] == pytest.approx(0.98)
    assert line["bbox"][0] == [0.0, 0.0]
    # v4/默认路径不加显式模型名（行为与 T-45 之前一致）
    kw = FakePaddleOCR.last_kwargs
    assert "text_detection_model_name" not in kw
    assert "text_recognition_model_name" not in kw
    assert kw["lang"] == "ch"
    assert kw["use_textline_orientation"] is True
    assert kw["text_det_box_thresh"] == pytest.approx(0.3)
    assert kw["device"] == "cpu"


def test_recognize_multipart_upload(fake_paddleocr, adapter_env):
    with _client() as c:
        r = c.post(
            "/predict/recognize",
            files={"file": ("in.png", PNG_1X1, "image/png")},
        )
    assert r.status_code == 200, r.text
    assert r.json()["result"]["lines"]
    # 上传落盘的临时文件在请求结束后清理
    tmp_used = adapter._ocr_engine.predict_calls[0]
    assert tmp_used.endswith(".png")
    assert not Path(tmp_used).exists()


def test_recognize_missing_input_is_422(adapter_env):
    with _client() as c:
        r = c.post("/predict/recognize", json={})
    assert r.status_code == 422


def test_recognize_missing_file_is_404(fake_paddleocr, adapter_env):
    with _client() as c:
        r = c.post("/predict/recognize", json={"input_path": str(adapter_env / "nope.png")})
    assert r.status_code == 404


def test_recognize_file_artifact_mode(fake_paddleocr, adapter_env):
    img = _write_image(adapter_env)
    out = adapter_env / "out" / "result.txt"
    with _client() as c:
        r = c.post(
            "/predict/recognize",
            json={"input_path": str(img), "params": {"output_format": "text", "output_path": str(out)}},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_type"] == "file"
    assert body["output_path"] == str(out)
    assert out.read_text(encoding="utf-8") == "入口点\nEntryPoint"


# ─── recognize：v5 变体分支（T-45） ──────────────────────────────────────────


def test_v5_variant_pins_explicit_model_names(fake_paddleocr, adapter_env, monkeypatch):
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v5-chinese")
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/recognize", json={"input_path": str(img)})
    assert r.status_code == 200, r.text
    kw = FakePaddleOCR.last_kwargs
    assert kw["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert kw["text_recognition_model_name"] == "PP-OCRv5_server_rec"
    # 其余基线参数与默认路径一致
    assert kw["lang"] == "ch"
    assert kw["use_textline_orientation"] is True
    assert kw["device"] == "cpu"


def test_v5_variant_result_shape_matches_v4(fake_paddleocr, adapter_env, monkeypatch):
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v5-chinese")
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/recognize", json={"input_path": str(img)})
    body = r.json()
    assert body["result"]["text"] == "入口点\nEntryPoint"
    assert body["result"]["language"] == "ch"


def test_variant_cache_key_recreates_engine(fake_paddleocr, adapter_env, monkeypatch):
    img = _write_image(adapter_env)
    e1 = adapter._get_ocr()
    e2 = adapter._get_ocr()
    assert e1 is e2  # 同参复用
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v5-chinese")
    e3 = adapter._get_ocr()
    assert e3 is not e1  # 变体切换重建
    assert e3.kwargs["text_detection_model_name"] == "PP-OCRv5_server_det"


def test_unknown_variant_keeps_default_path(fake_paddleocr, adapter_env, monkeypatch):
    # EP_MODEL_ID 不在 _VARIANT_MODEL_NAMES 时与空一致（v4 行为不回归的保险丝）
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v4-chinese")
    adapter._get_ocr()
    kw = FakePaddleOCR.last_kwargs
    assert "text_detection_model_name" not in kw


def test_det_thresh_out_of_range_not_passed(fake_paddleocr, adapter_env):
    # 注：现有缓存键只含 lang/angle/variant（det 不触发重建，启动预热即
    # 0.3），故越界值只在引擎构建时刻生效——此处直接测构建路径。
    adapter._get_ocr(det_db_thresh=7.0)
    assert "text_det_box_thresh" not in FakePaddleOCR.last_kwargs


# ─── 模型目录接线：inference.yml 模型名成对传入（paddlex 3.4 一致性校验） ─────


def _make_model_dir(base, subdir, model_name):
    d = base / subdir
    d.mkdir(parents=True)
    (d / "inference.yml").write_text(
        f"Global:\n  model_name: {model_name}\n", encoding="utf-8"
    )
    (d / "inference.pdiparams").write_bytes(b"\x01" * 8)
    return d


def test_read_model_name_from_yml(adapter_env):
    d = _make_model_dir(adapter_env, "m1", "PP-OCRv4_server_det")
    assert adapter._read_model_name(d) == "PP-OCRv4_server_det"
    empty = adapter_env / "m2"
    empty.mkdir()
    assert adapter._read_model_name(empty) == ""  # 无 yml → 空串
    broken = adapter_env / "m3"
    broken.mkdir()
    (broken / "inference.yml").write_text(":\nbad yaml [", encoding="utf-8")
    assert adapter._read_model_name(broken) == ""


def test_model_dir_wiring_passes_names(fake_paddleocr, adapter_env, monkeypatch):
    # v4 目录手动落位：名称必须随目录一起传入，否则 3.4 默认名撞 v4 目录
    monkeypatch.setattr(adapter, "EP_MODEL_DIR", str(adapter_env))
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v4-chinese")
    _make_model_dir(adapter_env, "det", "PP-OCRv4_server_det")
    _make_model_dir(adapter_env, "rec", "PP-OCRv4_server_rec")
    _make_model_dir(adapter_env, "cls", "PP-LCNet_x1_0_textline_orientation")
    adapter._get_ocr()
    kw = FakePaddleOCR.last_kwargs
    assert kw["text_detection_model_dir"].endswith("det")
    assert kw["text_detection_model_name"] == "PP-OCRv4_server_det"
    assert kw["text_recognition_model_name"] == "PP-OCRv4_server_rec"
    assert kw["textline_orientation_model_name"] == "PP-LCNet_x1_0_textline_orientation"


def test_dir_names_override_variant_pin(fake_paddleocr, adapter_env, monkeypatch):
    # 用户显式落位的目录内 yml 声明优先于变体 pin（落位什么装什么）
    monkeypatch.setattr(adapter, "EP_MODEL_ID", "v5-chinese")
    monkeypatch.setattr(adapter, "EP_MODEL_DIR", str(adapter_env))
    _make_model_dir(adapter_env, "det", "PP-OCRv4_server_det")
    adapter._get_ocr()
    assert FakePaddleOCR.last_kwargs["text_detection_model_name"] == "PP-OCRv4_server_det"


# ─── doc_vl：模型根探测（纯路径逻辑，无引擎） ────────────────────────────────


def test_vl_probe_canonical_layout(adapter_env):
    root = make_vl_root(adapter_env)
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    base, layout, vl = resolved
    assert base == root
    assert layout.name == "PP-DocLayoutV2"
    assert vl.name == "PaddleOCR-VL-0.9B"


def test_vl_probe_layout_alias(adapter_env):
    make_vl_root(adapter_env, layout_name="layout")
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    assert resolved[1].name == "layout"


def test_vl_probe_vl_alias_and_flat_root(adapter_env):
    make_vl_root(adapter_env, vl_name="vl")
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    assert resolved[2].name == "vl"

    # 平铺口径：safetensors 直接在根（ModelScope 整目录导入形态）
    target = make_vl_root(adapter_env, flat=True)
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    assert resolved[0] == target
    assert resolved[2] == target


def test_vl_probe_missing_safetensors(adapter_env):
    root = make_vl_root(adapter_env)
    for p in root.rglob("*.safetensors"):
        p.unlink()
    assert adapter._resolve_vl_model_root() is None


def test_vl_probe_missing_layout(adapter_env):
    root = make_vl_root(adapter_env)
    import shutil

    shutil.rmtree(root / "PP-DocLayoutV2")
    assert adapter._resolve_vl_model_root() is None


def test_vl_probe_ep_model_dir_fallback(adapter_env, monkeypatch):
    root = make_vl_root(adapter_env)
    monkeypatch.setattr(adapter, "EP_MODELS_ROOT", "")  # 契约位缺席
    monkeypatch.setattr(adapter, "EP_MODEL_DIR", str(root))  # 激活变体即 VL 目录
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    assert resolved[0] == root


def test_vl_probe_nested_layout_dir(adapter_env):
    # 版面目录顶层不是模型产物时，下钻一层取内部模型目录
    # （沿 pp-structure-v3 的 _probe_submodel_dir 同策略）
    target = make_vl_root(adapter_env)
    inner = target / "PP-DocLayoutV2" / "PP-DocLayoutV2"
    inner.mkdir()
    (inner / "inference.pdiparams").write_bytes(b"\x01" * 8)
    # 清空顶层的 paddle 产物，使顶层不再像模型目录
    for name in ("inference.pdiparams", "inference.pdmodel", "config.json"):
        p = target / "PP-DocLayoutV2" / name
        if p.exists():
            p.unlink()
    resolved = adapter._resolve_vl_model_root()
    assert resolved is not None
    assert resolved[1] == inner


# ─── doc_vl：端点契约 ────────────────────────────────────────────────────────


def test_doc_vl_not_placed_is_503(adapter_env):
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(img)})
    assert r.status_code == 503
    body = r.json()
    assert body["error_code"] == "MODEL_NOT_LOADED"
    assert adapter.PVL_TARGET_DIR in body["detail"]


def test_doc_vl_json_body_e2e(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(img)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["output_type"] == "json"
    result = body["result"]
    assert result["markdown"] == "# 标题\n\n正文段落"
    assert result["input"]["kind"] == "image"
    assert result["options"] == {"doc_orientation": False, "doc_unwarping": False}
    # 管线构造参数：v1 管线 + native 后端 + 探测到的两个模型目录
    kw = FakePaddleOCRVL.last_kwargs
    assert kw["pipeline_version"] == "v1"
    assert kw["vl_rec_backend"] == "native"
    assert kw["layout_detection_model_dir"].endswith("PP-DocLayoutV2")
    assert kw["vl_rec_model_dir"].endswith("PaddleOCR-VL-0.9B")
    assert kw["device"] == "cpu"
    assert kw["use_doc_orientation_classify"] is False
    # predict 收到单页输入与开关
    called_input, called_kwargs = adapter._vl_engine.predict_calls[0]
    assert called_input == str(img)
    assert "use_doc_orientation_classify" in called_kwargs


def test_doc_vl_predict_single_call_per_page(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)
    with _client() as c:
        c.post("/predict/doc_vl", json={"input_path": str(img)})
    # 引擎为模块级单例，直接断言其调用记录
    engine = adapter._vl_engine
    assert engine is not None
    assert len(engine.predict_calls) == 1
    path, kwargs = engine.predict_calls[0]
    assert path == str(img)
    assert kwargs["use_doc_orientation_classify"] is False


def test_doc_vl_md_file_artifact(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)
    out = adapter_env / "out" / "doc.md"
    with _client() as c:
        r = c.post(
            "/predict/doc_vl",
            json={"input_path": str(img), "params": {"output_format": "md", "output_path": str(out)}},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["output_type"] == "file"
    assert out.read_text(encoding="utf-8") == "# 标题\n\n正文段落"


def test_doc_vl_md_without_output_path_returns_text(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(img), "params": {"output_format": "md"}})
    body = r.json()
    assert body["output_type"] == "text"
    assert body["result"].startswith("# 标题")


def test_doc_vl_switches_forwarded(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)
    with _client() as c:
        r = c.post(
            "/predict/doc_vl",
            json={"input_path": str(img), "params": {"doc_orientation": True, "doc_unwarping": True}},
        )
    assert r.status_code == 200, r.text
    _, kwargs = adapter._vl_engine.predict_calls[0]
    assert kwargs["use_doc_orientation_classify"] is True
    assert kwargs["use_doc_unwarping"] is True
    assert r.json()["result"]["options"] == {"doc_orientation": True, "doc_unwarping": True}


def test_doc_vl_unsupported_extension_is_500(fake_paddleocr, adapter_env):
    make_vl_root(adapter_env)
    doc = adapter_env / "in.doc"
    doc.write_bytes(b"xx")
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(doc)})
    assert r.status_code == 500
    assert r.json()["error_code"] == "INFERENCE_ERROR"


def test_doc_vl_missing_input_is_422(adapter_env):
    with _client() as c:
        r = c.post("/predict/doc_vl", json={})
    assert r.status_code == 422


def test_doc_vl_missing_file_is_404(adapter_env):
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(adapter_env / "nope.png")})
    assert r.status_code == 404


def test_doc_vl_engine_error_is_500(fake_paddleocr, adapter_env, monkeypatch):
    make_vl_root(adapter_env)
    img = _write_image(adapter_env)

    def boom(**kwargs):
        raise RuntimeError("vl exploded")

    monkeypatch.setattr(adapter, "_get_vl", boom)
    with _client() as c:
        r = c.post("/predict/doc_vl", json={"input_path": str(img)})
    assert r.status_code == 500
    assert r.json()["error_code"] == "INFERENCE_ERROR"


# ─── 单元：markdown 提取兼容 VL 的 page_texts 键 ─────────────────────────────


def test_extract_markdown_page_texts_key():
    res = {"markdown": {"text": "", "page_texts": ["第一页", "第二页"]}}
    assert adapter._extract_markdown(res) == "第一页\n\n第二页"


def test_extract_markdown_legacy_keys_still_work():
    assert adapter._extract_markdown({"markdown": {"text": "T"}}) == "T"
    assert adapter._extract_markdown({"markdown": {"markdown_texts": ["a", "b"]}}) == "a\n\nb"
    assert adapter._extract_markdown({"markdown": None}) == ""
