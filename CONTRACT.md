# paddleocr 模块契约

## 1. 能力与端点

| 能力 | 输入 → 输出 | 引擎 | 激活变体 |
|---|---|---|---|
| `recognize` | image → json（text + lines[text/confidence/bbox]） | PaddleOCR 3.x 通用 OCR 管线 | v4-chinese / v4-multilingual / **v5-chinese** |
| `doc_understand` | image/PDF → Markdown+页面结构 | PPStructureV3（14 子模型） | pp-structure-v3（local_import） |
| `doc_vl`（T-45 新增） | image/PDF → Markdown+页面结构 | PaddleOCRVL v1（VL-0.9B + PP-DocLayoutV2） | paddleocr-vl（local_import） |

- 端点：`GET /health`、`GET /info`、`POST /predict/recognize`（multipart `file`/`input_path` +
  `params` JSON，或 JSON body；`/predict/recognize/json` 为纯 JSON 变体）、
  `POST /predict/doc_understand`、`POST /predict/doc_vl`（与 doc_understand 同输入契约）。
- 参数覆盖链：`params` JSON > 平铺表单字段（兼容旧版）。文件产物模式（MODULE_SPEC §5）：
  `params.output_format != json` + 执行器注入 `output_path` → 写文件，返回 `output_type=file`。
- 错误语义：404 输入不存在 / 413 超限（recognize 20MB、doc_understand 与 doc_vl 100MB）/
  422 缺输入 / **503 `MODEL_NOT_LOADED`**（pp-structure-v3、paddleocr-vl 模型未落位，不静默联网下载）/
  500 `INFERENCE_ERROR`。
- 变体激活：daemon 注入 `EP_MODEL_ID`（变体 id）+ `EP_MODEL_DIR`（`<models_cache>/<target_dir>`）
  + `EP_MODELS_ROOT`（模型缓存根）。**注意 `det_db_thresh` 不在引擎缓存键内**（缓存键 =
  lang/angle/variant），仅引擎（重）构建时刻生效，启动预热后改参不重建——存量语义，未在 T-45 改动。

## 2. 变体与识别引擎口径

| 变体 id | target_dir | 获取方式 | 引擎模型 |
|---|---|---|---|
| `v4-chinese`（default） | paddleocr-v4-chinese | url=auto（模块自管） | 见下「v4 澄清」 |
| `v4-multilingual` | paddleocr-v4-multilingual | url=auto | 同上（多语言 rec） |
| `v5-chinese`（T-45） | paddleocr-v5-chinese | url=auto | **显式 pin `PP-OCRv5_server_det` + `PP-OCRv5_server_rec`** |
| `pp-structure-v3` | paddleocr-pp-structure-v3 | local_import（14 子模型目录） | PPStructureV3 |
| `paddleocr-vl`（T-45） | paddleocr-vl-0.9b | local_import（版面 + HF 权重） | PaddleOCRVL `pipeline_version="v1"`、`vl_rec_backend="native"` |

- **v4 澄清（源码核查结论，诚实记录）**：paddleocr 3.4 的 auto 路径（不传模型名）对
  lang=ch 默认解析为 `PP-OCRv5_server_det` / `PP-OCRv5_server_rec`（`paddleocr/_pipelines/ocr.py`
  v3.4.1 L446-482）。故 v4 变体在「不手动落位模型目录」时实际跑的是 v5 server 权重——
  v4 标签只在 `EP_MODEL_DIR/det|rec` 放有 v4 模型时成立。T-45 未改动该行为（红线）；
  需要确定性 v4/v5 身份时用手动落位（目录接线会把目录内 `inference.yml` 声明的模型名
  成对传入，满足 paddlex 3.4 的 name/dir 一致性校验）。
- **v5 显式 pin 的动机**：上游默认漂移（paddleocr 3.7 起 auto 默认切 PP-OCRv6_medium_*），
  pin 死 `PP-OCRv5_server_*` 后变体身份不受升级影响。
- 依赖口径：全变体共用一套 venv（`paddleocr==3.4.*` + `paddlex[ocr]==3.4.*` +
  `paddlepaddle>=3.0,<4`）。VL 原生后端要求的 `paddlepaddle>=3.2.1`（下行上界内满足）与
  `safetensors`（paddlex[ocr] 3.4.x extras 自带，实测 0.8.0）均无需新增依赖行。

## 3. 密钥与模型

- **零密钥**：纯本地推理，无 API key。
- `url=auto` 语义：平台建空目录、不下载；paddleocr 首次构造时从 BOS 官方源
  （国内 CDN）按模型名单模型拉取到 `~/.paddlex/official_models/`。
- **手动落位口径（优先 ModelScope 单文件直链，国内直连实测秒级）**：

  | 模型 | ModelScope 直链（`https://modelscope.cn/models/<repo>/resolve/master/<file>`） |
  |---|---|
  | v5 检测 | `PaddlePaddle/PP-OCRv5_server_det`：config.json / inference.json / inference.pdiparams / inference.yml |
  | v5 识别 | `PaddlePaddle/PP-OCRv5_server_rec`：同上 4 文件 |
  | v4 检测（对照） | `PaddlePaddle/PP-OCRv4_server_det`：同上 4 文件 |
  | v4 识别（对照） | `PaddlePaddle/PP-OCRv4_server_rec`：同上 4 文件 |
  | VL 版面 | `PaddlePaddle/PaddleOCR-VL`：`PP-DocLayoutV2/`（config.json / inference.pdiparams / inference.pdmodel / inference.yml） |
  | VL 权重 | `PaddlePaddle/PaddleOCR-VL` 根：model.safetensors + config.json + generation_config.json + tokenizer*（.json/.model/_config.json）+ special_tokens_map.json + added_tokens.json + preprocessor/processor_config.json + chat_template.jinja + *_{image_processing,processing,modeling,configuration}_paddleocr_vl.py + inference.yml |
  | 行方向分类 | ModelScope 无官方单仓；url=auto 时 paddleocr 自取（~6.6MB），或手动放 `cls/` |

  - v4/v5 手动落位结构：`<target_dir>/det|rec|cls/`（目录名固定，adapter 按此接线）。
  - VL 落位结构（`paddleocr-vl-0.9b/`）：`PP-DocLayoutV2/`（或 `layout/`）+ `PaddleOCR-VL-0.9B/`
    （或 `vl/`，或 safetensors 平铺根目录）；adapter 探测目录内容、不硬编码内部文件名。
- **许可证核对（T-45 验收项）**：PaddleOCR-VL 权重仓 LICENSE = **Apache-2.0**（ModelScope
  `PaddlePaddle/PaddleOCR-VL` LICENSE 文件实读核对，2026-10-29）；PP-OCRv5 server 系列同为
  Apache-2.0（PaddlePaddle 官方模型仓统一许可）。

## 4. 占用与性能（Tesla P4 主机实测，i7-1165G7 8 线程，2026-10-29）

**重要口径：P4（Pascal，sm_61）上本模块全部实测为 CPU 路径**——paddlepaddle 3.x GPU 轮子
不支持 Pascal（官方架构表自 sm_70 起），且 PaddleOCR-VL 原生后端官方要求 GPU 计算能力
≥7.0（vLLM/SGLang ≥8.0）。测试期间 nvidia-smi 显存增量 0 MiB。**GPU 档位数据空缺，
待 sm_70+ 真机补测，不以任何估算值填充。**

| 项 | v4-chinese（真 v4 server 权重） | v5-chinese | paddleocr-vl |
|---|---|---|---|
| 模型落位体积 | 204 MB（det 113 + rec 90） | 174 MB（det 88 + rec 84） | 2.14 GB（版面 212 MB + safetensors 1.9 GB） |
| 行方向分类模型 | ~6.6 MB（`~/.paddlex/official_models`，url=auto 自取） | 同左 | —（VL 端到端） |
| 进程 RSS（装载后） | 842 MB | 901 MB | 5.98 GB（含 0.9B fp32 权重） |
| 单张耗时（CPU，zh 样张 3 行） | 2.2–2.4 s | **1.6–1.9 s** | — |
| 单张耗时（CPU，en 样张 3 行） | 2.7–2.9 s | **2.0–2.4 s** | — |
| 单页耗时（CPU，doc 样张 1 页） | — | — | 首呼 37.1 s（含管线装载+转换）；稳态 **23.7–24.3 s/页** |
| 显存增量（P4） | 0（CPU 路径） | 0（CPU 路径） | 0（CPU 路径；GPU 需 CC≥7.0，待真机） |
| venv 体积（cpu 轮子） | 1.5 GB（三变体共用） | 同左 | 同左 |

- 样张与判据：本机渲染 PNG（`tests/make_samples.py`，字体 msyh/arial），ground truth
  随样张（*.gt.txt）；**同批样张 v4 server vs v5 server 精度抽查：中文 3/3 行、英文 3/3 行
  双双全对**（清洗样张上二者不可分）；行置信度：zh v4 [0.9997,0.9992,0.9997] vs v5
  [0.9992,0.9995,0.9990]；en v4 [0.961,0.998,0.987] vs v5 [0.994,0.952,0.951]——量级相当。
  v5 的 CPU 速度优势约 25-30%（v5 为 v4 同架构升级、识别头更小）。
- VL 输出抽样：`sample_doc.png` → 标题提升为 `##` 级 Markdown 标题、段落顺序保持；
  `output_format=md + output_path` 文件产物模式实测通过；同实例 `doc_understand` 正确
  503（能力/模型集互相隔离）。
- 服务器部署记录：`bob@192.168.6.141:~/module-test/paddleocr-t45/`（venv = uv + python
  3.12 + 阿里镜像；paddleocr 3.4.1 / paddlex 3.4.3 / paddle 3.3.1 cpu / safetensors 0.8.0；
  模型经 ModelScope 单文件 resolve 直链落位；模块仓 pytest 服务器侧 42/42 通过）。

## 5. 边界与诚实声明

- **P4 GPU 不支持**：paddlepaddle 3.x 官方预编译轮子不含 Pascal（sm_61）；`module.toml`
  的 `default_backend="cuda"` 在此类主机上由 paddle 自动回退 CPU（仅告警）。GPU 数据
  （耗时/显存增量）在 sm_70+ 真机补测前一律留空。
- **PaddleOCR-VL 档位定位**：CPU 原生后端可用但慢（本样张 ~24 s/页），定位为「质量档/
  离线文档解析」，不建议实时管线使用；生产提速路径为 vLLM/SGLang 服务后端
  （`vl_rec_backend` 参数已留口，均需 CC≥8.0 GPU，本模块未实现拉起逻辑）。
- **VL 精度未做基准**：仅单样张抽查（输出结构正确），未跑 OmniDocBench 级评测；olmOCR-Bench
  96.3 为上游官方宣称值，非本机复测。
- `chart` 开关（doc_understand）保持默认 false：paddlex 3.4.3 官方 bug，开启即
  AttributeError（W2 已记录，与 T-45 无关）。
- PDF 页数上限：doc_vl 50 页（PVL_MAX_PDF_PAGES，VL 单页慢，上限较 doc_understand 的
  100 页收紧）；dpi 150-600 共用。
- `det_db_thresh` 引擎缓存键缺项（§1）为存量行为，修复涉及 recognize 缓存语义，
  非本票范围（如需另开小票）。
