# PaddleOCR 文字识别

本仓库是 [EntryPoint](https://github.com/PegionFish/EntryPoint) 对上游
[PaddleOCR 文字识别](https://github.com/PaddlePaddle/PaddleOCR) 的适配层镜像，反向同步自主仓库。

- 同步自: https://github.com/PegionFish/EntryPoint（source commit: `d74d40f1`，2026-08-26）
- 上游: https://github.com/PaddlePaddle/PaddleOCR
- 主仓库对应目录: modules/paddleocr
- 同步工具: scripts/sync-model-repos.sh
- 用法文档见仓库根 README.md；模块接口见主仓库 docs/MODULE_SPEC.md

## 内容

| 文件 | 说明 |
|------|------|
| `adapter.py` | HTTP 推理适配器（FastAPI 服务） |
| `module.toml` | EntryPoint 模块清单（模型注册、参数 schema、后端要求） |
| `requirements*.txt` | 依赖（默认/cuda/rocm/openvino 按后端分流） |
| `upstream.json` | 同步血缘元数据 |
| `README.md` | 模块用法文档 |

## 同步

```bash
# 从主仓库刷新
> /home/bob/EntryPoint/scripts/sync-model-repos.sh --only paddleocr

# 推送 GitHub（origin 自建）
git remote add origin git@github.com:<you>/paddleocr.git
git push -u origin main
```
