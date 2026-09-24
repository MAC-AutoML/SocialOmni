# SocialOmni：面向全模态模型的音视频社会交互评测

[论文（arXiv v3）](https://arxiv.org/abs/2603.16859v3) · [PDF](papers/socialomni-arxiv-v3.pdf) · [中英双语排行榜](https://mac-automl.github.io/SocialOmni/) · [数据集](https://huggingface.co/datasets/alexisty/SocialOmni) · [English](../README.md)

SocialOmni 是离线诊断评测，分别衡量模型能否识别**谁在说话**、判断指定参与者在标注时刻**是否应开口**、以及生成**合适的后续话语**。该协议不测量持续流式状态或真实运行延迟。

## 论文与复现材料

仓库收录 **2026 年 9 月 20 日的 arXiv:2603.16859v3**，以及原样保存的[公开复现附件](../reproducibility/arxiv-v3/README.md)。附件保留 arXiv 发布的 SHA-256 文件校验和，可离线核验已有结果；它不是全部模型 API 实验的完整重跑环境。

论文使用 **2,000 条感知样本**与 **200 条交互核心样本**（`video_0001`–`video_0200`，其中 128 条应回应）。公开交互数据集有 209 条，复现论文比较时应使用固定的 200 条核心标注。

离线核验不需要 GPU、视频或 API 密钥：

```bash
cd reproducibility/arxiv-v3
uv sync --python 3.13 --frozen
uv run python scripts/verify_package.py
```

来源及文件校验信息见[归档说明](papers/README.md)。

## 评测协议

- **Who**：四选一的说话者归属判断准确率。
- **When**：仅依据查询时刻及之前的音视频，判断指定参与者是否应开口。分类指标与回答质量分别报告。
- **QGold**：全部 128 条应回应样本的平均回答质量；即使模型预测 NO，也强制生成回答。
- **QEns**：应回应且模型预测 YES、生成非空回答的样本平均质量。
- **Cov+**：应回应样本中，模型预测 YES 且生成非空回答的比例。
- **QEns_joint**：`QEns × Cov+ / 100`；漏掉的应回应机会贡献零分。

新评测的回答质量分使用 **Gemini 3.8 Flash、Qwen3.8-Omni-Flash、GPT-5.6-Sol**。每条符合条件的回答必须具有三份完整评分，分档为 {0, 25, 50, 75, 100}，零分不能过滤。标准后续话语和人工核验的评委上下文不提供给被测模型。推理设置、提示词与解析规则见论文附录 A.8–A.10。

How 通过 QGold 和 QEns 衡量回答质量；Cov+ 和 QEns_joint 衡量开口决策与回答的联合表现。GPT-4o、Gemini 3 Pro 和 OmniVinci 保留论文整行指标及原评委来源。

## 主要结果

新评测的回答质量由 **Gemini 3.8 Flash、Qwen3.8-Omni-Flash 和 GPT-5.6-Sol** 评分，此前模型的已有回答也已重新评分。全部模型在[中英双语排行榜](https://mac-automl.github.io/SocialOmni/)中展示，可按各项指标排序。

| 模型 | Who | When | How: QGold | How: QEns | Cov+ | QEns_joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash | 25.40 | 82.50 | 86.52 | 86.48 | 82.81 | 71.61 |
| Gemini 3.1 Pro Preview | 45.05 | 84.00 | 80.01 | 81.92 | 82.81 | 67.84 |
| VITA-1.5 | 32.05 | 65.50 | — | 54.79 | 88.28 | 48.37 |
| Gemini 3.5 Flash | 6.00 | 60.50 | 79.56 | 82.95 | 50.78 | 42.12 |
| Qwen3.5-Omni-Plus | 91.05 | 58.50 | 80.27 | 77.82 | 48.44 | 37.70 |
| Qwen2.5-Omni | 4.15 | 61.50 | 49.35 | 49.90 | 64.06 | 31.97 |
| Qwen3-Omni-Thinking | 67.65 | 56.00 | — | 61.53 | 46.88 | 28.84 |
| Qwen3-Omni | 74.65 | 58.00 | — | 44.96 | 63.28 | 28.45 |
| OmniVinci | 29.75 | 64.50 | 40.82 | 38.52 | 70.31 | 27.08 |
| Qwen3.5-Omni-Flash (2026-03-15) | 86.55 | 71.00 | 33.14 | 34.72 | 70.31 | 24.41 |
| GPT-4o | 35.05 | 50.50 | 77.15 | 76.50 | 30.47 | 23.31 |
| Gemini 2.5 Flash | 33.70 | 55.50 | — | 30.30 | 42.97 | 13.02 |
| Qwen3.8-Omni-Flash | 89.45 | 44.00 | 77.02 | 69.93 | 17.97 | 12.57 |
| Gemini 2.5 Pro | 39.90 | 52.50 | — | 20.97 | 48.44 | 10.16 |
| Gemini 3 Pro | 45.40 | 52.00 | 21.55 | 25.00 | 32.03 | 8.01 |
| Gemini 3 Flash | 48.10 | 49.50 | — | 21.59 | 34.38 | 7.42 |
| Baichuan-Omni-1.5 | 22.45 | 36.50 | — | 31.25 | 12.50 | 3.91 |

所有指标为 0–100 标度，越高越好；— 表示缺少所需原始回答，暂不报告该指标。

[逐样本回答、评分与评测设置](../evaluation/results/modern-panel-20260924/README.md) · [历史材料归档](../evaluation/results/archive/README.md)

## API 模型评测

[评测程序与运行说明](../evaluation/hosted-omni/README.md)支持 Gemini 3.8 Flash、Qwen3.8-Omni-Flash、Qwen3.5-Omni-Plus 和 Qwen3.5-Omni-Flash，自动选择对应的音视频输入参数。新模型生成与已有回答重评分默认使用同一套三评委配置，支持断点恢复并保留逐次请求记录。

## ⚙️ 环境与安装

推荐环境如下：

- Python `>=3.10,<3.11`
- 支持 CUDA 的 PyTorch 运行环境
- 使用 `uv` 进行依赖和环境管理

安装方式：

```bash
git clone https://github.com/MAC-AutoML/SocialOmni.git
cd SocialOmni
uv sync
```

## 🚀 快速开始

以下开发入口使用公开数据集及各自配置的提示词、评委。默认运行不等于复现表 2；核验论文数字请使用上述固定快照。

### 1. 配置路径与运行参数

编辑 `config/config.yaml`，至少配置以下内容：

- API endpoint（密钥放在 `.env` 的 `OPENAI_API_KEY`，不写入 YAML）
- 本地模型路径或 `server_url`
- 数据集路径
- 输出目录和结果目录

常见环境变量包括：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`

### 2. 启动本地模型服务

例如：

```bash
uv run models/model_server/qwen3_omni/qwen3_omni_server.py
```

其他模型服务入口位于：

```text
models/model_server/*/*_server.py
```

### 3. 运行任务一评测

```bash
uv run run_benchmark.py --model qwen3_omni
```

### 4. 运行任务二评测

```bash
uv run run_benchmark_level2.py --model qwen3_omni --resume
```

## 🧱 仓库结构

```text
SocialOmni/
├── config/                  # 运行时、模型和评测配置
├── data/                    # 本地数据集（默认不纳入版本管理）
├── docs/                    # 文档和可视化素材
├── models/                  # 模型服务、客户端与共享 benchmark 逻辑
├── scripts/                 # 工具脚本
├── run_benchmark.py         # 任务一入口
├── run_benchmark_level2.py  # 任务二入口
├── pyproject.toml           # 依赖定义
└── README.md
```

## 🔑 支持的模型键

`--model` 可选值如下：

```text
gpt4o
gemini_2_5_flash
gemini_2_5_pro
gemini_3_flash_preview
gemini_3_pro_preview
qwen3_omni
qwen3_omni_thinking
qwen2_5_omni
miniomni_2
omnivinci
vita_1_5
baichuan_omni_1_5
ming
```

## 🧪 可复现建议

- 数据目录和结果目录建议本地保存，不纳入版本管理。
- 跨模型比较时固定 prompt 模板和运行配置。
- 报告改进时建议同时给出子集指标和置信区间。
- 生成任务评测时保持 judge 组合一致。

## ✏️ 引用

如果 SocialOmni 对你的研究有帮助，请引用：

```bibtex
@article{xie2026socialomni,
  title={SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models},
  author={Xie, Tianyu and Huang, Jinfa and Ma, Yuexiao and Luo, Rongfang and Yang, Yan and Ma, Qingchuan and Chen, Wang and Zeng, Yuhui and Zou, Yixuan and Lu, Zhiqiang and Fang, Ruize and Luo, Jiebo and Ji, Rongrong and Zheng, Xiawu},
  journal={arXiv preprint arXiv:2603.16859},
  year={2026}
}
```
