# SocialOmni：面向 Omni 模型的音视频社会交互基准

<p align="center">
  <img src="assets/socialomni_logo.png" alt="SocialOmni Logo" width="320" />
</p>

<h2 align="center">SocialOmni：面向 Omni 模型的音视频社会交互基准</h2>
<h5 align="center">一个联合评测 <i>who</i>、<i>when</i>、<i>how</i> 三个维度的 Omni 对话交互基准。</h5>

<p align="center">
  <a href="../README.md">English</a>
  ·
  <a href="#-基准概览">基准概览</a>
  ·
  <a href="#-快速开始">快速开始</a>
  ·
  <a href="#-主要结果">主要结果</a>
</p>

SocialOmni 是一个面向 omni-modal large language models（OLMs）的**音视频社会交互能力**评测基准。与只关心最终答案是否正确的静态评测不同，SocialOmni 聚焦模型在真实对话中的交互行为，联合评估三个紧密耦合的维度：

- **Who**：谁在说话，是否能正确识别当前说话人
- **When**：什么时候该介入，是否能把握合适的打断时机
- **How**：如何回应，是否能生成自然且符合语境的插话内容

本仓库提供完整的 benchmark 流水线、模型客户端与服务端、运行配置，以及可复现的感知任务与交互生成任务评测入口。

## 😮 亮点

### 1. 面向“社会交互”而不是静态理解

现有 omni 模型评测大多仍然围绕静态问答和答案正确率展开。SocialOmni 关注的是多方对话中的真实交互能力，因为在现实场景里，一个答案即使语义正确，也可能因为打断时机错误或续说不自然而导致整体交互失败。

### 2. 统一的 who-when-how 联合评测协议

SocialOmni 将对话交互能力具体化为一个联合画像：

- **Who**：模型能否在目标时刻定位真实说话人
- **When**：模型能否判断此刻是否适合介入
- **How**：模型能否给出上下文一致、语气自然的打断内容

这套设计可以直接暴露“感知强但交互差”或“能续说但不会择机”的模型失配问题。

### 3. 交互失败通过感知与生成联合刻画

SocialOmni 不只是给出总分，而是显式刻画：

- 音视频一致 / 不一致条件下的鲁棒性
- 打断时机判断的 Precision / Recall / F1
- 插话内容的 LLM Judge 质量评分
- 感知、时机、生成三轴之间的解耦现象

<p align="center">
  <img src="assets/socialomni_result_radar.png" alt="SocialOmni Cross-Axis Capability Profiles" width="88%" />
</p>

## 🔍 基准概览

<p align="center">
  <img src="assets/socialomni_overview.png" alt="SocialOmni Overview" width="100%" />
</p>

### 数据概况

- **2,209** 条 benchmark 样本
- **2,000** 条说话人感知样本
- **209** 条交互生成样本
- 覆盖 **15** 个对话子领域
- 显式划分音视频**一致 / 不一致**子集，用于鲁棒性分析

## 🧩 任务定义

### 任务一：感知（`who`）

给定一个视频片段和时间点 `t`，模型需要回答：

> 在时间点 `t`，谁在说话？

模型从 `{A, B, C, D}` 中选择一个选项。

### 任务二：交互生成（`when` + `how`）

给定视频前缀 `V[0:t]` 和候选说话者 `X`，模型需要完成两个子问题：

- **Q1（`when`）**：`X` 是否应该在 `t` 之后立刻打断
- **Q2（`how`）**：如果应该打断，合适的插话内容是什么

## 📏 评测协议

### 感知任务指标

- Top-1 Accuracy
- 一致 / 不一致子集准确率
- 差值指标：

```text
Δ = Acc_consistent - Acc_inconsistent
```

### 生成任务指标

- **Q1**：在容忍窗口（例如 `δ = 0.2s`）下计算 Accuracy / Precision / Recall / F1
- **Q2**：在 `{0, 25, 50, 75, 100}` 上进行 LLM Judge 打分

论文协议中，Q2 默认使用三位评审模型：

- GPT-4o
- Gemini 3 Pro
- Qwen3-Omni

## 🐳 主要结果

### SocialOmni 揭示了明显的跨轴解耦

感知能力强，并不意味着交互能力强；会自然续说，也不等于真的知道该在什么时候介入。SocialOmni 的核心价值就在于把这种解耦显式量化出来。

| 模型 | Who (%) | When Acc. (%) | How (/100) |
|---|---:|---:|---:|
| GPT-4o | 36.75 | 46.89 | 69.64 |
| Gemini 2.5 Pro | 44.69 | 55.67 | 72.32 |
| Gemini 2.5 Flash | 47.03 | 61.50 | **85.08** |
| Gemini 3 Flash Preview | 53.23 | 61.06 | 79.08 |
| Gemini 3 Pro Preview | 64.99 | **67.31** | 81.77 |
| Qwen3-Omni | **69.25** | 63.64 | 45.57 |

关键观察：

- **Who 最强**：Qwen3-Omni
- **When 最强**：Gemini 3 Pro Preview
- **How 最强**：Gemini 2.5 Flash

这说明单一总分不足以描述 omni 模型的真实对话能力，必须联合评测整个交互画像。

## ⚙️ 环境与安装

推荐环境如下：

- Python `>=3.10,<3.11`
- 支持 CUDA 的 PyTorch 运行环境
- 使用 `uv` 进行依赖和环境管理

安装方式：

```bash
git clone https://github.com/Alexisxty/SocialOmni.git
cd SocialOmni
uv sync
```

## 🚀 快速开始

### 1. 配置路径与运行参数

编辑 `config/config.yaml`，至少配置以下内容：

- API key / API endpoint
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
@misc{socialomni,
  title={SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models},
  author={Tianyu Xie and Jinfa Huang and Yuexiao Ma and Rongfang Luo and Yan Yang and Wang Chen and Yuhui Zeng and Ruize Fang and Yixuan Zou and Xiawu Zheng and Jiebo Luo and Rongrong Ji}
}
```
