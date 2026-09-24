<p align="center">
  <img src="docs/assets/socialomni_logo.png" alt="SocialOmni" width="320" />
</p>

# SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models

[Paper (arXiv v3)](https://arxiv.org/abs/2603.16859v3) · [PDF](docs/papers/socialomni-arxiv-v3.pdf) · [Leaderboard](https://teeryxie.github.io/socialomni/) · [Dataset](https://huggingface.co/datasets/alexisty/SocialOmni) · [中文](docs/README.zh-CN.md)

SocialOmni is an offline diagnostic benchmark for audio-visual social interaction. It evaluates **who** is speaking, **when** a designated participant should enter at an annotated query time, and **how** that participant should continue the dialogue. It does not measure persistent streaming state or wall-clock response latency.

## Paper and reproducibility

This repository includes the September 20, 2026 revision, **arXiv:2603.16859v3**, and its [public reproducibility package](reproducibility/arxiv-v3/README.md). The package is an unchanged copy of the arXiv ancillary files, with the original SHA-256 checksums. It supports offline verification of the recorded results; it is not a complete environment for rerunning all model APIs.

The paper uses **2,000 perception items** and a **200-item interaction core** (`video_0001`–`video_0200`, including 128 positive entry states). The broader public interaction dataset has 209 items. Use the frozen core annotations for paper comparisons.

To verify the archived results without GPUs, video files or API credentials:

```bash
cd reproducibility/arxiv-v3
uv sync --python 3.13 --frozen
uv run python scripts/verify_package.py
```

See the [archive manifest](docs/papers/README.md) for sources and file integrity.

## Evaluation protocol

- **Who:** four-choice speaker attribution, reported as accuracy.
- **When:** a fixed-time YES/NO decision from query-time-bounded audio and video. Report classification metrics separately from response quality.
- **QGold:** mean quality over all 128 gold-positive items, forcing generation even when the model predicts NO.
- **QEns:** mean quality of non-empty responses at true-positive entry decisions.
- **Cov+:** the percentage of gold-positive items with a predicted YES and a non-empty response.
- **QEns_joint:** `QEns × Cov+ / 100`; missed positive opportunities contribute zero.

Displayed response-quality results use **Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol**. Each eligible response requires all three scores from {0, 25, 50, 75, 100}; zero scores are retained. Human references and manually verified judge context are not inputs to the evaluated model. Appendix A.8–A.10 describes inference settings, prompts and parsing.

## Main results

All displayed quality scores use the September 24 judge panel, on a 0–100 scale. Hosted runs and archived paper answers retain separate groups because their input and classification protocols differ. See the [per-item results and settings](evaluation/results/modern-panel-20260924/README.md) for scores and the output-format diagnostic.

### Hosted candidates

| Model | Who | When | QGold | QEns | Cov+ | QEns_joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Omni-Flash | 89.45 | 44.00 | 77.02 | 69.93 | 17.97 | 12.57 |
| Gemini 3.8 Flash | 25.40 | 82.50 | 86.52 | 86.48 | 82.81 | 71.61 |
| Qwen3.5-Omni-Plus | 91.05 | 58.50 | 80.27 | 77.82 | 48.44 | 37.70 |
| Qwen3.5-Omni-Flash (2026-03-15) | 86.55 | 71.00 | 33.14 | 34.72 | 70.31 | 24.41 |

Who and When use strict label parsing; request failures remain in the denominators. Auxiliary format extraction is diagnostic and does not replace these scores.

### Archived paper answers, rescored

The 688 archived responses are reused. Who, When and coverage retain their original outputs. QGold is unavailable because the archive does not contain all 128 forced responses per model.

| Model | Who | When | QGold | QEns | Cov+ | QEns_joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o | 35.05 | 50.50 | — | 77.56 | 30.47 | 23.63 |
| Gemini 2.5 Pro | 39.90 | 52.50 | — | 20.97 | 48.44 | 10.16 |
| Gemini 2.5 Flash | 33.70 | 55.50 | — | 30.30 | 42.97 | 13.02 |
| Gemini 3 Flash | 48.10 | 49.50 | — | 21.59 | 34.38 | 7.42 |
| Gemini 3 Pro | 45.40 | 52.00 | — | 32.11 | 32.03 | 10.29 |
| Qwen3-Omni | 74.65 | 58.00 | — | 44.96 | 63.28 | 28.45 |
| Qwen3-Omni-Thinking | 67.65 | 56.00 | — | 61.53 | 46.88 | 28.84 |
| Qwen2.5-Omni | 40.95 | 60.50 | — | 34.67 | 67.97 | 23.57 |
| OmniVinci | 29.75 | 64.50 | — | 41.67 | 70.31 | 29.30 |
| VITA-1.5 | 32.05 | 65.50 | — | 54.79 | 88.28 | 48.37 |
| Baichuan-Omni-1.5 | 22.45 | 36.50 | — | 31.25 | 12.50 | 3.91 |

The original paper panel and the earlier substitute panel remain in the [historical materials archive](evaluation/results/archive/README.md), together with the original data, scores, paper and verification scripts.

## Hosted-model supplemental evaluation

The [standalone hosted-model runner](evaluation/hosted-omni/README.md) uses the v3 item lists, query-time prefixes and prompt cards with a complete three-judge scoring stage. It has a separate lightweight Python environment and an immutable request cache. Supplemental runs remain separate from the frozen paper results.

The [September 24 supplement](evaluation/results/modern-panel-20260924/README.md) rescores 688 archived answers with Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol. It also reports Gemini 3.8 Flash and both Qwen3.5-Omni variants alongside the existing Qwen3.8 run, with all 3,600 judge scores, candidate outputs and an auxiliary output-format diagnostic.

## ⚙️ Requirements and Installation

We recommend the following environment:

- Python `>=3.10,<3.11`
- CUDA-compatible PyTorch runtime for local omni models
- `uv` for dependency and environment management

Install with:

```bash
git clone https://github.com/MAC-AutoML/SocialOmni.git
cd SocialOmni
uv sync
```

## 🚀 Quick Start

These development entrypoints operate on the public dataset and use their configured prompts and judges. Running the defaults does not reproduce Table 2 automatically; use the frozen snapshot above to verify the published numbers.

### 1. Configure runtime and paths

Recommended setup:

- Put the single OpenAI-compatible API credential pair in `.env`
- Put non-sensitive defaults such as local model paths, `server_url`, dataset paths,
  output directories, and log directories in `config/config.yaml`

Start from the provided template:

```bash
cp .env.example .env
```

Then edit `.env` and set the API credential pair:

```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=...
```

Then edit `config/config.yaml` and set:

- local model path or `server_url`
- dataset path
- output and result directories

Notes:

- All hosted API models in this repo, including Gemini model keys, use the same
  OpenAI-compatible `OPENAI_API_KEY` and `OPENAI_API_BASE` configuration.
- API credentials should live in `.env`, not in `config/config.yaml`.
- API models do not require local weights.
- Local omni models require a valid `model_path` and usually a local `server_url`.
- If you leave `benchmark.level1.dataset_path`, `benchmark.level1.video_dir`,
  `benchmark.level2.dataset_path`, and `benchmark.level2.video_dir` empty, the
  benchmark uses the default `data/` layout shown below.

Dataset source:

- Hugging Face dataset: `alexisty/SocialOmni`
- Default local target directory: `data/`

If you keep the default benchmark paths, the runner will auto-download missing
benchmark data into `data/level_1` or `data/level_2` on first use.
To disable this behavior, set:

```bash
export SOCIALOMNI_AUTO_DOWNLOAD_DATASET=0
```

You can also download the benchmark data manually:

```bash
uv run python scripts/download_dataset.py --level all
```

Default expected layout:

```text
data/
├── level_1/
│   ├── dataset.json
│   └── videos/
└── level_2/
    ├── annotations.json
    └── videos/
```

Common environment variables:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `SOCIALOMNI_AUTO_DOWNLOAD_DATASET`

### 2. Start a local model server

Example:

```bash
uv run models/model_server/qwen3_omni/qwen3_omni_server.py
```

Other model server entrypoints are located under:

```text
models/model_server/*/*_server.py
```

### 3. Run Task I benchmark

```bash
uv run run_benchmark.py --model qwen3_omni
```

### 4. Run Task II benchmark

```bash
uv run run_benchmark_level2.py --model qwen3_omni --resume
```

## 🧱 Repository Structure

```text
SocialOmni/
├── config/                  # runtime, model, and evaluation configs
├── data/                    # local datasets (not tracked)
├── docs/                    # docs and visual assets
├── models/                  # model servers, clients, and shared benchmark logic
├── scripts/                 # utility scripts
├── run_benchmark.py         # Task I entrypoint
├── run_benchmark_level2.py  # Task II entrypoint
├── pyproject.toml           # dependency definition
└── README.md
```

## 🔑 Supported Model Keys

Use the following keys with `--model`:

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

## 🧪 Reproducibility Notes

- Keep dataset and result directories local and out of version control.
- Use fixed prompt templates and stable runtime configs for cross-model comparison.
- Report split-wise metrics and confidence intervals when claiming improvements.
- For generation evaluation, keep the judge set fixed across runs.

## ✏️ Citation

If you find SocialOmni useful in your research, please cite:

```bibtex
@article{xie2026socialomni,
  title={SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models},
  author={Xie, Tianyu and Huang, Jinfa and Ma, Yuexiao and Luo, Rongfang and Yang, Yan and Ma, Qingchuan and Chen, Wang and Zeng, Yuhui and Zou, Yixuan and Lu, Zhiqiang and Fang, Ruize and Luo, Jiebo and Ji, Rongrong and Zheng, Xiawu},
  journal={arXiv preprint arXiv:2603.16859},
  year={2026}
}
```
