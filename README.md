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

The fixed primary judges are **GPT-4o, Gemini 2.5 Pro and Qwen3-Omni**. Each eligible response requires all three scores from {0, 25, 50, 75, 100}; zero scores are retained. Human references and manually verified judge context are not inputs to the evaluated model. Appendix A.8–A.10 describes inference settings, prompts and parsing.

## Main results

Table 2 of arXiv v3. All values use a 0–100 scale. These are archived paper results, not new runs of the repository entrypoints.

| Model | Interface | Who | When | QGold | QEns | Cov+ | QEns_joint |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-4o | Cascade | 35.05 | 50.50 | 77.15 | 76.50 | 30.47 | 23.31 |
| Gemini 2.5 Pro | Visual-only | 39.90 | 52.50 | 15.62 | 12.37 | 48.44 | 5.99 |
| Gemini 2.5 Flash | Visual-only | 33.70 | 55.50 | 21.35 | 21.36 | 42.97 | 9.18 |
| Gemini 3 Flash | Visual-only | 48.10 | 49.50 | 13.93 | 17.61 | 34.38 | 6.05 |
| Gemini 3 Pro | Visual-only | 45.40 | 52.00 | 21.55 | 25.00 | 32.03 | 8.01 |
| Qwen3-Omni | Native AV | 74.65 | 58.00 | 44.47 | 42.59 | 63.28 | 26.95 |
| Qwen3-Omni-Thinking | Native AV | 67.65 | 56.00 | 67.77 | 64.86 | 46.88 | 30.40 |
| Qwen2.5-Omni | Native AV | 40.95 | 60.50 | 36.00 | 33.14 | 67.97 | 22.53 |
| OmniVinci | Native AV | 29.75 | 64.50 | 40.82 | 38.52 | 70.31 | 27.08 |
| VITA-1.5 | Native AV | 32.05 | 65.50 | 48.37 | 49.93 | 88.28 | 44.08 |
| Baichuan-Omni-1.5 | Native AV | 22.45 | 36.50 | 18.36 | 19.27 | 12.50 | 2.41 |

`Native AV` denotes joint audio-video input. GPT-4o uses a cascade of prefix transcription and video frames; the Gemini interfaces in this snapshot receive visual frames only. Input interfaces are part of each evaluated configuration.

The always-YES baseline reaches **64% When accuracy** on the core split. The six metrics describe different abilities and are not combined into an overall score. Newly evaluated configurations are listed separately on the [bilingual leaderboard](https://teeryxie.github.io/socialomni/) unless protocol equivalence has been established.

## Hosted-model supplemental evaluation

The [standalone hosted-model runner](evaluation/hosted-omni/README.md) uses the v3 item lists, query-time prefixes and prompt cards with a complete three-judge scoring stage. It has a separate lightweight Python environment and an immutable request cache. Supplemental runs remain separate from the frozen paper results.

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
