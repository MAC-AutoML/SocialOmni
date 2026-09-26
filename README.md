<p align="center">
  <img src="docs/assets/socialomni_logo.png" alt="SocialOmni" width="320" />
</p>

# SocialOmni: Benchmarking Audio-Visual Social Interactivity in Omni Models

[Paper (arXiv v3)](https://arxiv.org/abs/2603.16859v3) · [PDF](docs/papers/socialomni-arxiv-v3.pdf) · [Leaderboard](https://mac-automl.github.io/SocialOmni/) · [Dataset](https://huggingface.co/datasets/alexisty/SocialOmni) · [中文](docs/README.zh-CN.md)

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

New response-quality evaluations use **Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol**. Each eligible response requires all three scores from {0, 25, 50, 75, 100}; zero scores are retained. Human references and manually verified judge context are not inputs to the evaluated model. Appendix A.8–A.10 describes inference settings, prompts and parsing.

How response quality is reported as QGold and QEns; Cov+ and QEns_joint describe the combined entry decision and response.

## Main results

The [bilingual leaderboard](https://mac-automl.github.io/SocialOmni/) lists all models and supports sorting by each metric.

| Model | Who | When | How: QGold | How: QEns | Cov+ | QEns_joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash | 25.40 | 82.50 | 86.52 | 86.48 | 82.81 | 71.61 |
| Gemini 3.6 Flash | 13.20 | 88.00 | 79.23 | 79.50 | 88.28 | 70.18 |
| Gemini 3 Flash | 0.75 | 78.50 | 86.59 | 87.79 | 78.91 | 69.27 |
| Gemini 3.1 Pro Preview | 45.05 | 84.00 | 80.01 | 81.92 | 82.81 | 67.84 |
| Gemini 2.5 Pro | 5.80 | 78.00 | 80.40 | 82.30 | 82.03 | 67.51 |
| Gemini 3.7 Flash | 16.70 | 73.00 | 84.90 | 84.64 | 79.69 | 67.45 |
| Gemini 3.1 Flash-Lite | 73.90 | 47.00 | 83.98 | 83.81 | 27.34 | 22.92 |
| Gemini 2.5 Flash | 1.70 | 68.00 | 67.58 | 72.12 | 63.28 | 45.64 |
| Gemini 3.5 Flash | 6.00 | 60.50 | 79.56 | 82.95 | 50.78 | 42.12 |
| Qwen3.5-Omni-Plus | 91.05 | 58.50 | 80.27 | 77.82 | 48.44 | 37.70 |
| Qwen2.5-Omni | 4.15 | 61.50 | 49.35 | 49.90 | 64.06 | 31.97 |
| Qwen3-Omni | 70.85 | 64.00 | 49.22 | 45.49 | 66.41 | 30.21 |
| OmniVinci | 29.75 | 64.50 | 40.82 | 38.52 | 70.31 | 27.08 |
| Qwen3.5-Omni-Flash (2026-03-15) | 86.55 | 71.00 | 33.14 | 34.72 | 70.31 | 24.41 |
| Qwen3-Omni-Thinking | 75.65 | 50.50 | 63.15 | 78.85 | 30.47 | 24.02 |
| GPT-4o | 35.05 | 50.50 | 77.15 | 76.50 | 30.47 | 23.31 |
| VITA-1.5 | 34.65 | 56.00 | 50.20 | 49.44 | 46.09 | 22.79 |
| Qwen3.8-Omni-Flash | 89.45 | 44.00 | 77.02 | 69.93 | 17.97 | 12.57 |
| Gemini 3 Pro | 45.40 | 52.00 | 21.55 | 25.00 | 32.03 | 8.01 |
| Gemini 3.5 Flash-Lite | 78.45 | 38.50 | 74.61 | 65.28 | 4.69 | 3.06 |
| MiniCPM-o 4.5 | 72.80 | 38.00 | 57.81 | 47.92 | 6.25 | 2.99 |
| Baichuan-Omni-1.5 | 8.40 | 16.00 | 44.27 | 40.63 | 6.25 | 2.54 |

Classification metrics for MiniCPM-o 4.5: Who macro-F1 **72.52**; When macro-F1 **31.87**.

All scores use a 0–100 scale; higher is better. — indicates an unavailable score.

[Per-item responses, scores and evaluation settings](evaluation/results/) · [Historical materials](evaluation/results/archive/README.md)

## Hosted-model evaluation

The [evaluation runner and commands](evaluation/hosted-omni/README.md) support Gemini 3.8 Flash, Qwen3.8-Omni-Flash, Qwen3.5-Omni-Plus, Qwen3.5-Omni-Flash and other OpenAI-compatible model IDs with model-specific audio-video request defaults. New generation and rescoring of existing answers use the same default three-judge configuration, with resumable requests and per-attempt records. Native local MiniCPM-o 4.5 support is available through the optional `minicpmo_4_5` server adapter.

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

#### Use an SGLang-Omni server

SGLang-Omni can be used as the inference engine while SocialOmni remains the
single benchmark entrypoint. Start its OpenAI-compatible server (the server
must expose `/v1/chat/completions`), then run the official client:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --host 127.0.0.1 --port 8000

export SGLANG_OMNI_SERVER_URL=http://127.0.0.1:8000
export SOCIALOMNI_LEVEL1_OUTPUT_DIR=evaluation/results/<run-name>
uv run python run_benchmark.py --model sglang_omni --resume
```

Set `models.sglang_omni.model` in `config/config.yaml` (or
`SGLANG_OMNI_MODEL`) to the model name served by SGLang-Omni. The client uses
SGLang-Omni's native `videos`/`audios` fields and `modalities: ["text"]`; set
`use_audio_in_video` when the video contains the audio track. The default
`video_max_frames: 8` keeps the request within the model context window and can
be overridden per request or in `config/config.yaml`. Each result row
retains the model response, while request hashes and retry metadata are
recorded in the client result metadata. Keep each run under
`evaluation/results/<run-name>/` with its manifest and validation files.

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
sglang_omni
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
