# SocialOmni hosted-model evaluation

Evaluate hosted models on Who, When and How using the released SocialOmni item lists, audio-visual prefixes and prompt cards. Level 2 generation is scored by Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol. This is the default scoring protocol for both new candidates and archived model responses.

## Setup

The hosted runner has its own Python 3.12 environment; it does not require the repository's GPU environment. Install `ffmpeg` and `ffprobe`, then run:

```sh
cd evaluation/hosted-omni
uv sync --python 3.12 --frozen
```

Supported candidate models and their request defaults:

| Model ID | Thinking | Video input | Output modalities field |
| --- | --- | --- | --- |
| `gemini-3.8-flash` | Provider default | MP4 file | Omitted |
| `gemini-3.5-flash` | Provider default | MP4 file | Omitted |
| `gemini-3.1-pro-preview` | Provider default | MP4 file | Omitted |
| `gemini-3-flash-preview` | Provider default | MP4 file | Omitted |
| `gemini-2.5-flash` | Provider default | MP4 file | Omitted |
| `gemini-2.5-pro` | Provider default | MP4 file | Omitted |
| `dashscope/qwen3.8-omni-flash` | Off | `video_url` | Text |
| `dashscope/qwen3.5-omni-plus` | Off | `video_url` | Text |
| `dashscope/qwen3.5-omni-flash-2026-03-15` | Off | `video_url` | Text |

Both input forms preserve the audio track. Model defaults are selected by `--model`. Explicit `--thinking`, `--media-input-type`, and `--omit-modalities` / `--no-omit-modalities` options override them. Other OpenAI-compatible model IDs are accepted with provider-default thinking, `video_url` input and text output modalities; check that the endpoint accepts audio-bearing video.

The listed routes passed an [audio-track check](../input-checks/20260924/) with paired audible and silent videos: each recovered a spoken password and number from the audible video, without recovering them from the silent control. This verifies audio input through the tested route, not general benchmark accuracy. The earlier Gemini adapters under `models/model_server` send sampled images only; their archived scores must not be relabeled as audio-video results. Use a new output directory for a full evaluation with this runner.

## Generate answers

```sh
uv run python evaluate.py \
  --model gemini-3.8-flash \
  --annotations ../../reproducibility/arxiv-v3/final_source/data \
  --media /path/to/SocialOmni \
  --output runs/gemini38-candidates \
  --key-file /path/to/private/company.env \
  --key-field OPENAI_API_KEY \
  --base-url "$OPENAI_API_BASE" \
  --max-tokens 8192 --concurrency 8
```

Replace `--model` and `--output` to evaluate another model. Generation uses temperature 0.3, top-p 1, 2,000 Who items, 200 When items and forced How generation on all 128 gold-positive items. The default task order is When, How, Who. `--limit 1` selects one item per task for a separate smoke run.

Both audio and video are trimmed before encoding. Who uses the marked interval's end, or the stated query point; the query naming seconds 2 and 6 uses second 6. When and How use the annotated query timestamp. Candidate prompts contain neither reference responses nor full transcripts.

The manifest fixes model settings, prompts, media hashes and implementation hashes. Repeating a command resumes completed requests and retries failed ones while retaining attempts. Changed inputs or code require a new output directory. Empty completed answers are retained. Classification uses full-response parsing and a fixed denominator.

## Score Level 2 generation

Copy `judges.example.json` to a private configuration file, replace each endpoint, and set `SOCIALOMNI_API_KEY` in the environment. Each judge may use its own endpoint and credential variable. The checked-in example selects the default `standard` panel:

- `gemini-3.8-flash`
- `dashscope/qwen3.8-omni-flash`
- `gpt-5.6-sol`

The judge name `qwen3.8-omni` identifies the Qwen entry in score files; its served model ID is `dashscope/qwen3.8-omni-flash`. Judges use temperature 0, top-p 1 and an 8,192-token output limit. Qwen thinking is disabled. GPT-5.6-Sol uses `reasoning_effort: "none"`; Gemini uses provider-default thinking.

After all When and How records finish:

```sh
uv run python prepare_rescore.py \
  --candidates runs/gemini38-candidates \
  --contexts judge-contexts.json \
  --model gemini38 \
  --output runs/gemini38-input.json
uv run python rescore.py \
  --input runs/gemini38-input.json \
  --judges /path/to/private/judges.json \
  --output runs/gemini38-scores
```

Add `--candidate-output runs/gemini38-candidates.json` to preparation after all three tasks finish to export all 2,328 candidate records without private endpoint addresses. Judging can begin after When and How finish; Who remains null in that early input until the complete candidate artifact is available.

To score the 688 released model responses without generating answers again:

```sh
uv run python prepare_rescore.py \
  --paper ../../reproducibility/arxiv-v3 \
  --output runs/paper-input.json
uv run python rescore.py \
  --input runs/paper-input.json \
  --judges /path/to/private/judges.json \
  --output runs/paper-scores
```

The same rubric and three judges apply to both commands. Original contexts, references, candidate text, item IDs, When decisions and source hashes are preserved. Every non-empty answer requires all three valid scores; an incomplete panel never produces an average. Empty completed answers contribute zero. No same-family judge exclusion is applied.

The archived responses cover gold-positive items where the model predicted YES. They support QEns, Cov+ and QEns_joint. QGold requires forced responses for all 128 gold-positive items, so it remains null for those archives. Who and When measure accuracy against labels and do not use LLM judges.

The rubric implements Appendix A.6. `judge-contexts.json` contains observed contexts for all 128 positive items: 126 from archived primary records and two obtained by removing the unique reference continuation suffix from the released transcript. Judge prompts and configuration are recorded in each run manifest.

## Compatibility and exports

`judge_eval.py` also scores a complete hosted run directly, using the same default judges:

```sh
uv run python judge_eval.py \
  --candidates runs/gemini38-candidates \
  --contexts judge-contexts.json \
  --judges /path/to/private/judges.json \
  --output runs/gemini38-judges
uv run python export_results.py \
  --candidates runs/gemini38-candidates \
  --judges runs/gemini38-judges \
  --output runs/gemini38-results.json
```

Exports retain per-item predictions and scores while omitting credentials and private endpoints. Incomplete evaluations are rejected by default. `--allow-incomplete` exports a progress artifact without filling missing scores.

Historical panels remain available through explicit `--panel paper-v3` (GPT-4o, Gemini 2.5 Pro, Qwen3-Omni) and `--panel gpt56-sol` (GPT-5.6-Sol, Gemini 2.5 Pro, Qwen3-Omni). `modern-20260924` remains an alias for the current judge membership so existing manifests can be read. Use the panel and matching configuration recorded in an old run when reproducing it; do not mix cached scores from different requests.

`judge_eval.py` supports `--only-judge NAME` to run one configured judge. Rerun without it to assemble all scores from the request cache. Judge requests honor `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY`; candidate requests use the endpoint directly.

## Tests

```sh
uv run python -m unittest -v test_client test_evaluate test_judges test_export test_rescore
```
