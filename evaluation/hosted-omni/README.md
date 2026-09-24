# Qwen3.8-Omni SocialOmni supplemental evaluation

This standalone runner uses the arXiv v3 ancillary item lists, Appendix A.8 sampling settings and A.10 prompt cards. It reports supplemental evaluations separately from the frozen paper snapshot.

The current run configuration is `dashscope/qwen3.8-omni-flash` with `enable_thinking=false`, temperature 0.3, top-p 1 and an 8,192-token output limit. Non-thinking mode must be identified explicitly in any result table. Video requests include the audio track. Diagnostic smoke requests using an entire clip are not benchmark results.

Candidate generation uses 2,000 Who items, 200 When items and forced How generation on all 128 gold-positive items. Both audio and video are trimmed before encoding. Who uses the end of the marked interval, or the stated point for point queries; the query naming both seconds 2 and 6 uses second 6. When and How use the annotated query timestamp. Candidate prompts contain no reference response or full transcript.

```sh
cd evaluation/hosted-omni
uv sync --python 3.12 --frozen
uv run python evaluate.py \
  --annotations /path/to/ancillary/final_source/data \
  --media /path/to/SocialOmni \
  --output /path/to/new-run \
  --key-file /path/to/private/company.env \
  --key-field OPENAI_API_KEY \
  --base-url "$OPENAI_API_BASE" \
  --thinking off --max-tokens 8192 --concurrency 8
```

Default order is When, How, Who. `--limit 1` selects one item per requested task for a separate smoke output. A run directory has a fixed manifest; changes to metadata, prompts, model settings or harness code require a new directory. Re-running the same command reuses completed requests and retries failed ones without removing prior attempts.

The client interface is:

```python
client = StreamingClient(api_key, cache_dir, base_url=endpoint, model=model_id)
result = await client.complete(
    messages,
    media_sha256=[video_hash],
    temperature=0.3,
    top_p=1,
    max_tokens=8192,
    enable_thinking=False,
)
```

`result` contains `text`, `reasoning`, `finish_reason`, `usage`, `events`, `request_sha256`, `payload_sha256`, `media_sha256`, `model_sha256`, `elapsed_seconds` and `complete`. Raw streamed events and every completed attempt are stored separately. HTTP failures never store response bodies; credentials are not part of the saved manifest. Empty, successfully terminated text is preserved as empty; truncation or incomplete streams are failures. Classifiers use strict full-response parsing and fixed denominators.

Judging is separate. Candidate generation does not produce a complete three-judge score or authorize replacing one of the original judges. `items/how-*.json` provides candidate text and item identity for the judging stage. The classification summary reports request and parsing failures separately.

```sh
uv run python -m unittest -v test_client test_evaluate
```

## Three-judge scoring

`judge_eval.py` implements the rubric in Appendix A.6. Its prompt is recorded in the run manifest; it is not represented as a recovered verbatim primary API prompt. The context file takes 126 items from the archived primary response records. For `video_0046` and `video_0052`, it removes the unique exact reference suffix from the released annotation transcript. The resulting observed context excludes the reference continuation. No judge context or reference is sent to the candidate model.

Configure exactly `gpt-4o`, `gemini-2.5-pro` and `qwen3-omni`, including each actual served model identifier and endpoint. Set credentials through the named environment variables. The original Qwen3-Omni model is required; another Qwen generation is not a substitute.

```json
{"judges": [
  {"name": "gpt-4o", "model": "gpt-4o", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "GPT4O_API_KEY", "max_concurrency": 2},
  {"name": "gemini-2.5-pro", "model": "gemini-2.5-pro", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "GEMINI_API_KEY", "max_concurrency": 2},
  {"name": "qwen3-omni", "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "base_url": "http://127.0.0.1:8091/v1", "api_key_env": null, "max_concurrency": 1}
]}
```

After all 200 When and 128 forced How records have completed:

```sh
uv run python judge_eval.py --candidates runs/candidates --contexts judge-contexts.json --judges judges.json --output runs/judges
uv run python -m unittest -v test_client test_evaluate test_judges
```

No partial judge panel is averaged. Successfully empty responses contribute zero to the fixed-denominator metrics; transport failures keep the run incomplete. Source metadata, media, prompts, sampling and implementation hashes prevent reuse across different candidate protocols.

If one endpoint is temporarily unavailable, `--only-judge gemini-2.5-pro` (or another configured judge name) can save that judge’s work first. Rerun without `--only-judge` to assemble the complete panel using the same request cache. Until all three scores exist, quality metrics remain null and the command exits with an incomplete status. Judge requests honor `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY`; candidate requests use the endpoint directly.

Export a public result file after all stages finish:

```sh
uv run python export_results.py --candidates runs/candidates --judges runs/judges --output results.json
```

The export keeps per-item predictions and judge scores but omits credentials and private relay addresses. It refuses an incomplete evaluation by default. `--allow-incomplete` is for an explicitly labeled progress artifact; missing-panel quality remains null.

### Disclosed GPT-5.6-Sol panel

`--panel gpt56-sol` selects GPT-5.6-Sol, Gemini 2.5 Pro and Qwen3-Omni for a supplemental evaluation. GPT-5.6-Sol replaces GPT-4o only in this panel; the default `paper-v3` panel still requires GPT-4o. Results must identify the replacement and remain separate from paper-panel scores.

In the judge configuration, replace the GPT-4o entry with:

```json
{"name": "gpt-5.6-sol", "model": "gpt-5.6-sol", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "GPT56_API_KEY", "max_concurrency": 8, "include_modalities": false}
```

The text-only GPT-5.6-Sol endpoint does not accept `modalities`. Other judges retain their existing requests. Use a new output directory for the new panel; copy the existing Gemini and Qwen cache directories only when their endpoints, models, prompts and sampling settings are unchanged. Cache lookup uses the full request hash, so a changed payload cannot reuse those scores.

```sh
uv run python judge_eval.py --panel gpt56-sol --candidates runs/candidates --contexts judge-contexts.json --judges judges-sol.json --output runs/judges-sol
```

Exports record `judge_panel`, the three judge names and `judge_substitution`. Missing-panel quality remains null. With `--allow-incomplete`, a fully scored panel with retained candidate request failures has status `scored_with_request_failures`; those failures remain in the classification denominators.

Judge requests require a non-empty score response. An empty successful stream is retried with the identical payload and retained as an attempt; existing valid scores are reused. Candidate generation keeps its original behavior, including preserving empty successful generations.

## Rejudge archived answers with the September 24 panel

The `modern-20260924` panel uses Gemini 3.8 Flash (`gemini-3.8-flash`), Qwen3.8-Omni-Flash (`dashscope/qwen3.8-omni-flash`) and GPT-5.6-Sol (`gpt-5.6-sol`). It is a separate result cohort. Candidate answers and correctness labels are not rewritten when the judges change.

Each judge uses temperature 0, top-p 1 and an 8,192-token output limit. Qwen's thinking mode is disabled. GPT-5.6-Sol explicitly uses `reasoning_effort: "none"`, as required by the endpoint for top-p support. Gemini retains its provider-default thinking behavior. Set `include_modalities: false` for Gemini and GPT's text-only scoring requests.

Prepare the paper's canonical 688 responses:

```sh
uv run python prepare_rescore.py --paper ../../reproducibility/arxiv-v3 --output runs/paper-input.json
uv run python rescore.py --input runs/paper-input.json --judges judges-modern.json --output runs/paper-modern --panel modern-20260924
```

Or prepare all 128 forced responses from a completed hosted candidate run:

```sh
uv run python prepare_rescore.py --candidates runs/candidates --contexts judge-contexts.json --model MODEL_KEY --output runs/model-input.json
uv run python rescore.py --input runs/model-input.json --judges judges-modern.json --output runs/model-modern --panel modern-20260924
```

The normalized input records the original contexts, reference continuations, candidate text, item IDs and When decisions. Its source file hashes and the complete scoring configuration are frozen before judging. All three valid scores are required for each non-empty answer; zeros count. Incomplete panels never produce an averaged quality score.

The archived primary answers cover gold-positive items where the original model predicted YES. They support QEns, Cov+ and QEns_joint, but not QGold over all 128 gold-positive items. QGold therefore remains null when rejudging those 688 answers. New full-gold runs support all four metrics. Who and When remain accuracy against the original labels; they are not LLM-judge scores. Both these new models and the new judges may share a model family; no same-family judge exclusion is applied.

For the Gemini 3.8 Flash candidate, use `--model gemini-3.8-flash --thinking default --media-input-type file --omit-modalities`. The formal run uses an MP4 file content part because this relay's `video_url` route omitted media during verification. The Qwen3.5 candidates use `--thinking off` with the default `video_url` input and exact model IDs `dashscope/qwen3.5-omni-plus` and `dashscope/qwen3.5-omni-flash-2026-03-15`.

Add `--candidate-output candidates.json` to hosted input preparation to export all 2,328 candidate records without private relay addresses. Judging may begin after When and How finish; in that case the normalized input leaves Who null until the separate complete candidate artifact is available.

A `judges-modern.json` configuration has these three entries (replace each endpoint and set the named credential variable):

```json
{"judges": [
  {"name": "gemini-3.8-flash", "model": "gemini-3.8-flash", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "SOCIALOMNI_API_KEY", "include_modalities": false, "max_concurrency": 2},
  {"name": "qwen3.8-omni", "model": "dashscope/qwen3.8-omni-flash", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "SOCIALOMNI_API_KEY", "enable_thinking": false, "max_concurrency": 2},
  {"name": "gpt-5.6-sol", "model": "gpt-5.6-sol", "base_url": "https://YOUR_ENDPOINT/v1", "api_key_env": "SOCIALOMNI_API_KEY", "include_modalities": false, "reasoning_effort": "none", "max_concurrency": 2}
]}
```
