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
