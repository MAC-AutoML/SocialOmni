# SocialOmni — September 24 judge panel

This supplement evaluates response quality with **Gemini 3.8 Flash**, **Qwen3.8-Omni-Flash**, and **GPT-5.6-Sol**.

## Scope

The completed supplement contains 3,600 judge scores: 2,064 for archived paper answers and 1,536 for the four hosted runs. Fifteen missing streaming scores were recovered using non-streaming transport.

The paper-answer cohort reuses the 688 canonical responses from all 11 paper configurations. Neither the responses nor their When decisions are regenerated. All 688 responses receive the new three-judge panel. This supports QEns (quality when the model responded), Cov+ (positive-response coverage) and QEns_joint (quality weighted by that coverage). The complete forced-response set is unavailable for these archived runs, so QGold is not reported for this cohort.

The hosted cohort includes the existing Qwen3.8-Omni-Flash run and new Gemini 3.8 Flash, Qwen3.5-Omni-Plus and Qwen3.5-Omni-Flash runs. Each new candidate is evaluated on 2,000 Who items, 200 When items, and forced How generation on all 128 gold-positive items. All four response-quality metrics are available when their judge panels are complete. Who and When use the original correctness labels; they do not depend on the new judges.

Paper-answer and hosted cohorts remain separate because candidate input interfaces, preprocessing and classification handling have not been established as equivalent. The hosted runs explicitly use strict label parsing; a separate auxiliary diagnostic measures the effect of extracting labels from explanatory answers. The leaderboard does not combine their rankings or compute an overall score.

## Scoring

| Judge | Served model ID | Settings |
| --- | --- | --- |
| Gemini 3.8 Flash | `gemini-3.8-flash` | Provider-default thinking |
| Qwen3.8-Omni-Flash | `dashscope/qwen3.8-omni-flash` | `enable_thinking=false` |
| GPT-5.6-Sol | `gpt-5.6-sol` | `reasoning_effort=none` |

All judges use temperature 0, top-p 1 and an 8,192-token output limit. They receive the fixed observed context, target instruction, reference continuation and candidate answer. The rubric is reconstructed from arXiv v3 Appendix A.6; it is not claimed to be the unreleased original API prompt. Exact prompt, input, implementation and request hashes accompany the results.

Each non-empty answer requires three valid scores from {0, 25, 50, 75, 100}; its quality is their arithmetic mean. Zero scores are retained. Missing or invalid judge outputs leave quality incomplete. Empty judge streams and transient request failures may be retried with the same payload; valid scores are reused from exact-request caches. No same-family judge exclusion is applied, including when a candidate is also represented in the judge panel.

Some GPT requests repeatedly returned empty streams. Remaining missing scores could be recovered with ordinary, non-streaming HTTP responses, keeping the model, messages and all generation parameters unchanged. These scores carry `transport: non_streaming` and their own payload and request hashes; they are not represented as streaming responses. All previously valid scores are preserved. A diagnostic response reused during recovery explicitly identifies its request hash as reconstructed from the retained diagnostic program.

## Candidate interfaces

The hosted candidates use temperature 0.3, top-p 1 and an 8,192-token output limit. Both media tracks stop at the query boundary before encoding. Candidates receive neither judge references nor future transcript context. Qwen runs disable thinking; Gemini uses its default thinking behavior.

The Gemini route accepts an MP4 as a `file` content part. Its `video_url` route was found to omit media, so it is not used for the formal run. The chosen interface was checked against the actual scene and spoken dialogue before evaluation. Qwen uses `video_url` with the original audio track. The model configurations and frozen source snapshots identify these differences.

## Artifacts and verification

Normalized inputs preserve candidate answers, observed contexts, references and When decisions. Public score files include individual judge scores and their request hashes. New candidate exports preserve all per-item answers, parsing outcomes and request-success flags. Failed candidate requests remain in the fixed classification denominators; they are not silently dropped.

Unsuccessful requests and refusals count as incorrect in the fixed classification denominator. Request verification and bounded retry records are retained in the [request audit](technical-retry-audit.json). All 9,312 candidate records remained unchanged during this audit.

Private endpoints, credentials and raw service logs are excluded from the repository. Full request-attempt caches and the exact runtime sources are retained in the owner's persistent storage. The [runner documentation](../../hosted-omni/) provides the preparation, scoring and resumption commands.

The `runtime/` directory contains the exact candidate and judge source snapshots identified by the recorded hashes. Earlier Qwen runs predate the optional Gemini file-input support. These snapshots document the executed programs; the maintained implementation is in `evaluation/hosted-omni/`.

Each model directory contains `input.json` and `results.json`; hosted runs also contain `candidates.json` with all 2,328 candidate outputs. The paper directory contains the 688 archived response inputs and their new scores. Inputs prepared before Who generation finished may have a null Who summary; the completed classification results are in `candidates.json`.

Candidate harness hashes, transport-recovery implementation hashes and `input_file_sha256` hash raw file bytes. The rescorer's input, prompt, candidate and implementation digests use the canonical JSON serialization in `client.digest`; hashing the plain text bytes will give a different value. The legacy `archived_candidate_hash` is retained as an opaque source identifier.

## Output-format diagnostic

The primary classification scores require exact labels. For example, `The correct answer is **D**.` is invalid under this rule even when D is correct. The separate [format diagnostic](format-diagnostic.json) applies the repository's existing `extract_choice` function to successful requests with invalid strict predictions, using the same procedure for all four hosted models. Already valid predictions and failed requests are unchanged.

The diagnostic preserves per-item strict and extracted predictions, the extractor's source hash, and counts of recovered correct and incorrect answers. This permissive extraction can select an option mentioned in an explanation; it is not human adjudication. Its accuracies neither replace the strict leaderboard scores nor change When decisions used for response-quality metrics and coverage.

To reproduce the diagnostic from the repository root:

```sh
uv run --project evaluation/hosted-omni python \
  evaluation/results/modern-panel-20260924/runtime/format_diagnostic.py \
  --results evaluation/results/modern-panel-20260924 \
  --extractor models/pipeline/answer_extraction.py \
  --output /tmp/socialomni-format-diagnostic.json
```

## Results

Scores use a 0–100 scale. A dash denotes an unavailable metric. Classification uses the strict label parser.

### Hosted candidates

| Model | Who | When | QGold | QEns | Cov+ | QEns_joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Omni-Flash | 89.45 | 44.00 | 77.02 | 69.93 | 17.97 | 12.57 |
| Gemini 3.8 Flash | 25.40 | 82.50 | 86.52 | 86.48 | 82.81 | 71.61 |
| Qwen3.5-Omni-Plus | 91.05 | 58.50 | 80.27 | 77.82 | 48.44 | 37.70 |
| Qwen3.5-Omni-Flash (2026-03-15) | 86.55 | 71.00 | 33.14 | 34.72 | 70.31 | 24.41 |

### Archived paper answers, rescored

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

### Auxiliary format extraction

These values only diagnose output formatting; the primary tables above remain unchanged.

| Model | Who strict | Who auxiliary | When strict | When auxiliary |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Omni-Flash | 89.45 | 89.50 | 44.00 | 44.00 |
| Gemini 3.8 Flash | 25.40 | 94.50 | 82.50 | 86.00 |
| Qwen3.5-Omni-Plus | 91.05 | 91.05 | 58.50 | 58.50 |
| Qwen3.5-Omni-Flash (2026-03-15) | 86.55 | 86.70 | 71.00 | 71.00 |
