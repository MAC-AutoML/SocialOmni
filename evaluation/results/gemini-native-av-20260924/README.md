# Gemini audio-video evaluation

Each published model was evaluated on 2,000 Who items, 200 When items and 128 gold-positive How items. Inputs contain the original audio and video, trimmed to each item's observation boundary. Candidate prompts contain no transcript or reference answer.

Generation used the [hosted runner at ade4d8d](https://github.com/MAC-AutoML/SocialOmni/tree/ade4d8dc2eefe7a6f7d14dcc2076fbbb851eef86/evaluation/hosted-omni), with the exact model identifier recorded in each result directory, MP4 `file` input, provider-default thinking, temperature 0.3, top-p 1 and an 8,192-token output limit. The `modalities` field was omitted. Who and When use the runner's full-response parser and fixed denominators.

Nonempty How responses were scored by Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol, using the default rubric and judge settings in [the evaluation instructions](../../hosted-omni/README.md). Each nonempty response has all three scores; empty responses contribute zero over the fixed 128-item denominator. Failed GPT-5.6-Sol streams were recovered with the same prompt and generation parameters through non-streaming requests; their request hashes remain in the score records.

Each model directory contains:

- `candidates.json`: all 2,328 generated answers, predictions, labels and request/media hashes.
- `input.json`: the observed contexts, references and candidate continuations supplied for judging.
- `results.json`: per-response judge scores, metric aggregates and evaluation provenance.

Gemini 3.1 Pro Preview had 92 local preprocessing failures. Only these failed items were retried; the original 2,236 successful records and run manifest remained byte-identical. Each published run contains 2,328 successful requests.

`manifest.json` records the file hashes. Historical Gemini results remain in their original result directories.
