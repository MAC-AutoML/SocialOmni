# Qwen3.8-Omni-Flash — September 23, 2026

Supplemental hosted evaluation using the item lists released with [arXiv v3](https://arxiv.org/abs/2603.16859v3). This run is separate from Table 2: identical primary preprocessing and API envelopes have not been established. It evaluates `dashscope/qwen3.8-omni-flash` with thinking disabled.

| Metric | Result |
| --- | ---: |
| Who accuracy | 89.45% (1,789 / 2,000) |
| Who macro F1 | 89.43% |
| When accuracy | 44.00% (88 / 200) |
| When macro F1 | 41.42% |
| Cov+ | 17.97% (23 / 128) |
| QGold, QEns, QEns_joint | Pending complete three-judge scoring |

All 128 gold-positive items produced non-empty forced responses, including items predicted NO. When decisions contain 23 true positives, 7 false positives, 105 false negatives and 65 true negatives. The always-YES baseline on this split is 64%; the model's 44% When accuracy should be read alongside its low positive coverage.

Who item 1980 was rejected by the provider's input-video inspection (`data_inspection_failed`). Item 303 returned an explanation instead of a single permitted letter. Both count as incorrect in the fixed 2,000-item denominator. Resuming the original command recovered one transient request failure; the other 2,326 records were hash-checked as unchanged. Incorrect answers and the parsing failure were not regenerated.

## Configuration and provenance

- Temperature 0.3; top-p 1; maximum output 8,192 tokens; concurrency 8; `enable_thinking=false`.
- Media: `alexisty/SocialOmni` at `3b76009b45090eaa54007454c93a831f3cc8e1e6`.
- Metadata and prompt cards: the [archived v3 ancillary release](../../../reproducibility/arxiv-v3/).
- Both audio and video end at the query boundary before re-encoding (H.264 CRF 18, AAC 192 kb/s). Who interval queries use the marked interval's endpoint; point queries use that point. When and How use the annotated timestamp.
- Candidates receive neither reference continuations nor full transcripts. Judge prompts use the released contexts and the Appendix A.6 rubric; the reconstructed judge prompt is not claimed to be the original primary API prompt.
- Candidate code: [`c43ccec`](https://github.com/MAC-AutoML/SocialOmni/tree/c43ccec2c07dd9386cd26717eb710782b3f267b9/evaluation/hosted-omni). Candidate source, metadata, media and request hashes are recorded in `results.json`.
- Judge runner: [`70ada75`](https://github.com/MAC-AutoML/SocialOmni/tree/70ada75bc56b1266cfac472f226034531b2a57bc/evaluation/hosted-omni).

## Scoring status

Gemini 2.5 Pro and Qwen3-Omni each completed 128 valid scores, with no judge request or parsing errors. The 256 scores are preserved in the export.

GPT-4o is unavailable with the configured credentials/credits. Quality requires GPT-4o, Gemini 2.5 Pro and original Qwen3-Omni on every non-empty response; no two-judge average is reported. Cov+ depends only on decisions and non-empty responses, so it is available before judging completes. The export is explicitly marked incomplete and retains available individual judge scores for resumption.

`results.json` contains all 2,328 item records, available judge scores and provenance hashes. Private endpoints and credentials are omitted. See the [runner documentation](../../hosted-omni/) for commands and cache behavior.
