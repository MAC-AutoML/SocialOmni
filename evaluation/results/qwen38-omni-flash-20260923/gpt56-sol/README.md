# Qwen3.8-Omni-Flash — GPT-5.6-Sol judge panel

This supplemental evaluation replaces the paper's GPT-4o judge with **GPT-5.6-Sol**, as requested by the benchmark owner. The other judges are **Gemini 2.5 Pro** and **Qwen3-Omni-30B-A3B-Instruct**. These quality scores are not results from the paper's original judge panel and are displayed in a separate leaderboard group.

| Who | When | QGold | QEns | Cov+ | QEns_joint |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 89.45 | 44.00 | 73.96 | 68.12 | 17.97 | 12.24 |

All metrics use a 0–100 scale. There are 384 valid scores (128 per judge), with no missing or invalid final scores. The export status `scored_with_request_failures` records the retained Who input rejection; quality scoring is complete.

The 2,328 candidate records are unchanged from the [September 23 run](../). All 128 gold-positive responses are scored by all three judges. The existing Gemini and Qwen scores are reused from identical request caches; only the missing GPT slot is newly evaluated. Each response's quality is the arithmetic mean of its three scores, including zeros. No missing judge score is replaced with a two-judge mean.

The new judge uses the same Appendix A.6 rubric, contexts, references and candidate continuations, with temperature 0, top-p 1 and an 8,192-token output limit. Requests use the exact served model ID `gpt-5.6-sol`; the text-only interface omits the unsupported `modalities` field. Provider-default reasoning settings are used. The full rubric and implementation hashes are included in `results.json`.

Both historical Who failures remain counted as incorrect: one provider video rejection and one answer-format failure. Candidate response generation, metadata, media boundaries and classification denominators were not changed. The previous incomplete paper-panel export is retained in the parent directory.

The first GPT-5.6-Sol pass returned 119 valid scores and nine empty streams. Identical-payload recovery completed the nine missing scores; all 119 initial valid scores and all 256 Gemini/Qwen scores were verified unchanged. Failed and empty attempts are retained in the run cache.

See the [runner documentation](../../../hosted-omni/#disclosed-gpt-56-sol-panel) for the explicit `--panel gpt56-sol` command. A new output directory preserves the original panel manifest and caches.
