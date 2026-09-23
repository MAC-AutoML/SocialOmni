# SocialOmni: reproducibility materials

This package accompanies the updated SocialOmni arXiv manuscript (2603.16859).
The complete textual appendix is included in the paper PDF.

## Included

- Adjudicated annotations for 2,000 perception items and the 200-item interaction
  core split (`video_0001` through `video_0200`; 128 positive entry states).
- Per-item perception predictions and turn-entry decisions for all 11 systems.
- The canonical 688 generated responses, judge contexts, human references,
  candidate hashes, all 2,064 primary judge scores, and full model summaries.
- Frozen gold-state, modality, pause-baseline and judge-control summary tables.
- Three anonymous human ratings for each of 200 calibration IDs, the canonical
  response mapping, and calibration statistics. This subset covers ten systems
  and excludes GPT-4o; it is distinct from benchmark construction annotations.
- Offline validation, table/figure rendering and calibration-analysis scripts.
- The archived extension judge prompt template and held-out judge identifiers.

## Scope

This package supports offline auditing of the recorded predictions and primary
response scores. It is not a complete rerun environment for all multimodal API
experiments. Original video media (approximately 9.7 GB), model weights, private
API endpoints and credentials are not included.

Public code: https://github.com/MAC-AutoML/SocialOmni

Public dataset and video media: https://huggingface.co/datasets/alexisty/SocialOmni

These public resources are distinct from this frozen evaluation snapshot. Extension results are supplied as frozen summaries; their complete
raw response and API-log trees are not bundled. Do not interpret regenerated
summary tables as a fresh execution of those experiments.

Only local deployment paths and source-worker bookkeeping were anonymized.
Dialogue content, labels and scores were preserved. The interaction annotation
header was normalized to the actual 200 records; obsolete wrapper metadata is
not used as a benchmark count. Perception `asr_content` and interaction
`full_asr` are annotation/reference fields, not permission to expose future
context to evaluated models. Calibration ratings are mapped by calibration ID
to the final response table; obsolete candidate text from the returned rating
form is intentionally omitted. Archived intermediate generation outputs and
superseded judge caches are not used as the canonical response results.

## Validate and reproduce offline analyses

Use Python 3.10–3.13. Python 3.13 was used for package validation.

```sh
uv sync --python 3.13
uv run python scripts/verify_package.py
uv run python scripts/render_final_artifacts.py --snapshot-dir results/primary --source-root final_source --table-dir generated/tables --figure-dir generated/figures
uv run python scripts/render_judge_audit_tables.py --scores results/primary/ensemble_response_scores.csv --table-dir generated/judge_tables
uv run python scripts/render_extension_tables.py --snapshot-dir results/extension --table-dir generated/extension_tables --strict
uv run python scripts/analyze_three_rater_calibration.py --ratings results/human/ratings.csv --legacy-key results/human/calibration_key.csv --canonical results/primary/ensemble_response_scores.csv --output-dir generated/human
```

These commands require no API keys or video downloads. Generated TeX tables
preserve the analysis scripts' original styling; the submitted PDF is the
authoritative typeset version. Regeneration can differ slightly in numeric
formatting across library versions. `SHA256SUMS.json` records file integrity;
`VERIFICATION.json` records the checks completed before upload.
