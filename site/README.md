# SocialOmni leaderboard

Static English / Chinese project page for GitHub Pages. The published HTML includes the leaderboard, video examples and metric definitions. JavaScript adds sorting, search and language switching; it is not required to read the results. No third-party JavaScript dependencies.

After editing data, examples, translations or styles, run `node site/build.mjs` from the repository root and commit the generated HTML and stylesheet. The build embeds the JSON and updates content-based asset versions to prevent stale scripts and styles after deployment.

The table groups metrics by task: Who and When accuracy, How response quality (QGold and QEns), and When + How coverage and joint quality (Cov+ and QEns_joint). These labels do not introduce a new aggregate score.

All models appear in one table, initially sorted by QEns_joint. New evaluations use Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol, including rescoring of earlier models' existing responses. GPT-4o, Gemini 3 Pro and OmniVinci retain the six metrics and original judge panel reported in arXiv v3 Table 2. Each record identifies its source and judge panel. Original outputs, evaluation settings, diagnostics and historical panels remain in the source repository. Model names link to their supporting results.

## Results data

- `updated_at`: publication date.
- `links`: public paper, code and dataset URLs.
- `judges`: exact model identifiers for the current scoring panel.
- `records`: model identity, source, metrics and evaluation metadata.

Metrics are `who`, `when`, `qgold`, `qens`, `cov_plus` and `qens_joint`, on a 0–100 scale. Missing values are `null`, displayed as — and sorted last in either direction. Do not fill missing values with historical judge scores. No composite score is calculated beyond the defined QEns_joint metric.

Before publishing, verify model identities, complete judge scores, source responses and evaluation settings. Keep credentials and private endpoints out of public files. Preserve historical results in the repository archive when changing the scoring panel.

## Checks

Verify both languages, all-model display, metric sorting in both directions, missing values, search and no matches, source links, keyboard navigation and mobile table scrolling.

## Media and examples

`cases.json` preserves the selected dataset annotations, public source identifiers and video hashes. Chinese translations are separate `_zh` fields; original annotations are unchanged. The two Level 2 clips end at the annotated decision time. Reference answers are initially collapsed.

`media/socialomni-introduction.mp4` is the author-provided project video, remuxed for progressive playback without re-encoding. Videos load only on interaction. Case clips use H.264 video and AAC audio.

## How examples

`how-cases.json` presents two recorded answers to the same observation prefix, the original three-judge scores, later explanations from those judges, and an independent Codex evidence review. Original English answers and quoted context are preserved; Chinese translations are displayed separately. Highlighted quotes must occur verbatim in the stored context. The explanation requests and source hashes are archived under `evaluation/examples/how-20260924/`. Follow-up explanations do not change leaderboard scores.
