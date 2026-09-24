# Local audio-video evaluation

Each published model completed 2,000 Who items, 200 When items and 128 gold-positive How items. Inputs contain the original video and audio up to each observation boundary, without injected transcripts. Generation uses temperature 0.3, top-p 1 and a 9,000-token output limit. Who and When use full-response parsing and fixed denominators; refusals, malformed answers and ordinary output truncation remain model outcomes.

How responses are scored by Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol using the [evaluation rubric](../../hosted-omni/README.md). Every nonempty response requires all three scores. Empty final answers contribute zero. All six metrics in a model's result use the same generation run and its new When decisions.

Model directories contain the complete candidate answers, judge inputs, scores and provenance. Exact checkpoint revisions, preprocessing settings, package versions and implementation hashes are recorded with the results. Decoder thread limits changed execution concurrency; reused successful responses retained their original file hashes and manifests. Only technical failures were retried.

The previous published results and original paper materials remain in their existing directories. These new runs replace complete model rows on the leaderboard, without changing those historical artifacts.
