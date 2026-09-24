# Gemini audio-video evaluation

Each model is evaluated on 2,000 Who items, 200 When items and 128 How items. MP4 inputs include the original audio and video, trimmed to the observation boundary. Prompts, model identifiers, sampling parameters and source hashes are recorded in `candidates.json`.

Nonempty How responses receive scores from Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol. All three scores are required. Empty responses contribute zero over the fixed 128-item denominator. Who and When use full-response parsing and fixed denominators.

Each model directory includes the generated answers (`candidates.json`), judge inputs (`input.json`), individual scores and six aggregate metrics (`results.json`), source validation (`validation.json`) and file hashes (`manifest.json`). Technical failures are retried with the original request; model refusals and normal truncation remain model outcomes.
