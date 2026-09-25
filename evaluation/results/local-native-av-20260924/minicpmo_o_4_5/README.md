# MiniCPM-o 4.5

Native audio-visual evaluation on the SocialOmni core split. The run contains 2,000 Who items, 200 When items and 128 How items, with 384 completed judge scores from Gemini 3.8 Flash, Qwen3.8-Omni and GPT-5.6-Sol.

| Who accuracy | Who macro-F1 | When accuracy | When macro-F1 | QGold | QEns | Cov+ | QEns_joint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 72.80 | 72.52 | 38.00 | 31.87 | 57.81 | 47.92 | 6.25 | 2.99 |

The complete per-item result is in `results.json`; generation and judge manifests preserve the source and configuration hashes.
