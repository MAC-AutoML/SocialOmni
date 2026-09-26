# Ming-Omni 2.0

Native audio-visual evaluation on the SocialOmni core split. The run contains 2,000 Who items, 200 When items and 128 How items, with 384 completed judge scores from Gemini 3.8 Flash, Qwen3.8-Omni and GPT-5.6-Sol.

| Who accuracy | Who macro-F1 | When accuracy | When macro-F1 | QGold | QEns | Cov+ | QEns_joint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 52.30 | 50.47 | 49.50 | 47.90 | 75.78 | 77.34 | 25.00 | 19.34 |

The complete per-item result is in `results.json`; generation and judge manifests preserve source and configuration hashes.
