# MiniCPM-o 4.5

MiniCPM-o 4.5 is integrated as a native local audio-video model through the
`minicpmo_4_5` client and server. The server uses the model's own
`minicpmo-utils` frame/audio segmenter, keeps the original audio track, and
passes frames and aligned audio segments directly to `model.chat`.

The optional runtime is kept separate from the main SocialOmni environment
because MiniCPM-o 4.5 requires its compatible Transformers and Python 3.12
stack. On the Wulanchabu host, the verified JFS checkpoint is:

```text
/mnt/jfs/models/platform/openbmb/MiniCPM-o-4_5/503e754207c94da6bb26850b4469f367c9ea3582
```

Create an environment with the model's documented dependencies, then start
the adapter with `MINICPMO_4_5_MODEL_PATH` set when the checkpoint is elsewhere:

```sh
MINICPMO_4_5_MODEL_PATH=/path/to/MiniCPM-o-4_5 \
  uv run python models/model_server/minicpmo_4_5/minicpmo_4_5_server.py \
  --host 127.0.0.1 --port 5096
```

Run the existing benchmark with `--model minicpmo_4_5`. The adapter accepts the
same `/analyze` contract as the other native local models, so the benchmark
records and retry behavior stay unchanged.

For response-quality scoring, copy
`evaluation/hosted-omni/judges.minicpmo45.example.json`, fill the private
endpoints, and select `--panel minicpmo45-20260925`. This label keeps the
standard Gemini 3.8 Flash, Qwen3.8-Omni-Flash and GPT-5.6-Sol panel explicit;
it does not allow incomplete panels or two-judge averages.
