# Audio-video input check

`av.mp4` shows a red square on a navy background. Its audio says “The spoken password is silver lantern. The spoken number is forty seven.” `silent-av.mp4` contains the same video with its audio samples muted. No text appears in either video.

`hosted.json` records the prompt, media and request hashes, exact requested and returned model IDs, HTTP status, and answer for each request. Credentials and provider-specific response metadata are omitted. Each request used a fresh conversation. These are interface checks, not SocialOmni evaluation items or scores.

Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 3 Flash, Gemini 3.1 Pro Preview, Gemini 3.5 Flash, Gemini 3.8 Flash, Qwen3.5-Omni-Plus, Qwen3.5-Omni-Flash and Qwen3.8-Omni-Flash recovered the spoken password and number from the audible video. None recovered them from the silent control. Gemini 3 Flash, Gemini 3.1 Pro Preview and two Qwen Flash routes invented unrelated speech in the silent control; their original answers are retained.

Gemini 3 Pro Preview returned HTTP 404 because the original route was discontinued. This is an availability result, not evidence that the model lacks audio support. Availability and input handling can differ across providers and change over time.

For the Gemini routes, the MP4 was sent as an OpenAI-compatible `file` content part with `file_data` and `filename`. For Qwen, it was sent as `video_url`. In both cases the media was a base64 data URL containing the entire MP4, including its audio track.
