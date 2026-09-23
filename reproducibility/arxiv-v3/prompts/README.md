# Judge protocol

The primary judges are GPT-4o, Gemini 2.5 Pro and Qwen3-Omni. Each eligible
response has all three scores on {0, 25, 50, 75, 100}; quality is their mean.
The primary protocol is reference-aware. Evaluated systems do not receive the
human reference or the manually verified judge context.

`extension_judge_prompt.py` preserves the prompt-construction function from the
archived extension-judge runner. Its input object supplies `context`,
`target_question`, `reference`, and `candidate`; `reference_mode` selects
`reference-aware` or `reference-free`. This is a prompt template, not an API
client. The extension audit records held-out judges as `claude-sonnet-4-6` and
`deepseek-v3.2`; these are supplementary controls, not primary judges.

The package contains the final 688 response records and 2,064 primary score
values. It does not claim to include every original API invocation log or the
item-level raw outputs of all extension conditions.
