from __future__ import annotations

def build_prompt(item: JudgeItem, reference_mode: str) -> str:
    base = (
        "You are a strict evaluator of a target participant's continuation in a social dialogue.\n"
        "Score contextual appropriateness, target-role consistency, pragmatic/social fit, "
        "grounding in the provided context, and coherence.\n"
        "Use exactly one score: 0, 25, 50, 75, or 100. Output only the number.\n\n"
        f"[Observed dialogue context]\n{item.context or '(not provided)'}\n\n"
        f"[Target instruction]\n{item.target_question or '(not provided)'}\n\n"
    )
    if reference_mode == "reference-aware":
        base += (
            "The human reference clarifies the intended role and intent, but valid paraphrases "
            "and alternative socially appropriate continuations must receive credit.\n\n"
            f"[Human reference continuation]\n{item.reference or '(not provided)'}\n\n"
        )
    else:
        base += "Evaluate without using a human reference answer.\n\n"
    return base + f"[Candidate continuation]\n{item.candidate}\n"
