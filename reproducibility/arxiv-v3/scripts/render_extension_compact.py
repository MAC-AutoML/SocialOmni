"""Compact main-paper tables for SocialOmni extension results."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable


def _number(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: object, completeness, label: str, *, percent: bool = False) -> str:
    number = _number(value)
    if number is None:
        completeness.mark(f"missing value {label}")
        return "--"
    if percent and abs(number) <= 1:
        number *= 100
    return f"{number:.2f}"


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_modality_compact(
    rows: list[dict[str, str]],
    how_rows: list[dict[str, str]],
    path: Path,
    completeness,
    display_name: Callable[[str], str],
) -> None:
    if not rows or not how_rows:
        completeness.mark("empty modality inputs")
        return
    metric_lookup = {
        (row.get("model", ""), row.get("task", ""), row.get("modality", "")): row.get("accuracy")
        for row in rows
    }
    gold_lookup = {
        (row.get("model", ""), row.get("modality", "")): row.get("Q_Gold")
        or row.get("primary_aware_mean_failures_zero")
        for row in how_rows
    }
    models = [
        "gpt4o", "qwen3_omni", "qwen3_omni_thinking", "qwen2_5_omni",
        "omnivinci", "vita_1_5", "baichuan_omni_1_5",
    ]
    body = []
    for model in models:
        audio_mode = "transcript-only" if model == "gpt4o" else "audio-only"
        video_mode = "frames-only" if model == "gpt4o" else "video-only"
        values = [
            metric_lookup.get((model, "who", audio_mode)),
            metric_lookup.get((model, "who", video_mode)),
            metric_lookup.get((model, "when", audio_mode)),
            metric_lookup.get((model, "when", video_mode)),
        ]
        cells = [
            display_name(model),
            "T/F" if model == "gpt4o" else "A/V",
            *[
                _fmt(value, completeness, f"{model} modality metric {index}", percent=True)
                for index, value in enumerate(values)
            ],
            _fmt(gold_lookup.get((model, audio_mode)), completeness, f"{model} QGold audio"),
            _fmt(gold_lookup.get((model, video_mode)), completeness, f"{model} QGold video"),
        ]
        body.append(" & ".join(cells) + " \\\\")
    _write(
        path,
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Input-channel ablations. A/V denotes audio-only/video-only; T/F denotes GPT-4o transcript-only/frames-only adapters. Primary-interface scores remain in Table~\ref{tab:main_triplet}.}",
            r"\label{tab:extension_modality_ablation}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr}\toprule",
            r"Model & Pair & \multicolumn{2}{c}{Who} & \multicolumn{2}{c}{When} & \multicolumn{2}{c}{$Q_{\rm Gold}$} \\",
            r" & & A/T & V/F & A/T & V/F & A/T & V/F \\\midrule",
            *body,
            r"\bottomrule\end{tabular*}",
            r"\end{table*}",
        ],
    )


def write_judge_compact(
    correlations: list[dict[str, str]],
    rank_shifts: list[dict[str, str]],
    sanity_rows: list[dict[str, str]],
    path: Path,
    completeness,
) -> None:
    if not correlations or not rank_shifts or not sanity_rows:
        completeness.mark("empty judge audit inputs")
        return
    body = []
    settings = (
        ("GPT-4o", "reference-aware:gpt4o", "gpt4o", "reference-aware"),
        ("Gemini", "reference-aware:gemini_2_5_pro", "gemini_2_5_pro", "reference-aware"),
        ("Qwen3", "reference-aware:qwen3_omni", "qwen3_omni", "reference-aware"),
    )
    for label, metric, sanity_judge, sanity_mode in settings:
        response = next(
            (row for row in correlations if row.get("scope") == "response" and row.get("metric") == metric),
            {},
        )
        model_rank = next(
            (row for row in correlations if row.get("scope") == "model-rank" and row.get("metric") == metric),
            {},
        )
        shifts = [
            abs(_number(row.get("rank_shift")) or 0)
            for row in rank_shifts
            if row.get("metric") == metric
        ]
        sanity = next(
            (
                row for row in sanity_rows
                if row.get("judge") == sanity_judge and row.get("reference_mode") == sanity_mode
            ),
            {},
        )
        cells = [
            label,
            _fmt(response.get("spearman"), completeness, label + " response Spearman"),
            _fmt(response.get("mae"), completeness, label + " response MAE"),
            _fmt(model_rank.get("kendall"), completeness, label + " model Kendall"),
            str(int(max(shifts))) if shifts else "--",
            _fmt(sanity.get("auc_gold_vs_shuffled"), completeness, label + " sanity AUC"),
            _fmt(
                sanity.get("shuffled_high_score_false_positive_rate"),
                completeness,
                label + " sanity FPR",
                percent=True,
            ),
        ]
        body.append(" & ".join(cells) + " \\\\")
    _write(
        path,
        [
            r"\begin{table}[t]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{Primary-judge robustness audit. Top: each judge against the complete ensemble and independent sanity controls; bottom: leave-one-out rank sensitivity and pairwise response agreement.}",
            r"\label{tab:extension_judge_audit}",
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr}",
            r"\toprule",
            r"Audit & $\rho_r$ & MAE & $\tau_m$ & Shift & AUC & FPR \\\midrule",
            *body,
            r"\bottomrule\end{tabular*}",
            r"\vspace{2pt}",
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrlrrr}",
            r"\toprule",
            r"Removal & $\tau_m$ & Shift & Pair & $\rho_r$ & $\tau_r$ & MAE \\\midrule",
            r"\input{tables/judge_combined_rows}",
            r"\end{tabular*}",
            r"\end{table}",
        ],
    )
