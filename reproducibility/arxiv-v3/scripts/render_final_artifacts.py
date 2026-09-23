#!/usr/bin/env python3
"""Render final SocialOmni LaTeX tables and diagnostic figures from one snapshot."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr


MODEL_ORDER = (
    "gpt4o",
    "gemini_2_5_pro",
    "gemini_2_5_flash",
    "gemini_3_flash_preview",
    "gemini_3_pro_preview",
    "qwen3_omni",
    "qwen3_omni_thinking",
    "qwen2_5_omni",
    "omnivinci",
    "vita_1_5",
    "baichuan_omni_1_5",
)
DISPLAY = {
    "gpt4o": "GPT-4o",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_3_flash_preview": "Gemini 3 Flash",
    "gemini_3_pro_preview": "Gemini 3 Pro",
    "qwen3_omni": "Qwen3-Omni",
    "qwen3_omni_thinking": "Qwen3-Omni-Thinking",
    "qwen2_5_omni": "Qwen2.5-Omni",
    "omnivinci": "OmniVinci",
    "vita_1_5": "VITA-1.5",
    "baichuan_omni_1_5": "Baichuan-Omni-1.5",
}
CITATIONS = {
    "gpt4o": "hurst2024gpt",
    "gemini_2_5_pro": "comanici2025gemini",
    "gemini_2_5_flash": "comanici2025gemini",
    "gemini_3_flash_preview": "google2025gemini3",
    "gemini_3_pro_preview": "google2025gemini3",
    "qwen3_omni": "Qwen3-Omni",
    "qwen3_omni_thinking": "Qwen3-Omni",
    "qwen2_5_omni": "xu2025qwen25omni",
    "omnivinci": "ye2025omnivinci",
    "vita_1_5": "fu2025vita",
    "baichuan_omni_1_5": "li2025baichuan",
}
INTERFACES = {
    "gpt4o": "Cascade",
    "gemini_2_5_pro": "Visual-only",
    "gemini_2_5_flash": "Visual-only",
    "gemini_3_flash_preview": "Visual-only",
    "gemini_3_pro_preview": "Visual-only",
    "qwen3_omni": "Native AV",
    "qwen3_omni_thinking": "Native AV",
    "qwen2_5_omni": "Native AV",
    "omnivinci": "Native AV",
    "vita_1_5": "Native AV",
    "baichuan_omni_1_5": "Native AV",
}
JUDGES = ("gpt4o", "gemini_2_5_pro", "qwen3_omni")
COLORS = {"who": "#267A78", "when": "#D27A38", "quality": "#4F6699", "joint": "#8A5A44"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--human-dir", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def fmt(value: float | None) -> str:
    return "N.A." if value is None else f"{value:.2f}"


def cited_name(model: str) -> str:
    return f"{DISPLAY[model]}~\\cite{{{CITATIONS[model]}}}"


def bold_max(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
    return {
        field: max(float(row[field]) for row in rows if row.get(field) not in (None, ""))
        for field in fields
    }


def tex_value(value: float | None, maximum: float | None) -> str:
    text = fmt(value)
    return f"\\textbf{{{text}}}" if value is not None and maximum is not None and abs(value - maximum) < 1e-9 else text


def write_main_table(rows: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "who_accuracy",
        "who_macro_f1",
        "when_accuracy",
        "when_macro_f1",
        "q_ensemble",
        "coverage_plus",
        "q_joint_ensemble",
    )
    maxima = bold_max(rows, fields)
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [cited_name(row["model"]), INTERFACES[row["model"]]]
                + [tex_value(f(row[field]), maxima[field]) for field in fields]
            )
            + r" \\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{2.4pt}",
                r"\caption{\textbf{SocialOmni main performance.} Who/When use strict 2,000/200-item denominators. Interface denotes native audio-video input (Native AV), sampled frames without media audio (Visual-only), or query-bounded audio transcription with sampled frames (Cascade). $Q_{\rm Ens}$ is the complete three-judge mean over non-empty true-positive responses; $Q_{\rm joint}^{\rm Ens}=({\rm Cov.}^{+}/100)Q_{\rm Ens}$.}",
                r"\label{tab:main_triplet}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrr}",
                r"\toprule",
                r"\textbf{Model} & \textbf{Interface} & \multicolumn{2}{c}{\textbf{Who}} & \multicolumn{2}{c}{\textbf{When}} & \textbf{$Q_{\rm Ens}$} & \textbf{Cov.$^{+}$} & \textbf{$Q_{\rm joint}^{\rm Ens}$} \\",
                r" & & Acc. & F1 & Acc. & F1 & & & \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular*}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return 100 * (center - radius), 100 * (center + radius)


def write_classification_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        low, high = wilson(int(float(row["who_correct"])), 2000)
        when_low, when_high = wilson(
            int(float(row["when_tp"])) + int(float(row["when_tn"])), 200
        )
        body.append(
            f"{DISPLAY[row['model']]} & {float(row['who_accuracy']):.2f} & {float(row['who_macro_f1']):.2f} & "
            f"[{low:.2f}, {high:.2f}] & {float(row['when_accuracy']):.2f} & "
            f"{float(row['when_macro_f1']):.2f} & [{when_low:.2f}, {when_high:.2f}] & "
            f"{int(float(row['when_tp']))}/{int(float(row['when_fp']))}/"
            f"{int(float(row['when_tn']))}/{int(float(row['when_fn']))} \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering\small",
                r"\caption{Strict classification results. CIs are Wilson 95\% intervals for accuracy; When TP/FP/TN/FN counts cover parsed outputs, while failures remain in the 200-item metric denominator and are itemized separately.}",
                r"\label{tab:perception_f1_ci}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrr}",
                r"\toprule",
                r"Model & Who Acc. & Who F1 & Who CI & When Acc. & When F1 & When CI & TP/FP/TN/FN \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular*}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def visibility_rows(source_root: Path) -> list[dict[str, Any]]:
    dataset = json.loads((source_root / "data" / "level_1" / "dataset.json").read_text(encoding="utf-8"))
    visibility = {
        int(row["id"]): str(row.get("metadata", {}).get("consistency") or "").lower() == "consistent"
        for row in dataset
    }
    output = []
    for model in MODEL_ORDER:
        payload = json.loads(
            (source_root / "results" / f"results_{model}_level1_audio-video_no-asr.json").read_text(encoding="utf-8")
        )
        counts = {True: [0, 0], False: [0, 0]}
        for row in payload["results"]:
            group = visibility[int(row["id"])]
            counts[group][1] += 1
            counts[group][0] += int(row.get("correct_answer") == row.get("prediction"))
        visible = 100 * counts[True][0] / counts[True][1]
        mismatch = 100 * counts[False][0] / counts[False][1]
        output.append({"model": model, "visible": visible, "mismatch": mismatch, "gap": visible - mismatch})
    return output


def write_visibility_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = [
        f"{DISPLAY[row['model']]} & {row['visible']:.2f} & {row['mismatch']:.2f} & {row['gap']:+.2f} \\\\"
        for row in rows
    ]
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\small",
                r"\setlength{\tabcolsep}{3pt}",
                r"\caption{Speaker-visible and naturally occurring visibility-mismatch accuracy. The gap is descriptive, not causal.}",
                r"\label{tab:visibility_results}",
                r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrr}\toprule",
                r"Model & Visible & Mismatch & $\Delta_{\rm vis}$ \\\midrule",
                *body,
                r"\bottomrule\end{tabular*}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_generation_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        coverage = float(row["coverage_plus"])
        coverage_low, coverage_high = wilson(
            int(float(row["responses"])), int(float(row["when_gold_yes"]))
        )
        quality = fmt(f(row["q_ensemble"]))
        quality_ci = (
            "N.A."
            if row.get("q_ensemble_ci_low") in (None, "")
            else f"[{float(row['q_ensemble_ci_low']):.2f}, {float(row['q_ensemble_ci_high']):.2f}]"
        )
        body.append(
            f"{DISPLAY[row['model']]} & {quality} & {quality_ci} & {coverage:.2f} & "
            f"[{coverage_low:.2f}, {coverage_high:.2f}] & {float(row['q_joint_ensemble']):.2f} \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering\small",
                r"\caption{Three-judge conditional response quality and positive-opportunity adjustment.}",
                r"\label{tab:appx_generation_coverage}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr}",
                r"\toprule",
                r"Model & $Q_{\rm Ens}$ & Quality CI & Cov.$^{+}$ & Coverage CI & $Q_{\rm joint}^{\rm Ens}$ \\\midrule",
                *body,
                r"\bottomrule\end{tabular*}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_generation_uncertainty_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        coverage_low, coverage_high = wilson(
            int(float(row["responses"])), int(float(row["when_gold_yes"]))
        )
        quality_ci = (
            "N.A."
            if row.get("q_ensemble_ci_low") in (None, "")
            else f"[{float(row['q_ensemble_ci_low']):.1f},{float(row['q_ensemble_ci_high']):.1f}]"
        )
        body.append(
            f"{DISPLAY[row['model']]} & {quality_ci} & "
            f"[{coverage_low:.1f},{coverage_high:.1f}] \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\small",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\caption{Bootstrap 95\% intervals for conditional response quality and Wilson 95\% intervals for positive-opportunity coverage. Point estimates are in Table~\ref{tab:main_triplet}.}",
                r"\label{tab:generation_uncertainty}",
                r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr}",
                r"\toprule Model & $Q_{\rm Ens}$ CI & Cov.$^{+}$ CI \\\midrule",
                *body,
                r"\bottomrule\end{tabular*}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_classification_uncertainty_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        who_low, who_high = wilson(int(float(row["who_correct"])), int(float(row["who_n"])))
        when_correct = int(float(row["when_tp"])) + int(float(row["when_tn"]))
        when_low, when_high = wilson(when_correct, int(float(row["when_n"])))
        body.append(
            f"{DISPLAY[row['model']]} & {float(row['who_macro_f1']):.1f} & "
            f"[{who_low:.1f},{who_high:.1f}] & {float(row['when_macro_f1']):.1f} & "
            f"[{when_low:.1f},{when_high:.1f}] \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\footnotesize",
                r"\setlength{\tabcolsep}{2pt}",
                r"\caption{Classification reliability. F1 is macro-F1; brackets are Wilson 95\% accuracy intervals.}",
                r"\label{tab:classification_uncertainty}",
                r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr}",
                r"\toprule & \multicolumn{2}{c}{Who} & \multicolumn{2}{c}{When} \\",
                r"Model & F1 & Acc. CI & F1 & Acc. CI \\\midrule",
                *body,
                r"\bottomrule\end{tabular*}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_compact_reliability_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = []
    for row in rows:
        who_low, who_high = wilson(int(float(row["who_correct"])), int(float(row["who_n"])))
        when_correct = int(float(row["when_tp"])) + int(float(row["when_tn"]))
        when_low, when_high = wilson(when_correct, int(float(row["when_n"])))
        coverage_low, coverage_high = wilson(
            int(float(row["responses"])), int(float(row["when_gold_yes"]))
        )
        quality_ci = (
            "N.A."
            if row.get("q_ensemble_ci_low") in (None, "")
            else f"[{float(row['q_ensemble_ci_low']):.1f},{float(row['q_ensemble_ci_high']):.1f}]"
        )
        body.append(
            f"{DISPLAY[row['model']]} & {float(row['who_macro_f1']):.1f} & [{who_low:.1f},{who_high:.1f}] & "
            f"{float(row['when_macro_f1']):.1f} & [{when_low:.1f},{when_high:.1f}] & "
            f"{quality_ci} & [{coverage_low:.1f},{coverage_high:.1f}] \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\caption{Reliability of the principal estimates. Classification columns report macro-F1 and Wilson 95\% accuracy intervals; generation columns report bootstrap 95\% $Q_{\rm Ens}$ intervals and Wilson 95\% coverage intervals. Point estimates are in Table~\ref{tab:main_triplet}.}",
                r"\label{tab:compact_reliability}",
                r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr}",
                r"\toprule & \multicolumn{2}{c}{Who} & \multicolumn{2}{c}{When} & \multicolumn{2}{c}{How} \\",
                r"Model & F1 & Acc. CI & F1 & Acc. CI & $Q_{\rm Ens}$ CI & Cov.$^{+}$ CI \\\midrule",
                *body,
                r"\bottomrule\end{tabular*}",
                r"\end{table*}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_failure_table(rows: list[dict[str, Any]], path: Path) -> None:
    body = [
        f"{DISPLAY[row['model']]} & {int(float(row['who_failures']))} & "
        f"{int(float(row['who_parse_failures']))} & {int(float(row['when_failures']))} & "
        f"{int(float(row['when_parse_failures']))} & {int(float(row['responses']))} \\\\"
        for row in rows
    ]
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{2.5pt}",
                r"\caption{Strict-denominator failure accounting. Request/runtime and parse failures are disjoint; all remain in the fixed denominators.}",
                r"\label{tab:failure_accounting}",
                r"\begin{tabular}{@{}lrrrrr@{}}",
                r"\toprule Model & Who req. & Who parse & When req. & When parse & Scored \\\midrule",
                *body,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def rank_map(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda model: (-values[model], MODEL_ORDER.index(model)))
    return {model: index + 1 for index, model in enumerate(ordered)}


def write_judge_sensitivity(response_rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = {model: [] for model in MODEL_ORDER}
    for row in response_rows:
        grouped[row["model"]].append(row)
    full = {
        model: sum(float(row["q_ensemble"]) for row in rows) / len(rows)
        for model, rows in grouped.items()
        if rows
    }
    full_ranks = rank_map(full)
    leave_out_ranks: list[dict[str, int]] = []
    for omitted in JUDGES:
        values = {
            model: sum(
                sum(float(row[f"score_{judge}"]) for judge in JUDGES if judge != omitted) / 2
                for row in rows
            )
            / len(rows)
            for model, rows in grouped.items()
            if rows
        }
        leave_out_ranks.append(rank_map(values))
    body = []
    for model in MODEL_ORDER:
        rows = grouped[model]
        if not rows:
            body.append(f"{DISPLAY[model]} & 0 & N.A. & N.A. & N.A. & N.A. \\\\")
            continue
        no_family = sum(float(row["q_no_family"]) for row in rows) / len(rows)
        max_shift = max(abs(ranks[model] - full_ranks[model]) for ranks in leave_out_ranks)
        body.append(
            f"{DISPLAY[model]} & {len(rows)} & {full[model]:.2f} & {no_family:.2f} & "
            f"{no_family - full[model]:+.2f} & {max_shift} \\\\"
        )
    path.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\footnotesize",
                r"\setlength{\tabcolsep}{2.2pt}",
                r"\caption{Judge-family sensitivity over eligible responses. N.A. denotes no eligible response; shift is the maximum leave-one-judge-out rank change.}",
                r"\label{tab:judge_sensitivity}",
                r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
                r"Model & Resp. & Full & No-fam. & $\Delta$ & Shift \\\midrule",
                *body,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def judge_statistics(response_rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for index, left in enumerate(JUDGES):
        for right in JUDGES[index + 1 :]:
            x = [float(row[f"score_{left}"]) for row in response_rows]
            y = [float(row[f"score_{right}"]) for row in response_rows]
            stats[f"{left}_vs_{right}"] = {
                "pearson": float(pearsonr(x, y).statistic),
                "spearman": float(spearmanr(x, y).statistic),
                "kendall": float(kendalltau(x, y).statistic),
                "mae": sum(abs(a - b) for a, b in zip(x, y)) / len(x),
            }
    grouped = {model: [row for row in response_rows if row["model"] == model] for model in MODEL_ORDER}
    included = [model for model in MODEL_ORDER if grouped[model]]
    full_values = {
        model: sum(float(row["q_ensemble"]) for row in grouped[model]) / len(grouped[model])
        for model in included
    }
    full_ranks = rank_map(full_values)
    stats["leave_one_judge_out_ranking"] = {}
    for omitted in JUDGES:
        values = {
            model: sum(
                sum(float(row[f"score_{judge}"]) for judge in JUDGES if judge != omitted) / 2
                for row in grouped[model]
            )
            / len(grouped[model])
            for model in included
        }
        ranks = rank_map(values)
        stats["leave_one_judge_out_ranking"][omitted] = {
            "kendall": float(kendalltau([full_values[m] for m in included], [values[m] for m in included]).statistic),
            "maximum_rank_shift": max(abs(full_ranks[m] - ranks[m]) for m in included),
        }
    no_family_values = {
        model: sum(float(row["q_no_family"]) for row in grouped[model]) / len(grouped[model])
        for model in included
    }
    no_family_ranks = rank_map(no_family_values)
    stats["same_family_removal_ranking"] = {
        "kendall": float(
            kendalltau(
                [full_values[model] for model in included],
                [no_family_values[model] for model in included],
            ).statistic
        ),
        "maximum_rank_shift": max(
            abs(full_ranks[model] - no_family_ranks[model]) for model in included
        ),
    }
    output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def write_judge_removal_rows(stats: dict[str, Any], path: Path) -> None:
    labels = {
        "gpt4o": "w/o GPT",
        "gemini_2_5_pro": "w/o Gem.",
        "qwen3_omni": "w/o Qwen",
    }
    rows = []
    for judge in JUDGES:
        values = stats["leave_one_judge_out_ranking"][judge]
        rows.append(
            f"{labels[judge]} & {values['kendall']:.2f} & "
            f"{values['maximum_rank_shift']} \\\\"
        )
    family = stats["same_family_removal_ranking"]
    rows.append(
        f"Same-family & {family['kendall']:.2f} & "
        f"{family['maximum_rank_shift']} \\\\"
    )
    path.write_text("\n".join([*rows, r"\bottomrule"]) + "\n", encoding="utf-8")


def write_judge_pairwise_rows(stats: dict[str, Any], path: Path) -> None:
    pairs = (
        ("GPT / Gem.", "gpt4o_vs_gemini_2_5_pro"),
        ("GPT / Qwen", "gpt4o_vs_qwen3_omni"),
        ("Gem. / Qwen", "gemini_2_5_pro_vs_qwen3_omni"),
    )
    rows = []
    for label, key in pairs:
        values = stats[key]
        rows.append(
            f"{label} & {values['spearman']:.2f} & "
            f"{values['kendall']:.2f} & {values['mae']:.2f} \\\\"
        )
    path.write_text("\n".join([*rows, r"\bottomrule"]) + "\n", encoding="utf-8")


def write_judge_combined_rows(stats: dict[str, Any], path: Path) -> None:
    removal_labels = {
        "gpt4o": "w/o GPT",
        "gemini_2_5_pro": "w/o Gem.",
        "qwen3_omni": "w/o Qwen",
    }
    pair_specs = (
        ("GPT / Gem.", "gpt4o_vs_gemini_2_5_pro"),
        ("GPT / Qwen", "gpt4o_vs_qwen3_omni"),
        ("Gem. / Qwen", "gemini_2_5_pro_vs_qwen3_omni"),
    )
    removal_rows = []
    for judge in JUDGES:
        values = stats["leave_one_judge_out_ranking"][judge]
        removal_rows.append(
            (removal_labels[judge], values["kendall"], values["maximum_rank_shift"])
        )
    family = stats["same_family_removal_ranking"]
    removal_rows.append(("Same-family", family["kendall"], family["maximum_rank_shift"]))

    rows = []
    for index, (removal, rank_tau, shift) in enumerate(removal_rows):
        if index < len(pair_specs):
            pair_label, key = pair_specs[index]
            pair = stats[key]
            pair_cells = f"{pair_label} & {pair['spearman']:.2f} & {pair['kendall']:.2f} & {pair['mae']:.2f}"
        else:
            pair_cells = r" &  &  & "
        rows.append(
            f"{removal} & {rank_tau:.2f} & {shift} & {pair_cells} \\\\"
        )
    path.write_text("\n".join([*rows, r"\bottomrule"]) + "\n", encoding="utf-8")


def horizontal_plot(labels: list[str], series: list[tuple[str, list[float], str]], path: Path) -> None:
    positions = list(range(len(labels)))
    offset_step = 0.20
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for index, (name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * offset_step
        y_values = [position + offset for position in positions]
        ax.scatter(values, y_values, s=34, label=name, color=color, zorder=3)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)")
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncol=max(1, len(series)))
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def diagnostic_profiles(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [DISPLAY[row["model"]] for row in rows]
    panels = (
        (
            "Who",
            [("Accuracy", [float(row["who_accuracy"]) for row in rows], COLORS["who"])],
        ),
        (
            "When",
            [
                ("Accuracy", [float(row["when_accuracy"]) for row in rows], COLORS["when"]),
                ("Macro-F1", [float(row["when_macro_f1"]) for row in rows], COLORS["who"]),
            ],
        ),
        (
            "How",
            [
                ("Conditional quality", [f(row["q_ensemble"]) or 0.0 for row in rows], COLORS["quality"]),
                ("Coverage-adjusted", [float(row["q_joint_ensemble"]) for row in rows], COLORS["joint"]),
            ],
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), sharex=True)
    positions = list(range(len(labels)))
    for axis, (title, series) in zip(axes, panels):
        offset_step = 0.20
        for index, (name, values, color) in enumerate(series):
            offset = (index - (len(series) - 1) / 2) * offset_step
            axis.scatter(
                values,
                [position + offset for position in positions],
                s=28,
                label=name,
                color=color,
                zorder=3,
            )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_yticks(positions, labels, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Score (%)", fontsize=8)
        axis.grid(axis="x", color="#D8D8D8", linewidth=0.6, zorder=0)
        axis.tick_params(axis="x", labelsize=7)
        axis.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def render_human(human_dir: Path, table_dir: Path, figure_dir: Path) -> None:
    scored = read_csv(human_dir / "human_calibration_scored.csv")
    model_rows = read_csv(human_dir / "human_calibration_model_summary.csv")
    table_body = [
        f"{DISPLAY[row['model']]} & {row['n']} & {float(row['q_ensemble_mean']):.2f} & "
        f"{float(row['human_mean']):.2f} & {float(row['mean_difference']):+.2f} & "
        f"{float(row['mae']):.2f} \\\\"
        for row in model_rows
    ]
    table_dir.joinpath("human_calibration.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{2pt}",
                r"\caption{Blind 200-response human calibration. Difference is ensemble minus human mean; agreement is computed at response level.}",
                r"\label{tab:human_calibration}",
                r"\begin{tabular}{@{}lrrrrr@{}}",
                r"\toprule Model & $n$ & Ens. & Human & Diff. & MAE \\\midrule",
                *table_body,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    scatter_ax, difference_ax = axes
    scatter_ax.scatter(
        [float(row["q_ensemble"]) for row in scored],
        [float(row["human_score"]) for row in scored],
        color=COLORS["quality"],
        edgecolors=COLORS["quality"],
        alpha=0.50,
        s=24,
    )
    scatter_ax.plot([0, 100], [0, 100], color="#666666", linewidth=1, linestyle="--")
    scatter_ax.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xlabel="Three-judge ensemble",
        ylabel="Human score",
        title="Response-level agreement",
    )
    labels = [DISPLAY[row["model"]] for row in model_rows]
    differences = [float(row["mean_difference"]) for row in model_rows]
    colors = [COLORS["quality"] if value >= 0 else COLORS["joint"] for value in differences]
    difference_ax.barh(labels, differences, color=colors)
    difference_ax.axvline(0, color="#444444", linewidth=0.8)
    difference_ax.invert_yaxis()
    difference_ax.set(
        xlabel="Ensemble mean - human mean",
        title="Per-model mean difference",
    )
    difference_ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(figure_dir / "S1_human_calibration.pdf")
    plt.close(fig)
    (figure_dir / "S1_human_calibration_scatter.pdf").unlink(missing_ok=True)
    (figure_dir / "S2_human_model_difference.pdf").unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.snapshot_dir / "model_summary.csv")
    by_model = {row["model"]: row for row in rows}
    ordered = [by_model[model] for model in MODEL_ORDER]
    response_rows = read_csv(args.snapshot_dir / "ensemble_response_scores.csv")
    if any(row[f"score_{judge}"] == "" for row in response_rows for judge in JUDGES):
        raise SystemExit("Cannot render final artifacts with missing judge scores")

    write_main_table(ordered, args.table_dir / "main_triplet.tex")
    write_classification_table(ordered, args.table_dir / "timing_and_perception_combined.tex")
    write_visibility_table(visibility_rows(args.source_root), args.table_dir / "visibility_results.tex")
    write_generation_table(ordered, args.table_dir / "generation_coverage.tex")
    write_generation_uncertainty_table(ordered, args.table_dir / "generation_uncertainty.tex")
    write_classification_uncertainty_table(ordered, args.table_dir / "classification_uncertainty.tex")
    write_compact_reliability_table(ordered, args.table_dir / "compact_reliability.tex")
    write_failure_table(ordered, args.table_dir / "failure_accounting.tex")
    write_judge_sensitivity(response_rows, args.table_dir / "judge_sensitivity.tex")
    judge_stats = judge_statistics(response_rows, args.snapshot_dir / "judge_agreement.json")
    write_judge_removal_rows(judge_stats, args.table_dir / "judge_removal_rows.tex")
    write_judge_pairwise_rows(judge_stats, args.table_dir / "judge_pairwise_rows.tex")
    write_judge_combined_rows(judge_stats, args.table_dir / "judge_combined_rows.tex")

    labels = [DISPLAY[model] for model in MODEL_ORDER]
    diagnostic_profiles(ordered, args.figure_dir / "figS01_diagnostic_profiles.pdf")
    (args.figure_dir / "3_diagnostic_profiles.pdf").unlink(missing_ok=True)
    for old_name in ("2a_who_profile.pdf", "2b_when_profile.pdf", "2c_how_profile.pdf"):
        (args.figure_dir / old_name).unlink(missing_ok=True)
    if args.human_dir and (args.human_dir / "human_calibration_scored.csv").exists():
        render_human(args.human_dir, args.table_dir, args.figure_dir)


if __name__ == "__main__":
    main()
