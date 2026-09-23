#!/usr/bin/env python3
"""Render SocialOmni final-extension snapshot summaries as compact AAAI LaTeX."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_extension_compact import write_judge_compact, write_modality_compact


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
TABLES = {
    "main": "extension_main_triplet.tex",
    "how": "extension_how_gold_main.tex",
    "when": "extension_when_baselines.tex",
    "modality": "extension_modality_ablation.tex",
    "judge": "extension_judge_audit.tex",
    "supp": "extension_supplementary_details.tex",
}


class Completeness:
    def __init__(self, *, strict: bool) -> None:
        self.strict = strict
        self.missing: list[str] = []

    def mark(self, message: str) -> None:
        self.missing.append(message)

    def require(self) -> None:
        if self.strict and self.missing:
            details = "\n".join(f"- {item}" for item in self.missing[:30])
            extra = "" if len(self.missing) <= 30 else f"\n- ... {len(self.missing) - 30} more"
            raise SystemExit(f"Cannot render complete extension tables with missing data:\n{details}{extra}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_csv_optional(path: Path, completeness: Completeness) -> list[dict[str, str]]:
    if not path.exists():
        completeness.mark(f"missing input {path.name}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_optional(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def complete_from_report(snapshot_dir: Path) -> bool:
    report = read_json_optional(snapshot_dir / "audit_report.json")
    return bool(isinstance(report, dict) and report.get("complete") is True)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def display_name(model: str) -> str:
    return tex_escape(DISPLAY.get(model, model or "TODO"))


def cited_display_name(model: str) -> str:
    name = display_name(model)
    citation = CITATIONS.get(model)
    return f"{name}~\\cite{{{citation}}}" if citation else name


def how_gold_interface(model: str, modality: str) -> str:
    """Expose the evaluated primary interface rather than storage modality."""
    if model == "gpt4o" and modality == "audio-video":
        return "cascade"
    if model.startswith("gemini") and modality == "audio-video":
        return "visual-only"
    return modality


def number(value: object) -> float | None:
    if value in (None, "", "None", "N.A.", "TODO", "--"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def fmt(value: object, completeness: Completeness, label: str, *, percent: bool = False) -> str:
    parsed = number(value)
    if parsed is None:
        completeness.mark(f"missing value {label}")
        return "--"
    if percent and abs(parsed) <= 1.0:
        parsed *= 100
    return f"{parsed:.2f}"


def fmt_int(value: object, completeness: Completeness, label: str) -> str:
    parsed = number(value)
    if parsed is None:
        completeness.mark(f"missing value {label}")
        return "--"
    return str(int(parsed))


def fmt_ci(row: dict[str, str], low: str, high: str, completeness: Completeness, label: str) -> str:
    left = number(row.get(low))
    right = number(row.get(high))
    if left is None or right is None:
        completeness.mark(f"missing CI {label}")
        return "--"
    return f"[{left:.2f}, {right:.2f}]"


def write_table(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def placeholder_table(path: Path, label: str, caption: str) -> None:
    write_table(
        path,
        [
            r"\begin{table}[t]",
            r"\centering\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{lc}\toprule",
            r"Artifact & Status \\\midrule",
            r"TODO & -- \\",
            r"\bottomrule\end{tabular}",
            r"\end{table}",
        ],
    )


def write_how_gold(rows: list[dict[str, str]], path: Path, completeness: Completeness) -> None:
    if not rows:
        completeness.mark("empty input how_gold_summary.csv")
        placeholder_table(path, "tab:extension_how_gold", r"\textbf{How@Gold extension results.} TODO.")
        return
    ordered = sorted(rows, key=lambda item: (item.get("model", ""), item.get("modality", "")))
    primary = [row for row in ordered if row.get("modality", "") not in {"audio-only", "video-only", "frames-only", "transcript-only"}]
    variants = [row for row in ordered if row not in primary]

    def row_tex(row: dict[str, str], compact: bool = False) -> str:
        model = row.get("model", "")
        modality = how_gold_interface(model, row.get("modality", ""))
        quality = fmt(row.get("primary_aware_mean_failures_zero"), completeness, f"{model} How@Gold")
        ci = fmt_ci(row, "primary_aware_ci_low", "primary_aware_ci_high", completeness, f"{model} How@Gold")
        if compact:
            return " & ".join([display_name(model), tex_escape(modality or "TODO"), quality, ci]) + r" \\"
        valid = fmt_int(row.get("successes"), completeness, f"{model} successes")
        failures = fmt_int(row.get("failures"), completeness, f"{model} failures")
        return " & ".join([display_name(model), tex_escape(modality or "TODO"), f"{valid} ({failures} fail)", quality, ci]) + r" \\"

    primary_body = [row_tex(row) for row in primary]
    variant_body = [row_tex(row, compact=True) for row in variants]
    write_table(
        path,
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\caption{\textbf{Complete gold-state generation results.} $Q_{\rm Gold}$ is the three-judge reference-aware mean over the fixed 128 positive states; failed or missing responses score zero. CI is a 95\% item bootstrap interval.}",
            r"\label{tab:extension_how_gold}",
            r"\begin{minipage}[t]{0.49\textwidth}\centering",
            r"\begin{tabular}{@{}lcrrl@{}}\toprule Model & Interface & Valid & $Q_{\rm Gold}$ & 95\% CI \\\midrule",
            *primary_body,
            r"\bottomrule\end{tabular}\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.49\textwidth}\centering",
            r"\begin{tabular}{@{}lcrl@{}}\toprule Model & Variant & $Q_{\rm Gold}$ & 95\% CI \\\midrule",
            *variant_body,
            r"\bottomrule\end{tabular}\end{minipage}",
            r"\end{table*}",
        ],
    )


def write_main(rows: list[dict[str, str]], path: Path, completeness: Completeness) -> None:
    if not rows:
        completeness.mark("empty input main_extension_summary.csv")
        placeholder_table(path, "tab:main_triplet", "Primary SocialOmni results. TODO.")
        return
    metrics = ("who_accuracy", "when_accuracy", "q_gold", "q_ensemble", "coverage_plus", "q_joint_ensemble")
    maxima = {
        metric: max((value for row in rows if (value := number(row.get(metric))) is not None), default=None)
        for metric in metrics
    }

    def main_value(row: dict[str, str], metric: str, label: str) -> str:
        if metric == "q_ensemble" and number(row.get(metric)) is None and number(row.get("responses")) == 0:
            return "--"
        rendered = fmt(row.get(metric), completeness, label)
        value = number(row.get(metric))
        maximum = maxima[metric]
        return f"\\textbf{{{rendered}}}" if value is not None and maximum is not None and abs(value - maximum) < 1e-9 else rendered

    body = []
    for row in rows:
        model = row.get("model", "")
        body.append(
            " & ".join(
                [
                    cited_display_name(model),
                    tex_escape(row.get("interface") or "TODO"),
                    main_value(row, "who_accuracy", f"{model} Who"),
                    main_value(row, "when_accuracy", f"{model} When"),
                    main_value(row, "q_gold", f"{model} QGold"),
                    main_value(row, "q_ensemble", f"{model} QEns"),
                    main_value(row, "coverage_plus", f"{model} coverage"),
                    main_value(row, "q_joint_ensemble", f"{model} Qjoint"),
                ]
            )
            + r" \\"
        )
    write_table(
        path,
        [
            r"\begin{table*}[t]",
            r"\centering\small",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{Primary strict-denominator results. $Q_{\rm Gold}$ forces generation at all 128 gold-positive opportunities; $Q_{\rm Ens}$ is conditional on a non-empty true-positive response; Cov.$^{+}$ and $Q_{\rm joint}^{\rm Ens}$ retain the model's entry policy.}",
            r"\label{tab:main_triplet}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr}",
            r"\toprule",
            r"Model & Interface & Who & When & $Q_{\rm Gold}$ & $Q_{\rm Ens}$ & Cov.$^{+}$ & $Q_{\rm joint}^{\rm Ens}$ \\\midrule",
            *body,
            r"\bottomrule\end{tabular*}",
            r"\end{table*}",
        ],
    )


def write_when_baselines(rows: list[dict[str, str]], path: Path, completeness: Completeness) -> None:
    if not rows:
        completeness.mark("empty input when_baselines_summary.csv")
        placeholder_table(path, "tab:extension_when_baselines", "When baseline extension results. TODO.")
        return
    body = []
    baseline_names = {
        "always_no": "Always-NO",
        "always_yes": "Always-YES",
        "majority": "Majority",
        "random_seed_13": "Random",
        "silence_0.2s_dbfs_-35": "Pause 0.2 s",
        "silence_0.5s_dbfs_-35": "Pause 0.5 s",
        "silence_1.0s_dbfs_-35": "Pause 1.0 s",
        "whisper_small_punctuation": "Whisper punct.",
        "whisper_small_punctuation_silence_0.2s_dbfs_-35": "Whisper + pause",
    }
    for row in sorted(rows, key=lambda item: item.get("baseline", "")):
        baseline = row.get("baseline", "")
        body.append(
            f"{tex_escape(baseline_names.get(baseline, baseline or 'TODO'))} & "
            f"{fmt(row.get('metric_accuracy'), completeness, baseline + ' accuracy', percent=True)} & "
            f"{fmt(row.get('metric_macro_f1'), completeness, baseline + ' macro-F1', percent=True)} & "
            f"{fmt_int(row.get('metric_tp'), completeness, baseline + ' TP')} & "
            f"{fmt_int(row.get('metric_fp'), completeness, baseline + ' FP')} & "
            f"{fmt_int(row.get('metric_tn'), completeness, baseline + ' TN')} & "
            f"{fmt_int(row.get('metric_fn'), completeness, baseline + ' FN')} \\\\"
        )
    write_table(
        path,
        [
            r"\begin{table}[t]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{When-only controls. Pause rules use $-35$ dBFS; Random uses seed 13; Whisper + pause combines punctuation with a 0.2-s pause.}",
            r"\label{tab:extension_when_baselines}",
            r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrrr}\toprule",
            r"Baseline & Acc. & F1 & TP & FP & TN & FN \\\midrule",
            *body,
            r"\bottomrule\end{tabular*}",
            r"\end{table}",
        ],
    )


def write_modality(
    rows: list[dict[str, str]],
    how_rows: list[dict[str, str]],
    path: Path,
    completeness: Completeness,
) -> None:
    if not rows or not how_rows:
        completeness.mark("empty modality inputs")
        placeholder_table(path, "tab:extension_modality_ablation", "Modality ablation extension results. TODO.")
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
        body.append(
            " & ".join(
                [
                    display_name(model),
                    "T/F" if model == "gpt4o" else "A/V",
                    *[
                        fmt(value, completeness, f"{model} modality metric {index}", percent=True)
                        for index, value in enumerate(values)
                    ],
                    fmt(gold_lookup.get((model, audio_mode)), completeness, f"{model} QGold audio"),
                    fmt(gold_lookup.get((model, video_mode)), completeness, f"{model} QGold video"),
                ]
            )
            + r" \\"
        )
    write_table(
        path,
        [
            r"\begin{table*}[t]",
            r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{2.5pt}",
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


def write_judge_audit(
    aggregations: list[dict[str, str]],
    correlations: list[dict[str, str]],
    rank_shifts: list[dict[str, str]],
    path: Path,
    completeness: Completeness,
) -> None:
    if not aggregations and not correlations and not rank_shifts:
        completeness.mark("empty judge audit CSVs")
        placeholder_table(path, "tab:extension_judge_audit", "Combined judge-audit extension results. TODO.")
        return
    aggregate_body = []
    for row in sorted(aggregations, key=lambda item: (item.get("source", ""), item.get("reference_mode", ""), item.get("judge", ""), item.get("model", "")))[:12]:
        label = "/".join(part for part in (row.get("source"), row.get("reference_mode"), row.get("judge")) if part)
        aggregate_body.append(
            f"{tex_escape(label or 'TODO')} & {display_name(row.get('model', ''))} & "
            f"{fmt_int(row.get('n'), completeness, label + ' n')} & "
            f"{fmt(row.get('mean'), completeness, label + ' mean')} & "
            f"{fmt(row.get('median'), completeness, label + ' median')} \\\\"
        )
    corr_body = []
    for row in sorted(correlations, key=lambda item: item.get("metric", ""))[:8]:
        metric = row.get("metric", "")
        corr_body.append(
            f"{tex_escape(metric or 'TODO')} & {fmt_int(row.get('n'), completeness, metric + ' n')} & "
            f"{fmt(row.get('pearson'), completeness, metric + ' Pearson')} & "
            f"{fmt(row.get('spearman'), completeness, metric + ' Spearman')} \\\\"
        )
    rank_body = []
    for row in sorted(rank_shifts, key=lambda item: (item.get("metric", ""), item.get("metric_rank", ""), item.get("model", "")))[:8]:
        metric = row.get("metric", "")
        rank_body.append(
            f"{tex_escape(metric or 'TODO')} & {display_name(row.get('model', ''))} & "
            f"{fmt_int(row.get('base_rank'), completeness, metric + ' base rank')} & "
            f"{fmt_int(row.get('metric_rank'), completeness, metric + ' rank')} & "
            f"{fmt(row.get('rank_shift'), completeness, metric + ' rank shift')} \\\\"
        )
    write_table(
        path,
        [
            r"\begin{table*}[t]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\caption{Combined extension judge audit. Aggregates summarize judge score distributions; correlations and rank shifts compare extension judge settings against the base ensemble where overlap exists.}",
            r"\label{tab:extension_judge_audit}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr}",
            r"\toprule",
            r"Setting & Model & $n$ & Mean & Median \\\midrule",
            *aggregate_body,
            r"\bottomrule\end{tabular*}",
            r"\vspace{2pt}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrr}",
            r"\toprule",
            r"Metric & $n$ & Pearson & Spearman \\\midrule",
            *corr_body,
            r"\bottomrule\end{tabular*}",
            r"\vspace{2pt}",
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr}",
            r"\toprule",
            r"Metric & Model & Base rank & Audit rank & Shift \\\midrule",
            *rank_body,
            r"\bottomrule\end{tabular*}",
            r"\end{table*}",
        ],
    )


def write_supplement(
    how_rows: list[dict[str, str]],
    modality_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    sanity_rows: list[dict[str, str]],
    path: Path,
    completeness: Completeness,
) -> None:
    body = [
        f"How@Gold groups & {len(how_rows) if how_rows else '--'} \\\\",
        f"Modality groups & {len(modality_rows) if modality_rows else '--'} \\\\",
        f"When baselines & {len(baseline_rows) if baseline_rows else '--'} \\\\",
        f"Sanity judge groups & {len(sanity_rows) if sanity_rows else '--'} \\\\",
    ]
    if not how_rows:
        completeness.mark("missing supplementary How@Gold rows")
    if not modality_rows:
        completeness.mark("missing supplementary modality rows")
    if not baseline_rows:
        completeness.mark("missing supplementary baseline rows")
    sanity_priority = {
        (judge, mode): index
        for index, (judge, mode) in enumerate(
            (
                ("primary-mean", "reference-aware"),
                ("primary-mean", "reference-free"),
                ("held-out-mean", "reference-aware"),
                ("held-out-mean", "reference-free"),
                ("gpt4o", "reference-aware"),
                ("gpt4o", "reference-free"),
                ("gemini_2_5_pro", "reference-aware"),
                ("gemini_2_5_pro", "reference-free"),
                ("qwen3_omni", "reference-aware"),
                ("qwen3_omni", "reference-free"),
            )
        )
    }
    selected_sanity = sorted(
        (
            row
            for row in sanity_rows
            if (row.get("judge", ""), row.get("reference_mode", "")) in sanity_priority
        ),
        key=lambda row: sanity_priority[(row.get("judge", ""), row.get("reference_mode", ""))],
    )
    sanity_body = []
    for row in selected_sanity:
        label = "/".join(part for part in (row.get("judge"), row.get("reference_mode")) if part)
        sanity_body.append(
            f"{tex_escape(label or 'TODO')} & {fmt_int(row.get('positive_n'), completeness, label + ' positive n')} & "
            f"{fmt_int(row.get('negative_n'), completeness, label + ' negative n')} & "
            f"{fmt(row.get('auc_gold_vs_shuffled'), completeness, label + ' gold/shuffled AUC')} & "
            f"{fmt(row.get('paired_margin_gold_vs_shuffled'), completeness, label + ' gold/shuffled margin')} \\\\"
        )
    if not sanity_body:
        sanity_body = [r"TODO & -- & -- & -- & -- \\"]
    write_table(
        path,
        [
            r"\begin{table}[t]",
            r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{1.8pt}",
            r"\caption{Supplementary final-extension inventory and selected sanity-control results. Detailed CSV/JSON snapshots remain the source of record.}",
            r"\label{tab:extension_supplementary_details}",
            r"\begin{tabular}{lr}\toprule",
            r"Snapshot family & Rows \\\midrule",
            *body,
            r"\bottomrule\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{tabular}{lrrrr}\toprule",
            r"Sanity setting & Pos. & Neg. & G/S AUC & G--S \\\midrule",
            *sanity_body,
            r"\bottomrule\end{tabular}",
            r"\end{table}",
        ],
    )


def render(snapshot_dir: Path, table_dir: Path, *, strict: bool = False) -> list[Path]:
    completeness = Completeness(strict=strict or complete_from_report(snapshot_dir))
    table_dir.mkdir(parents=True, exist_ok=True)
    how_rows = read_csv_optional(snapshot_dir / "how_gold_summary.csv", completeness)
    main_rows = read_csv_optional(snapshot_dir / "main_extension_summary.csv", completeness)
    modality_rows = read_csv_optional(snapshot_dir / "modality_metrics.csv", completeness)
    baseline_rows = read_csv_optional(snapshot_dir / "when_baselines_summary.csv", completeness)
    judge_aggs = read_csv_optional(snapshot_dir / "judge_audit_aggregations.csv", completeness)
    judge_corrs = read_csv_optional(snapshot_dir / "judge_audit_correlations.csv", completeness)
    judge_ranks = read_csv_optional(snapshot_dir / "judge_audit_rank_shifts.csv", completeness)
    sanity_rows = read_csv_optional(snapshot_dir / "sanity_summary.csv", completeness)

    outputs = [table_dir / name for name in TABLES.values()]
    write_main(main_rows, table_dir / TABLES["main"], completeness)
    write_how_gold(how_rows, table_dir / TABLES["how"], completeness)
    write_when_baselines(baseline_rows, table_dir / TABLES["when"], completeness)
    write_modality_compact(
        modality_rows,
        how_rows,
        table_dir / TABLES["modality"],
        completeness,
        display_name,
    )
    write_judge_compact(
        judge_corrs,
        judge_ranks,
        sanity_rows,
        table_dir / TABLES["judge"],
        completeness,
    )
    write_supplement(how_rows, modality_rows, baseline_rows, sanity_rows, table_dir / TABLES["supp"], completeness)
    completeness.require()
    return outputs


def main() -> None:
    args = parse_args()
    for output in render(args.snapshot_dir, args.table_dir, strict=args.strict):
        print(output)


if __name__ == "__main__":
    main()
