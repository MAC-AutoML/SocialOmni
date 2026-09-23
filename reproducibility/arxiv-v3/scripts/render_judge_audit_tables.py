#!/usr/bin/env python3
"""Render compact judge-distribution and pairwise-agreement LaTeX tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scipy.stats import kendalltau, pearsonr, spearmanr


JUDGES = (
    ("gpt4o", "GPT-4o"),
    ("gemini_2_5_pro", "Gemini 2.5 Pro"),
    ("qwen3_omni", "Qwen3-Omni"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.scores)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    distribution = []
    values: dict[str, list[float]] = {}
    for judge, display in JUDGES:
        scores = [float(row[f"score_{judge}"]) for row in rows]
        values[judge] = scores
        distribution.append(
            f"{display} & {len(scores)} & {sum(scores) / len(scores):.2f} & "
            f"{100 * scores.count(0) / len(scores):.2f} & "
            f"{100 * scores.count(100) / len(scores):.2f} \\\\"
        )
    (args.table_dir / "judge_distribution.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\small",
                rf"\caption{{Score distributions over all {len(rows)} eligible responses. Extreme columns report the percentage assigned 0 or 100.}}",
                r"\label{tab:judge_distribution}",
                r"\begin{tabular}{lrrrr}\toprule",
                r"Judge & $n$ & Mean & 0 (\%) & 100 (\%) \\\midrule",
                *distribution,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pairwise = []
    pearson_matrix = {(judge, judge): 1.0 for judge, _ in JUDGES}
    for index, (left, left_name) in enumerate(JUDGES):
        for right, right_name in JUDGES[index + 1 :]:
            x, y = values[left], values[right]
            pearson = float(pearsonr(x, y).statistic)
            pearson_matrix[left, right] = pearson
            pearson_matrix[right, left] = pearson
            pairwise.append(
                f"{left_name} / {right_name} & {pearson:.3f} & "
                f"{spearmanr(x, y).statistic:.3f} & {kendalltau(x, y).statistic:.3f} & "
                f"{sum(abs(a - b) for a, b in zip(x, y)) / len(x):.2f} \\\\"
            )
    (args.table_dir / "judge_pairwise.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{3pt}",
                r"\caption{Response-level pairwise agreement across the complete judge outputs.}",
                r"\label{tab:judge_pairwise}",
                r"\begin{tabular}{@{}lrrrr@{}}\toprule",
                r"Judge pair & Pearson & Spearman & Kendall & MAE \\\midrule",
                *pairwise,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    compact = []
    for judge, display in JUDGES:
        scores = values[judge]
        correlations = [pearson_matrix[judge, other] for other, _ in JUDGES]
        compact.append(
            f"{display} & {sum(scores) / len(scores):.2f} & "
            f"{100 * scores.count(0) / len(scores):.2f} & "
            f"{100 * scores.count(100) / len(scores):.2f} & "
            + " & ".join(f"{value:.3f}" for value in correlations)
            + r" \\\\"
        )

    (args.table_dir / "judge_audit_compact.tex").write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering\scriptsize",
                r"\setlength{\tabcolsep}{1.8pt}",
                rf"\caption{{Judge score distributions and response-level Pearson agreement over {len(rows)} eligible responses. Full Spearman, Kendall, and MAE results are reported in the supplementary material.}}",
                r"\label{tab:judge_audit_compact}",
                r"\begin{tabular}{@{}lrrrrrr@{}}\toprule",
                r"Judge & Mean & 0 (\%) & 100 (\%) & $r_G$ & $r_M$ & $r_Q$ \\\midrule",
                *compact,
                r"\bottomrule\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
