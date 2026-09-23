#!/usr/bin/env python3
"""Analyze three-rater scores against the current canonical candidates."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

SCORES = {0, 25, 50, 75, 100}
RATERS = ("human1_score", "human2_score", "human3_score")
DISPLAY = {
    "baichuan_omni_1_5": "Baichuan-Omni-1.5",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "gemini_2_5_pro": "Gemini 2.5 Pro",
    "gemini_3_flash_preview": "Gemini 3 Flash",
    "gemini_3_pro_preview": "Gemini 3 Pro",
    "omnivinci": "OmniVinci",
    "qwen2_5_omni": "Qwen2.5-Omni",
    "qwen3_omni": "Qwen3-Omni",
    "qwen3_omni_thinking": "Qwen3-Omni-Thinking",
    "vita_1_5": "VITA-1.5",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlations(x: list[float], y: list[float]) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
        "mae": float(np.mean(np.abs(np.asarray(x) - np.asarray(y)))),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def bootstrap(
    rows: list[dict[str, object]], iterations: int, seed: int
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    estimates: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        stats = correlations(
            [float(row["q_ensemble"]) for row in sample],
            [float(row["human_mean"]) for row in sample],
        )
        for name, value in stats.items():
            if value == value:
                estimates[name].append(value)
    return {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in estimates.items()
    }


def icc_two_way_absolute(scores: np.ndarray) -> dict[str, float]:
    n, k = scores.shape
    grand = float(scores.mean())
    row_means = scores.mean(axis=1)
    column_means = scores.mean(axis=0)
    ms_rows = k * float(np.square(row_means - grand).sum()) / (n - 1)
    ms_columns = n * float(np.square(column_means - grand).sum()) / (k - 1)
    residual = scores - row_means[:, None] - column_means[None, :] + grand
    ms_error = float(np.square(residual).sum()) / ((n - 1) * (k - 1))
    icc_single = (ms_rows - ms_error) / (
        ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    )
    icc_average = (ms_rows - ms_error) / (
        ms_rows + (ms_columns - ms_error) / n
    )
    return {"icc_2_1": icc_single, "icc_2_k": icc_average}


def quality_tier(value: float) -> str:
    if value >= 50:
        return "strong"
    if value >= 25:
        return "moderate"
    return "poor"


def write_latex_table(
    path: Path,
    model_rows: list[dict[str, object]],
    response_stats: dict[str, object],
    model_stats: dict[str, object],
    rater_stats: dict[str, object],
) -> None:
    response_ci = response_stats["bootstrap_95_ci"]
    model_body = []
    for row in model_rows:
        model_body.append(
            f"{DISPLAY[str(row['model'])]} & {int(row['n'])} & "
            f"{float(row['human_mean']):.2f} & {float(row['q_ensemble_mean']):.2f} & "
            f"{str(row['human_tier']).title()}/{str(row['llm_tier']).title()} \\\\"
        )
    lines = [
        r"\begin{table}[!t]",
        r"\footnotesize\centering",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\caption{Blind three-annotator calibration on 200 final responses from ten systems. Annotator and LLM values are three-rater and three-judge means; tiers are Poor ($<25$), Moderate ($25$--$49.99$), and Strong ($\geq50$). GPT-4o is absent from this fixed subset.}",
        r"\label{tab:human_feedback}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr}\toprule",
        r"Statistic & Estimate & 95\% CI \\\midrule",
        f"Response Pearson & {float(response_stats['pearson']):.3f} & [{response_ci['pearson'][0]:.3f}, {response_ci['pearson'][1]:.3f}] \\\\ ",
        f"Response Spearman & {float(response_stats['spearman']):.3f} & [{response_ci['spearman'][0]:.3f}, {response_ci['spearman'][1]:.3f}] \\\\ ",
        f"Model Pearson & {float(model_stats['pearson']):.3f} & -- \\\\ ",
        f"Tier agreement & {100 * float(model_stats['tier_agreement']):.1f}\\% & -- \\\\ ",
        f"ICC(2,$k$) & {float(rater_stats['icc_2_k']):.3f} & -- \\\\ ",
        f"Within 25 points & {100 * float(rater_stats['within_25_fraction']):.1f}\\% & -- \\\\ ",
        r"\bottomrule\end{tabular*}",
        r"\par\vspace{4pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrl}\toprule",
        r"Model & $n$ & Annotator & LLM & A/LLM tier \\\midrule",
        *model_body,
        r"\bottomrule\end{tabular*}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--legacy-key", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270727)
    args = parser.parse_args()

    ratings = read_csv(args.ratings)
    key_rows = read_csv(args.legacy_key)
    canonical_rows = read_csv(args.canonical)
    key = {row["calibration_id"]: row for row in key_rows}
    current = {(row["model"], row["item_id"]): row for row in canonical_rows}
    if len(ratings) != 200 or set(key) != {row["calibration_id"] for row in ratings}:
        raise SystemExit("Ratings and key must contain the same 200 calibration IDs")

    merged: list[dict[str, object]] = []
    for row in ratings:
        metadata = key[row["calibration_id"]]
        candidate = current[(metadata["model"], metadata["item_id"])]
        scores = [int(row[name]) for name in RATERS]
        if any(score not in SCORES for score in scores):
            raise SystemExit(f"Invalid score support for {row['calibration_id']}")
        merged.append(
            {
                "calibration_id": row["calibration_id"],
                "model": metadata["model"],
                "item_id": metadata["item_id"],
                "candidate_hash": candidate["candidate_hash"],
                "q_ensemble": float(candidate["q_ensemble"]),
                "human1_score": scores[0],
                "human2_score": scores[1],
                "human3_score": scores[2],
                "human_mean": sum(scores) / len(scores),
            }
        )

    response_stats = correlations(
        [float(row["q_ensemble"]) for row in merged],
        [float(row["human_mean"]) for row in merged],
    )
    response_stats["bootstrap_95_ci"] = bootstrap(
        merged, args.bootstrap, args.seed
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in merged:
        grouped[str(row["model"])].append(row)
    model_rows: list[dict[str, object]] = []
    for model, rows in grouped.items():
        llm = float(np.mean([float(row["q_ensemble"]) for row in rows]))
        human = float(np.mean([float(row["human_mean"]) for row in rows]))
        model_rows.append(
            {"model": model, "n": len(rows), "q_ensemble_mean": llm,
             "human_mean": human, "difference": llm - human}
        )
    human_ranks = rankdata([-float(row["human_mean"]) for row in model_rows], method="min")
    llm_ranks = rankdata([-float(row["q_ensemble_mean"]) for row in model_rows], method="min")
    for row, human_rank, llm_rank in zip(model_rows, human_ranks, llm_ranks):
        row["human_rank"] = int(human_rank)
        row["llm_rank"] = int(llm_rank)
        row["rank_shift"] = abs(int(human_rank) - int(llm_rank))
        row["human_tier"] = quality_tier(float(row["human_mean"]))
        row["llm_tier"] = quality_tier(float(row["q_ensemble_mean"]))
    model_rows.sort(key=lambda row: int(row["human_rank"]))
    model_stats = correlations(
        [float(row["q_ensemble_mean"]) for row in model_rows],
        [float(row["human_mean"]) for row in model_rows],
    )
    model_stats["max_rank_shift"] = max(int(row["rank_shift"]) for row in model_rows)
    tier_labels = ("poor", "moderate", "strong")
    tier_agreement = float(
        np.mean([row["human_tier"] == row["llm_tier"] for row in model_rows])
    )
    human_tier_rate = {
        label: np.mean([row["human_tier"] == label for row in model_rows])
        for label in tier_labels
    }
    llm_tier_rate = {
        label: np.mean([row["llm_tier"] == label for row in model_rows])
        for label in tier_labels
    }
    expected_tier_agreement = sum(
        human_tier_rate[label] * llm_tier_rate[label] for label in tier_labels
    )
    model_stats["tier_thresholds"] = {"poor": "<25", "moderate": "25-49.99", "strong": ">=50"}
    model_stats["tier_agreement"] = tier_agreement
    model_stats["tier_kappa"] = (
        (tier_agreement - expected_tier_agreement) / (1 - expected_tier_agreement)
    )

    matrix = np.asarray([[float(row[name]) for name in RATERS] for row in ratings])
    rater_stats: dict[str, object] = icc_two_way_absolute(matrix)
    rater_stats["unanimous_fraction"] = float(np.mean(np.ptp(matrix, axis=1) == 0))
    rater_stats["within_25_fraction"] = float(np.mean(np.ptp(matrix, axis=1) <= 25))
    pairwise = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pairwise[f"human{left + 1}_human{right + 1}"] = correlations(
            matrix[:, left].tolist(), matrix[:, right].tolist()
        )
    rater_stats["pairwise"] = pairwise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "human_calibration_scored.csv", merged)
    write_csv(args.output_dir / "human_calibration_model_summary.csv", model_rows)
    report = {
        "n": len(merged), "models": len(model_rows),
        "mapping_contract": "scores joined by calibration_id to current model/item candidate",
        "response_level": response_stats, "model_level": model_stats,
        "human_rater_agreement": rater_stats,
    }
    (args.output_dir / "human_calibration_statistics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_latex_table(
        args.output_dir / "human_feedback_three_rater.tex",
        model_rows,
        response_stats,
        model_stats,
        rater_stats,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
