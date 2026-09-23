from __future__ import annotations
from collections import Counter
from typing import Any

def multiclass_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("A", "B", "C", "D")
    confusion = {gold: Counter() for gold in labels}
    invalid = failures = parse_failures = correct = 0
    for row in rows:
        gold = str(row.get("correct_answer") or "")
        pred = str(row.get("prediction") or "")
        system_failure = bool(row.get("error") or row.get("skip_reason") or not row.get("scored", True))
        failures += int(system_failure)
        if pred not in labels:
            invalid += 1
            parse_failures += int(not system_failure)
        else:
            confusion[gold][pred] += 1
        correct += int(gold == pred)
    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label) + sum(
            1 for row in rows if row.get("correct_answer") == label and row.get("prediction") not in labels
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": 100 * correct / len(rows),
        "macro_f1": 100 * sum(f1s) / len(f1s),
        "invalid": invalid,
        "failures": failures,
        "parse_failures": parse_failures,
    }

def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = invalid_yes = invalid_no = failures = parse_failures = 0
    for row in rows:
        gold, pred = row.get("q1_answer"), row.get("q1_prediction")
        system_failure = bool(row.get("error") or row.get("skip_reason"))
        failures += int(system_failure)
        if pred not in {"A", "B"}:
            invalid_yes += int(gold == "A")
            invalid_no += int(gold == "B")
            parse_failures += int(not system_failure)
        elif gold == "A" and pred == "A":
            tp += 1
        elif gold == "B" and pred == "A":
            fp += 1
        elif gold == "B" and pred == "B":
            tn += 1
        elif gold == "A" and pred == "B":
            fn += 1
    f1_yes_denominator = 2 * tp + fp + fn + invalid_yes
    f1_no_denominator = 2 * tn + fp + fn + invalid_no
    f1_yes = 2 * tp / f1_yes_denominator if f1_yes_denominator else 0.0
    f1_no = 2 * tn / f1_no_denominator if f1_no_denominator else 0.0
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "gold_yes": tp + fn + invalid_yes,
        "gold_no": tn + fp + invalid_no,
        "accuracy": 100 * (tp + tn) / len(rows),
        "precision_yes": 100 * tp / (tp + fp) if tp + fp else 0.0,
        "recall_yes": 100 * tp / (tp + fn + invalid_yes) if tp + fn + invalid_yes else 0.0,
        "f1_yes": 100 * f1_yes,
        "macro_f1": 50 * (f1_yes + f1_no),
        "invalid": invalid_yes + invalid_no,
        "failures": failures,
        "parse_failures": parse_failures,
    }
