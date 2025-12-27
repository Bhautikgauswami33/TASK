#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


def calculate_character_overlap(span1: str, span2: str) -> float:
    if not span1 or not span2:
        return 0.0
    
    s1 = span1.lower().strip()
    s2 = span2.lower().strip()
    
    if not s1 or not s2:
        return 0.0
    
    if s1 in s2 or s2 in s1:
        return 1.0
    
    lcs_length = _longest_common_substring_length(s1, s2)
    min_len = min(len(s1), len(s2))
    
    if min_len == 0:
        return 0.0
    
    return lcs_length / min_len


def _longest_common_substring_length(s1: str, s2: str) -> int:
    if not s1 or not s2:
        return 0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
    
    return max_length


def is_match(pred_item: Dict, gold_item: Dict, overlap_threshold: float = 0.5) -> bool:
    if pred_item.get("domain") != gold_item.get("domain"):
        return False
    
    pred_span = pred_item.get("evidence_span", "")
    gold_span = gold_item.get("evidence_span", "")
    overlap = calculate_character_overlap(pred_span, gold_span)
    
    return overlap > overlap_threshold


def normalize_bucket(item: Dict) -> Optional[str]:
    return item.get("intensity_bucket") or item.get("arousal_bucket")


def load_jsonl(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_journals(filepath: str) -> Dict[str, str]:
    journals = {}
    for entry in load_jsonl(filepath):
        journals[entry["journal_id"]] = entry["text"]
    return journals


def validate_evidence_span(evidence_span: str, journal_text: str) -> bool:
    if not evidence_span or not journal_text:
        return False
    return evidence_span in journal_text


def calculate_evidence_coverage(predictions: List[Dict], journals: Dict[str, str]) -> Tuple[int, int, float]:
    valid_count = 0
    total_count = 0
    
    for pred in predictions:
        journal_id = pred["journal_id"]
        journal_text = journals.get(journal_id, "")
        
        for item in pred.get("items", []):
            total_count += 1
            evidence_span = item.get("evidence_span", "")
            if validate_evidence_span(evidence_span, journal_text):
                valid_count += 1
    
    coverage_rate = valid_count / total_count if total_count > 0 else 0.0
    return valid_count, total_count, coverage_rate


def match_items(pred_items: List[Dict], gold_items: List[Dict], overlap_threshold: float = 0.5) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    matched_pairs = []
    used_gold_indices = set()
    unmatched_predictions = []
    
    for pred in pred_items:
        best_match_idx = None
        best_overlap = overlap_threshold
        
        for idx, gold in enumerate(gold_items):
            if idx in used_gold_indices:
                continue
            
            if pred.get("domain") != gold.get("domain"):
                continue
            
            overlap = calculate_character_overlap(pred.get("evidence_span", ""), gold.get("evidence_span", ""))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_idx = idx
        
        if best_match_idx is not None:
            matched_pairs.append((pred, gold_items[best_match_idx]))
            used_gold_indices.add(best_match_idx)
        else:
            unmatched_predictions.append(pred)
    
    unmatched_gold = [gold for idx, gold in enumerate(gold_items) if idx not in used_gold_indices]
    
    return matched_pairs, unmatched_predictions, unmatched_gold


def compute_attribute_accuracy(matched_pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    if not matched_pairs:
        return {"polarity_accuracy": 0.0, "bucket_accuracy": 0.0, "time_accuracy": 0.0}
    
    n = len(matched_pairs)
    polarity_correct = 0
    bucket_correct = 0
    time_correct = 0
    
    for pred, gold in matched_pairs:
        if pred.get("polarity") == gold.get("polarity"):
            polarity_correct += 1
        
        pred_bucket = normalize_bucket(pred)
        gold_bucket = normalize_bucket(gold)
        if pred_bucket == gold_bucket:
            bucket_correct += 1
        
        if pred.get("time_bucket") == gold.get("time_bucket"):
            time_correct += 1
    
    return {
        "polarity_accuracy": polarity_correct / n,
        "bucket_accuracy": bucket_correct / n,
        "time_accuracy": time_correct / n
    }


def evaluate_journal(pred: Dict, gold: Dict, overlap_threshold: float = 0.5) -> Dict:
    pred_items = pred.get("items", [])
    gold_items = gold.get("items", [])
    
    matched_pairs, unmatched_preds, unmatched_gold = match_items(pred_items, gold_items, overlap_threshold)
    
    tp = len(matched_pairs)
    fp = len(unmatched_preds)
    fn = len(unmatched_gold)
    
    attr_accuracy = compute_attribute_accuracy(matched_pairs)
    
    return {
        "journal_id": pred.get("journal_id"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pairs": [
            {
                "pred_span": p.get("evidence_span"),
                "gold_span": g.get("evidence_span"),
                "domain": p.get("domain"),
                "polarity_match": p.get("polarity") == g.get("polarity"),
                "bucket_match": normalize_bucket(p) == normalize_bucket(g),
                "time_match": p.get("time_bucket") == g.get("time_bucket")
            }
            for p, g in matched_pairs
        ],
        "false_positives": [{"evidence_span": p.get("evidence_span"), "domain": p.get("domain")} for p in unmatched_preds],
        "false_negatives": [{"evidence_span": g.get("evidence_span"), "domain": g.get("domain")} for g in unmatched_gold],
        **attr_accuracy
    }


def evaluate_all(predictions: List[Dict], golds: List[Dict], journals: Dict[str, str] = None, overlap_threshold: float = 0.5) -> Tuple[Dict, List[Dict]]:
    gold_by_id = {g["journal_id"]: g for g in golds}
    
    per_journal_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_polarity_correct = 0
    total_bucket_correct = 0
    total_time_correct = 0
    total_matched = 0
    
    for pred in predictions:
        journal_id = pred["journal_id"]
        gold = gold_by_id.get(journal_id, {"journal_id": journal_id, "items": []})
        
        result = evaluate_journal(pred, gold, overlap_threshold)
        per_journal_results.append(result)
        
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        
        if result["tp"] > 0:
            total_polarity_correct += result["polarity_accuracy"] * result["tp"]
            total_bucket_correct += result["bucket_accuracy"] * result["tp"]
            total_time_correct += result["time_accuracy"] * result["tp"]
            total_matched += result["tp"]
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    polarity_accuracy = total_polarity_correct / total_matched if total_matched > 0 else 0.0
    bucket_accuracy = total_bucket_correct / total_matched if total_matched > 0 else 0.0
    time_accuracy = total_time_correct / total_matched if total_matched > 0 else 0.0
    
    evidence_coverage = 0.0
    if journals:
        _, _, evidence_coverage = calculate_evidence_coverage(predictions, journals)
    
    summary = {
        "total_journals_evaluated": len(predictions),
        "object_precision": round(precision, 4),
        "object_recall": round(recall, 4),
        "object_f1": round(f1, 4),
        "polarity_accuracy": round(polarity_accuracy, 4),
        "bucket_accuracy": round(bucket_accuracy, 4),
        "time_bucket_accuracy": round(time_accuracy, 4),
        "evidence_coverage_rate": round(evidence_coverage, 4),
        "tp_count": total_tp,
        "fp_count": total_fp,
        "fn_count": total_fn,
        "overlap_threshold": overlap_threshold
    }
    
    return summary, per_journal_results


def main():
    parser = argparse.ArgumentParser(description="Ashwam Evidence-Grounded Extraction Evaluator")
    parser.add_argument("--gold", "-g", required=True, help="Path to gold.jsonl file")
    parser.add_argument("--pred", "-p", required=True, help="Path to predictions.jsonl file")
    parser.add_argument("--journals", "-j", required=False, help="Path to journals.jsonl for evidence coverage")
    parser.add_argument("--out", "-o", required=True, help="Output directory for results")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Overlap threshold (default: 0.5)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading gold from: {args.gold}")
    golds = load_jsonl(args.gold)
    print(f"  Loaded {len(golds)} gold entries")
    
    print(f"Loading predictions from: {args.pred}")
    predictions = load_jsonl(args.pred)
    print(f"  Loaded {len(predictions)} prediction entries")
    
    journals = None
    if args.journals:
        print(f"Loading journals from: {args.journals}")
        journals = load_journals(args.journals)
        print(f"  Loaded {len(journals)} journals")
    
    print(f"\nEvaluating with overlap threshold: {args.threshold}")
    summary, per_journal = evaluate_all(predictions, golds, journals, args.threshold)
    
    summary_path = out_dir / "score_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote summary to: {summary_path}")
    
    per_journal_path = out_dir / "per_journal_scores.jsonl"
    with open(per_journal_path, 'w', encoding='utf-8') as f:
        for result in per_journal:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote per-journal results to: {per_journal_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Precision:           {summary['object_precision']:.2%}")
    print(f"  Recall:              {summary['object_recall']:.2%}")
    print(f"  F1 Score:            {summary['object_f1']:.2%}")
    print(f"  Polarity Accuracy:   {summary['polarity_accuracy']:.2%}")
    print(f"  Bucket Accuracy:     {summary['bucket_accuracy']:.2%}")
    print(f"  Time Bucket Acc:     {summary['time_bucket_accuracy']:.2%}")
    if journals:
        print(f"  Evidence Coverage:   {summary['evidence_coverage_rate']:.2%}")
    print(f"\n  TP: {summary['tp_count']}  FP: {summary['fp_count']}  FN: {summary['fn_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
