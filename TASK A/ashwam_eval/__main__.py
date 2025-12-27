#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator import load_jsonl, load_journals, evaluate_all


def run_command(args):
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    
    gold_path = data_dir / "gold.jsonl"
    journals_path = data_dir / "journals.jsonl"
    
    pred_path = data_dir / "predictions.jsonl"
    if not pred_path.exists():
        pred_path = data_dir / "sample_predictions.jsonl"
    
    if not gold_path.exists():
        print(f"Error: gold.jsonl not found in {data_dir}")
        sys.exit(1)
    
    if not pred_path.exists():
        print(f"Error: No predictions file found in {data_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading gold from: {gold_path}")
    golds = load_jsonl(str(gold_path))
    print(f"  Loaded {len(golds)} gold entries")
    
    print(f"Loading predictions from: {pred_path}")
    predictions = load_jsonl(str(pred_path))
    print(f"  Loaded {len(predictions)} prediction entries")
    
    journals = None
    if journals_path.exists():
        print(f"Loading journals from: {journals_path}")
        journals = load_journals(str(journals_path))
        print(f"  Loaded {len(journals)} journals")
    
    print(f"\nEvaluating with overlap threshold: 0.5")
    summary, per_journal = evaluate_all(predictions, golds, journals, 0.5)
    
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


def main():
    parser = argparse.ArgumentParser(prog="ashwam_eval", description="Ashwam Evidence-Grounded Extraction & Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    run_parser = subparsers.add_parser("run", help="Run the evaluation pipeline")
    run_parser.add_argument("--data", required=True, help="Path to data directory")
    run_parser.add_argument("--out", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
