#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter


VALID_DOMAINS = {'symptom', 'food', 'emotion', 'mind'}
VALID_POLARITIES = {'present', 'absent', 'uncertain'}
VALID_BUCKETS = {'low', 'medium', 'high', 'unknown'}
VALID_TIME_BUCKETS = {'today', 'last_night', 'past_week', 'unknown'}


def check_attribute_validity(item: Dict) -> List[str]:
    errors = []
    
    domain = item.get('domain')
    if domain not in VALID_DOMAINS:
        errors.append(f"Invalid domain: '{domain}'")
    
    polarity = item.get('polarity')
    if polarity not in VALID_POLARITIES:
        errors.append(f"Invalid polarity: '{polarity}'")
    
    intensity = item.get('intensity_bucket')
    arousal = item.get('arousal_bucket')
    time = item.get('time_bucket')
    
    if intensity and intensity not in VALID_BUCKETS:
        errors.append(f"Invalid intensity_bucket: '{intensity}'")
    
    if arousal and arousal not in VALID_BUCKETS:
        errors.append(f"Invalid arousal_bucket: '{arousal}'")
    
    if time and time not in VALID_TIME_BUCKETS:
        errors.append(f"Invalid time_bucket: '{time}'")
    
    return errors


def check_evidence_validity(item: Dict, journal_text: str) -> List[str]:
    errors = []
    
    evidence = item.get('evidence_span', '')
    if not evidence:
        errors.append("Missing evidence_span")
    elif evidence not in journal_text:
        errors.append(f"Hallucinated evidence_span: '{evidence[:50]}...' not in journal")
    
    return errors


def check_required_fields(item: Dict) -> List[str]:
    errors = []
    required = ['domain', 'evidence_span', 'polarity']
    
    for field in required:
        if not item.get(field):
            errors.append(f"Missing required field: '{field}'")
    
    return errors


def check_contradictions(items: List[Dict]) -> List[Dict]:
    contradictions = []
    by_evidence = defaultdict(list)
    
    for item in items:
        evidence = item.get('evidence_span', '')
        if evidence:
            by_evidence[evidence].append(item)
    
    for evidence, grouped_items in by_evidence.items():
        polarities = set(item.get('polarity') for item in grouped_items)
        
        if 'present' in polarities and 'absent' in polarities:
            contradictions.append({
                'evidence_span': evidence,
                'polarities': list(polarities),
                'items': grouped_items
            })
    
    return contradictions


def run_invariant_checks(parser_outputs: List[Dict], journals: Dict[str, str]) -> Dict:
    total_items = 0
    attribute_failures = []
    evidence_failures = []
    field_failures = []
    contradictions = []
    
    for output in parser_outputs:
        journal_id = output.get('journal_id')
        journal_text = journals.get(journal_id, '')
        items = output.get('items', [])
        
        journal_contradictions = check_contradictions(items)
        for c in journal_contradictions:
            contradictions.append({**c, 'journal_id': journal_id})
        
        for item in items:
            total_items += 1
            
            attr_errors = check_attribute_validity(item)
            for err in attr_errors:
                attribute_failures.append({'journal_id': journal_id, 'error': err, 'item': item})
            
            evidence_errors = check_evidence_validity(item, journal_text)
            for err in evidence_errors:
                evidence_failures.append({'journal_id': journal_id, 'error': err, 'item': item})
            
            field_errors = check_required_fields(item)
            for err in field_errors:
                field_failures.append({'journal_id': journal_id, 'error': err, 'item': item})
    
    total_failures = len(attribute_failures) + len(evidence_failures) + len(field_failures) + len(contradictions)
    
    return {
        'total_items_checked': total_items,
        'total_failures': total_failures,
        'failure_rate': round(total_failures / total_items, 4) if total_items > 0 else 0,
        'checks': {
            'attribute_validity': {'failures': len(attribute_failures), 'details': attribute_failures[:10]},
            'evidence_validity': {'failures': len(evidence_failures), 'details': evidence_failures[:10]},
            'required_fields': {'failures': len(field_failures), 'details': field_failures[:10]},
            'contradictions': {'failures': len(contradictions), 'details': contradictions}
        },
        'passed': total_failures == 0
    }


def compute_distribution(outputs: List[Dict], key_func) -> Dict[str, float]:
    counts = Counter()
    total = 0
    
    for output in outputs:
        for item in output.get('items', []):
            value = key_func(item)
            if value is not None:
                counts[value] += 1
                total += 1
    
    return {k: round(v / total, 4) if total > 0 else 0 for k, v in counts.items()}


def compute_items_per_journal(outputs: List[Dict]) -> Dict[str, float]:
    counts = [len(output.get('items', [])) for output in outputs]
    
    if not counts:
        return {'mean': 0, 'min': 0, 'max': 0, 'total': 0}
    
    return {'mean': round(sum(counts) / len(counts), 2), 'min': min(counts), 'max': max(counts), 'total': sum(counts)}


def compute_drift_metrics(day0_outputs: List[Dict], day1_outputs: List[Dict]) -> Dict:
    day0_items = compute_items_per_journal(day0_outputs)
    day1_items = compute_items_per_journal(day1_outputs)
    
    day0_domains = compute_distribution(day0_outputs, lambda x: x.get('domain'))
    day1_domains = compute_distribution(day1_outputs, lambda x: x.get('domain'))
    
    day0_polarity = compute_distribution(day0_outputs, lambda x: x.get('polarity'))
    day1_polarity = compute_distribution(day1_outputs, lambda x: x.get('polarity'))
    
    day0_arousal = compute_distribution(day0_outputs, lambda x: x.get('arousal_bucket') if x.get('domain') == 'emotion' else None)
    day1_arousal = compute_distribution(day1_outputs, lambda x: x.get('arousal_bucket') if x.get('domain') == 'emotion' else None)
    
    day0_time = compute_distribution(day0_outputs, lambda x: x.get('time_bucket'))
    day1_time = compute_distribution(day1_outputs, lambda x: x.get('time_bucket'))
    
    def distribution_shift(d0: Dict, d1: Dict) -> float:
        all_keys = set(d0.keys()) | set(d1.keys())
        return sum(abs(d0.get(k, 0) - d1.get(k, 0)) for k in all_keys) / 2
    
    return {
        'items_per_journal': {'day0': day0_items, 'day1': day1_items, 'mean_change': round(day1_items['mean'] - day0_items['mean'], 2)},
        'domain_distribution': {'day0': day0_domains, 'day1': day1_domains, 'shift': round(distribution_shift(day0_domains, day1_domains), 4)},
        'polarity_distribution': {'day0': day0_polarity, 'day1': day1_polarity, 'shift': round(distribution_shift(day0_polarity, day1_polarity), 4)},
        'emotion_arousal_distribution': {'day0': day0_arousal, 'day1': day1_arousal, 'shift': round(distribution_shift(day0_arousal, day1_arousal), 4)},
        'time_bucket_distribution': {'day0': day0_time, 'day1': day1_time, 'shift': round(distribution_shift(day0_time, day1_time), 4)},
        'alerts': []
    }


def add_drift_alerts(drift_report: Dict, thresholds: Dict = None) -> None:
    if thresholds is None:
        thresholds = {'domain_shift': 0.1, 'polarity_shift': 0.1, 'arousal_shift': 0.15, 'items_mean_change': 0.5}
    
    alerts = []
    
    if drift_report['domain_distribution']['shift'] > thresholds['domain_shift']:
        alerts.append({
            'type': 'DOMAIN_DRIFT', 'severity': 'MEDIUM',
            'message': f"Domain distribution shifted by {drift_report['domain_distribution']['shift']:.2%}",
            'day0': drift_report['domain_distribution']['day0'],
            'day1': drift_report['domain_distribution']['day1']
        })
    
    if drift_report['polarity_distribution']['shift'] > thresholds['polarity_shift']:
        alerts.append({'type': 'POLARITY_DRIFT', 'severity': 'HIGH', 'message': f"Polarity distribution shifted by {drift_report['polarity_distribution']['shift']:.2%}"})
    
    if drift_report['emotion_arousal_distribution']['shift'] > thresholds['arousal_shift']:
        alerts.append({'type': 'AROUSAL_DRIFT', 'severity': 'MEDIUM', 'message': f"Emotion arousal distribution shifted by {drift_report['emotion_arousal_distribution']['shift']:.2%}"})
    
    items_change = abs(drift_report['items_per_journal']['mean_change'])
    if items_change > thresholds['items_mean_change']:
        alerts.append({'type': 'EXTRACTION_RATE_CHANGE', 'severity': 'LOW', 'message': f"Mean items per journal changed by {items_change:.2f}"})
    
    drift_report['alerts'] = alerts


def calculate_evidence_overlap(span1: str, span2: str) -> float:
    if not span1 or not span2:
        return 0.0
    
    s1 = span1.lower().strip()
    s2 = span2.lower().strip()
    
    if s1 in s2 or s2 in s1:
        return 1.0
    
    chars1 = set(s1)
    chars2 = set(s2)
    intersection = len(chars1 & chars2)
    union = len(chars1 | chars2)
    
    return intersection / union if union > 0 else 0.0


def evaluate_canary(predictions: List[Dict], gold: List[Dict]) -> Dict:
    gold_by_id = {g['journal_id']: g for g in gold}
    pred_by_id = {p['journal_id']: p for p in predictions}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    polarity_correct = 0
    per_journal = []
    
    for journal_id, gold_entry in gold_by_id.items():
        pred_entry = pred_by_id.get(journal_id, {'items': []})
        
        gold_items = gold_entry.get('items', [])
        pred_items = pred_entry.get('items', [])
        
        used_gold = set()
        tp = 0
        pol_correct = 0
        
        for pred in pred_items:
            matched = False
            for i, g in enumerate(gold_items):
                if i in used_gold:
                    continue
                if pred.get('domain') != g.get('domain'):
                    continue
                
                overlap = calculate_evidence_overlap(pred.get('evidence_span', ''), g.get('evidence_span', ''))
                
                if overlap > 0.5:
                    matched = True
                    used_gold.add(i)
                    tp += 1
                    if pred.get('polarity') == g.get('polarity'):
                        pol_correct += 1
                    break
        
        fp = len(pred_items) - tp
        fn = len(gold_items) - tp
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        polarity_correct += pol_correct
        
        per_journal.append({'journal_id': journal_id, 'tp': tp, 'fp': fp, 'fn': fn, 'polarity_correct': pol_correct})
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    polarity_acc = polarity_correct / total_tp if total_tp > 0 else 0
    
    return {
        'total_journals': len(gold_by_id),
        'metrics': {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), 'polarity_accuracy': round(polarity_acc, 4)},
        'counts': {'tp': total_tp, 'fp': total_fp, 'fn': total_fn},
        'per_journal': per_journal,
        'alert': {'triggered': f1 < 0.7, 'message': 'Canary F1 below 70% threshold!' if f1 < 0.7 else 'Canary performance acceptable'}
    }


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
        journals[entry['journal_id']] = entry['text']
    return journals


def main():
    parser = argparse.ArgumentParser(description="Ashwam Production Monitoring Without Ground Truth")
    subparsers = parser.add_subparsers(dest='command')
    
    run_parser = subparsers.add_parser('run', help='Run monitoring pipeline')
    run_parser.add_argument('--data', required=True, help='Path to data directory')
    run_parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.command != 'run':
        parser.print_help()
        return
    
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    journals = load_journals(str(data_dir / 'journals.jsonl'))
    print(f"  Loaded {len(journals)} journals")
    
    day0_outputs = load_jsonl(str(data_dir / 'parser_outputs_day0.jsonl'))
    day1_outputs = load_jsonl(str(data_dir / 'parser_outputs_day1.jsonl'))
    print(f"  Loaded Day0: {len(day0_outputs)} outputs, Day1: {len(day1_outputs)} outputs")
    
    canary_dir = data_dir / 'canary'
    canary_gold = load_jsonl(str(canary_dir / 'gold.jsonl'))
    print(f"  Loaded canary gold: {len(canary_gold)} entries")
    
    print("\nRunning invariant checks on Day1...")
    invariant_report = run_invariant_checks(day1_outputs, journals)
    
    invariant_path = out_dir / 'invariant_report.json'
    with open(invariant_path, 'w', encoding='utf-8') as f:
        json.dump(invariant_report, f, indent=2, ensure_ascii=False)
    print(f"  Wrote invariant report to: {invariant_path}")
    
    print("\nComputing drift metrics (Day0 vs Day1)...")
    drift_report = compute_drift_metrics(day0_outputs, day1_outputs)
    add_drift_alerts(drift_report)
    
    drift_path = out_dir / 'drift_report.json'
    with open(drift_path, 'w', encoding='utf-8') as f:
        json.dump(drift_report, f, indent=2, ensure_ascii=False)
    print(f"  Wrote drift report to: {drift_path}")
    
    print("\nEvaluating canary set...")
    canary_report = evaluate_canary(day1_outputs, canary_gold)
    
    canary_path = out_dir / 'canary_report.json'
    with open(canary_path, 'w', encoding='utf-8') as f:
        json.dump(canary_report, f, indent=2, ensure_ascii=False)
    print(f"  Wrote canary report to: {canary_path}")
    
    print("\n" + "=" * 60)
    print("PRODUCTION MONITORING SUMMARY")
    print("=" * 60)
    
    print("\n[INVARIANT CHECKS]")
    print(f"  Total Items Checked:    {invariant_report['total_items_checked']}")
    print(f"  Total Failures:         {invariant_report['total_failures']}")
    print(f"  Failure Rate:           {invariant_report['failure_rate']:.2%}")
    print(f"  - Attribute Validity:   {invariant_report['checks']['attribute_validity']['failures']}")
    print(f"  - Evidence Validity:    {invariant_report['checks']['evidence_validity']['failures']}")
    print(f"  - Required Fields:      {invariant_report['checks']['required_fields']['failures']}")
    print(f"  - Contradictions:       {invariant_report['checks']['contradictions']['failures']}")
    
    print("\n[DRIFT METRICS]")
    print(f"  Domain Shift:           {drift_report['domain_distribution']['shift']:.2%}")
    print(f"  Polarity Shift:         {drift_report['polarity_distribution']['shift']:.2%}")
    print(f"  Arousal Shift:          {drift_report['emotion_arousal_distribution']['shift']:.2%}")
    print(f"  Items/Journal Change:   {drift_report['items_per_journal']['mean_change']:+.2f}")
    print(f"  Alerts Triggered:       {len(drift_report['alerts'])}")
    
    print("\n[CANARY EVALUATION]")
    print(f"  Precision:              {canary_report['metrics']['precision']:.2%}")
    print(f"  Recall:                 {canary_report['metrics']['recall']:.2%}")
    print(f"  F1 Score:               {canary_report['metrics']['f1']:.2%}")
    print(f"  Polarity Accuracy:      {canary_report['metrics']['polarity_accuracy']:.2%}")
    print(f"  Alert:                  {canary_report['alert']['message']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
