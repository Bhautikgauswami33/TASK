#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict


def load_json(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(filepath: str) -> List[Dict]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_all_runs(llm_runs_dir: str) -> Dict[str, List[Dict]]:
    runs_by_journal = defaultdict(list)
    
    for filename in os.listdir(llm_runs_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(llm_runs_dir, filename)
            run_data = load_json(filepath)
            journal_id = run_data.get('journal_id')
            if journal_id:
                runs_by_journal[journal_id].append(run_data)
    
    for journal_id in runs_by_journal:
        runs_by_journal[journal_id].sort(key=lambda x: x.get('run_id', ''))
    
    return dict(runs_by_journal)


def calculate_evidence_overlap(span1: str, span2: str) -> float:
    if not span1 or not span2:
        return 0.0
    
    s1 = span1.lower().strip()
    s2 = span2.lower().strip()
    
    if s1 in s2 or s2 in s1:
        return 1.0
    
    lcs = _lcs_length(s1, s2)
    min_len = min(len(s1), len(s2))
    
    return lcs / min_len if min_len > 0 else 0.0


def _lcs_length(s1: str, s2: str) -> int:
    if not s1 or not s2:
        return 0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
    
    return max_len


def match_items_across_runs(runs: List[Dict], overlap_threshold: float = 0.5) -> Dict[str, List[Dict]]:
    matched_groups = defaultdict(list)
    
    for run in runs:
        run_id = run.get('run_id', 'unknown')
        for item in run.get('items', []):
            evidence = item.get('evidence_span', '')
            domain = item.get('domain', '')
            
            if not evidence:
                continue
            
            matched = False
            for key in list(matched_groups.keys()):
                existing_items = matched_groups[key]
                if not existing_items:
                    continue
                
                sample = existing_items[0]
                if sample.get('domain') != domain:
                    continue
                
                sample_evidence = sample.get('evidence_span', '')
                overlap = calculate_evidence_overlap(evidence, sample_evidence)
                
                if overlap > overlap_threshold:
                    matched_groups[key].append({**item, '_run_id': run_id})
                    matched = True
                    break
            
            if not matched:
                key = f"{domain}::{evidence}"
                matched_groups[key].append({**item, '_run_id': run_id})
    
    return dict(matched_groups)


def normalize_bucket(item: Dict) -> Optional[str]:
    return item.get('intensity_bucket') or item.get('arousal_bucket')


def compute_agreement_rate(matched_groups: Dict[str, List[Dict]], num_runs: int) -> float:
    if not matched_groups:
        return 0.0
    
    total_items = len(matched_groups)
    agreed_items = sum(1 for items in matched_groups.values() if len(items) >= num_runs)
    
    return agreed_items / total_items if total_items > 0 else 0.0


def compute_polarity_flip_rate(matched_groups: Dict[str, List[Dict]]) -> Tuple[float, List[Dict]]:
    flips = []
    total_groups = 0
    
    for key, items in matched_groups.items():
        if len(items) < 2:
            continue
        
        total_groups += 1
        polarities = set(item.get('polarity') for item in items)
        
        if len(polarities) > 1:
            flips.append({
                'evidence_span': items[0].get('evidence_span'),
                'domain': items[0].get('domain'),
                'polarities_seen': list(polarities),
                'runs': [item.get('_run_id') for item in items]
            })
    
    flip_rate = len(flips) / total_groups if total_groups > 0 else 0.0
    return flip_rate, flips


def compute_bucket_drift_rate(matched_groups: Dict[str, List[Dict]]) -> Tuple[float, List[Dict]]:
    drifts = []
    total_groups = 0
    
    for key, items in matched_groups.items():
        if len(items) < 2:
            continue
        
        total_groups += 1
        buckets = set(normalize_bucket(item) for item in items)
        time_buckets = set(item.get('time_bucket') for item in items)
        
        has_drift = len(buckets) > 1 or len(time_buckets) > 1
        
        if has_drift:
            drifts.append({
                'evidence_span': items[0].get('evidence_span'),
                'domain': items[0].get('domain'),
                'intensity_buckets_seen': list(buckets),
                'time_buckets_seen': list(time_buckets),
                'runs': [item.get('_run_id') for item in items]
            })
    
    drift_rate = len(drifts) / total_groups if total_groups > 0 else 0.0
    return drift_rate, drifts


def compute_recall_variance(matched_groups: Dict[str, List[Dict]], num_runs: int) -> float:
    if not matched_groups:
        return 0.0
    
    partial_items = sum(1 for items in matched_groups.values() if len(items) < num_runs)
    return partial_items / len(matched_groups)


def analyze_risk(polarity_flips: List[Dict], bucket_drifts: List[Dict]) -> Dict:
    risk_report = {'critical_risks': [], 'high_risks': [], 'medium_risks': [], 'low_risks': []}
    
    for flip in polarity_flips:
        domain = flip.get('domain')
        polarities = flip.get('polarities_seen', [])
        
        if domain == 'symptom' and 'present' in polarities and 'absent' in polarities:
            risk_report['critical_risks'].append({
                **flip,
                'risk_reason': 'Symptom polarity flip (present/absent) can mislead health decisions',
                'recommendation': 'Require human review or consensus voting'
            })
        elif domain == 'emotion' and 'present' in polarities and 'absent' in polarities:
            risk_report['high_risks'].append({
                **flip,
                'risk_reason': 'Emotion polarity flip may misrepresent mental state',
                'recommendation': 'Flag for review'
            })
        else:
            risk_report['medium_risks'].append({
                **flip,
                'risk_reason': 'Polarity variance on non-critical domain',
                'recommendation': 'Log for monitoring'
            })
    
    for drift in bucket_drifts:
        if len(drift.get('intensity_buckets_seen', [])) <= 1:
            risk_report['low_risks'].append({
                **drift,
                'risk_reason': 'Only time bucket drift detected',
                'recommendation': 'Acceptable variance'
            })
        else:
            risk_report['medium_risks'].append({
                **drift,
                'risk_reason': 'Intensity/arousal bucket variance',
                'recommendation': 'Monitor for patterns'
            })
    
    return risk_report


def compute_stable_output(matched_groups: Dict[str, List[Dict]], num_runs: int) -> List[Dict]:
    stable_items = []
    majority_threshold = num_runs / 2
    
    for key, items in matched_groups.items():
        if len(items) < majority_threshold:
            continue
        
        polarities = [item.get('polarity') for item in items]
        polarity_counts = defaultdict(int)
        for p in polarities:
            polarity_counts[p] += 1
        
        max_polarity_count = max(polarity_counts.values())
        majority_polarities = [p for p, c in polarity_counts.items() if c == max_polarity_count]
        final_polarity = 'uncertain' if len(majority_polarities) > 1 else majority_polarities[0]
        
        buckets = [normalize_bucket(item) for item in items]
        bucket_counts = defaultdict(int)
        for b in buckets:
            bucket_counts[b] += 1
        
        max_bucket_count = max(bucket_counts.values())
        majority_buckets = [b for b, c in bucket_counts.items() if c == max_bucket_count]
        final_bucket = 'unknown' if len(majority_buckets) > 1 else majority_buckets[0]
        
        longest_span = max(items, key=lambda x: len(x.get('evidence_span', '')))
        
        stable_item = {
            'domain': items[0].get('domain'),
            'evidence_span': longest_span.get('evidence_span'),
            'polarity': final_polarity,
            'bucket': final_bucket,
            'time_bucket': items[0].get('time_bucket'),
            'agreement_count': len(items),
            'total_runs': num_runs
        }
        
        if items[0].get('domain') == 'emotion':
            stable_item['arousal_bucket'] = final_bucket
        else:
            stable_item['intensity_bucket'] = final_bucket
        
        stable_items.append(stable_item)
    
    return stable_items


def analyze_journal_stability(journal_id: str, runs: List[Dict]) -> Dict:
    num_runs = len(runs)
    matched_groups = match_items_across_runs(runs)
    
    agreement_rate = compute_agreement_rate(matched_groups, num_runs)
    polarity_flip_rate, polarity_flips = compute_polarity_flip_rate(matched_groups)
    bucket_drift_rate, bucket_drifts = compute_bucket_drift_rate(matched_groups)
    recall_variance = compute_recall_variance(matched_groups, num_runs)
    risk_report = analyze_risk(polarity_flips, bucket_drifts)
    stable_output = compute_stable_output(matched_groups, num_runs)
    
    return {
        'journal_id': journal_id,
        'num_runs': num_runs,
        'num_unique_items': len(matched_groups),
        'metrics': {
            'agreement_rate': round(agreement_rate, 4),
            'polarity_flip_rate': round(polarity_flip_rate, 4),
            'bucket_drift_rate': round(bucket_drift_rate, 4),
            'recall_variance': round(recall_variance, 4)
        },
        'polarity_flips': polarity_flips,
        'bucket_drifts': bucket_drifts,
        'risk_summary': {
            'critical': len(risk_report['critical_risks']),
            'high': len(risk_report['high_risks']),
            'medium': len(risk_report['medium_risks']),
            'low': len(risk_report['low_risks'])
        },
        'stable_output': stable_output
    }


def analyze_all_stability(runs_by_journal: Dict[str, List[Dict]]) -> Tuple[Dict, List[Dict], Dict]:
    per_journal_results = []
    total_agreement = 0
    total_polarity_flip = 0
    total_bucket_drift = 0
    total_recall_var = 0
    all_critical_risks = []
    all_high_risks = []
    
    for journal_id, runs in runs_by_journal.items():
        result = analyze_journal_stability(journal_id, runs)
        per_journal_results.append(result)
        
        total_agreement += result['metrics']['agreement_rate']
        total_polarity_flip += result['metrics']['polarity_flip_rate']
        total_bucket_drift += result['metrics']['bucket_drift_rate']
        total_recall_var += result['metrics']['recall_variance']
        
        for flip in result['polarity_flips']:
            if flip.get('domain') == 'symptom':
                all_critical_risks.append({**flip, 'journal_id': journal_id})
            elif flip.get('domain') == 'emotion':
                all_high_risks.append({**flip, 'journal_id': journal_id})
    
    n = len(runs_by_journal)
    
    summary = {
        'total_journals': n,
        'total_runs_analyzed': sum(len(runs) for runs in runs_by_journal.values()),
        'aggregate_metrics': {
            'mean_agreement_rate': round(total_agreement / n, 4) if n > 0 else 0,
            'mean_polarity_flip_rate': round(total_polarity_flip / n, 4) if n > 0 else 0,
            'mean_bucket_drift_rate': round(total_bucket_drift / n, 4) if n > 0 else 0,
            'mean_recall_variance': round(total_recall_var / n, 4) if n > 0 else 0
        },
        'risk_summary': {
            'total_critical_risks': len(all_critical_risks),
            'total_high_risks': len(all_high_risks)
        }
    }
    
    risk_analysis = {
        'critical_risks': all_critical_risks,
        'high_risks': all_high_risks,
        'production_implications': {
            'guardrail_design': 'Implement polarity consistency check before output',
            'cost': 'Consider running 3+ times with voting for safety-critical extractions',
            'scalability': 'Cache stable outputs to reduce redundant LLM calls'
        }
    }
    
    return summary, per_journal_results, risk_analysis


def main():
    parser = argparse.ArgumentParser(description="Ashwam Run-to-Run Variance & Stability Analyzer")
    parser.add_argument("--data", "-d", required=True, help="Path to data directory")
    parser.add_argument("--out", "-o", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    llm_runs_dir = data_dir / "llm_runs"
    print(f"Loading LLM runs from: {llm_runs_dir}")
    runs_by_journal = load_all_runs(str(llm_runs_dir))
    print(f"  Loaded runs for {len(runs_by_journal)} journals")
    
    print("\nAnalyzing stability...")
    summary, per_journal, risk_analysis = analyze_all_stability(runs_by_journal)
    
    summary_path = out_dir / "stability_report.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote stability report to: {summary_path}")
    
    per_journal_path = out_dir / "per_journal_stability.jsonl"
    with open(per_journal_path, 'w', encoding='utf-8') as f:
        for result in per_journal:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote per-journal results to: {per_journal_path}")
    
    risk_path = out_dir / "risk_analysis.json"
    with open(risk_path, 'w', encoding='utf-8') as f:
        json.dump(risk_analysis, f, indent=2, ensure_ascii=False)
    print(f"Wrote risk analysis to: {risk_path}")
    
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Journals Analyzed:      {summary['total_journals']}")
    print(f"  Total Runs:             {summary['total_runs_analyzed']}")
    print(f"\n  Mean Agreement Rate:    {summary['aggregate_metrics']['mean_agreement_rate']:.2%}")
    print(f"  Mean Polarity Flip:     {summary['aggregate_metrics']['mean_polarity_flip_rate']:.2%}")
    print(f"  Mean Bucket Drift:      {summary['aggregate_metrics']['mean_bucket_drift_rate']:.2%}")
    print(f"  Mean Recall Variance:   {summary['aggregate_metrics']['mean_recall_variance']:.2%}")
    print(f"\n  Critical Risks Found:   {summary['risk_summary']['total_critical_risks']}")
    print(f"  High Risks Found:       {summary['risk_summary']['total_high_risks']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
