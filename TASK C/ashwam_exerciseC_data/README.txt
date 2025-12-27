Ashwam ML/AI Take-Home — Exercise C Data Package
=================================================

This folder contains synthetic "production-like" batches to support:
Exercise C — Production Monitoring Without Ground Truth

Files
-----
- data/journals.jsonl
  20 synthetic Ashwam journal entries (mixed English/Hinglish).

- data/parser_outputs_day0.jsonl
  Simulated parser outputs for Day 0 (baseline behavior).

- data/parser_outputs_day1.jsonl
  Simulated parser outputs for Day 1 (drift/breakage behavior).
  Intentional issues included:
    - evidence spans that do NOT exist in the source text (hallucination/invariant violation)
    - missing required fields on some items (schema validity degradation)
    - increased 'mind' extraction rate (domain mix drift)
    - elevated high-arousal emotion rates (distribution drift)
    - contradictions: same evidence span with conflicting polarity (high risk)

- data/canary/
  A small labeled canary set to run post-deployment.
    - journals.jsonl (5 journals)
    - gold.jsonl (evidence-grounded labels; no canonical symptom/food/emotion/mind labels)

How candidates might use this
-----------------------------
- Run hard checks on Day0 vs Day1 and report invariant failure rates.
- Compute drift metrics comparing Day0 distribution vs Day1.
- Demonstrate a canary runner that scores a small fixed set and triggers alerts.

All data is synthetic (no real user content).

Created: 2025-12-19
