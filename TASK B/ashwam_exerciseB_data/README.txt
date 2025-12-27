Ashwam ML/AI Take-Home — Exercise B Data Package
=================================================

This folder contains synthetic journaling data + simulated LLM outputs for:
Exercise B — Run-to-Run Variance & Stability Analysis

Files
-----
- data/journals.jsonl
  5 synthetic Ashwam journal entries.

- data/llm_runs/
  For each journal_id, there are 3 JSON files:
    - <journal_id>.run1.json
    - <journal_id>.run2.json
    - <journal_id>.run3.json

  Each run file contains:
    - journal_id
    - run_id
    - items: list of semantic objects with:
        - domain: "symptom" | "food" | "emotion" | "mind"
        - text: free text summary (may vary across runs)
        - evidence_span: exact substring from journal text (anchor)
        - polarity: "present" | "absent" | "uncertain"
        - intensity_bucket OR arousal_bucket
        - time_bucket
        - confidence (optional numeric)

Intentional Variance Patterns Included
--------------------------------------
- synonym / paraphrase drift in `text`
- missing objects in some runs (recall variance)
- bucket drift (intensity/arousal/time)
- one deliberate high-risk polarity flip on a negated emotion case

All data is synthetic (no real user content). Evidence spans were validated as exact substrings.

Created: 2025-12-19
