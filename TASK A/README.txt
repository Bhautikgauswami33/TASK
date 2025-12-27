Ashwam ML/AI Take-Home — Exercise A Data Package
=================================================

This folder contains synthetic journaling data for:
Exercise A — Evidence-Grounded Extraction & Evaluation

Files
-----
- data/journals.jsonl
  Synthetic Ashwam journal entries.
  Each line is a JSON object with:
    - journal_id (string)
    - created_at (ISO8601 string)
    - text (string)
    - lang_hint (string, optional)

- data/gold.jsonl
  Gold references for evaluation.
  IMPORTANT: No canonical labels are used for symptoms/food/emotion/mind.
  Each line is a JSON object with:
    - journal_id
    - items: list of objects with:
        - domain: "symptom" | "food" | "emotion" | "mind"
        - evidence_span: exact substring from the journal text
        - polarity: "present" | "absent" | "uncertain"
        - time_bucket: "today" | "last_night" | "past_week" | "unknown"
        - intensity_bucket (for symptom/mind/food) OR arousal_bucket (for emotion)

- data/sample_predictions.jsonl (optional)
  A tiny set of example predicted outputs for 2 journals.
  Provided only to help candidates validate their scorer quickly.

Notes
-----
- Evidence spans are exact substrings and were validated programmatically.
- All content is synthetic; no real user data.

Created: 2025-12-19
