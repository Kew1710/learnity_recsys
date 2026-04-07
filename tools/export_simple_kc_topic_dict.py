#!/usr/bin/env python3
"""Export a simple kc_id -> topic_id|null dictionary from the draft mapping."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = ROOT / "docs" / "kc_topic_mapping_draft.jsonl"
DICT_PATH = ROOT / "docs" / "kc_topic_mapping_simple.json"
REVIEW_PATH = ROOT / "docs" / "kc_topic_mapping_review.json"


def main() -> int:
    rows = [
        json.loads(line)
        for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    simple: dict[str, int | None] = {}
    review: dict[str, dict] = {}

    for row in rows:
        status = row["status"]
        kc_id = row["kc_id"]
        if status in {"exact", "heuristic", "ambiguous"}:
            simple[kc_id] = row.get("topic_id")
        else:
            simple[kc_id] = None

        if status in {"ambiguous", "no_match"}:
            review[kc_id] = {
                "kc_name": row["kc_name"],
                "status": status,
                "topic_id": row.get("topic_id"),
                "topic_name": row.get("topic_name"),
                "confidence": row.get("confidence"),
                "rationale": row.get("rationale"),
                "alternative_topics": row.get("alternative_topics", []),
            }

    DICT_PATH.write_text(
        json.dumps(dict(sorted(simple.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    REVIEW_PATH.write_text(
        json.dumps(dict(sorted(review.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote simple dict to {DICT_PATH}")
    print(f"Wrote review file to {REVIEW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
