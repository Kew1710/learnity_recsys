#!/usr/bin/env python3
"""Builds task_bank import artifacts from a staging JSONL file.

Workflow:
1. Bootstrap placeholder rows from exam_data leaf directories.
2. Fill the generated JSONL with real task content and topic bindings.
3. Export CSV or SQL matching task_bank_schema.md.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAM_DATA_DIR = ROOT / "exam_data"
DEFAULT_STAGING_PATH = DEFAULT_EXAM_DATA_DIR / "task_bank_staging.jsonl"
DEFAULT_CSV_PATH = ROOT / "task_bank_import.csv"
DEFAULT_SQL_PATH = ROOT / "task_bank_import.sql"
GRAPH_SEED_PATH = ROOT / "services" / "graph" / "seed.py"


REQUIRED_EXPORT_FIELDS = (
    "tutor_id",
    "subject_id",
    "task_type",
    "answer_type",
    "difficulty",
    "body",
    "answer",
    "solution",
    "topics",
    "is_public",
    "is_generated",
)


@dataclass(frozen=True)
class KcTopic:
    kc_id: str
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create placeholder staging rows from exam_data.")
    bootstrap.add_argument("--exam-data-dir", type=Path, default=DEFAULT_EXAM_DATA_DIR)
    bootstrap.add_argument("--staging-path", type=Path, default=DEFAULT_STAGING_PATH)
    bootstrap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite staging file if it already exists.",
    )

    export = subparsers.add_parser("export", help="Export staging rows to CSV or SQL.")
    export.add_argument("--staging-path", type=Path, default=DEFAULT_STAGING_PATH)
    export.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    export.add_argument("--sql-path", type=Path, default=DEFAULT_SQL_PATH)
    export.add_argument(
        "--format",
        choices=("csv", "sql", "both"),
        default="both",
        help="Which export artifacts to write.",
    )
    export.add_argument(
        "--allow-empty-topics",
        action="store_true",
        help="Allow rows without resolved integer topic ids.",
    )

    validate = subparsers.add_parser("validate", help="Validate staging rows without writing exports.")
    validate.add_argument("--staging-path", type=Path, default=DEFAULT_STAGING_PATH)
    validate.add_argument(
        "--allow-empty-topics",
        action="store_true",
        help="Allow rows without resolved integer topic ids.",
    )
    return parser.parse_args()


def load_kc_catalog(seed_path: Path) -> dict[str, KcTopic]:
    module = ast.parse(seed_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "KCS":
                    value = ast.literal_eval(node.value)
                    return {
                        item["kc_id"]: KcTopic(kc_id=item["kc_id"], name=item["name"])
                        for item in value
                    }
    raise RuntimeError(f"Could not find KCS in {seed_path}")


def detect_leaf_dirs(exam_data_dir: Path) -> list[Path]:
    leaf_dirs: list[Path] = []
    for path in sorted(exam_data_dir.rglob("*")):
        if not path.is_dir():
            continue
        children = [child for child in path.iterdir() if child.is_dir()]
        if children:
            continue
        if path == exam_data_dir:
            continue
        leaf_dirs.append(path)
    return leaf_dirs


def build_placeholder_rows(exam_data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for leaf_dir in detect_leaf_dirs(exam_data_dir):
        rel = leaf_dir.relative_to(exam_data_dir)
        parts = rel.parts
        exam_kind = parts[0] if len(parts) > 0 else ""
        package_slug = parts[1] if len(parts) > 1 else ""
        task_number = parts[2] if len(parts) > 2 else leaf_dir.name
        rows.append(
            {
                "source_path": str(rel),
                "exam_kind": exam_kind,
                "package_slug": package_slug,
                "task_number": task_number,
                "tutor_id": "",
                "subject_id": "",
                "task_type": "exam",
                "answer_type": "open",
                "difficulty": "medium",
                "body": "",
                "answer": [],
                "solution": [],
                "topics": [],
                "topic_kc_ids": [],
                "is_public": True,
                "is_generated": False,
                "deadline": None,
                "notes": (
                    "Заполни body/answer/solution и либо topics (integer[]), "
                    "либо topic_kc_ids с последующим ручным сопоставлением."
                ),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Field '{field_name}' must be boolean.")


def normalize_text_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Field '{field_name}' must be a list of strings.")
    return value


def normalize_int_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Field '{field_name}' must be a list of integers.")
    return value


def validate_uuid(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Field '{field_name}' must be a non-empty UUID string.")
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"Field '{field_name}' must be a valid UUID string.") from exc
    return value


def validate_row(row: dict[str, Any], kc_catalog: dict[str, KcTopic], allow_empty_topics: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    normalized["id"] = str(uuid.uuid4())

    for field_name in ("tutor_id", "subject_id", "task_type", "answer_type", "difficulty", "body"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field '{field_name}' must be a non-empty string.")
        normalized[field_name] = value.strip()

    normalized["tutor_id"] = validate_uuid(normalized["tutor_id"], "tutor_id")
    normalized["subject_id"] = validate_uuid(normalized["subject_id"], "subject_id")
    normalized["answer"] = normalize_text_list(row.get("answer"), "answer")
    normalized["solution"] = normalize_text_list(row.get("solution"), "solution")
    normalized["is_public"] = normalize_bool(row.get("is_public"), "is_public")
    normalized["is_generated"] = normalize_bool(row.get("is_generated"), "is_generated")

    topics = row.get("topics", [])
    normalized["topics"] = normalize_int_list(topics, "topics")

    topic_kc_ids = row.get("topic_kc_ids", [])
    if topic_kc_ids:
        if not isinstance(topic_kc_ids, list) or not all(isinstance(item, str) for item in topic_kc_ids):
            raise ValueError("Field 'topic_kc_ids' must be a list of strings.")
        unknown = [kc_id for kc_id in topic_kc_ids if kc_id not in kc_catalog]
        if unknown:
            raise ValueError(f"Unknown kc ids in topic_kc_ids: {', '.join(sorted(unknown))}")
        normalized["topic_kc_ids"] = topic_kc_ids
    else:
        normalized["topic_kc_ids"] = []

    if not allow_empty_topics and not normalized["topics"]:
        raise ValueError("Field 'topics' must contain at least one integer topic id.")

    deadline = row.get("deadline")
    if deadline is not None and not isinstance(deadline, str):
        raise ValueError("Field 'deadline' must be null or ISO timestamp string.")
    normalized["deadline"] = deadline

    normalized["source_path"] = row.get("source_path", "")
    normalized["exam_kind"] = row.get("exam_kind", "")
    normalized["package_slug"] = row.get("package_slug", "")
    normalized["task_number"] = row.get("task_number", "")
    return normalized


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_text_array(values: list[str]) -> str:
    return "ARRAY[" + ", ".join(sql_quote(item) for item in values) + "]"


def sql_int_array(values: list[int]) -> str:
    return "ARRAY[" + ", ".join(str(item) for item in values) + "]"


def sql_bool(value: bool) -> str:
    return "true" if value else "false"


def export_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=("id",) + REQUIRED_EXPORT_FIELDS + ("deadline",))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "tutor_id": row["tutor_id"],
                    "subject_id": row["subject_id"],
                    "task_type": row["task_type"],
                    "answer_type": row["answer_type"],
                    "difficulty": row["difficulty"],
                    "body": row["body"],
                    "answer": json.dumps(row["answer"], ensure_ascii=False),
                    "solution": json.dumps(row["solution"], ensure_ascii=False),
                    "topics": json.dumps(row["topics"], ensure_ascii=False),
                    "is_public": row["is_public"],
                    "is_generated": row["is_generated"],
                    "deadline": row["deadline"] or "",
                }
            )


def export_sql(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "INSERT INTO task_bank "
            "(id, tutor_id, subject_id, task_type, answer_type, difficulty, body, answer, solution, topics, is_public, is_generated, deadline)\n"
            "VALUES\n"
        )
        values_sql: list[str] = []
        for row in rows:
            deadline_sql = "NULL" if row["deadline"] is None else sql_quote(row["deadline"])
            values_sql.append(
                "("
                + ", ".join(
                    [
                        sql_quote(row["id"]),
                        sql_quote(row["tutor_id"]),
                        sql_quote(row["subject_id"]),
                        sql_quote(row["task_type"]),
                        sql_quote(row["answer_type"]),
                        sql_quote(row["difficulty"]),
                        sql_quote(row["body"]),
                        sql_text_array(row["answer"]),
                        sql_text_array(row["solution"]),
                        sql_int_array(row["topics"]),
                        sql_bool(row["is_public"]),
                        sql_bool(row["is_generated"]),
                        deadline_sql,
                    ]
                )
                + ")"
            )
        fh.write(",\n".join(values_sql))
        fh.write(";\n")


def warn_subject_mismatch(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        package_slug = str(row.get("package_slug", ""))
        if package_slug and "math" not in package_slug and "mat" not in package_slug:
            warnings.append(
                f"{row.get('source_path', '')}: package '{package_slug}' is not obviously a math package, "
                "while services/graph/seed.py contains only math topics."
            )
    return warnings


def main() -> int:
    args = parse_args()
    kc_catalog = load_kc_catalog(GRAPH_SEED_PATH)

    if args.command == "bootstrap":
        if args.staging_path.exists() and not args.force:
            print(f"Refusing to overwrite existing staging file: {args.staging_path}", file=sys.stderr)
            return 1
        rows = build_placeholder_rows(args.exam_data_dir)
        write_jsonl(args.staging_path, rows)
        print(f"Wrote {len(rows)} placeholder rows to {args.staging_path}")
        if not rows:
            print("No leaf task directories found under exam_data.", file=sys.stderr)
            return 1
        return 0

    rows = load_jsonl(args.staging_path)
    if not rows:
        print(f"No rows found in {args.staging_path}", file=sys.stderr)
        return 1

    warnings = warn_subject_mismatch(rows)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    try:
        normalized_rows = [
            validate_row(row, kc_catalog=kc_catalog, allow_empty_topics=args.allow_empty_topics)
            for row in rows
        ]
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if args.command == "validate":
        print(f"Validated {len(normalized_rows)} rows from {args.staging_path}")
        return 0

    if args.format in ("csv", "both"):
        export_csv(normalized_rows, args.csv_path)
        print(f"Wrote CSV export to {args.csv_path}")
    if args.format in ("sql", "both"):
        export_sql(normalized_rows, args.sql_path)
        print(f"Wrote SQL export to {args.sql_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
