#!/usr/bin/env python3
"""Generate draft mapping from graph KC ids to common topic ids.

Default mode is local-only heuristic matching:
1. Parse KCS from services/graph/seed.py.
2. Parse topics from common_topics.yaml.
3. Resolve deterministic exact/normalized matches.
4. For the rest, build a heuristic shortlist and classify:
   - heuristic
   - ambiguous
   - no_match

Outputs:
- JSONL with one mapping decision per KC
- Markdown summary with grouped counts
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GRAPH_SEED_PATH = ROOT / "services" / "graph" / "seed.py"
COMMON_TOPICS_PATH = ROOT / "common_topics.yaml"
DEFAULT_JSONL = ROOT / "docs" / "kc_topic_mapping_draft.jsonl"
DEFAULT_MD = ROOT / "docs" / "kc_topic_mapping_draft.md"
STOPWORDS = {
    "и", "или", "с", "со", "на", "по", "для", "в", "во", "к", "ко", "из", "у", "о", "об",
    "теорема", "свойства", "решение", "задач", "задача", "задачи", "формула", "формулы",
    "понятие", "основы", "метод", "элементы", "виды", "график", "графики",
}

TOKEN_SYNONYMS = {
    "натуральные": "натуральн",
    "натуральных": "натуральн",
    "натуральныех": "натуральн",
    "числа": "числ",
    "чисел": "числ",
    "дроби": "дроб",
    "дробей": "дроб",
    "дробями": "дроб",
    "рациональные": "рациональн",
    "рациональных": "рациональн",
    "иррациональные": "иррациональн",
    "иррациональных": "иррациональн",
    "квадратные": "квадратн",
    "квадратного": "квадратн",
    "квадратных": "квадратн",
    "линейное": "линейн",
    "линейные": "линейн",
    "линейных": "линейн",
    "уравнение": "уравн",
    "уравнений": "уравн",
    "уравнения": "уравн",
    "неравенства": "неравенств",
    "неравенство": "неравенств",
    "углов": "угл",
    "угол": "угл",
    "угла": "угл",
    "треугольник": "треугольн",
    "треугольника": "треугольн",
    "треугольников": "треугольн",
    "окружность": "окружн",
    "окружности": "окружн",
    "круг": "круг",
    "круга": "круг",
    "функция": "функц",
    "функции": "функц",
    "график": "график",
    "графики": "график",
    "корень": "корень",
    "корней": "корень",
    "корня": "корень",
    "логарифмы": "логарифм",
    "логарифмические": "логарифм",
    "прогрессия": "прогресс",
    "прогрессии": "прогресс",
    "арифметическая": "арифметическ",
    "геометрическая": "геометрическ",
    "площадь": "площад",
    "периметр": "периметр",
    "координатная": "координатн",
    "координатной": "координатн",
    "плоскость": "плоскост",
    "плоскости": "плоскост",
    "подобия": "подоби",
    "подобие": "подоби",
    "вероятность": "вероятност",
    "вероятностей": "вероятност",
    "комбинаторики": "комбинаторик",
    "комбинаторика": "комбинаторик",
    "преобразования": "преобразован",
    "преобразование": "преобразован",
}


@dataclass(frozen=True)
class KC:
    kc_id: str
    name: str
    grade_introduced: int
    subject: str


@dataclass(frozen=True)
class Topic:
    topic_id: int
    name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--max-kcs", type=int, default=0, help="Limit unmatched KC count for test runs.")
    return parser.parse_args()


def load_kcs(path: Path) -> list[KC]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "KCS":
                    raw = ast.literal_eval(node.value)
                    return [
                        KC(
                            kc_id=item["kc_id"],
                            name=item["name"],
                            grade_introduced=item["grade_introduced"],
                            subject=item["subject"],
                        )
                        for item in raw
                    ]
    raise RuntimeError(f"Could not find KCS in {path}")


def load_topics(path: Path) -> list[Topic]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Topic(topic_id=int(k), name=v) for k, v in raw["topics"].items()]


def normalize_text(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("√", "корень")
    value = re.sub(r"[+*/=(),.:;!?\"'`[\]{}]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-zа-я0-9]+", normalize_text(value))
    result = []
    for token in tokens:
        if len(token) <= 1 or token in STOPWORDS:
            continue
        result.append(TOKEN_SYNONYMS.get(token, token))
    return result


def build_exact_matches(kcs: list[KC], topics: list[Topic]) -> tuple[dict[str, dict[str, Any]], list[KC]]:
    by_normalized_topic: dict[str, list[Topic]] = defaultdict(list)
    for topic in topics:
        by_normalized_topic[normalize_text(topic.name)].append(topic)

    matches: dict[str, dict[str, Any]] = {}
    unmatched: list[KC] = []
    for kc in kcs:
        norm = normalize_text(kc.name)
        candidates = by_normalized_topic.get(norm, [])
        if len(candidates) == 1:
            topic = candidates[0]
            matches[kc.kc_id] = {
                "kc_id": kc.kc_id,
                "kc_name": kc.name,
                "grade_introduced": kc.grade_introduced,
                "subject": kc.subject,
                "status": "exact",
                "topic_id": topic.topic_id,
                "topic_name": topic.name,
                "confidence": 1.0,
                "rationale": "Normalized names are identical.",
                "candidates": [{"topic_id": topic.topic_id, "topic_name": topic.name}],
            }
        else:
            unmatched.append(kc)
    return matches, unmatched


def shortlist_topics(kc: KC, topics: list[Topic], limit: int) -> list[Topic]:
    kc_tokens = set(tokenize(kc.name))
    scored: list[tuple[float, Topic]] = []
    for topic in topics:
        topic_tokens = set(tokenize(topic.name))
        if not topic_tokens:
            continue
        overlap = len(kc_tokens & topic_tokens)
        if overlap == 0:
            continue
        jaccard = overlap / len(kc_tokens | topic_tokens)
        containment = overlap / max(1, min(len(kc_tokens), len(topic_tokens)))
        score = 0.65 * containment + 0.35 * jaccard
        if str(kc.grade_introduced)[:1] == str(topic.topic_id)[:1]:
            score += 0.05
        scored.append((score, topic))

    if not scored:
        for topic in topics:
            topic_tokens = set(tokenize(topic.name))
            if any(token.startswith("уравн") for token in kc_tokens) and any(token.startswith("уравн") for token in topic_tokens):
                scored.append((0.05, topic))

    scored.sort(key=lambda item: (-item[0], item[1].topic_id))
    shortlisted = [topic for _, topic in scored[:limit]]
    if not shortlisted:
        shortlisted = topics[:limit]
    return shortlisted


def classify_shortlist(kc: KC, shortlist: list[Topic]) -> dict[str, Any]:
    kc_tokens = set(tokenize(kc.name))
    scored_candidates: list[tuple[float, Topic, int, int]] = []
    for topic in shortlist:
        topic_tokens = set(tokenize(topic.name))
        overlap = len(kc_tokens & topic_tokens)
        union = len(kc_tokens | topic_tokens)
        containment = overlap / max(1, min(len(kc_tokens), len(topic_tokens)))
        jaccard = overlap / max(1, union)
        score = 0.7 * containment + 0.3 * jaccard
        if str(kc.grade_introduced)[:1] == str(topic.topic_id)[:1]:
            score += 0.05
        if any(token in topic_tokens for token in ("уравн", "неравенств", "корень", "логарифм", "прогресс", "комбинаторик", "вероятност", "окружн", "треугольн")):
            score += 0.03 * len(kc_tokens & topic_tokens)
        scored_candidates.append((score, topic, overlap, len(topic_tokens)))

    scored_candidates.sort(key=lambda item: (-item[0], -item[2], item[1].topic_id))
    if not scored_candidates:
        return {
            "status": "no_match",
            "topic_id": None,
            "topic_name": None,
            "confidence": 0.0,
            "rationale": "No lexical overlap with topic catalog.",
            "alternative_topic_ids": [],
            "alternative_topics": [],
        }

    top_score, top_topic, top_overlap, _ = scored_candidates[0]
    alternatives = [
        {"topic_id": topic.topic_id, "topic_name": topic.name}
        for score, topic, _, _ in scored_candidates[1:3]
        if score >= max(0.45, top_score - 0.08)
    ]

    if top_overlap == 0 or top_score < 0.42:
        return {
            "status": "no_match",
            "topic_id": None,
            "topic_name": None,
            "confidence": round(top_score, 2),
            "rationale": "Top candidate is too weak for safe mapping.",
            "alternative_topic_ids": [item["topic_id"] for item in alternatives],
            "alternative_topics": alternatives,
        }

    if len(alternatives) >= 2:
        return {
            "status": "ambiguous",
            "topic_id": top_topic.topic_id,
            "topic_name": top_topic.name,
            "confidence": round(top_score, 2),
            "rationale": "Several nearby topics look similarly plausible.",
            "alternative_topic_ids": [item["topic_id"] for item in alternatives],
            "alternative_topics": alternatives,
        }

    second_score = scored_candidates[1][0] if len(scored_candidates) > 1 else 0.0
    if top_score - second_score < 0.06 and second_score >= 0.45:
        return {
            "status": "ambiguous",
            "topic_id": top_topic.topic_id,
            "topic_name": top_topic.name,
            "confidence": round(top_score, 2),
            "rationale": "Top two candidates are too close for an automatic decision.",
            "alternative_topic_ids": [scored_candidates[1][1].topic_id],
            "alternative_topics": [{"topic_id": scored_candidates[1][1].topic_id, "topic_name": scored_candidates[1][1].name}],
        }

    return {
        "status": "heuristic",
        "topic_id": top_topic.topic_id,
        "topic_name": top_topic.name,
        "confidence": round(top_score, 2),
        "rationale": "Chosen by token-overlap heuristic and grade proximity.",
        "alternative_topic_ids": [item["topic_id"] for item in alternatives],
        "alternative_topics": alternatives,
    }


def run_heuristic_mapping(
    unmatched: list[KC],
    topics: list[Topic],
    candidate_count: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for kc in unmatched:
        shortlist = shortlist_topics(kc, topics, candidate_count)
        decision = classify_shortlist(kc, shortlist)
        results[kc.kc_id] = {
            "kc_id": kc.kc_id,
            "kc_name": kc.name,
            "grade_introduced": kc.grade_introduced,
            "subject": kc.subject,
            "status": decision["status"],
            "topic_id": decision["topic_id"],
            "topic_name": decision["topic_name"],
            "confidence": decision["confidence"],
            "rationale": decision["rationale"],
            "alternative_topic_ids": decision["alternative_topic_ids"],
            "alternative_topics": decision["alternative_topics"],
            "candidates": [
                {"topic_id": topic.topic_id, "topic_name": topic.name}
                for topic in shortlist
            ],
        }
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["status"] for row in rows)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_status[row["status"]].append(row)

    lines = [
        "# KC -> Common Topic Mapping Draft",
        "",
        f"Всего KC: {len(rows)}",
        "",
        "## Summary",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")

    for status in sorted(by_status):
        lines.extend(["", f"## {status}", ""])
        for row in sorted(by_status[status], key=lambda r: r["kc_id"]):
            topic_part = f"{row['topic_id']} — {row['topic_name']}" if row.get("topic_id") else "—"
            lines.append(
                f"- `{row['kc_id']}` | {row['kc_name']} | {topic_part} | conf={row['confidence']:.2f}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    kcs = load_kcs(GRAPH_SEED_PATH)
    topics = load_topics(COMMON_TOPICS_PATH)
    exact, unmatched = build_exact_matches(kcs, topics)

    if args.max_kcs > 0:
        unmatched = unmatched[:args.max_kcs]

    heuristic_results = run_heuristic_mapping(
        unmatched=unmatched,
        topics=topics,
        candidate_count=args.candidate_count,
    )

    merged = {**exact, **heuristic_results}
    rows = [merged[kc.kc_id] for kc in kcs if kc.kc_id in merged]
    write_jsonl(args.jsonl_out, rows)
    write_markdown(args.md_out, rows)
    print(f"Wrote {len(rows)} rows to {args.jsonl_out}")
    print(f"Wrote summary to {args.md_out}")
    counts = Counter(row["status"] for row in rows)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
