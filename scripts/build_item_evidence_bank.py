from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance
from typing import Any, Iterable, Iterator

JsonRow = dict[str, Any]


def read_jsonl(path: str | Path) -> Iterator[JsonRow]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[JsonRow], overwrite: bool) -> int:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return count


def load_aspect_names(path: str | Path) -> list[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        records = data.get("lexicon", data.get("aspects", data.get("aspect_names", [])))
    else:
        records = data

    if not isinstance(records, list):
        raise ValueError(f"{path}: aspect lexicon must be a list or contain a lexicon/aspects list")

    names: list[str] = []
    seen: set[str] = set()
    for record in records:
        if isinstance(record, str):
            name = record.strip()
        elif isinstance(record, dict):
            name = str(record.get("aspect_name", record.get("aspect", record.get("name", record.get("label", ""))))).strip()
        else:
            name = ""
        if name and name not in seen:
            names.append(name)
            seen.add(name)

    if not names:
        raise ValueError(f"No aspect names found in {path}")
    return names


def load_items(path: str | Path) -> dict[str, JsonRow]:
    items: dict[str, JsonRow] = {}
    for row_index, row in enumerate(read_jsonl(path), start=1):
        item_id = row.get("item_id")
        if item_id is None:
            raise ValueError(f"{path}:{row_index}: missing required field 'item_id'")
        items[str(item_id)] = row
    return items


def load_review_id_set(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    review_ids: set[str] = set()
    for row_index, row in enumerate(read_jsonl(path), start=1):
        review_id = row.get("review_id")
        if review_id is None:
            raise ValueError(f"{path}:{row_index}: missing required field 'review_id'")
        review_ids.add(str(review_id))
    return review_ids


def parse_split_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"all", "*"}:
        return None
    return {split.strip() for split in value.split(",") if split.strip()}


def split_allowed(split: Any, allowed_splits: set[str] | None) -> bool:
    return allowed_splits is None or str(split) in allowed_splits


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_descriptor(value: Any) -> str:
    return " ".join(as_text(value).strip().lower().split())


def sentiment(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"positive", "pos", "+", "+1"}:
            return 1
        if normalized in {"negative", "neg", "-", "-1"}:
            return -1
        if normalized in {"neutral", "neu", "0"}:
            return 0
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return 1
    if parsed < 0:
        return -1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build item evidence banks from sentence-level preference labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labeled_sentences_path", required=True)
    parser.add_argument("--items_path", required=True)
    parser.add_argument("--aspect_lexicon_path", "--lexicon_path", dest="aspect_lexicon_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--allowed_splits", default="train")
    parser.add_argument("--exclude_review_ids_path", default=None)
    parser.add_argument("--max_descriptors_per_polarity", type=int, default=10)
    parser.add_argument("--max_examples_per_descriptor", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_descriptors_per_polarity < 0:
        raise ValueError("--max_descriptors_per_polarity must be non-negative")
    if args.max_examples_per_descriptor < 0:
        raise ValueError("--max_examples_per_descriptor must be non-negative")

    allowed_splits = parse_split_set(args.allowed_splits)
    excluded_review_ids = load_review_id_set(args.exclude_review_ids_path)
    items = load_items(args.items_path)
    aspect_names = load_aspect_names(args.aspect_lexicon_path)
    aspect_set = set(aspect_names)
    observed_splits: set[str] = set()

    item_review_ids: dict[str, set[str]] = defaultdict(set)
    aspect_review_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    aspect_sentiments: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    descriptor_counts: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"positive": Counter(), "negative": Counter(), "neutral": Counter()})
    )
    descriptor_examples: dict[str, dict[str, dict[str, dict[str, list[JsonRow]]]]] = defaultdict(
        lambda: defaultdict(lambda: {"positive": defaultdict(list), "negative": defaultdict(list), "neutral": defaultdict(list)})
    )

    for row_index, row in enumerate(read_jsonl(args.labeled_sentences_path), start=1):
        split = row.get("split")
        if not split_allowed(split, allowed_splits):
            continue
        if split is not None:
            observed_splits.add(str(split))
        if row.get("item_id") is None or row.get("review_id") is None:
            raise ValueError(f"{args.labeled_sentences_path}:{row_index}: missing item_id or review_id")

        item_id = str(row["item_id"])
        review_id = str(row["review_id"])
        if review_id in excluded_review_ids:
            continue
        item_review_ids[item_id].add(review_id)

        labels = row.get("labels", [])
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise ValueError(f"{args.labeled_sentences_path}:{row_index}: field 'labels' must be a list")

        for label in labels:
            if not isinstance(label, dict):
                continue
            aspect = as_text(label.get("a")).strip()
            if aspect not in aspect_set:
                continue
            polarity = sentiment(label.get("s", 0))
            bucket = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
            descriptor = normalize_descriptor(label.get("d", ""))
            aspect_review_ids[item_id][aspect].add(review_id)
            aspect_sentiments[item_id][aspect].append(polarity)

            if not descriptor:
                continue
            descriptor_counts[item_id][aspect][bucket][descriptor] += 1
            examples = descriptor_examples[item_id][aspect][bucket][descriptor]
            if len(examples) < args.max_examples_per_descriptor:
                examples.append(
                    {
                        "review_id": review_id,
                        "sent_id": row.get("sent_id"),
                        "sentence": as_text(row.get("sentence", "")),
                        "evidence": label.get("e"),
                        "descriptor": as_text(label.get("d", "")).strip(),
                    }
                )

    source_splits = sorted(observed_splits if allowed_splits is None else allowed_splits)
    outputs: list[JsonRow] = []

    for item_id in sorted(item_review_ids):
        total_reviews = len(item_review_ids[item_id])
        item_meta = items.get(item_id, {})
        aspects: dict[str, JsonRow] = {}

        for aspect in aspect_names:
            support_reviews = len(aspect_review_ids[item_id][aspect])
            values = aspect_sentiments[item_id][aspect]
            bucket_rows: dict[str, list[JsonRow]] = {}

            for bucket in ("positive", "negative", "neutral"):
                bucket_rows[bucket] = [
                    {
                        "descriptor": descriptor,
                        "count": count,
                        "examples": descriptor_examples[item_id][aspect][bucket][descriptor],
                    }
                    for descriptor, count in descriptor_counts[item_id][aspect][bucket].most_common(
                        args.max_descriptors_per_polarity
                    )
                ]

            aspects[aspect] = {
                "support_reviews": support_reviews,
                "support_labels": len(values),
                "mention_rate": float(support_reviews / total_reviews) if total_reviews else 0.0,
                "mean_sentiment": float(mean(values)) if values else 0.0,
                "sentiment_variance": float(pvariance(values)) if len(values) >= 2 else 0.0,
                "positive_descriptors": bucket_rows["positive"],
                "negative_descriptors": bucket_rows["negative"],
                "neutral_descriptors": bucket_rows["neutral"],
            }

        outputs.append(
            {
                "item_id": item_id,
                "item_title": item_meta.get("title", item_meta.get("item_title", "")),
                "description": item_meta.get("description", ""),
                "categories": item_meta.get("categories", item_meta.get("category", [])),
                "review_count": total_reviews,
                "bank_source_splits": source_splits,
                "aspects": aspects,
            }
        )

    count = write_jsonl(args.output_path, outputs, overwrite=args.overwrite)
    print(f"Wrote {count:,} item evidence banks to {args.output_path}")


if __name__ == "__main__":
    main()
