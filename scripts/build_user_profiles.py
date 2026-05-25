from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
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


def write_json(path: str | Path, row: JsonRow, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build multi-aspect user preference profiles from sentence-level preference labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labeled_sentences_path", required=True)
    parser.add_argument("--aspect_lexicon_path", "--lexicon_path", dest="aspect_lexicon_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--summary_output_path", default=None)
    parser.add_argument("--allowed_splits", default="train")
    parser.add_argument("--exclude_review_ids_path", default=None)
    parser.add_argument("--shrinkage_c", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.shrinkage_c < 0:
        raise ValueError("--shrinkage_c must be non-negative")

    aspect_names = load_aspect_names(args.aspect_lexicon_path)
    aspect_set = set(aspect_names)
    allowed_splits = parse_split_set(args.allowed_splits)
    excluded_review_ids = load_review_id_set(args.exclude_review_ids_path)
    observed_splits: set[str] = set()

    user_review_ids: dict[str, set[str]] = defaultdict(set)
    user_ratings: dict[str, dict[str, Any]] = defaultdict(dict)
    user_mentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    user_sentiments: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    global_mentions: dict[str, int] = defaultdict(int)
    global_sentiments: dict[str, list[int]] = defaultdict(list)
    global_total_mentions = 0

    for row_index, row in enumerate(read_jsonl(args.labeled_sentences_path), start=1):
        split = row.get("split")
        if not split_allowed(split, allowed_splits):
            continue
        if split is not None:
            observed_splits.add(str(split))
        if row.get("user_id") is None or row.get("review_id") is None:
            raise ValueError(f"{args.labeled_sentences_path}:{row_index}: missing user_id or review_id")

        user_id = str(row["user_id"])
        review_id = str(row["review_id"])
        if review_id in excluded_review_ids:
            continue
        user_review_ids[user_id].add(review_id)
        if review_id not in user_ratings[user_id]:
            user_ratings[user_id][review_id] = row.get("rating")

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
            user_mentions[user_id][aspect] += 1
            user_sentiments[user_id][aspect].append(polarity)
            global_mentions[aspect] += 1
            global_sentiments[aspect].append(polarity)
            global_total_mentions += 1

    global_importance: dict[str, float] = {}
    global_mean_sentiment: dict[str, float] = {}
    for aspect in aspect_names:
        global_importance[aspect] = float(global_mentions[aspect] / global_total_mentions) if global_total_mentions else 0.0
        global_mean_sentiment[aspect] = float(mean(global_sentiments[aspect])) if global_sentiments[aspect] else 0.0

    source_splits = sorted(observed_splits if allowed_splits is None else allowed_splits)
    outputs: list[JsonRow] = []

    for user_id in sorted(user_review_ids):
        total_mentions = sum(user_mentions[user_id].values())
        num_reviews = len(user_review_ids[user_id])
        alpha = float(num_reviews / (num_reviews + args.shrinkage_c)) if num_reviews + args.shrinkage_c > 0 else 0.0
        aspect_importance: dict[str, float] = {}
        aspect_sentiment: dict[str, float] = {}
        aspect_importance_shrunk: dict[str, float] = {}
        aspect_sentiment_shrunk: dict[str, float] = {}
        aspect_difference_from_global: dict[str, float] = {}

        for aspect in aspect_names:
            importance = float(user_mentions[user_id][aspect] / total_mentions) if total_mentions else 0.0
            values = user_sentiments[user_id][aspect]
            mean_sentiment = float(mean(values)) if values else 0.0
            importance_shrunk = alpha * importance + (1.0 - alpha) * global_importance[aspect]
            sentiment_shrunk = alpha * mean_sentiment + (1.0 - alpha) * global_mean_sentiment[aspect]
            aspect_importance[aspect] = importance
            aspect_sentiment[aspect] = mean_sentiment
            aspect_importance_shrunk[aspect] = importance_shrunk
            aspect_sentiment_shrunk[aspect] = sentiment_shrunk
            aspect_difference_from_global[aspect] = importance_shrunk - global_importance[aspect]

        ratings = [value for value in (safe_float(value) for value in user_ratings[user_id].values()) if value is not None]
        outputs.append(
            {
                "user_id": user_id,
                "profile_source_splits": source_splits,
                "num_reviews": num_reviews,
                "num_labeled_mentions": total_mentions,
                "avg_rating": float(mean(ratings)) if ratings else None,
                "shrinkage_alpha": alpha,
                "aspect_importance": aspect_importance,
                "aspect_sentiment": aspect_sentiment,
                "aspect_importance_shrunk": aspect_importance_shrunk,
                "aspect_sentiment_shrunk": aspect_sentiment_shrunk,
                "aspect_difference_from_global": aspect_difference_from_global,
            }
        )

    count = write_jsonl(args.output_path, outputs, overwrite=args.overwrite)
    print(f"Wrote {count:,} user profiles to {args.output_path}")

    if args.summary_output_path:
        summary = {
            "profile_source_splits": source_splits,
            "num_users": len(outputs),
            "num_aspects": len(aspect_names),
            "global_total_mentions": global_total_mentions,
            "global_importance": global_importance,
            "global_mean_sentiment": global_mean_sentiment,
            "shrinkage_c": args.shrinkage_c,
        }
        write_json(args.summary_output_path, summary, overwrite=args.overwrite)
        print(f"Wrote profile summary to {args.summary_output_path}")


if __name__ == "__main__":
    main()
