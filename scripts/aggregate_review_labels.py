from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
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


def as_text(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_descriptor(value: Any) -> str:
    return " ".join(as_text(value).strip().split())


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


def required(row: JsonRow, key: str, path: str | Path, row_index: int) -> Any:
    if key not in row or row[key] is None:
        raise ValueError(f"{path}:{row_index}: missing required field '{key}'")
    return row[key]


def build_review_labels(
    labeled_sentences_path: str | Path,
    aspect_lexicon_path: str | Path,
    max_descriptors_per_aspect: int,
    max_evidence_examples_per_aspect: int,
) -> list[JsonRow]:
    if max_descriptors_per_aspect < 0:
        raise ValueError("--max_descriptors_per_aspect must be non-negative")
    if max_evidence_examples_per_aspect < 0:
        raise ValueError("--max_evidence_examples_per_aspect must be non-negative")

    aspect_names = load_aspect_names(aspect_lexicon_path)
    aspect_set = set(aspect_names)
    grouped: dict[str, JsonRow] = {}

    for row_index, row in enumerate(read_jsonl(labeled_sentences_path), start=1):
        review_id = str(required(row, "review_id", labeled_sentences_path, row_index))
        sent_id = required(row, "sent_id", labeled_sentences_path, row_index)
        sentence = as_text(required(row, "sentence", labeled_sentences_path, row_index))

        if review_id not in grouped:
            grouped[review_id] = {
                "review_id": review_id,
                "user_id": str(required(row, "user_id", labeled_sentences_path, row_index)),
                "item_id": str(required(row, "item_id", labeled_sentences_path, row_index)),
                "split": row.get("split"),
                "rating": row.get("rating"),
                "title": row.get("title", ""),
                "item_title": row.get("item_title", ""),
                "domain": row.get("domain"),
                "sentences": [],
                "labels": [],
            }

        labels = row.get("labels", [])
        if labels is None:
            labels = []
        if not isinstance(labels, list):
            raise ValueError(f"{labeled_sentences_path}:{row_index}: field 'labels' must be a list")

        valid_labels = []
        for label in labels:
            if not isinstance(label, dict):
                continue
            aspect = as_text(label.get("a")).strip()
            if aspect not in aspect_set:
                continue
            valid_labels.append(
                {
                    "sent_id": sent_id,
                    "sentence": sentence,
                    "a": aspect,
                    "d": normalize_descriptor(label.get("d")),
                    "s": sentiment(label.get("s", 0)),
                    "e": label.get("e"),
                }
            )

        grouped[review_id]["sentences"].append(
            {
                "sent_id": sent_id,
                "sentence": sentence,
                "num_labels": len(valid_labels),
            }
        )
        grouped[review_id]["labels"].extend(valid_labels)

    outputs: list[JsonRow] = []
    for review_id, data in grouped.items():
        labels = data["labels"]
        total_labels = len(labels)
        aspect_counter: Counter[str] = Counter()
        aspect_sentiments: dict[str, list[int]] = defaultdict(list)
        aspect_descriptor_counter: dict[str, Counter[str]] = defaultdict(Counter)
        aspect_examples: dict[str, list[JsonRow]] = defaultdict(list)

        for label in labels:
            aspect = label["a"]
            polarity = sentiment(label.get("s", 0))
            descriptor = normalize_descriptor(label.get("d", ""))
            aspect_counter[aspect] += 1
            aspect_sentiments[aspect].append(polarity)
            if descriptor:
                aspect_descriptor_counter[aspect][descriptor] += 1
            if len(aspect_examples[aspect]) < max_evidence_examples_per_aspect:
                aspect_examples[aspect].append(
                    {
                        "sent_id": label["sent_id"],
                        "sentence": label["sentence"],
                        "descriptor": descriptor,
                        "sentiment": polarity,
                        "evidence": label.get("e"),
                    }
                )

        aspect_stats: JsonRow = {}
        aspect_vector: JsonRow = {}
        sentiment_vector: JsonRow = {}

        for aspect in aspect_names:
            values = aspect_sentiments[aspect]
            mentions = aspect_counter[aspect]
            mention_fraction = float(mentions / total_labels) if total_labels else 0.0
            aspect_stats[aspect] = {
                "mentions": mentions,
                "pos": sum(1 for value in values if value > 0),
                "neg": sum(1 for value in values if value < 0),
                "neu": sum(1 for value in values if value == 0),
                "mean_sentiment": float(mean(values)) if values else 0.0,
                "mention_fraction": mention_fraction,
                "top_descriptors": [
                    {"descriptor": descriptor, "count": count}
                    for descriptor, count in aspect_descriptor_counter[aspect].most_common(max_descriptors_per_aspect)
                ],
                "evidence_examples": aspect_examples[aspect],
            }
            aspect_vector[aspect] = mention_fraction
            sentiment_vector[aspect] = aspect_stats[aspect]["mean_sentiment"]

        sentences = data["sentences"]
        outputs.append(
            {
                "review_id": review_id,
                "user_id": data["user_id"],
                "item_id": data["item_id"],
                "split": data.get("split"),
                "rating": data.get("rating"),
                "title": data.get("title", ""),
                "item_title": data.get("item_title", ""),
                "domain": data.get("domain"),
                "num_sentences": len(sentences),
                "num_labeled_sentences": sum(1 for sentence_row in sentences if sentence_row["num_labels"] > 0),
                "num_labels": total_labels,
                "aspect_vector": aspect_vector,
                "sentiment_vector": sentiment_vector,
                "aspect_stats": aspect_stats,
            }
        )

    outputs.sort(key=lambda row: (as_text(row.get("user_id")), as_text(row.get("review_id"))))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate sentence-level preference labels into review-level summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labeled_sentences_path", required=True)
    parser.add_argument("--aspect_lexicon_path", "--lexicon_path", dest="aspect_lexicon_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_descriptors_per_aspect", type=int, default=8)
    parser.add_argument("--max_evidence_examples_per_aspect", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = build_review_labels(
        labeled_sentences_path=args.labeled_sentences_path,
        aspect_lexicon_path=args.aspect_lexicon_path,
        max_descriptors_per_aspect=args.max_descriptors_per_aspect,
        max_evidence_examples_per_aspect=args.max_evidence_examples_per_aspect,
    )
    count = write_jsonl(args.output_path, rows, overwrite=args.overwrite)
    print(f"Wrote {count:,} review-level rows to {args.output_path}")


if __name__ == "__main__":
    main()
