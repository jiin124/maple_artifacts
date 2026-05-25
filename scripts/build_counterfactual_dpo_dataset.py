import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, Optional, Sequence

from tqdm import tqdm


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row


def write_jsonl(path: str, rows: Sequence[Dict], overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".jsonl", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_json(path: str, obj: Dict, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_splits(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def row_review_id(row: Dict) -> str:
    for key in ["review_id", "target_review_id", "chosen_review_id"]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def load_sft_rows(path: str) -> Dict[str, Dict]:
    out = {}
    for row in read_jsonl(path):
        rid = row_review_id(row)
        if rid:
            out[rid] = row
    return out


def prompt_messages(sft_row: Dict) -> list:
    messages_prompt = sft_row.get("messages_prompt")
    if isinstance(messages_prompt, list) and messages_prompt:
        return messages_prompt
    messages = sft_row.get("messages")
    if isinstance(messages, list) and messages:
        if messages[-1].get("role") == "assistant":
            return messages[:-1]
        return messages
    prompt = sft_row.get("prompt", "")
    return [{"role": "user", "content": prompt}]


def make_pair(
    sft_row: Dict,
    same_item_row: Dict,
    min_chosen_chars: int,
    min_rejected_chars: int,
    idx: int,
) -> Optional[Dict]:
    chosen = clean_text(same_item_row.get("chosen") or sft_row.get("target") or sft_row.get("target_review_text"))
    rejected = clean_text(same_item_row.get("rejected") or same_item_row.get("negative") or same_item_row.get("rejected_review_text"))
    if len(chosen) < min_chosen_chars or len(rejected) < min_rejected_chars:
        return None
    rid = row_review_id(sft_row)
    return {
        "id": f"{rid}__same_item_other_user__{idx}",
        "split": sft_row.get("split"),
        "prompt": sft_row.get("prompt", ""),
        "messages_prompt": prompt_messages(sft_row),
        "chosen": chosen,
        "rejected": rejected,
        "pair_type": "same_item_counterfactual",
        "negative_type": "same_item_other_user",
        "user_id": sft_row.get("user_id"),
        "item_id": sft_row.get("item_id"),
        "review_id": rid,
        "rating": sft_row.get("rating"),
        "review_title": sft_row.get("review_title", ""),
        "item_title": sft_row.get("item_title", ""),
        "personalized_evidence_plan": sft_row.get("personalized_evidence_plan", []),
        "top_user_aspects": sft_row.get("top_user_aspects", []),
        "style_examples": sft_row.get("style_examples", []),
        "prompt_version": sft_row.get("prompt_version", "maple_personalized_evidence_plan_style_examples"),
        "rejected_user_id": same_item_row.get("rejected_user_id") or same_item_row.get("negative_user_id") or same_item_row.get("peer_user_id"),
        "rejected_review_id": same_item_row.get("rejected_review_id") or same_item_row.get("negative_review_id") or same_item_row.get("peer_review_id"),
        "profile_distance": same_item_row.get("profile_distance"),
        "rating_diff": same_item_row.get("rating_diff"),
        "length_ratio": same_item_row.get("length_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-item counterfactual DPO rows from generator SFT rows.")
    parser.add_argument("--sft_all_path", required=True)
    parser.add_argument("--same_item_pairs_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="counterfactual_dpo")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--min_chosen_chars", type=int, default=40)
    parser.add_argument("--min_rejected_chars", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wanted_splits = parse_splits(args.splits)
    sft_rows = load_sft_rows(args.sft_all_path)
    rows_by_split = defaultdict(list)
    all_pairs = []
    stats = Counter()

    for same_item_row in tqdm(list(read_jsonl(args.same_item_pairs_path)), desc="Building counterfactual DPO rows", unit="pair"):
        rid = row_review_id(same_item_row)
        if not rid:
            stats["skip_missing_review_id"] += 1
            continue
        sft_row = sft_rows.get(rid)
        if sft_row is None:
            stats["skip_missing_sft_row"] += 1
            continue
        split = sft_row.get("split")
        if wanted_splits and split not in wanted_splits:
            stats["skip_split"] += 1
            continue
        pair = make_pair(sft_row, same_item_row, args.min_chosen_chars, args.min_rejected_chars, len(all_pairs))
        if pair is None:
            stats["skip_short_pair"] += 1
            continue
        rows_by_split[split].append(pair)
        all_pairs.append(pair)
        stats["same_item_other_user_pairs"] += 1

    for split in sorted(wanted_splits):
        write_jsonl(os.path.join(args.output_dir, f"{args.prefix}_{split}.jsonl"), rows_by_split.get(split, []), overwrite=args.overwrite)
    write_jsonl(os.path.join(args.output_dir, f"{args.prefix}_all.jsonl"), all_pairs, overwrite=args.overwrite)
    summary = {
        "num_pairs": len(all_pairs),
        "stats": dict(stats),
        "split_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "config": vars(args),
    }
    write_json(os.path.join(args.output_dir, f"{args.prefix}_summary.json"), summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
