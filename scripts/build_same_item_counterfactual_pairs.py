import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]], overwrite: bool = False) -> None:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def parse_split_set(value: Optional[str]) -> Set[str]:
    if value is None:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def normalize_id(value: Any) -> str:
    return str(value).strip()


def get_review_id(row: Dict[str, Any]) -> str:
    value = row.get("review_id") or row.get("reviewer_review_id") or row.get("id")
    if value is not None:
        return normalize_id(value)
    user_id = normalize_id(row.get("user_id", ""))
    item_id = normalize_id(row.get("item_id", ""))
    timestamp = normalize_id(row.get("timestamp", row.get("unixReviewTime", "0")))
    return f"{user_id}__{item_id}__{timestamp}"


def get_review_text(row: Dict[str, Any]) -> str:
    for key in ("target_review", "review_text", "text", "review", "chosen"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def load_jsonl_by_id(path: str, key: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        if key in row and row[key] is not None:
            rows[normalize_id(row[key])] = row
    return rows


def load_profiles(path: str) -> Dict[str, Dict[str, Any]]:
    return load_jsonl_by_id(path, "user_id")


def load_items(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return load_jsonl_by_id(path, "item_id")


def load_item_banks(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return load_jsonl_by_id(path, "item_id")


def load_aspect_names(path: Optional[str], profiles: Dict[str, Dict[str, Any]]) -> List[str]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if isinstance(data.get("lexicon"), list):
                values = data["lexicon"]
            elif isinstance(data.get("aspects"), list):
                values = data["aspects"]
            else:
                values = list(data.keys())
        elif isinstance(data, list):
            values = data
        else:
            values = []
        names = []
        for value in values:
            if isinstance(value, dict):
                name = value.get("aspect_name") or value.get("name") or value.get("aspect")
            else:
                name = value
            name = str(name).strip() if name is not None else ""
            if name:
                names.append(name)
        if names:
            return list(dict.fromkeys(names))
    for profile in profiles.values():
        values = profile.get("aspect_importance_shrunk") or profile.get("aspect_importance") or {}
        if isinstance(values, dict) and values:
            return list(values.keys())
    return []


def get_profile_map(profile: Dict[str, Any], primary_key: str, fallback_key: str) -> Dict[str, float]:
    values = profile.get(primary_key)
    if not isinstance(values, dict):
        values = profile.get(fallback_key)
    if not isinstance(values, dict):
        return {}
    output = {}
    for key, value in values.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            output[str(key)] = 0.0
    return output


def profile_distance(
    target_profile: Dict[str, Any],
    peer_profile: Dict[str, Any],
    aspect_names: Sequence[str],
    importance_weight: float,
    sentiment_weight: float,
) -> float:
    target_importance = get_profile_map(target_profile, "aspect_importance_shrunk", "aspect_importance")
    peer_importance = get_profile_map(peer_profile, "aspect_importance_shrunk", "aspect_importance")
    target_sentiment = get_profile_map(target_profile, "aspect_sentiment_shrunk", "aspect_sentiment")
    peer_sentiment = get_profile_map(peer_profile, "aspect_sentiment_shrunk", "aspect_sentiment")
    total = 0.0
    for aspect in aspect_names:
        total += importance_weight * abs(target_importance.get(aspect, 0.0) - peer_importance.get(aspect, 0.0))
        total += sentiment_weight * abs(target_sentiment.get(aspect, 0.0) - peer_sentiment.get(aspect, 0.0))
    return float(total)


def top_user_aspects(user_profile: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    importance = get_profile_map(user_profile, "aspect_importance_shrunk", "aspect_importance")
    sentiment = get_profile_map(user_profile, "aspect_sentiment_shrunk", "aspect_sentiment")
    rows = []
    for aspect, value in importance.items():
        rows.append({"aspect": aspect, "importance": float(value), "sentiment": float(sentiment.get(aspect, 0.0))})
    rows.sort(key=lambda item: (-item["importance"], item["aspect"]))
    return rows[:top_k]


def descriptor_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("descriptor", "text", "d"):
            if value.get(key):
                return str(value[key]).strip()
    if value is not None:
        return str(value).strip()
    return ""


def evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        if isinstance(value.get("evidence"), dict):
            nested = value["evidence"]
            return str(nested.get("text", "")).strip()
        if value.get("evidence"):
            return str(value["evidence"]).strip()
        examples = value.get("examples")
        if isinstance(examples, list) and examples:
            first = examples[0]
            if isinstance(first, dict):
                nested = first.get("evidence")
                if isinstance(nested, dict):
                    return str(nested.get("text", "")).strip()
                if nested:
                    return str(nested).strip()
                if first.get("sentence"):
                    return str(first["sentence"]).strip()
    return ""


def first_descriptor(bucket: Any) -> Tuple[str, str]:
    if not isinstance(bucket, list):
        return "", ""
    for value in bucket:
        descriptor = descriptor_text(value)
        evidence = evidence_text(value)
        if descriptor or evidence:
            return descriptor, evidence
    return "", ""


def select_descriptor(aspect_bank: Dict[str, Any], action: str) -> Tuple[str, str]:
    if action == "highlight":
        buckets = ("positive_descriptors", "neutral_descriptors", "negative_descriptors")
    elif action == "warn":
        buckets = ("negative_descriptors", "neutral_descriptors", "positive_descriptors")
    elif action == "hedge":
        buckets = ("neutral_descriptors", "positive_descriptors", "negative_descriptors")
    else:
        buckets = ("neutral_descriptors", "positive_descriptors", "negative_descriptors")
    for bucket in buckets:
        descriptor, evidence = first_descriptor(aspect_bank.get(bucket))
        if descriptor or evidence:
            return descriptor, evidence
    return "", ""


def infer_action(mean_sentiment: float, sentiment_variance: float, positive_threshold: float, negative_threshold: float, hedge_variance_threshold: float) -> str:
    if mean_sentiment >= positive_threshold:
        return "highlight"
    if mean_sentiment <= negative_threshold:
        return "warn"
    if sentiment_variance >= hedge_variance_threshold:
        return "hedge"
    return "mention"


def build_personalized_evidence_plan(
    user_profile: Dict[str, Any],
    item_bank: Dict[str, Any],
    aspect_names: Sequence[str],
    top_k: int,
    positive_threshold: float,
    negative_threshold: float,
    hedge_variance_threshold: float,
) -> List[Dict[str, Any]]:
    importance = get_profile_map(user_profile, "aspect_importance_shrunk", "aspect_importance")
    aspects = item_bank.get("aspects") if isinstance(item_bank.get("aspects"), dict) else {}
    rows = []
    for aspect in aspect_names:
        aspect_bank = aspects.get(aspect)
        if not isinstance(aspect_bank, dict):
            continue
        user_importance = float(importance.get(aspect, 0.0))
        mention_rate = float(aspect_bank.get("mention_rate", 0.0) or 0.0)
        mean_sentiment = float(aspect_bank.get("mean_sentiment", 0.0) or 0.0)
        sentiment_variance = float(aspect_bank.get("sentiment_variance", 0.0) or 0.0)
        score = user_importance * mention_rate * (1.0 + abs(mean_sentiment))
        if score <= 0.0:
            continue
        action = infer_action(mean_sentiment, sentiment_variance, positive_threshold, negative_threshold, hedge_variance_threshold)
        descriptor, evidence = select_descriptor(aspect_bank, action)
        rows.append(
            {
                "aspect": aspect,
                "action": action,
                "descriptor": descriptor,
                "evidence": evidence,
                "user_importance": user_importance,
                "mention_rate": mention_rate,
                "mean_sentiment": mean_sentiment,
                "sentiment_variance": sentiment_variance,
                "score": float(score),
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["aspect"]))
    return rows[:top_k]


def normalize_categories(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(
    row: Dict[str, Any],
    item_row: Dict[str, Any],
    item_bank: Dict[str, Any],
    user_aspects: Sequence[Dict[str, Any]],
    plan: Sequence[Dict[str, Any]],
) -> str:
    existing_prompt = row.get("prompt")
    if existing_prompt is not None and str(existing_prompt).strip():
        return str(existing_prompt).strip()
    title = item_row.get("title") or item_bank.get("item_title") or row.get("item_title") or ""
    description = item_row.get("description") or item_bank.get("description") or ""
    categories = normalize_categories(item_row.get("categories", item_row.get("category", item_bank.get("categories", ""))))
    user_lines = []
    for item in user_aspects:
        user_lines.append(f"- {item['aspect']}: importance={item['importance']:.4f}, sentiment={item['sentiment']:.4f}")
    plan_lines = []
    for item in plan:
        plan_lines.append(
            f"- {item['aspect']}: action={item['action']}, descriptor={item['descriptor']}, evidence={item['evidence']}"
        )
    if not user_lines:
        user_lines = ["- No strong aspect signal available"]
    if not plan_lines:
        plan_lines = ["- No strong item evidence available"]
    return "\n".join(
        [
            "Write a personalized review for the target user about the target item.",
            "Use the multi-aspect preference profile and the personalized evidence plan.",
            "",
            "[Target Item]",
            f"- title: {title}",
            f"- categories: {categories}",
            f"- description: {description}",
            "",
            "[Multi-aspect Preference Profile]",
            *user_lines,
            "",
            "[Personalized Evidence Plan]",
            *plan_lines,
            "",
            "[Output Constraints]",
            "- write one coherent review paragraph",
            "- do not mention aspect labels explicitly",
            "- do not invent unsupported item facts",
            f"- rating context: {row.get('rating', '')}",
            f"- review title context: {row.get('title', '')}",
        ]
    ).strip()


def normalize_review_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if row.get("user_id") is None or row.get("item_id") is None:
        return None
    output = dict(row)
    output["user_id"] = normalize_id(row["user_id"])
    output["item_id"] = normalize_id(row["item_id"])
    output["review_id"] = get_review_id(row)
    output["review_text"] = get_review_text(row)
    return output


def load_review_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        normalized = normalize_review_row(row)
        if normalized is not None:
            rows.append(normalized)
    return rows


def index_by_item(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_item = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    return dict(by_item)


def split_allowed(row: Dict[str, Any], allowed_splits: Set[str]) -> bool:
    return not allowed_splits or row.get("split") in allowed_splits


def choose_hard_negative(
    target: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    profiles: Dict[str, Dict[str, Any]],
    aspect_names: Sequence[str],
    peer_splits: Set[str],
    min_review_chars: int,
    importance_weight: float,
    sentiment_weight: float,
) -> Optional[Tuple[Dict[str, Any], float]]:
    target_user = target["user_id"]
    target_review_id = target["review_id"]
    target_profile = profiles.get(target_user)
    if target_profile is None:
        return None
    scored = []
    for candidate in candidates:
        if candidate["user_id"] == target_user:
            continue
        if candidate["review_id"] == target_review_id:
            continue
        if not split_allowed(candidate, peer_splits):
            continue
        if len(candidate.get("review_text", "")) < min_review_chars:
            continue
        peer_profile = profiles.get(candidate["user_id"])
        if peer_profile is None:
            continue
        distance = profile_distance(target_profile, peer_profile, aspect_names, importance_weight, sentiment_weight)
        scored.append((distance, len(candidate.get("review_text", "")), candidate["review_id"], candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = scored[0]
    return best[3], float(best[0])


def build_pair_id(review_id: str) -> str:
    return f"{review_id}__same_item_counterfactual"


def build_pairs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    input_path = args.sft_rows_path or args.interactions_path
    if not input_path:
        raise ValueError("One of --sft_rows_path or --interactions_path is required.")
    review_pool_path = args.review_pool_path or input_path
    profiles = load_profiles(args.user_profiles_path)
    aspect_names = load_aspect_names(args.aspect_lexicon_path or args.lexicon_path, profiles)
    if not aspect_names:
        raise ValueError("No aspect names were found. Provide --aspect_lexicon_path or user profiles with aspect fields.")
    sft_rows = load_review_rows(input_path)
    review_pool = load_review_rows(review_pool_path)
    by_item = index_by_item(review_pool)
    items = load_items(args.items_path)
    item_banks = load_item_banks(args.item_evidence_bank_path)
    target_splits = parse_split_set(args.target_splits)
    peer_splits = parse_split_set(args.peer_splits)
    outputs = []
    stats = defaultdict(int)
    for row in sft_rows:
        stats["input_rows"] += 1
        if not split_allowed(row, target_splits):
            stats["skipped_split"] += 1
            continue
        chosen = row.get("review_text", "")
        if len(chosen) < args.min_review_chars:
            stats["skipped_short_chosen"] += 1
            continue
        user_id = row["user_id"]
        item_id = row["item_id"]
        user_profile = profiles.get(user_id)
        if user_profile is None:
            stats["skipped_missing_user_profile"] += 1
            continue
        negative = choose_hard_negative(
            row,
            by_item.get(item_id, []),
            profiles,
            aspect_names,
            peer_splits,
            args.min_review_chars,
            args.importance_weight,
            args.sentiment_weight,
        )
        if negative is None:
            stats["skipped_no_negative"] += 1
            continue
        hard_peer, distance = negative
        item_bank = item_banks.get(item_id, {})
        existing_prompt = row.get("prompt")
        has_prompt = existing_prompt is not None and str(existing_prompt).strip()
        plan = row.get("personalized_evidence_plan")
        if not isinstance(plan, list):
            plan = row.get("item_plan")
        if not isinstance(plan, list):
            if item_bank:
                plan = build_personalized_evidence_plan(
                    user_profile,
                    item_bank,
                    aspect_names,
                    args.top_k_plan_items,
                    args.positive_sentiment_threshold,
                    args.negative_sentiment_threshold,
                    args.hedge_variance_threshold,
                )
            else:
                plan = []
        if not has_prompt and not plan:
            stats["skipped_missing_prompt_or_evidence_plan"] += 1
            continue
        user_aspects = row.get("top_user_aspects")
        if not isinstance(user_aspects, list):
            user_aspects = top_user_aspects(user_profile, args.top_k_user_aspects)
        prompt = build_prompt(row, items.get(item_id, {}), item_bank, user_aspects, plan)
        outputs.append(
            {
                "pair_id": build_pair_id(row["review_id"]),
                "split": row.get("split"),
                "user_id": user_id,
                "item_id": item_id,
                "review_id": row["review_id"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": hard_peer.get("review_text", ""),
                "pair_type": "same_item_counterfactual",
                "negative_type": "same_item_other_user",
                "rejected_user_id": hard_peer["user_id"],
                "rejected_review_id": hard_peer["review_id"],
                "profile_distance": distance,
                "top_user_aspects": user_aspects,
                "personalized_evidence_plan": plan,
                "rating": row.get("rating"),
                "title": row.get("title", ""),
                "item_title": items.get(item_id, {}).get("title", item_bank.get("item_title", row.get("item_title", ""))),
            }
        )
        stats["written_pairs"] += 1
        if args.max_pairs is not None and len(outputs) >= args.max_pairs:
            break
    return outputs, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build same-item counterfactual DPO pairs.")
    parser.add_argument("--sft_rows_path", default=None)
    parser.add_argument("--interactions_path", default=None)
    parser.add_argument("--review_pool_path", default=None)
    parser.add_argument("--items_path", default=None)
    parser.add_argument("--user_profiles_path", required=True)
    parser.add_argument("--item_evidence_bank_path", default=None)
    parser.add_argument("--aspect_lexicon_path", default=None)
    parser.add_argument("--lexicon_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_splits", default="train")
    parser.add_argument("--peer_splits", default="train")
    parser.add_argument("--min_review_chars", type=int, default=80)
    parser.add_argument("--top_k_user_aspects", type=int, default=4)
    parser.add_argument("--top_k_plan_items", type=int, default=5)
    parser.add_argument("--importance_weight", type=float, default=0.7)
    parser.add_argument("--sentiment_weight", type=float, default=0.3)
    parser.add_argument("--positive_sentiment_threshold", type=float, default=0.15)
    parser.add_argument("--negative_sentiment_threshold", type=float, default=-0.15)
    parser.add_argument("--hedge_variance_threshold", type=float, default=0.25)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs, stats = build_pairs(args)
    write_jsonl(args.output_path, outputs, overwrite=args.overwrite)
    print(f"Wrote {len(outputs):,} same-item counterfactual DPO pairs to {args.output_path}")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
