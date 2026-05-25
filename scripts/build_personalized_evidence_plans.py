import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def load_jsonl_map(path: str, key: str) -> Dict[str, Dict]:
    out = {}
    for row in read_jsonl(path):
        value = row.get(key)
        if value is not None:
            out[str(value)] = row
    return out


def load_aspect_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj
    if isinstance(obj, dict):
        raw = obj.get("lexicon") or obj.get("aspects") or obj.get("aspect_names") or []
    aspects = []
    for item in raw:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("aspect_name") or item.get("name") or item.get("aspect") or "").strip()
        else:
            name = ""
        if name and name not in aspects:
            aspects.append(name)
    if not aspects:
        raise ValueError(f"No aspect names found in {path}")
    return aspects


def parse_splits(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    return re.sub(r"\s+", " ", str(value)).strip()


def review_id(row: Dict) -> str:
    rid = row.get("review_id")
    if rid is not None and str(rid).strip():
        return str(rid)
    return f"{row.get('user_id')}__{row.get('item_id')}__{row.get('timestamp', '')}"


def to_float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def evidence_text(example: Dict, descriptor: str) -> str:
    evidence = example.get("evidence")
    if isinstance(evidence, dict):
        text = clean_text(evidence.get("text"))
    else:
        text = clean_text(evidence)
    return text or clean_text(example.get("descriptor")) or descriptor


def descriptor_entries(aspect_bank: Dict, polarity: int) -> List[Dict]:
    if polarity > 0:
        buckets = ["positive_descriptors", "neutral_descriptors", "negative_descriptors"]
    elif polarity < 0:
        buckets = ["negative_descriptors", "neutral_descriptors", "positive_descriptors"]
    else:
        buckets = ["neutral_descriptors", "positive_descriptors", "negative_descriptors"]
    entries = []
    for bucket in buckets:
        for item in aspect_bank.get(bucket, []) or []:
            if isinstance(item, str):
                entries.append({"descriptor": item, "count": 1, "examples": []})
            elif isinstance(item, dict):
                entries.append(item)
    return entries


def valid_descriptor(text: str, min_words: int, max_words: int) -> bool:
    text = clean_text(text)
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    generic = {
        "great", "good", "bad", "fun", "nice", "okay", "ok", "excellent", "terrible",
        "great game", "good game", "bad game", "fun game", "great book", "good book",
        "this game", "this book", "must have", "worth it"
    }
    return normalized not in generic


def choose_descriptor(
    aspect: str,
    aspect_bank: Dict,
    polarity: int,
    forbidden_review_ids: set[str],
    min_descriptor_words: int,
    max_descriptor_words: int,
) -> Optional[Dict]:
    for item in descriptor_entries(aspect_bank, polarity):
        descriptor = clean_text(item.get("descriptor"))
        if not valid_descriptor(descriptor, min_descriptor_words, max_descriptor_words):
            continue
        examples = item.get("examples", []) or []
        if not examples:
            return {
                "descriptor": descriptor,
                "evidence": descriptor,
                "source_sentence": "",
                "source_review_id": None,
                "source_sent_id": None,
            }
        for example in examples:
            rid = str(example.get("review_id", ""))
            if rid and rid in forbidden_review_ids:
                continue
            ev = evidence_text(example, descriptor)
            if not valid_descriptor(ev, min_descriptor_words, max_descriptor_words + 8):
                ev = descriptor
            return {
                "descriptor": descriptor,
                "evidence": ev,
                "source_sentence": clean_text(example.get("sentence")),
                "source_review_id": rid or None,
                "source_sent_id": example.get("sent_id"),
            }
    return None


def decide_action(
    user_importance: float,
    mean_sentiment: float,
    variance: float,
    importance_high: float,
    importance_mid: float,
    positive_threshold: float,
    negative_threshold: float,
    hedge_variance_threshold: float,
) -> Tuple[str, int]:
    if mean_sentiment >= positive_threshold:
        polarity = 1
    elif mean_sentiment <= negative_threshold:
        polarity = -1
    else:
        polarity = 0
    if variance >= hedge_variance_threshold and abs(mean_sentiment) < 0.50:
        return "hedge", polarity
    if polarity > 0 and user_importance >= importance_mid:
        return "highlight", polarity
    if polarity < 0 and user_importance >= importance_mid:
        return "warn", polarity
    if user_importance >= importance_high and polarity == 0:
        return "mention", polarity
    return "mention", polarity


def action_reason(action: str, aspect: str) -> str:
    if action == "highlight":
        return f"user cares about {aspect} and the item evidence is positive"
    if action == "warn":
        return f"user cares about {aspect} and the item evidence is negative"
    if action == "hedge":
        return f"available evidence about {aspect} varies across reviews"
    return f"item evidence is available for {aspect}"


def summarize_user_profile(profile: Dict, aspect_names: List[str], top_k: int) -> List[Dict]:
    importance = profile.get("aspect_importance_shrunk") or profile.get("aspect_importance") or {}
    sentiment = profile.get("aspect_sentiment_shrunk") or profile.get("aspect_sentiment") or {}
    difference = profile.get("aspect_difference_from_global") or {}
    rows = []
    for aspect in aspect_names:
        rows.append({
            "aspect": aspect,
            "importance": to_float(importance.get(aspect)),
            "sentiment": to_float(sentiment.get(aspect)),
            "difference_from_global": to_float(difference.get(aspect)),
        })
    rows.sort(key=lambda x: (x["importance"], abs(x["difference_from_global"])), reverse=True)
    return rows[:top_k]


def build_plan(
    profile: Dict,
    bank: Dict,
    aspect_names: List[str],
    top_k: int,
    min_mention_rate: float,
    min_score: float,
    importance_high: float,
    importance_mid: float,
    positive_threshold: float,
    negative_threshold: float,
    hedge_variance_threshold: float,
    forbidden_review_ids: set[str],
    min_descriptor_words: int,
    max_descriptor_words: int,
) -> List[Dict]:
    importance = profile.get("aspect_importance_shrunk") or profile.get("aspect_importance") or {}
    sentiment = profile.get("aspect_sentiment_shrunk") or profile.get("aspect_sentiment") or {}
    aspects_obj = bank.get("aspects") or {}
    candidates = []
    for aspect in aspect_names:
        aspect_bank = aspects_obj.get(aspect) or {}
        mention_rate = to_float(aspect_bank.get("mention_rate"))
        mean_sentiment = to_float(aspect_bank.get("mean_sentiment"))
        variance = to_float(aspect_bank.get("sentiment_variance"))
        support_reviews = to_int(aspect_bank.get("support_reviews"))
        support_labels = to_int(aspect_bank.get("support_labels"))
        user_importance = to_float(importance.get(aspect))
        user_sentiment = to_float(sentiment.get(aspect))
        if support_labels <= 0 or support_reviews <= 0 or mention_rate < min_mention_rate:
            continue
        action, polarity = decide_action(
            user_importance=user_importance,
            mean_sentiment=mean_sentiment,
            variance=variance,
            importance_high=importance_high,
            importance_mid=importance_mid,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            hedge_variance_threshold=hedge_variance_threshold,
        )
        descriptor = choose_descriptor(
            aspect=aspect,
            aspect_bank=aspect_bank,
            polarity=polarity,
            forbidden_review_ids=forbidden_review_ids,
            min_descriptor_words=min_descriptor_words,
            max_descriptor_words=max_descriptor_words,
        )
        if descriptor is None:
            continue
        score = (0.20 + user_importance) * mention_rate * (1.0 + abs(mean_sentiment)) * math.log1p(max(support_labels, support_reviews))
        if score < min_score:
            continue
        candidates.append({
            "aspect": aspect,
            "action": action,
            "descriptor": descriptor["descriptor"],
            "evidence": descriptor["evidence"],
            "polarity": polarity,
            "reason": action_reason(action, aspect),
            "source_sentence": descriptor["source_sentence"],
            "source_review_id": descriptor["source_review_id"],
            "source_sent_id": descriptor["source_sent_id"],
            "user_importance": user_importance,
            "user_sentiment": user_sentiment,
            "item_mention_rate": mention_rate,
            "item_mean_sentiment": mean_sentiment,
            "item_sentiment_variance": variance,
            "support_reviews": support_reviews,
            "support_labels": support_labels,
            "score": score,
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def categories_text(row: Dict, bank: Dict) -> str:
    value = row.get("categories", row.get("category", bank.get("categories", [])))
    if isinstance(value, list):
        return ", ".join(clean_text(x) for x in value if clean_text(x))
    return clean_text(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build personalized evidence plans from user profiles and item evidence banks.")
    parser.add_argument("--interactions_path", required=True)
    parser.add_argument("--items_path", required=True)
    parser.add_argument("--user_profiles_path", required=True)
    parser.add_argument("--item_evidence_bank_path", required=True)
    parser.add_argument("--aspect_lexicon_path", "--lexicon_path", dest="aspect_lexicon_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_splits", default="train,val,test")
    parser.add_argument("--top_k_user_aspects", type=int, default=5)
    parser.add_argument("--top_k_plan_items", type=int, default=5)
    parser.add_argument("--min_review_chars", type=int, default=20)
    parser.add_argument("--min_mention_rate", type=float, default=0.0)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--importance_high", type=float, default=0.12)
    parser.add_argument("--importance_mid", type=float, default=0.06)
    parser.add_argument("--positive_sentiment_threshold", type=float, default=0.15)
    parser.add_argument("--negative_sentiment_threshold", type=float, default=-0.15)
    parser.add_argument("--hedge_variance_threshold", type=float, default=0.60)
    parser.add_argument("--min_descriptor_words", type=int, default=2)
    parser.add_argument("--max_descriptor_words", type=int, default=12)
    parser.add_argument("--allow_target_review_evidence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    aspect_names = load_aspect_names(args.aspect_lexicon_path)
    profiles = load_jsonl_map(args.user_profiles_path, "user_id")
    items = load_jsonl_map(args.items_path, "item_id")
    banks = load_jsonl_map(args.item_evidence_bank_path, "item_id")
    target_splits = parse_splits(args.target_splits)
    outputs = []
    stats = Counter()

    for row in tqdm(list(read_jsonl(args.interactions_path)), desc="Building personalized evidence plans", unit="row"):
        split = row.get("split")
        if target_splits and split not in target_splits:
            continue
        target_text = clean_text(row.get("review_text") or row.get("target_review_text"))
        if len(target_text) < args.min_review_chars:
            stats["skip_short_review"] += 1
            continue
        user_id = str(row.get("user_id"))
        item_id = str(row.get("item_id"))
        rid = review_id(row)
        profile = profiles.get(user_id)
        bank = banks.get(item_id)
        if profile is None:
            stats["skip_missing_profile"] += 1
            continue
        if bank is None:
            stats["skip_missing_item_evidence_bank"] += 1
            continue
        forbidden = set() if args.allow_target_review_evidence else {rid}
        plan = build_plan(
            profile=profile,
            bank=bank,
            aspect_names=aspect_names,
            top_k=args.top_k_plan_items,
            min_mention_rate=args.min_mention_rate,
            min_score=args.min_score,
            importance_high=args.importance_high,
            importance_mid=args.importance_mid,
            positive_threshold=args.positive_sentiment_threshold,
            negative_threshold=args.negative_sentiment_threshold,
            hedge_variance_threshold=args.hedge_variance_threshold,
            forbidden_review_ids=forbidden,
            min_descriptor_words=args.min_descriptor_words,
            max_descriptor_words=args.max_descriptor_words,
        )
        item_row = items.get(item_id, {})
        outputs.append({
            "plan_id": f"{rid}__plan",
            "review_id": rid,
            "user_id": user_id,
            "item_id": item_id,
            "split": split,
            "rating": row.get("rating"),
            "review_title": row.get("title", row.get("review_title", "")),
            "target_review_text": target_text,
            "timestamp": row.get("timestamp"),
            "item_title": clean_text(item_row.get("title") or item_row.get("item_title") or bank.get("item_title")),
            "item_description": clean_text(item_row.get("description") or bank.get("description")),
            "item_categories": categories_text(item_row, bank),
            "top_user_aspects": summarize_user_profile(profile, aspect_names, args.top_k_user_aspects),
            "personalized_evidence_plan": plan,
            "num_plan_items": len(plan),
            "target_review_evidence_allowed": bool(args.allow_target_review_evidence),
            "profile_num_reviews": profile.get("num_reviews"),
            "profile_shrinkage_alpha": profile.get("shrinkage_alpha"),
        })
        stats["written"] += 1
        if not plan:
            stats["empty_plan"] += 1

    write_jsonl(args.output_path, outputs, overwrite=args.overwrite)
    print(json.dumps({"num_rows": len(outputs), "stats": dict(stats)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
