import argparse
import html
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


SYSTEM_PROMPT = """You are an evidence-plan-conditioned personalized review writer.
Write a natural Amazon-style review that reflects the target user's likely priorities and writing style.
Use the user's historical reviews only as examples of tone, verbosity, and phrasing.
Do not copy item-specific facts from historical examples.
All item-specific claims about the target item must be supported by the clean target metadata or the personalized evidence plan.
If evidence is negative, express it honestly as a concern rather than an unsupported positive claim.
Do not mention JSON, aspect labels, scores, or that you are following a plan."""


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


def review_id(row: Dict) -> str:
    rid = row.get("review_id")
    if rid is not None and str(rid).strip():
        return str(rid)
    return f"{row.get('user_id')}__{row.get('item_id')}__{row.get('timestamp', '')}"


def parse_splits(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def load_aspect_names(path: Optional[str], plans_path: str) -> List[str]:
    aspects = []
    if path:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        raw = obj
        if isinstance(obj, dict):
            raw = obj.get("lexicon") or obj.get("aspects") or obj.get("aspect_names") or []
        for item in raw:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("aspect_name") or item.get("name") or item.get("aspect") or "").strip()
            else:
                name = ""
            if name and name not in aspects:
                aspects.append(name)
    if aspects:
        return aspects
    for row in read_jsonl(plans_path):
        for item in row.get("personalized_evidence_plan", []) or []:
            aspect = str(item.get("aspect", "")).strip()
            if aspect and aspect not in aspects:
                aspects.append(aspect)
        for item in row.get("top_user_aspects", []) or []:
            aspect = str(item.get("aspect", "")).strip()
            if aspect and aspect not in aspects:
                aspects.append(aspect)
    if not aspects:
        raise ValueError("Could not infer aspect names. Provide --aspect_lexicon_path.")
    return aspects


def strip_html(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\?xml.*?\?>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(li|br|p|div|tr|td|th|ul|ol|table|center|blockquote|h1|h2|h3|h4|b|strong)[^>]*>", ". ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"file:///\S+", " ", text)
    text = re.sub(r"\b(mso-[A-Za-z-]+|font-family|font-size|border|padding|margin|width|height|background|color)\s*:[^.;]+[.;]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[^{}]*\}", " ", text)
    text = re.sub(r"\s*View larger\s*\.?,?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\.\s*){2,}", ". ", text)
    return re.sub(r"\s+", " ", text).strip()


def dedupe_sentences(text: str, max_sentences: int = 12) -> str:
    out = []
    seen = set()
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = clean_text(part)
        key = re.sub(r"[^a-z0-9]+", " ", part.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(part)
        if len(out) >= max_sentences:
            break
    return " ".join(out)


def truncate_words(text: str, max_words: int) -> str:
    text = clean_text(text)
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + " ..."


def clean_metadata_text(text: str, max_words: int) -> str:
    return truncate_words(dedupe_sentences(strip_html(text)), max_words)


def clean_categories(value, max_chars: int = 240) -> str:
    if isinstance(value, list):
        text = ", ".join(clean_metadata_text(str(x), 30) for x in value if clean_text(x))
    else:
        text = clean_metadata_text(str(value or ""), 60)
    chunks = []
    for chunk in re.split(r",|>", text):
        chunk = clean_text(chunk)
        if not chunk:
            continue
        if chunk.startswith("</") or chunk in {"span", "/span"}:
            continue
        if "style" in chunk.lower() or "font" in chunk.lower():
            continue
        chunks.append(chunk)
    return ", ".join(dict.fromkeys(chunks))[:max_chars]


def count_words(text: str) -> int:
    return len(clean_text(text).split())


def normalize_vec(vec: Dict[str, float], aspects: List[str]) -> List[float]:
    arr = [float(vec.get(a, 0.0)) for a in aspects]
    total = sum(abs(x) for x in arr)
    if total <= 1e-12:
        return [0.0 for _ in arr]
    return [x / total for x in arr]


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def plan_items(row: Dict) -> List[Dict]:
    return row.get("personalized_evidence_plan", []) or []


def plan_aspect_distribution(plan: List[Dict], aspects: List[str], use_score: bool) -> Dict[str, float]:
    counts = Counter()
    for item in plan:
        aspect = item.get("aspect")
        if aspect not in aspects:
            continue
        weight = float(item.get("score", 1.0)) if use_score else 1.0
        if item.get("action", item.get("decision")) in {"highlight", "warn"}:
            weight *= 1.25
        counts[aspect] += max(weight, 0.0)
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {aspect: counts[aspect] / total for aspect in aspects}


def load_review_aspect_distributions(path: Optional[str], aspects: List[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    counts = defaultdict(Counter)
    for row in tqdm(list(read_jsonl(path)), desc="Loading review aspect distributions", unit="row"):
        rid = str(row.get("review_id", ""))
        if not rid:
            continue
        for label in row.get("labels", []) or []:
            aspect = label.get("a")
            if aspect in aspects:
                counts[rid][aspect] += 1.0
    out = {}
    for rid, counter in counts.items():
        total = sum(counter.values())
        if total > 0:
            out[rid] = {aspect: counter[aspect] / total for aspect in aspects}
    return out


def load_interactions(path: str) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, Dict]]:
    rows = []
    by_user = defaultdict(list)
    by_review = {}
    for row in tqdm(list(read_jsonl(path)), desc="Loading interactions", unit="row"):
        row = dict(row)
        row["review_id"] = review_id(row)
        row["review_text"] = clean_text(row.get("review_text"))
        row["title"] = clean_text(row.get("title"))
        row["user_id"] = str(row.get("user_id"))
        row["item_id"] = str(row.get("item_id"))
        rows.append(row)
        by_user[row["user_id"]].append(row)
        by_review[row["review_id"]] = row
    for user_id in by_user:
        by_user[user_id].sort(key=lambda x: int(x.get("timestamp") or 0), reverse=True)
    return rows, dict(by_user), by_review


def compute_user_length_stats(by_user: Dict[str, List[Dict]], source_split: str, min_chars: int) -> Dict[str, Dict]:
    out = {}
    global_lengths = []
    for user_id, rows in by_user.items():
        lengths = [count_words(r.get("review_text")) for r in rows if r.get("split") == source_split and len(r.get("review_text", "")) >= min_chars]
        global_lengths.extend(lengths)
        if lengths:
            sorted_lengths = sorted(lengths)
            out[user_id] = {
                "avg_train_review_words": sum(lengths) / len(lengths),
                "median_train_review_words": sorted_lengths[len(sorted_lengths) // 2],
                "num_train_reviews_for_length": len(lengths),
            }
    if global_lengths:
        sorted_global = sorted(global_lengths)
        out["__GLOBAL__"] = {
            "avg_train_review_words": sum(global_lengths) / len(global_lengths),
            "median_train_review_words": sorted_global[len(sorted_global) // 2],
            "num_train_reviews_for_length": len(global_lengths),
        }
    else:
        out["__GLOBAL__"] = {"avg_train_review_words": 120.0, "median_train_review_words": 100.0, "num_train_reviews_for_length": 0}
    return out


def length_fit(candidate_words: int, user_avg_words: float) -> float:
    candidate_words = max(candidate_words, 1)
    user_avg_words = max(user_avg_words, 1.0)
    return math.exp(-abs(math.log(candidate_words / user_avg_words)))


def recency_fit(candidate_row: Dict, user_rows: List[Dict]) -> float:
    train_rows = [r for r in user_rows if r.get("split") == "train"]
    rid = candidate_row.get("review_id")
    for idx, row in enumerate(train_rows):
        if row.get("review_id") == rid:
            return 1.0 / (1.0 + idx)
    return 0.0


def select_style_examples(
    target_row: Dict,
    by_user: Dict[str, List[Dict]],
    review_aspect_dist: Dict[str, Dict[str, float]],
    aspects: List[str],
    num_examples: int,
    max_style_words: int,
    min_style_chars: int,
    exclude_same_item: bool,
    source_split: str,
    lambda_aspect: float,
    lambda_length: float,
    lambda_recency: float,
    use_plan_score: bool,
    user_length_stats: Dict[str, Dict],
) -> List[Dict]:
    user_id = str(target_row.get("user_id"))
    target_review_id = str(target_row.get("review_id"))
    target_item_id = str(target_row.get("item_id"))
    user_rows = by_user.get(user_id, [])
    query_dist = plan_aspect_distribution(plan_items(target_row), aspects, use_score=use_plan_score)
    query_vec = normalize_vec(query_dist, aspects)
    avg_words = float((user_length_stats.get(user_id) or user_length_stats.get("__GLOBAL__", {})).get("avg_train_review_words", 120.0))
    candidates = []
    for row in user_rows:
        if row.get("split") != source_split:
            continue
        if str(row.get("review_id")) == target_review_id:
            continue
        if exclude_same_item and str(row.get("item_id")) == target_item_id:
            continue
        text = clean_text(row.get("review_text"))
        if len(text) < min_style_chars:
            continue
        rid = str(row.get("review_id"))
        row_dist = review_aspect_dist.get(rid, {})
        row_vec = normalize_vec(row_dist, aspects)
        aspect_score = cosine(row_vec, query_vec) if any(query_vec) and any(row_vec) else 0.0
        len_score = length_fit(count_words(text), avg_words)
        rec_score = recency_fit(row, user_rows)
        score = lambda_aspect * aspect_score + lambda_length * len_score + lambda_recency * rec_score
        candidates.append({
            "row": row,
            "score": score,
            "aspect_score": aspect_score,
            "length_fit": len_score,
            "recency_fit": rec_score,
            "word_count": count_words(text),
            "review_aspect_distribution": row_dist,
        })
    candidates.sort(key=lambda x: (x["score"], x["aspect_score"], x["length_fit"], x["recency_fit"]), reverse=True)
    selected = []
    for item in candidates[:num_examples]:
        row = item["row"]
        selected.append({
            "review_id": row.get("review_id"),
            "item_id": row.get("item_id"),
            "rating": row.get("rating"),
            "title": clean_text(row.get("title")),
            "text": truncate_words(row.get("review_text", ""), max_style_words),
            "score": item["score"],
            "aspect_score": item["aspect_score"],
            "length_fit": item["length_fit"],
            "recency_fit": item["recency_fit"],
            "word_count": item["word_count"],
            "review_aspect_distribution": item["review_aspect_distribution"],
        })
    return selected


def render_user_profile(top_user_aspects: List[Dict]) -> str:
    if not top_user_aspects:
        return "- No reliable user aspect profile is available; use a cautious general tone."
    lines = []
    for item in top_user_aspects:
        lines.append(
            f"- {item.get('aspect')}: importance={float(item.get('importance', 0.0)):.4f}, "
            f"user_sentiment={float(item.get('sentiment', 0.0)):.4f}, "
            f"difference_from_global={float(item.get('difference_from_global', 0.0)):.4f}"
        )
    return "\n".join(lines)


def target_length_instruction(avg_words: float) -> str:
    if avg_words <= 60:
        return f"The user's train reviews are usually short, about {avg_words:.0f} words. Write a concise review."
    if avg_words <= 180:
        return f"The user's train reviews are usually medium length, about {avg_words:.0f} words. Write a medium-length review."
    return f"The user's train reviews are often long, about {avg_words:.0f} words. Write a fuller review, but avoid repetition."


def render_style_examples(examples: List[Dict]) -> str:
    if not examples:
        return "- No style examples are available. Use the user aspect profile and rating/title context."
    lines = [
        "The following reviews are from the same user's train history.",
        "Use them only for tone, verbosity, phrasing, and review structure.",
        "Do not copy item-specific facts from these examples.",
    ]
    for idx, example in enumerate(examples, start=1):
        title = json.dumps(example.get("title", ""), ensure_ascii=False)
        lines.append(f"\nExample {idx} | past rating={example.get('rating')} | past review title={title}:\n{example.get('text', '')}")
    return "\n".join(lines)


def render_evidence_plan(plan: List[Dict], max_items: int) -> str:
    if not plan:
        return "- No strong personalized evidence item is available. Avoid specific unsupported item claims."
    lines = []
    for idx, item in enumerate(plan[:max_items], start=1):
        action = item.get("action", item.get("decision", "mention"))
        lines.append(
            f"{idx}. action={action} | aspect={item.get('aspect')} | polarity={item.get('polarity')} | "
            f"descriptor={json.dumps(item.get('descriptor', ''), ensure_ascii=False)} | "
            f"evidence={json.dumps(item.get('evidence', ''), ensure_ascii=False)} | "
            f"reason={json.dumps(item.get('reason', ''), ensure_ascii=False)}"
        )
    return "\n".join(lines)


def build_user_prompt(row: Dict, style_examples: List[Dict], max_plan_items: int, max_description_words: int, user_length_stats: Dict[str, Dict]) -> str:
    item_title = clean_text(row.get("item_title"))
    item_description = clean_metadata_text(row.get("item_description", ""), max_description_words)
    categories = clean_categories(row.get("item_categories", ""))
    user_id = str(row.get("user_id"))
    avg_words = float((user_length_stats.get(user_id) or user_length_stats.get("__GLOBAL__", {})).get("avg_train_review_words", 120.0))
    rating = row.get("rating")
    review_title = clean_text(row.get("review_title"))
    plan = plan_items(row)
    empty_note = ""
    if not plan:
        empty_note = "\nImportant: The personalized evidence plan is empty. Do not assert target-item-specific qualities unless they are clearly present in the clean metadata."
    return f"""Write a personalized Amazon-style review for the target user and item.

[Target Item]
- title: {item_title}
- categories: {categories}
- clean description: {item_description}

[Target User Multi-Aspect Preference]
{render_user_profile(row.get("top_user_aspects", []))}

[Target User Writing Pattern]
- length guidance: {target_length_instruction(avg_words)}

[User Writing Examples: Style Only]
{render_style_examples(style_examples)}

[Personalized Evidence Plan]
{render_evidence_plan(plan, max_items=max_plan_items)}

[Output Conditions]
- rating context: {rating}
- review title context: {review_title}

[Writing Requirements]
- Write one coherent review in the user's likely style.
- Match the user's tone, verbosity, and phrasing using the style examples.
- Use personalized evidence plan items naturally when available.
- If an item is marked warn, include that concern honestly.
- If an item is marked hedge, use cautious language.
- Do not copy facts from style examples; they are not about the target item.
- Do not invent target-item-specific facts beyond the clean metadata and personalized evidence plan.
- Do not output bullets, JSON, or aspect labels.
- Avoid repetitive phrasing.
- Output only the review text.{empty_note}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generator SFT rows with personalized evidence plans and style examples.")
    parser.add_argument("--personalized_evidence_plans_path", dest="plans_path", required=True)
    parser.add_argument("--interactions_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="generator_sft")
    parser.add_argument("--aspect_labels_path", "--asde_sentences_path", "--labeled_sentences_path", dest="aspect_labels_path", default=None)
    parser.add_argument("--aspect_lexicon_path", "--lexicon_path", dest="aspect_lexicon_path", default=None)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--num_style_examples", type=int, default=5)
    parser.add_argument("--style_source_split", default="train")
    parser.add_argument("--allow_same_item_style", action="store_true")
    parser.add_argument("--max_style_words", type=int, default=140)
    parser.add_argument("--min_style_chars", type=int, default=80)
    parser.add_argument("--max_description_words", type=int, default=160)
    parser.add_argument("--max_plan_items", type=int, default=5)
    parser.add_argument("--min_target_chars", type=int, default=40)
    parser.add_argument("--lambda_aspect", type=float, default=0.70)
    parser.add_argument("--lambda_length", type=float, default=0.25)
    parser.add_argument("--lambda_recency", type=float, default=0.05)
    parser.add_argument("--use_plan_score_for_aspect_query", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits = parse_splits(args.splits)
    aspects = load_aspect_names(args.aspect_lexicon_path, args.plans_path)
    _, by_user, _ = load_interactions(args.interactions_path)
    user_length_stats = compute_user_length_stats(by_user, args.style_source_split, args.min_style_chars)
    review_aspect_dist = load_review_aspect_distributions(args.aspect_labels_path, aspects)
    rows_by_split = defaultdict(list)
    all_rows = []
    stats = Counter()
    selected_counts = Counter()

    for row in tqdm(list(read_jsonl(args.plans_path)), desc="Building generator SFT rows", unit="row"):
        split = row.get("split")
        if splits and split not in splits:
            continue
        target = clean_text(row.get("target_review_text"))
        if len(target) < args.min_target_chars:
            stats["skip_short_target"] += 1
            continue
        row = dict(row)
        row["review_id"] = str(row.get("review_id"))
        style_examples = select_style_examples(
            target_row=row,
            by_user=by_user,
            review_aspect_dist=review_aspect_dist,
            aspects=aspects,
            num_examples=args.num_style_examples,
            max_style_words=args.max_style_words,
            min_style_chars=args.min_style_chars,
            exclude_same_item=not args.allow_same_item_style,
            source_split=args.style_source_split,
            lambda_aspect=args.lambda_aspect,
            lambda_length=args.lambda_length,
            lambda_recency=args.lambda_recency,
            use_plan_score=args.use_plan_score_for_aspect_query,
            user_length_stats=user_length_stats,
        )
        selected_counts[len(style_examples)] += 1
        prompt = build_user_prompt(row, style_examples, args.max_plan_items, args.max_description_words, user_length_stats)
        plan = plan_items(row)
        out = {
            "id": row.get("plan_id", row.get("review_id")),
            "split": split,
            "user_id": row.get("user_id"),
            "item_id": row.get("item_id"),
            "review_id": row.get("review_id"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target},
            ],
            "messages_prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "prompt": prompt,
            "target": target,
            "rating": row.get("rating"),
            "review_title": clean_text(row.get("review_title")),
            "item_title": clean_text(row.get("item_title")),
            "clean_item_description": clean_metadata_text(row.get("item_description", ""), args.max_description_words),
            "top_user_aspects": row.get("top_user_aspects", []),
            "personalized_evidence_plan": plan,
            "style_examples": style_examples,
            "user_length_profile": user_length_stats.get(str(row.get("user_id"))) or user_length_stats.get("__GLOBAL__", {}),
            "prompt_version": "maple_personalized_evidence_plan_style_examples",
            "style_selection_config": {
                "num_style_examples": args.num_style_examples,
                "lambda_aspect": args.lambda_aspect,
                "lambda_length": args.lambda_length,
                "lambda_recency": args.lambda_recency,
                "exclude_same_item_style": not args.allow_same_item_style,
                "style_source_split": args.style_source_split,
                "max_style_words": args.max_style_words,
                "min_style_chars": args.min_style_chars,
                "use_plan_score_for_aspect_query": args.use_plan_score_for_aspect_query,
            },
        }
        rows_by_split[split].append(out)
        all_rows.append(out)
        stats[f"split_{split}"] += 1
        if not style_examples:
            stats["rows_without_style_examples"] += 1
        if not plan:
            stats["empty_plan_rows"] += 1

    for split in sorted(splits):
        write_jsonl(os.path.join(args.output_dir, f"{args.prefix}_{split}.jsonl"), rows_by_split.get(split, []), overwrite=args.overwrite)
    all_path = os.path.join(args.output_dir, f"{args.prefix}_all.jsonl")
    write_jsonl(all_path, all_rows, overwrite=args.overwrite)
    summary = {
        "num_rows": len(all_rows),
        "stats": dict(stats),
        "selected_example_count_distribution": dict(selected_counts),
        "num_aspects": len(aspects),
        "aspects": aspects,
        "paths": {
            "personalized_evidence_plans_path": args.plans_path,
            "interactions_path": args.interactions_path,
            "aspect_labels_path": args.aspect_labels_path,
            "output_dir": args.output_dir,
        },
        "config": vars(args),
    }
    write_json(os.path.join(args.output_dir, f"{args.prefix}_summary.json"), summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
