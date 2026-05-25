import argparse
import inspect
import json
import os
from typing import Dict, Iterable


SYSTEM_PROMPT = """You are an evidence-plan-conditioned personalized review writer.
Write a natural review that matches the target user's likely priorities.
Use item-specific claims only when they are supported by the provided metadata or personalized evidence plan.
Do not invent unsupported item facts."""


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


def latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-")[-1])
        except Exception:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def dpo_config_kwargs(config_cls, kwargs: Dict) -> Dict:
    params = inspect.signature(config_cls.__init__).parameters
    working = dict(kwargs)
    if "eval_strategy" in params and "evaluation_strategy" in working:
        working["eval_strategy"] = working.pop("evaluation_strategy")
    if "evaluation_strategy" in params and "eval_strategy" in working:
        working["evaluation_strategy"] = working.pop("eval_strategy")
    return {k: v for k, v in working.items() if k in params}


def build_chat_prompt(row: Dict, tokenizer) -> str:
    messages = row.get("messages_prompt")
    if isinstance(messages, list) and messages:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = row.get("prompt", "")
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_dpo_dataset(path: str, tokenizer, dataset_cls, max_samples: int = 0):
    rows = []
    for idx, row in enumerate(read_jsonl(path)):
        rows.append({
            "prompt": build_chat_prompt(row, tokenizer),
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "negative_type": row.get("negative_type", ""),
            "pair_type": row.get("pair_type", ""),
            "id": row.get("id", str(idx)),
        })
        if max_samples > 0 and len(rows) >= max_samples:
            break
    return dataset_cls.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA DPO training for the personalized review generator.")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--max_prompt_length", type=int, default=2300)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--report_to", default="tensorboard")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dpo_dataset(args.train_path, tokenizer, Dataset, max_samples=args.max_samples)
    eval_dataset = load_dpo_dataset(args.eval_path, tokenizer, Dataset) if args.eval_path else None
    raw_dpo_kwargs = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "optim": "paged_adamw_32bit" if args.load_in_4bit else "adamw_torch",
        "report_to": args.report_to.split(",") if args.report_to else [],
        "remove_unused_columns": False,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "loss_type": "sigmoid",
    }
    dpo_args = DPOConfig(**dpo_config_kwargs(DPOConfig, raw_dpo_kwargs))
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "auto":
        resume_checkpoint = latest_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint if resume_checkpoint else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
