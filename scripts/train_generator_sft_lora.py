import argparse
import inspect
import json
import os
from typing import Dict, Iterable, List, Optional


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


class ChatSFTDataset:
    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows = list(read_jsonl(path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import torch

        row = self.rows[idx]
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Each SFT row must contain messages ending with an assistant message.")
        prompt_text = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.max_length:
            overflow = len(full_ids) - self.max_length
            full_ids = full_ids[overflow:]
            prompt_len = max(0, len(prompt_ids) - overflow)
        else:
            prompt_len = len(prompt_ids)
        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class CausalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]):
        import torch

        max_len = max(len(x["input_ids"]) for x in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for item in features:
            n = len(item["input_ids"])
            pad_n = max_len - n
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_n, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad_n,), -100, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def training_arguments_kwargs(training_arguments_cls, kwargs: Dict) -> Dict:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    working = dict(kwargs)
    if "eval_strategy" in params and "evaluation_strategy" in working:
        working["eval_strategy"] = working.pop("evaluation_strategy")
    if "evaluation_strategy" in params and "eval_strategy" in working:
        working["evaluation_strategy"] = working.pop("eval_strategy")
    return {k: v for k, v in working.items() if k in params}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for the personalized review generator.")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "right"
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

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

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

    train_dataset = ChatSFTDataset(args.train_path, tokenizer, max_length=args.max_length)
    eval_dataset = ChatSFTDataset(args.eval_path, tokenizer, max_length=args.max_length) if args.eval_path else None
    collator = CausalDataCollator(tokenizer)

    raw_training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
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
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": 2,
    }
    training_args = TrainingArguments(**training_arguments_kwargs(TrainingArguments, raw_training_kwargs))
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint == "auto":
        resume_checkpoint = latest_checkpoint(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint if resume_checkpoint else None)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved SFT LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
