from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _format_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
    assistant = messages[1].get("content", "") if isinstance(messages[1], dict) else ""
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    if not user.strip() or not assistant.strip():
        return None
    return user.strip(), assistant.strip()


def _build_text(tokenizer: Any, user: str, assistant: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return f"User: {user}\nAssistant: {assistant}"


def _auto_target_modules(model: Any) -> list[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "fc_in",
        "fc_out",
    ]
    names = [name for name, _ in model.named_modules()]
    hits: list[str] = []
    for cand in preferred:
        if any(n.endswith(cand) for n in names):
            hits.append(cand)
    if hits:
        # Keep stable order, dedup.
        out: list[str] = []
        seen = set()
        for x in hits:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out
    # Last-resort fallback for unknown architectures.
    return ["c_attn", "c_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT trainer for tool-calling dataset.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-file", default="finetune/train_tool_use_full.jsonl")
    parser.add_argument("--val-file", default="finetune/val_tool_use_full.jsonl")
    parser.add_argument("--output-dir", default="finetune/output/lora_tool_use")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except Exception as e:
        print("Missing training dependencies.")
        print("Install with:")
        print("  pip install -U torch transformers datasets peft trl accelerate")
        print(f"Import error: {e}")
        raise SystemExit(1)

    train_path = Path(args.train_file)
    val_path = Path(args.val_file)
    if not train_path.exists():
        raise SystemExit(f"train file not found: {train_path}")
    if not val_path.exists():
        raise SystemExit(f"val file not found: {val_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    target_modules = _auto_target_modules(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    train_texts: list[str] = []
    for row in train_rows:
        pair = _format_pair(row)
        if not pair:
            continue
        train_texts.append(_build_text(tokenizer, pair[0], pair[1]))

    val_texts: list[str] = []
    for row in val_rows:
        pair = _format_pair(row)
        if not pair:
            continue
        val_texts.append(_build_text(tokenizer, pair[0], pair[1]))

    if not train_texts:
        raise SystemExit("No train samples available after formatting.")

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts if val_texts else train_texts[: max(1, len(train_texts) // 50)]})

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=["text"])

    try:
        import inspect

        sig = inspect.signature(TrainingArguments.__init__)
        strategy_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
    except Exception:
        strategy_key = "evaluation_strategy"

    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": max(1, args.batch_size),
        "gradient_accumulation_steps": args.grad_accum,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.save_steps,
        "save_strategy": "steps",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": [],
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": True,
        strategy_key: "steps",
    }
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
