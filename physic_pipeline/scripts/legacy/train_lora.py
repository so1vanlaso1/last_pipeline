#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from the repository without installing the package first.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import inspect

def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT for Qwen3-8B on EXACT-FAMA data")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--eval_file", default="")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", default="artifacts/qwen3_8b_exact_lora")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Use fixed-step training; -1 means full epochs")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Reduce VRAM usage during training")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint folder to resume training")
    args = parser.parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Run: pip install -r requirements-train.txt\n"
            f"Original error: {exc}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
        trust_remote_code=True,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    ds = load_dataset("json", data_files=data_files)

    def render(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = ds.map(render, remove_columns=ds["train"].column_names, num_proc=args.num_proc)

    def tokenize(example):
        enc = tokenizer(example["text"], truncation=True, max_length=args.max_seq_length)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = ds.map(tokenize, remove_columns=["text"], num_proc=args.num_proc)

    has_eval = "validation" in tokenized
    save_strategy = "steps" if args.max_steps > 0 else "epoch"

    # Transformers has used both `evaluation_strategy` and `eval_strategy` across versions.
    import inspect
    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps if has_eval else None,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        report_to="none",
        group_by_length=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        ta_kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

        
    supported_args = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported_args}

    dropped_args = sorted(set(ta_kwargs.keys()) - set(filtered_ta_kwargs.keys()))
    if dropped_args:
        print(f"[train_lora] Dropped unsupported TrainingArguments: {dropped_args}")

    training_args = TrainingArguments(**filtered_ta_kwargs)   

    # training_args = TrainingArguments(**ta_kwargs)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") if has_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
