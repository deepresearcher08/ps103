"""QLoRA finetune of a small open-source LLM on the MoSPI explanation dataset.

Target: a ~3B model (e.g. Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct) so 4-bit QLoRA
fits on a 6GB laptop GPU. Output is a PEFT adapter + merged GGUF for Ollama.

Usage:
  pip install -r requirements-finetune.txt
  python -m src.finetune                    # train + merge adapter
  python -m src.finetune --export-gguf      # also build GGUF for `ollama create`

Notes:
  - 4-bit QLoRA (bitsandbytes) keeps memory low.
  - Record `model_name` so `explainer.py` knows which LoRA/dedicated Ollama name to load.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "finetune_train.jsonl"
OUT_DIR = ROOT / "models" / "mospi-llm"

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM = (
    "You are MoSPI's infrastructure project risk analyst. Given a JSON of a project's "
    "risk features, explain concisely why it is at risk and what to do, in 2-3 sentences."
)


def _format(example: dict) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['output']}<|im_end|>"
    )


def load_dataset(tokenizer, path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    texts = [_format(r) for r in rows]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=768,
        padding=False,
        return_tensors=None,
    )
    ds = Dataset.from_list([{"input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}
                            for t in tokenized])
    ds = ds.map(lambda b: {"labels": b["input_ids"]}, batched=True)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-acc", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--export-gguf", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(tok, Path(args.data))
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=None,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))

    # merge adapter back into the base for Ollama GGUF export
    merged = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    from peft import PeftModel
    merged = PeftModel.from_pretrained(merged, str(out / "adapter")).merge_and_unload()
    merged.save_pretrained(str(out / "merged"))
    tok.save_pretrained(str(out / "merged"))

    cfg = {"base_model": args.model, "adapter": str(out / "adapter"), "merged": str(out / "merged")}
    (out / "info.json").write_text(json.dumps(cfg, indent=2))
    print("Adapter + merged saved ->", out)
    print("To import into Ollama: convert 'merged' to GGUF (llama.cpp convert.py) then")
    print("  ollama create mospi-llm -f Modelfile   (pointing to ./merged)")


if __name__ == "__main__":
    main()