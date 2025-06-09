#!/usr/bin/env python
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, json
from datasets import load_dataset
from tqdm import tqdm
import torch
from unsloth import FastLanguageModel
from utils.data_utils import SYSTEM_PROMPT, extract_hash_answer, extract_xml_answer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=2048,
        fast_inference=True,
        load_in_4bit=True,
        device_map="auto",
    )
    model.eval()
    ds = load_dataset("openai/gsm8k", "main", split="test[:100]")

    results, num_correct = [], 0
    for i, ex in enumerate(tqdm(ds, desc="Evaluating")):
        gold = extract_hash_answer(ex["answer"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["question"]}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            ids = model.generate(inputs, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(ids[0][inputs.shape[-1]:], skip_special_tokens=True)
        pred = extract_xml_answer(completion)
        correct = pred == gold
        num_correct += int(correct)
        results.append({"q": ex["question"], "gold": gold, "pred": pred, "full": completion, "correct": correct})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Accuracy: {num_correct / len(results):.2%}")

if __name__ == "__main__":
    main()
