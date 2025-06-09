#!/usr/bin/env python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
from unsloth import FastLanguageModel
from utils.data_utils import SYSTEM_PROMPT

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--max_tokens", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_tokens + 256,
        fast_inference=True,
        load_in_4bit=True,
        device_map="auto",
    )
    model.eval()

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.question},
    ]
    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        ids = model.generate(inputs, max_new_tokens=args.max_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(ids[0][inputs.shape[-1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
