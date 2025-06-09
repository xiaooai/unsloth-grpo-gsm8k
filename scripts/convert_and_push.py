#!/usr/bin/env python
import argparse
from unsloth import FastLanguageModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--hf_repo", required=True)
    p.add_argument("--hf_token", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=2048,
        fast_inference=True,
        device_map="auto",
    )
    model.save_pretrained_merge("model", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(args.hf_repo, tokenizer, save_method="merged_16bit", token=args.hf_token)
    model.push_to_hub_gguf(args.hf_repo, tokenizer, quantization_method=["q4_k_m", "q5_k_m", "q8_0"], token=args.hf_token)

if __name__ == "__main__":
    main()
