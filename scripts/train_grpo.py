#!/usr/bin/env python
from __future__ import annotations
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, yaml
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from utils.data_utils import get_gsm8k, SYSTEM_PROMPT
from utils.rewards import correctness_reward, int_reward, strict_xml_reward, soft_xml_reward, xml_count_reward
import wandb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--output_dir", default="outputs", type=str)
    p.add_argument("--max_steps", default=None, type=int)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps:
        cfg["max_steps"] = args.max_steps

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        device_map="auto",
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        fast_inference=cfg["fast_inference"],
        max_lora_rank=cfg["lora_rank"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg["lora_rank"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train_ds = get_gsm8k("train")

    training_args = GRPOConfig(
        learning_rate=cfg["learning_rate"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=cfg["log_every"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_seq_length"] // 4,
        max_completion_length=cfg["max_seq_length"] // 4 * 3,
        num_train_epochs=1,
        max_steps=cfg["max_steps"],
        save_steps=cfg["save_every"],
        max_grad_norm=cfg["max_grad_norm"],
        report_to=cfg["report_to"],
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[xml_count_reward, soft_xml_reward, strict_xml_reward, int_reward, correctness_reward],
        args=training_args,
        train_dataset=train_ds,
    )

    # wandb.login()
    # wandb.init(project=cfg["wandb_project"])
    trainer.train()

if __name__ == "__main__":
    main()
