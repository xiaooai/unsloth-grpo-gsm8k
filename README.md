
# Unsloth-GRPO-GSM8K

这是一个参考实现项目，使用 **GRPO（Guided Reinforcement Policy Optimization）** 算法对 **Google Gemma-3B-IT** 模型在 **GSM8K** 小学数学推理数据集上进行微调。项目支持：

- 💬 使用结构化的 XML 输出格式（`<reasoning>...</reasoning><answer>...</answer>`）
- 🎯 定义多种奖励函数（正确性、格式规范、是否为整数、XML标签匹配等）
- ⚙️ 基于 [Unsloth](https://github.com/unslothai/unsloth) 高效实现 LoRA 微调（支持 4-bit）
- 🧪 支持对 GSM8K 测试集进行评估
- ☁️ 模型权重合并、量化（GGUF）并上传至 Hugging Face Hub

---

## 🔍 什么是 GRPO？

GRPO 是一种面向奖励引导的微调算法（Guided Reinforcement Policy Optimization），相比 PPO 或 DPO，它支持多个同时生效的奖励函数，可以引导模型输出更规范、结构更清晰、符合任务需求的回答。

适用于如下场景：
- 输出必须有固定结构，例如 XML / JSON / SQL / 代码片段
- 需要引导模型输出特定格式，如结尾带换行或特定 token
- 用于数学、逻辑、编程等高精度场景

---

## 📚 什么是 GSM8K？

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) 是由 OpenAI 提供的小学数学问答数据集，适用于检验模型的数学推理和计算能力。

每条样本都包含一段自然语言问题和一段包含最终答案的详解过程，答案通过特殊符号 `#### 42` 标注。我们通过 XML 格式输出使模型回答更清晰、便于评估：

```xml
<reasoning>
首先，我们需要计算...
</reasoning>
<answer>
42
</answer>
```

---

## 💡 为什么选择本项目？

如果你想要：

- ✅ 使用 GRPO 多重奖励函数引导模型输出结构化内容
- ✅ 对 Gemma 系列模型进行高效、轻量的微调
- ✅ 在数学推理任务上做训练 + 推理 + 评估一体化流程
- ✅ 将模型权重合并并发布至 Hugging Face
- ✅ 快速复用此结构用于其他任务（如 SQL、代码、JSON QA）

那这个项目将非常适合你！

---

## 🛠️ 项目特性一览

- ✅ 基于 `trl` 实现 GRPO 训练，支持 5 种奖励函数
- ✅ 采用 Unsloth 加速微调、节省显存、兼容 4bit
- ✅ 完善的 prompt 格式和结构标签解析工具
- ✅ 支持 wandb 实时训练可视化
- ✅ 提供推理、评估、上传 Hugging Face 等脚本
- ✅ 支持 GGUF 量化与多种量化方案上传

---

## 📂 项目结构

```
unsloth-grpo-gsm8k/
├── README.md               # 当前说明文档
├── requirements.txt        # 安装依赖列表
├── .gitignore              # 忽略文件配置
├── configs/
│   └── training_config.yaml        # 训练参数配置
├── scripts/
│   ├── train_grpo.py               # 执行 GRPO 微调
│   ├── inference.py                # 推理单个样本
│   ├── evaluate_gsm8k.py           # GSM8K 评估脚本
│   └── convert_and_push.py         # 合并、量化并推送模型
└── utils/
    ├── data_utils.py               # 数据处理与提示构造
    └── rewards.py                  # 各类奖励函数定义
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 微调模型

```bash
python scripts/train_grpo.py \
  --output_dir outputs/gemma-grpo \
  --max_steps 250
```

### 3. 单条推理

```bash
python scripts/inference.py \
  --model_dir outputs/gemma-grpo \
  --question "3个苹果共9元，7个苹果多少钱？"
```

### 4. GSM8K评估

```bash
python scripts/evaluate_gsm8k.py \
  --model_dir outputs/gemma-grpo
```

### 5. 合并权重并推送 Hugging Face

```bash
python scripts/convert_and_push.py \
  --model_dir outputs/gemma-grpo \
  --hf_repo your_name/merged_16bit \
  --hf_token your_hf_token
```

---

## 🧪 评估示例输出

```
[001/100] ✅=1 | Acc=100.00% | Pred: 27 | GT: 27
[002/100] ✅=2 | Acc=100.00% | Pred: 5 | GT: 5
...
Final accuracy: 88.00% (88/100)
```

---

## 📝 许可证

本项目基于 MIT License 开源，欢迎 Fork、改进和反馈。

---


