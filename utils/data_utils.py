import re
from datasets import load_dataset

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
  ...
</reasoning>
<answer>
  ...
</answer>"""

def extract_xml_answer(text: str) -> str:
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[-1].strip() if "####" in text else None

def get_gsm8k(split: str = "train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def _map(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": extract_hash_answer(example["answer"]),
        }

    return ds.map(_map)
