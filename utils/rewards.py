import re
from typing import List
from .data_utils import extract_xml_answer

def correctness_reward(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward(completions, **_) -> List[float]:
    extracted = [extract_xml_answer(c[0]["content"]) for c in completions]
    return [0.5 if e.isdigit() else 0.0 for e in extracted]

def strict_xml_reward(completions, **_) -> List[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

def soft_xml_reward(completions, **_) -> List[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(pattern, r, flags=re.S) else 0.0 for r in responses]

def _xml_tag_score(text: str) -> float:
    score = 0.0
    score += 0.125 if text.count("<reasoning>\n") == 1 else 0.0
    score += 0.125 if text.count("\n</reasoning>\n") == 1 else 0.0
    score += 0.125 if text.count("\n<answer>\n") == 1 else 0.0
    score += 0.125 if text.count("\n</answer>") == 1 else 0.0
    trail = text.split("\n</answer>")[-1]
    score -= 0.001 * len(trail)
    return score

def xml_count_reward(completions, **_) -> List[float]:
    contents = [c[0]["content"] for c in completions]
    return [_xml_tag_score(c) for c in contents]
