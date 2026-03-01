"""Main experiment runner for 'An LLM That's Careful With Its Words'."""

import json
import random
import time
import sys
import os
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import (
    OPEN_ENDED_PROMPTS,
    format_open_ended,
    format_truthfulqa,
    format_gsm8k,
)
from src.api_client import call_llm, call_judge, call_truthfulness_judge
from src.text_analysis import analyze_text, strip_think_tags, count_think_blocks

RESULTS_DIR = Path("/workspaces/careful-llm-words-aaec-claude/results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["standard", "cot", "inter_sentence_cot"]
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def run_experiment1_open_ended(n_prompts=30):
    """Experiment 1: Open-ended generation quality comparison."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Open-Ended Generation Quality")
    print("="*70)

    results = []
    prompts = OPEN_ENDED_PROMPTS[:n_prompts]

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        entry = {"prompt": prompt, "responses": {}, "text_analysis": {}}

        for cond in CONDITIONS:
            messages = format_open_ended(prompt, cond)
            print(f"  Condition: {cond}...", end=" ", flush=True)
            response = call_llm(messages, temperature=0.7, max_tokens=1024)
            entry["responses"][cond] = response
            visible_text = strip_think_tags(response["text"])
            entry["text_analysis"][cond] = analyze_text(response["text"])
            entry["text_analysis"][cond]["think_blocks"] = count_think_blocks(response["text"])
            entry["text_analysis"][cond]["total_tokens"] = response["usage"]["completion_tokens"]
            print(f"done ({response['usage']['completion_tokens']} tokens)")

        results.append(entry)

    # Save raw results
    out_path = RESULTS_DIR / "experiment1_open_ended.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return results


def run_experiment1_judging(results: list):
    """Run LLM-as-judge pairwise comparisons for Experiment 1."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: LLM-as-Judge Pairwise Comparisons")
    print("="*70)

    comparisons = [
        ("standard", "inter_sentence_cot"),
        ("cot", "inter_sentence_cot"),
        ("standard", "cot"),
    ]
    criteria_list = [
        ("coherence", "Which response is more coherent, logical, and well-structured? Consider the flow between sentences and the overall organization."),
        ("depth", "Which response provides more depth, nuance, and insightful analysis? Consider whether claims are well-supported and carefully considered."),
        ("accuracy", "Which response is more likely to be factually accurate and avoids making unsupported or overconfident claims?"),
    ]

    judge_results = []
    for i, entry in enumerate(results):
        print(f"\n[{i+1}/{len(results)}] {entry['prompt'][:50]}...")
        entry_judgments = {"prompt": entry["prompt"], "comparisons": []}

        for cond_a, cond_b in comparisons:
            text_a = strip_think_tags(entry["responses"][cond_a]["text"])
            text_b = strip_think_tags(entry["responses"][cond_b]["text"])

            for criteria_name, criteria_desc in criteria_list:
                # Randomize order to control for position bias
                if random.random() < 0.5:
                    judgment = call_judge(entry["prompt"], text_a, text_b, criteria_desc)
                    winner_mapped = cond_a if judgment["winner"] == "A" else (cond_b if judgment["winner"] == "B" else "TIE")
                else:
                    judgment = call_judge(entry["prompt"], text_b, text_a, criteria_desc)
                    winner_mapped = cond_b if judgment["winner"] == "A" else (cond_a if judgment["winner"] == "B" else "TIE")

                entry_judgments["comparisons"].append({
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "criteria": criteria_name,
                    "winner": winner_mapped,
                    "raw_explanation": judgment["explanation"],
                })
                print(f"  {cond_a} vs {cond_b} ({criteria_name}): {winner_mapped}")

        judge_results.append(entry_judgments)

    out_path = RESULTS_DIR / "experiment1_judgments.json"
    with open(out_path, "w") as f:
        json.dump(judge_results, f, indent=2)
    print(f"\nSaved judgments to {out_path}")
    return judge_results


def run_experiment2_truthfulqa(n_questions=100):
    """Experiment 2: TruthfulQA factual accuracy comparison."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: TruthfulQA Factual Accuracy")
    print("="*70)

    # Load TruthfulQA dataset
    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/careful-llm-words-aaec-claude/datasets/truthfulqa")
    data = ds["validation"]

    # Sample questions
    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = indices[:n_questions]

    results = []
    for idx_i, idx in enumerate(indices):
        item = data[idx]
        question = item["question"]
        best_answer = item["best_answer"]
        print(f"\n[{idx_i+1}/{n_questions}] {question[:60]}...")

        entry = {"question": question, "best_answer": best_answer, "responses": {}, "judgments": {}}

        for cond in CONDITIONS:
            messages = format_truthfulqa(question, cond)
            print(f"  {cond}...", end=" ", flush=True)
            response = call_llm(messages, temperature=0.7, max_tokens=512)
            entry["responses"][cond] = response

            visible_text = strip_think_tags(response["text"])
            judgment = call_truthfulness_judge(question, visible_text, best_answer)
            entry["judgments"][cond] = judgment
            t = "T" if judgment["truthful"] else "F"
            i = "I" if judgment["informative"] else "U"
            print(f"{t}{i}", end=" ")
            print(f"({response['usage']['completion_tokens']} tokens)")

        results.append(entry)

    out_path = RESULTS_DIR / "experiment2_truthfulqa.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return results


def extract_gsm8k_answer(text: str) -> str:
    """Extract numerical answer after #### from GSM8k response."""
    # Look for #### pattern
    match = re.search(r'####\s*(\-?[\d,]+)', text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: look for last number in text
    numbers = re.findall(r'\-?\d+', text)
    return numbers[-1] if numbers else ""


def run_experiment3_gsm8k(n_questions=100):
    """Experiment 3: GSM8k math reasoning comparison."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: GSM8k Math Reasoning")
    print("="*70)

    import re
    # Make extract_gsm8k_answer available
    global extract_gsm8k_answer

    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/careful-llm-words-aaec-claude/datasets/gsm8k")
    data = ds["test"]

    indices = list(range(len(data)))
    random.shuffle(indices)
    indices = indices[:n_questions]

    results = []
    for idx_i, idx in enumerate(indices):
        item = data[idx]
        question = item["question"]
        gold_answer = item["answer"].split("####")[-1].strip().replace(",", "")
        print(f"\n[{idx_i+1}/{n_questions}] {question[:60]}...")

        entry = {"question": question, "gold_answer": gold_answer, "responses": {}, "correct": {}}

        for cond in CONDITIONS:
            messages = format_gsm8k(question, cond)
            print(f"  {cond}...", end=" ", flush=True)
            response = call_llm(messages, temperature=0.0, max_tokens=1024)
            entry["responses"][cond] = response

            visible_text = strip_think_tags(response["text"])
            predicted = extract_gsm8k_answer(visible_text)
            is_correct = predicted == gold_answer
            entry["correct"][cond] = is_correct
            entry[f"predicted_{cond}"] = predicted

            mark = "+" if is_correct else "-"
            print(f"{mark}({predicted} vs {gold_answer})", end=" ")
            print(f"({response['usage']['completion_tokens']} tok)")

        results.append(entry)

    out_path = RESULTS_DIR / "experiment3_gsm8k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")
    return results


import re  # needed for extract_gsm8k_answer


def main():
    set_seed(SEED)
    print(f"Python: {sys.version}")
    print(f"Seed: {SEED}")
    print(f"Model: gpt-4.1")
    print(f"Conditions: {CONDITIONS}")

    # Save config
    config = {
        "seed": SEED,
        "model": "gpt-4.1",
        "conditions": CONDITIONS,
        "temperature_generation": 0.7,
        "temperature_gsm8k": 0.0,
        "temperature_judge": 0.0,
        "max_tokens_generation": 1024,
        "max_tokens_truthfulqa": 512,
        "max_tokens_gsm8k": 1024,
        "n_open_ended": 30,
        "n_truthfulqa": 100,
        "n_gsm8k": 100,
    }
    config_path = RESULTS_DIR.parent / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments
    exp1_results = run_experiment1_open_ended(n_prompts=30)
    exp1_judgments = run_experiment1_judging(exp1_results)
    exp2_results = run_experiment2_truthfulqa(n_questions=100)
    exp3_results = run_experiment3_gsm8k(n_questions=100)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
