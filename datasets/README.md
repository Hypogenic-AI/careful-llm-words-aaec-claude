# Downloaded Datasets

This directory contains datasets for the research project "An LLM That's Careful With Its Words." Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: GSM8k (Grade School Math 8K)

### Overview
- **Source**: HuggingFace `openai/gsm8k`
- **Size**: 8,792 examples (train: 7,473, test: 1,319), ~2.7 MB
- **Format**: HuggingFace Dataset
- **Task**: Multi-step mathematical reasoning
- **Splits**: train (7,473), test (1,319)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

### Notes
- Used in the Pause Tokens paper (Goyal et al., 2024) — showed ~1% accuracy gain with pause-training
- Each example includes step-by-step solutions enabling evaluation of reasoning quality
- Key benchmark for chain-of-thought reasoning evaluation

---

## Dataset 2: TruthfulQA

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa`
- **Size**: 817 questions, ~223 KB
- **Format**: HuggingFace Dataset
- **Task**: Factual accuracy evaluation (adversarial)
- **Splits**: validation (817)
- **License**: Apache-2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation")
dataset.save_to_disk("datasets/truthfulqa")
```

### Notes
- Adversarially crafted questions designed to exploit common misconceptions
- Directly tests whether "careful" generation improves factual accuracy
- Covers 38 categories: health, law, finance, politics, etc.

---

## Dataset 3: CommonSenseQA

### Overview
- **Source**: HuggingFace `tau/commonsense_qa`
- **Size**: 12,102 examples, ~1.6 MB
- **Format**: HuggingFace Dataset
- **Task**: Multiple-choice commonsense reasoning
- **Splits**: train (9,741), validation (1,221), test (1,140)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("tau/commonsense_qa")
dataset.save_to_disk("datasets/commonsense_qa")
```

### Notes
- Used in Pause Tokens paper — showed ~8% improvement with pause-training (1B model)
- Questions constructed using ConceptNet requiring commonsense world knowledge
- One of the highest-signal benchmarks for this research direction

---

## Dataset 4: HellaSwag

### Overview
- **Source**: HuggingFace `Rowan/hellaswag`
- **Size**: ~59,950 examples, ~71 MB
- **Format**: HuggingFace Dataset
- **Task**: Commonsense NLI / sentence completion
- **Splits**: train (39,905), validation (10,042), test (10,003)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Rowan/hellaswag")
dataset.save_to_disk("datasets/hellaswag")
```

### Notes
- Used in Pause Tokens paper — notably, this was the one task where pause-training did NOT help
- Tests sequential text coherence: given context, select most plausible continuation
- Critical replication target for understanding where thinking tokens help vs. don't

---

## Additional Recommended Datasets (Not Downloaded)

For extended evaluation, consider also downloading:

| Dataset | HuggingFace ID | Size | Task |
|---------|---------------|------|------|
| XSum | `EdinburghNLP/xsum` | ~226K examples | Abstractive summarization |
| AlpacaEval 2.0 | `tatsu-lab/alpaca_eval` | 805 instructions | Open-ended generation quality |
| MT-Bench | `lmsys/mt_bench_human_judgments` | 80 questions | Multi-turn conversation quality |
