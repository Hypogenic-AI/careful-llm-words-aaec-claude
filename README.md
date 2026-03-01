# An LLM That's Careful With Its Words

Can we use chain-of-thought to make an LLM that considers every sentence carefully before stating it?

## Key Findings

- **Inter-sentence CoT produces qualitatively different text**: Longer, denser sentences with higher lexical diversity (TTR 0.700 vs 0.640, p<0.0001) and more hedging/self-correction
- **Truthfulness improves**: 98% truthful on TruthfulQA vs 95% standard — the model catches false presuppositions and avoids overconfident claims
- **But the text is shorter**: Thinking tokens consume the token budget, producing ~35% fewer visible words (130 vs 201)
- **LLM judges prefer longer responses**: Standard and traditional CoT win pairwise comparisons due to length bias, not per-sentence quality
- **Math reasoning is unaffected**: All three conditions achieve ~91% on GSM8k

## Method

We compare three prompting strategies with GPT-4.1:

1. **Standard**: Direct generation
2. **Standard CoT**: "Think step by step" before answering
3. **Inter-Sentence CoT** (ours): `<think>` block required before every sentence

Tested on 30 open-ended questions, 100 TruthfulQA questions, and 100 GSM8k math problems.

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy matplotlib seaborn pandas datasets nltk

# Set API key
export OPENAI_API_KEY=your_key_here

# Run experiments (~45 min, ~960 API calls)
python src/run_experiments.py

# Run analysis
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan
├── src/
│   ├── prompts.py               # Prompt templates for 3 conditions
│   ├── api_client.py            # OpenAI API client with caching
│   ├── text_analysis.py         # Text property measurement
│   ├── run_experiments.py       # Main experiment runner
│   └── analyze_results.py       # Analysis and visualization
├── results/
│   ├── config.json              # Experiment configuration
│   ├── analysis_results.json    # Aggregated analysis
│   ├── raw/                     # Raw experiment outputs
│   │   ├── experiment1_open_ended.json
│   │   ├── experiment1_judgments.json
│   │   ├── experiment2_truthfulqa.json
│   │   └── experiment3_gsm8k.json
│   ├── plots/                   # Generated visualizations
│   └── cache/                   # API response cache
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── papers/                      # Downloaded research papers (22)
├── datasets/                    # Downloaded datasets (4)
└── code/                        # Cloned repositories (3)
```

See [REPORT.md](REPORT.md) for the full research report.
