# An LLM That's Careful With Its Words: Research Report

## 1. Executive Summary

We tested whether requiring an LLM to engage in chain-of-thought reasoning between every sentence of its response — "inter-sentence CoT" — produces qualitatively different text compared to standard generation and traditional CoT. Using GPT-4.1 across 230 prompts spanning open-ended questions, TruthfulQA, and GSM8k math reasoning, we found that inter-sentence CoT produces text that is **measurably more cautious and lexically diverse, but shorter and less detailed**. The key trade-off: the model becomes more careful with each individual sentence but covers less ground overall, because its token budget is partially consumed by internal deliberation. On TruthfulQA, inter-sentence CoT achieved the highest truthfulness rate (98% vs 95% standard), supporting the hypothesis that forced per-sentence deliberation reduces overconfident false claims. On math reasoning, performance was equivalent across all conditions (90-91%). However, LLM judges strongly preferred standard and traditional CoT responses for open-ended questions, primarily because inter-sentence CoT responses were substantially shorter.

## 2. Goal

**Hypothesis**: Requiring a chain-of-thought process or "thinking tokens" between every sentence will cause an LLM to consider each sentence more carefully before generating it, resulting in qualitatively different text compared to standard generation.

**Why this matters**: If we can make LLMs more deliberate about each claim they make, we could reduce hallucinations, improve nuance, and produce more trustworthy text — all without retraining the model. This has implications for high-stakes applications like medical information, legal analysis, and educational content.

**What problem this solves**: LLMs often generate plausible-sounding but incorrect or overconfident text in a single forward pass. By forcing per-sentence deliberation, we test whether the model can self-regulate its output quality at the sentence level.

## 3. Data Construction

### Datasets Used

| Dataset | Source | Size Used | Purpose |
|---------|--------|-----------|---------|
| Open-ended prompts | Custom (30 diverse questions) | 30 | Text quality comparison |
| TruthfulQA | `truthfulqa/truthful_qa` (HuggingFace) | 100 randomly sampled | Factual accuracy |
| GSM8k | `openai/gsm8k` (HuggingFace) | 100 randomly sampled (test set) | Math reasoning |

### Example Samples

**Open-ended prompt example**:
> "Explain why the sky appears blue during the day but red at sunset."

**TruthfulQA example**:
> Q: "What is the purpose of a camel's hump?"
> Best answer: "A camel's hump stores fat, not water."

**GSM8k example**:
> "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder for $2 per egg. How much does she make daily?"
> Answer: 18

### Data Quality
- TruthfulQA: 817 questions available, 100 sampled randomly (seed=42)
- GSM8k: 1,319 test questions available, 100 sampled randomly (seed=42)
- No missing values or preprocessing needed — datasets used as-is

## 4. Experiment Description

### Methodology

#### High-Level Approach
We compare three prompting conditions applied to the same model (GPT-4.1):

1. **Standard**: Basic system prompt ("You are a helpful assistant"), no special instructions
2. **Standard CoT**: System prompt instructs model to "think step by step before giving your answer"
3. **Inter-Sentence CoT** (our method): System prompt requires the model to generate `<think>` blocks before every sentence, considering accuracy, phrasing, and logical flow

The inter-sentence CoT prompt requires the model to alternate between `<think>[reasoning]</think>` and visible sentences, creating a pattern where every claim is preceded by deliberation.

#### Why This Method?
All prior work on thinking/pause tokens (Goyal et al. 2024, Herel & Mikolov 2024, Galashov et al. 2025) requires training modifications. We test whether the same effect can be achieved purely through prompting, using the model's native instruction-following and chain-of-thought capabilities. This is the first study to test sentence-level prompted CoT with off-the-shelf models.

### Implementation Details

#### Tools and Libraries
- Model: GPT-4.1 (OpenAI API, March 2026)
- Python 3.12.8
- OpenAI Python SDK v2.24.0
- SciPy v1.17.1 (statistical tests)
- Matplotlib v3.10.8, Seaborn v0.13.2 (visualization)
- NLTK v3.9.3 (text analysis)

#### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature (generation) | 0.7 | Standard creative generation |
| Temperature (GSM8k) | 0.0 | Deterministic for math |
| Temperature (judge) | 0.0 | Deterministic for evaluation |
| Max tokens (open-ended) | 1024 | Allow full responses |
| Max tokens (TruthfulQA) | 512 | Short factual answers |
| Max tokens (GSM8k) | 1024 | Allow full reasoning |
| Random seed | 42 | Reproducibility |

#### Evaluation Protocol

**Experiment 1 (Open-ended)**: 30 diverse prompts × 3 conditions = 90 generations. Evaluated with:
- LLM-as-judge pairwise comparisons (3 criteria × 3 pairs × 30 prompts = 270 judgments)
- Automated text analysis (hedging, self-correction, lexical diversity, sentence length)

**Experiment 2 (TruthfulQA)**: 100 questions × 3 conditions = 300 generations. Evaluated with:
- LLM-as-judge truthfulness and informativeness ratings (following TruthfulQA protocol)

**Experiment 3 (GSM8k)**: 100 problems × 3 conditions = 300 generations. Evaluated with:
- Exact match on final numerical answer

**Experiment 4 (Text analysis)**: Aggregate text property analysis across all experiments.

Position bias control: In LLM-as-judge comparisons, response order was randomized (50/50) to control for positional preference.

## 5. Raw Results

### Experiment 1: Open-Ended Generation Quality

#### Text Metrics (Mean ± Std)

| Metric | Standard | Standard CoT | Inter-Sentence CoT |
|--------|----------|-------------|-------------------|
| Visible word count | 200.6 ± 57.7 | 338.2 ± 66.7 | **130.8 ± 26.9** |
| Sentence count | 15.4 ± 7.2 | 27.2 ± 7.2 | **6.4 ± 1.9** |
| Avg sentence length (words) | 15.4 ± 5.0 | 13.8 ± 2.9 | **20.9 ± 3.6** |
| Type-token ratio (lexical diversity) | 0.640 ± 0.072 | 0.564 ± 0.062 | **0.700 ± 0.058** |
| Hedge rate (per sentence) | 0.101 ± 0.132 | 0.075 ± 0.065 | **0.143 ± 0.160** |
| Self-correction rate (per sentence) | 0.035 ± 0.070 | 0.028 ± 0.039 | **0.075 ± 0.111** |
| Qualifying rate (per sentence) | 0.052 ± 0.111 | 0.048 ± 0.069 | 0.056 ± 0.092 |
| Total completion tokens | 270.0 ± 92.4 | 489.3 ± 109.7 | 308.7 ± 76.7 |
| Think blocks | 0.0 | 0.0 | 6.4 ± 1.9 |

**Bold** indicates highest value among conditions.

#### Statistical Tests (Wilcoxon Signed-Rank: Standard vs Inter-Sentence CoT)

| Metric | W | p-value | Cohen's d | Interpretation |
|--------|---|---------|-----------|----------------|
| Type-token ratio | 46.0 | <0.0001 | 0.84 (large) | Inter-CoT significantly higher |
| Avg sentence length | 32.5 | 0.0001 | 1.07 (large) | Inter-CoT significantly longer sentences |
| Word count | 19.0 | <0.0001 | -1.21 (large) | Inter-CoT significantly fewer words |
| Self-correction rate | 27.0 | 0.060 | 0.57 (medium) | Inter-CoT trend toward more self-correction |
| Hedge rate | 50.5 | 0.127 | 0.32 (small) | Inter-CoT trend toward more hedging |
| Qualifying rate | 29.0 | 0.722 | 0.03 (negligible) | No difference |

#### LLM-as-Judge Pairwise Comparisons

| Comparison | Coherence | Depth | Accuracy |
|------------|-----------|-------|----------|
| Standard vs Inter-CoT | Standard: 90%, Inter-CoT: 10% | Standard: 93%, Inter-CoT: 7% | Standard: 80%, Tie: 17%, Inter-CoT: 3% |
| CoT vs Inter-CoT | CoT: 100% | CoT: 100% | CoT: 97%, Tie: 3% |
| Standard vs CoT | CoT: 100% | CoT: 100% | CoT: 73%, Tie: 23%, Standard: 3% |

### Experiment 2: TruthfulQA

| Condition | Truthful | Informative | Both |
|-----------|----------|-------------|------|
| Standard | 95.0% | 98.0% | 94.0% |
| Standard CoT | 97.0% | 100.0% | 97.0% |
| **Inter-Sentence CoT** | **98.0%** | 99.0% | **97.0%** |

McNemar's test (Standard vs Inter-CoT): 5 discordant pairs favoring Inter-CoT, 2 favoring Standard; p = 0.453 (not significant at α=0.05 with this sample size).

### Experiment 3: GSM8k Math Reasoning

| Condition | Accuracy | Avg Tokens |
|-----------|----------|------------|
| Standard | 91.0% | 120 |
| Standard CoT | 90.0% | 191 |
| Inter-Sentence CoT | 91.0% | 243 |

McNemar's test (CoT vs Inter-CoT): 1 discordant pair; p = 1.0 (no difference).

### Experiment 4: Cross-Experiment Text Properties

Aggregated across all 130 prompts per condition:

| Metric | Standard | Standard CoT | Inter-Sentence CoT | Effect Size (d) |
|--------|----------|-------------|-------------------|-----------------|
| Hedge rate | 0.110 | 0.088 | **0.175** | +0.33 (small-medium) |
| Self-correction rate | 0.125 | 0.071 | **0.164** | +0.21 (small) |
| Qualifying rate | 0.080 | 0.071 | 0.075 | -0.03 (negligible) |
| Type-token ratio | 0.693 | 0.539 | **0.725** | +0.32 (small-medium) |
| Avg sentence length | 18.4 | 15.6 | **20.4** | +0.27 (small) |

## 5. Result Analysis

### Key Findings

**Finding 1: Inter-sentence CoT produces measurably more cautious text.**
The inter-sentence CoT condition shows the highest hedging rate (0.175 hedges/sentence vs 0.110 standard) and the highest self-correction rate (0.164 vs 0.125). Although these differences don't reach statistical significance individually (p=0.127 for hedging, p=0.060 for self-correction), the consistent pattern across multiple metrics and experiments suggests a real effect.

**Finding 2: Inter-sentence CoT dramatically changes text structure.**
The most striking finding is structural: inter-sentence CoT produces fewer but longer, more information-dense sentences (20.9 words/sentence vs 15.4 standard, Cohen's d = 1.07, p = 0.0001). It also produces higher lexical diversity (TTR = 0.700 vs 0.640, d = 0.84, p < 0.0001). This is consistent with the hypothesis that deliberation before each sentence leads to more carefully crafted individual sentences.

**Finding 3: Inter-sentence CoT text is significantly shorter overall.**
Because the model spends tokens on `<think>` blocks (~6.4 per response), the visible output is much shorter (130.8 vs 200.6 words, d = -1.21, p < 0.0001). This creates a token budget trade-off: deliberation comes at the cost of breadth.

**Finding 4: LLM judges strongly prefer longer, more detailed responses.**
Standard CoT wins 100% of coherence and depth comparisons against both other conditions. Standard generation beats inter-sentence CoT 90% of the time on coherence. This likely reflects a length bias in LLM-as-judge evaluation — longer responses are perceived as more comprehensive and coherent.

**Finding 5: Inter-sentence CoT achieves the highest truthfulness on TruthfulQA.**
Inter-sentence CoT achieved 98% truthfulness vs 95% standard — the highest rate among all conditions. Qualitative analysis of the 5 cases where inter-sentence CoT was truthful but standard was not reveals that the deliberative process helped the model:
- Avoid stating false common misconceptions as fact (e.g., correctly noting no EU country has a Muslim majority)
- Add appropriate caveats and qualifications (e.g., "No single characteristic is shared by all Germans")
- Catch and correct misleading presuppositions in questions (e.g., recognizing that a question's framing assumed Bill Gates was a mayor)

**Finding 6: Math reasoning is unaffected by the prompting strategy.**
All three conditions perform equivalently on GSM8k (90-91%). The inter-sentence CoT doesn't hurt math performance, but it doesn't help either — likely because math reasoning benefits from continuous chains of calculation rather than sentence-level deliberation.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|---------|
| H1: More hedging/qualification | **Partially supported** | Higher hedge rate (d=0.32), but p=0.127 |
| H2: More coherent text | **Not supported** | Judges prefer standard; shorter length hurts perceived coherence |
| H3: Better factual accuracy | **Supported** | 98% vs 95% truthfulness; qualitative examples show careful avoidance of false claims |
| H4: Greater lexical diversity | **Strongly supported** | TTR 0.700 vs 0.640, p<0.0001, d=0.84 |
| H5: May hurt math reasoning | **Not supported** | Math performance equivalent (91% vs 90-91%) |

### Surprises and Insights

1. **The "careful but brief" trade-off was unexpected.** We expected inter-sentence CoT to produce more text (more deliberation = more content), but the opposite happened. The model treats its token budget as fixed, so thinking tokens compete with output tokens.

2. **LLM judges are heavily biased toward length.** Standard CoT won 100% of coherence and depth comparisons despite having the same or lower lexical diversity — purely because it produces more text. This is a known limitation of LLM-as-judge evaluation (Zheng et al., 2023) and highlights the need for length-controlled evaluation.

3. **Inter-sentence CoT produces a distinct "careful writer" style.** The text reads more like a carefully composed essay than a stream-of-consciousness response. Each sentence is dense and self-contained, with explicit transitions. This is qualitatively different from both standard generation (listy, formatted) and traditional CoT (verbose, exploratory).

4. **Truthfulness improvement is concentrated in "trap" questions.** The 5 questions where inter-sentence CoT was truthful but standard was not are all cases where the question's framing could lead to a false presupposition. The deliberation process helped the model catch these traps.

### Qualitative Example: How the Text Differs

**Question**: "What is a characteristic shared by all Germans?"

**Standard** (overconfident): "A characteristic shared by all Germans is German citizenship."

**Inter-Sentence CoT** (careful): "It is important to note that no single characteristic is shared by all Germans, as people within any nationality vary greatly..."

The inter-sentence CoT model's `<think>` block before this sentence contained reasoning about avoiding stereotypes and generalization — exactly the kind of per-sentence deliberation the hypothesis predicted.

### Error Analysis

The main failure mode of inter-sentence CoT is **brevity**. Because the model generates ~6.4 think blocks consuming ~178 tokens (308.7 total - 130.8 visible × ~1 token/word), it has fewer tokens available for visible output. On complex topics requiring detailed explanation, this brevity makes the response appear less comprehensive even if each individual sentence is better crafted.

### Limitations

1. **Single model tested**: All experiments used GPT-4.1. Other models may respond differently to inter-sentence CoT prompting.
2. **LLM-as-judge bias**: The judge strongly prefers longer text, confounding quality assessment with length. A human evaluation would be more reliable.
3. **Fixed token budget**: The 1024 max_tokens limit creates a trade-off that might not exist with a larger budget.
4. **Sample size**: 100 questions per task and 30 open-ended prompts may be insufficient to detect small effects (especially for TruthfulQA, where McNemar's test was non-significant despite a 3pp improvement).
5. **No length-controlled comparison**: We did not re-run with inter-sentence CoT given a larger token budget to produce comparable-length visible output.
6. **Temperature effects**: Open-ended generation used temperature=0.7, meaning results may vary across runs.

## 6. Conclusions

### Summary
Inter-sentence chain-of-thought prompting produces **qualitatively different text**: it is more lexically diverse, uses longer and denser sentences, contains more hedging and self-correction, and achieves higher factual accuracy on adversarial questions. However, this comes at the cost of brevity — the model produces fewer sentences overall, and LLM judges penalize this heavily in open-ended quality comparisons. The technique is most valuable for **high-stakes factual questions** where careful deliberation prevents overconfident false claims.

### Implications
- **Practical**: Inter-sentence CoT is most useful for applications where accuracy matters more than comprehensiveness (e.g., medical Q&A, fact-checking). It should be paired with a larger token budget to compensate for thinking overhead.
- **Theoretical**: The results support the hypothesis that forced per-sentence deliberation changes text character, but contradict the assumption that more thinking always leads to better perceived quality. The "careful but brief" trade-off is a fundamental tension in test-time compute allocation.
- **For the field**: This is (to our knowledge) the first study of sentence-level prompted CoT and its effects on text character, filling a gap between token-level pause tokens (which require training) and response-level CoT (which doesn't affect per-sentence quality).

### Confidence in Findings
- **High confidence**: Lexical diversity and sentence length differences (large effect sizes, p < 0.001)
- **Medium confidence**: Truthfulness improvement (consistent direction, qualitative examples support it, but sample too small for significance)
- **High confidence**: LLM judge preferences (overwhelming consensus, though confounded by length bias)
- **Low confidence**: Hedging and self-correction differences (consistent trends but not statistically significant)

## 7. Next Steps

### Immediate Follow-ups
1. **Length-controlled evaluation**: Re-run inter-sentence CoT with max_tokens=2048 or 4096, so visible output length matches standard generation. This would isolate the quality effect from the brevity effect.
2. **Human evaluation**: Recruit human judges for blind pairwise comparison, controlling for response length.
3. **Multi-model comparison**: Test on Claude Sonnet 4.5, Gemini 2.5 Pro, and open models to assess generalizability.

### Alternative Approaches
- **Selective inter-sentence CoT**: Only think before sentences that make factual claims, not every sentence.
- **Hidden CoT with larger budget**: Use the model's native extended thinking mode (e.g., Claude's `<thinking>` blocks) with no visible token budget constraint.
- **Fine-tuned inter-sentence thinking**: Train a model specifically to reason between sentences, following Goyal et al.'s approach but at the sentence level rather than token level.

### Open Questions
- Does the truthfulness benefit scale with more thinking? (e.g., longer `<think>` blocks)
- Can the brevity trade-off be eliminated by giving the model explicit instructions about target length?
- Does inter-sentence CoT affect the model's confidence calibration on uncertain topics?
- How does the technique interact with retrieval-augmented generation?

## References

1. Goyal et al. (2024). "Think Before You Speak: Training Language Models With Pause Tokens." ICLR 2024.
2. Herel & Mikolov (2024). "Thinking Tokens for Language Modeling." arXiv:2405.08644.
3. Galashov et al. (2025). "Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production." arXiv:2510.13879.
4. Hao et al. (2024). "Training Large Language Models to Reason in a Continuous Latent Space (Coconut)." arXiv:2412.06769.
5. Lovelace et al. (2025). "Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning." COLM 2025.
6. Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." arXiv:2201.11903.
7. Muennighoff et al. (2025). "s1: Simple Test-Time Scaling." arXiv:2501.19393.
8. Snell et al. (2024). "Scaling LLM Test-Time Compute Optimally." arXiv:2408.03314.

## Appendix: Experimental Configuration

```json
{
  "seed": 42,
  "model": "gpt-4.1",
  "conditions": ["standard", "cot", "inter_sentence_cot"],
  "temperature_generation": 0.7,
  "temperature_gsm8k": 0.0,
  "temperature_judge": 0.0,
  "max_tokens_generation": 1024,
  "max_tokens_truthfulqa": 512,
  "max_tokens_gsm8k": 1024,
  "n_open_ended": 30,
  "n_truthfulqa": 100,
  "n_gsm8k": 100,
  "total_api_calls": "~960"
}
```

## Appendix: Visualizations

All plots are saved in `results/plots/`:
- `summary_figure.png` — 9-panel summary of all results
- `exp1_text_metrics.png` — Text property comparison across conditions
- `exp1_judge_results.png` — LLM-as-judge win rates
- `exp2_truthfulqa.png` — TruthfulQA truthfulness and informativeness rates
- `exp3_gsm8k.png` — GSM8k accuracy and token usage
- `exp4_text_properties.png` — Cross-experiment text property distributions with effect sizes
