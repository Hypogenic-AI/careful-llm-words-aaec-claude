# Research Plan: An LLM That's Careful With Its Words

## Motivation & Novelty Assessment

### Why This Research Matters
Current LLMs generate text sentence-by-sentence in a single forward pass without pausing to reconsider. Humans, by contrast, often plan what they'll say next, reconsider phrasing, and adjust for coherence. If we can prompt an LLM to deliberate between every sentence, the resulting text may be more coherent, more nuanced, more factually careful, and qualitatively different from standard generation. This has practical implications for any application where text quality matters: writing assistants, scientific communication, journalism, and educational content.

### Gap in Existing Work
Based on the literature review, all prior work on "thinking tokens" or "pause tokens" requires training modifications (Goyal et al. 2024, Herel & Mikolov 2024, Galashov et al. 2025). No work studies whether prompting an off-the-shelf LLM to reason between every sentence — using its native chain-of-thought capability — produces qualitatively different text. The literature focuses on benchmark accuracy improvements, not on how the *style, structure, and character* of generated text changes.

### Our Novel Contribution
We test whether **inter-sentence chain-of-thought prompting** (requiring thinking between every sentence in the output) produces qualitatively and quantitatively different text from standard generation, using off-the-shelf LLMs via API. We analyze not just accuracy but the textual properties: hedging, specificity, sentence complexity, coherence, and self-correction patterns.

### Experiment Justification
- **Experiment 1 (Open-ended generation)**: Tests how inter-sentence thinking affects essay/explanation quality — the most direct test of "qualitatively different text."
- **Experiment 2 (Factual accuracy - TruthfulQA)**: Tests whether careful deliberation reduces hallucination and improves truthfulness.
- **Experiment 3 (Math reasoning - GSM8k subset)**: Tests whether per-sentence thinking helps or hurts structured reasoning tasks, compared to standard CoT.
- **Experiment 4 (Qualitative text analysis)**: Measures concrete textual properties (vocabulary diversity, hedging frequency, sentence length, self-corrections) to characterize *how* the text differs.

## Research Question
Does requiring an LLM to engage in chain-of-thought reasoning between every sentence produce text that is qualitatively different from standard generation? If so, how?

## Hypothesis Decomposition
1. **H1**: Inter-sentence CoT will produce more hedged, qualified claims compared to standard generation.
2. **H2**: Inter-sentence CoT will produce more coherent, better-structured text (higher inter-sentence logical flow).
3. **H3**: Inter-sentence CoT will improve factual accuracy on TruthfulQA.
4. **H4**: Inter-sentence CoT will produce text with greater lexical diversity and more complex sentence structures.
5. **H5**: Inter-sentence CoT may hurt efficiency on math reasoning (GSM8k) compared to standard CoT, due to forced thinking at suboptimal points.

## Proposed Methodology

### Approach
We use a **prompting-based approach** with off-the-shelf LLMs (GPT-4.1 via OpenAI API). We compare three conditions:
1. **Standard**: Normal generation with no special instructions
2. **Standard CoT**: "Think step by step" before answering (traditional chain-of-thought)
3. **Inter-sentence CoT (our method)**: The model must generate internal reasoning between every sentence of its response

For inter-sentence CoT, we use a structured format where the model alternates between `<think>` blocks (hidden reasoning) and visible sentences. This leverages the model's native reasoning capability without any training modifications.

### Experimental Steps

#### Experiment 1: Open-Ended Generation Quality
- **Prompts**: 30 diverse open-ended prompts (explanations, essays, advice)
- **Conditions**: Standard, Standard CoT, Inter-sentence CoT
- **Evaluation**: LLM-as-judge (pairwise comparisons using a separate model), plus qualitative text analysis
- **Metrics**: Win rate, coherence rating, informativeness rating

#### Experiment 2: Factual Accuracy (TruthfulQA)
- **Dataset**: 100 randomly sampled TruthfulQA questions
- **Conditions**: Standard, Standard CoT, Inter-sentence CoT
- **Evaluation**: LLM-as-judge for truthfulness + informativeness (following TruthfulQA methodology)
- **Metrics**: Truthfulness rate, informativeness rate

#### Experiment 3: Math Reasoning (GSM8k)
- **Dataset**: 100 randomly sampled GSM8k problems
- **Conditions**: Standard, Standard CoT, Inter-sentence CoT
- **Evaluation**: Exact match on final numerical answer
- **Metrics**: Accuracy, token count (efficiency)

#### Experiment 4: Qualitative Text Analysis
- **Data**: All outputs from Experiments 1-3
- **Analyses**:
  - Hedging language frequency (words like "might", "perhaps", "likely")
  - Self-correction patterns ("actually", "however", "on second thought")
  - Lexical diversity (type-token ratio)
  - Average sentence length and complexity
  - Conditional/qualifying clause frequency

### Baselines
1. **Standard generation**: No special prompting (zero-shot)
2. **Standard CoT**: "Let's think step by step" prefix
3. **Inter-sentence CoT**: Our method — forced thinking between every sentence

### Evaluation Metrics
- **Accuracy**: Exact match for GSM8k
- **Truthfulness**: LLM-judge following TruthfulQA protocol
- **Text quality**: LLM-as-judge pairwise comparisons
- **Textual properties**: Hedging rate, TTR, sentence length, self-correction count
- **Efficiency**: Token count per response

### Statistical Analysis Plan
- Paired comparisons using McNemar's test (for accuracy) and Wilcoxon signed-rank test (for ratings)
- Bootstrap confidence intervals for win rates
- Effect sizes (Cohen's d) for continuous metrics
- Significance level: α = 0.05

## Expected Outcomes
- Inter-sentence CoT text will be more hedged, nuanced, and self-correcting
- It will show improved factual accuracy on TruthfulQA
- It may show equivalent or slightly lower accuracy on GSM8k compared to standard CoT
- Text will have higher lexical diversity and longer sentences
- LLM judges will prefer inter-sentence CoT text for coherence and depth

## Timeline and Milestones
1. Environment setup: 10 min
2. Implement prompting framework: 30 min
3. Run Experiment 1 (open-ended): 30 min
4. Run Experiment 2 (TruthfulQA): 30 min
5. Run Experiment 3 (GSM8k): 20 min
6. Run Experiment 4 (text analysis): 20 min
7. Statistical analysis & visualization: 30 min
8. Documentation: 30 min

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and batching
- **Cost**: ~300 API calls × 3 conditions = ~900 calls. Manageable with GPT-4.1.
- **Inter-sentence CoT format compliance**: Model may not follow the think-between-sentences format perfectly. Will validate and retry.
- **Judge bias**: LLM judges may prefer longer text. Will control for length.

## Success Criteria
1. Clear qualitative differences documented between conditions
2. Statistical significance on at least 2 of 4 experiments
3. Concrete characterization of how inter-sentence CoT changes text
