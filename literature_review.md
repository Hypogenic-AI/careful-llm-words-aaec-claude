# Literature Review: An LLM That's Careful With Its Words

## Research Area Overview

This research explores whether requiring a chain-of-thought process or "thinking tokens" between every sentence causes an LLM to generate qualitatively different text. The core idea is that by forcing the model to reason between sentences, rather than generating them in immediate succession, the resulting text may exhibit improved coherence, factual accuracy, and reasoning quality.

This sits at the intersection of several active research areas: (1) **test-time compute scaling** — using additional inference-time computation to improve LLM outputs; (2) **pause/thinking tokens** — inserting special tokens to give models additional processing steps; (3) **latent reasoning** — reasoning in continuous hidden states rather than discrete tokens; and (4) **chain-of-thought prompting** — eliciting step-by-step reasoning in natural language.

---

## Key Papers

### Paper 1: Think Before You Speak: Training Language Models With Pause Tokens
- **Authors**: Goyal, Ji, Rawat, Menon, Kumar, Nagarajan
- **Year**: 2024 (ICLR 2024)
- **Source**: arXiv:2310.02226
- **Key Contribution**: First rigorous study of learnable `<pause>` tokens in Transformer LMs, demonstrating that appending dummy tokens to delay output generation improves downstream task performance — but only when used during both pretraining and finetuning.
- **Methodology**: Decoder-only models (1B, 130M params) pretrained on C4 (200B tokens). During pause-pretraining, `<pause>` tokens are inserted at 10% of random positions. During finetuning, 10 or 50 `<pause>` tokens are appended to the input prefix. Loss on `<pause>` token predictions is ignored.
- **Datasets Used**: GSM8k, SQuAD, CoQA, CommonSenseQA, PhysicalIQA, LAMBADA, HellaSwag, WebQuestions, NaturalQuestions
- **Results**: PausePT+PauseFT improves over standard training on 8/9 tasks for 1B model. Most prominent gains: +18% EM on SQuAD, +8% on CommonSenseQA, +1% on GSM8k. Only HellaSwag showed no gain. Filler characters (periods) at inference-only do NOT help.
- **Code Available**: No (Google internal infrastructure)
- **Relevance**: Most directly relevant prior work. Demonstrates that thinking tokens can help but require training integration. Our hypothesis extends this by using prompted thinking (between sentences) with off-the-shelf models rather than trained pause tokens.

### Paper 2: Thinking Tokens for Language Modeling
- **Authors**: Herel, Mikolov
- **Year**: 2024
- **Source**: arXiv:2405.08644
- **Key Contribution**: Proposes inserting `<T>` thinking tokens after each word in LSTM-based language models to give the model more computation time. Proof-of-concept showing per-sentence perplexity improvements on reasoning-heavy sentences.
- **Methodology**: LSTM LM with 1 hidden layer (450 neurons). N=1 thinking token after each word. Trained on Penn TreeBank, WikiText-2, MacroEconomics textbook, and math dataset.
- **Results**: Overall validation perplexity slightly worsens (68.2→68.4 on PTB), but per-sentence analysis shows improvements on complex reasoning sentences (e.g., math: 16.8→13.1 perplexity). More thinking tokens (2, 3) hurt due to LSTM forgetting.
- **Code Available**: No
- **Relevance**: Earliest articulation of the thinking tokens idea for language models. Limited by LSTM architecture and small scale. Our work extends this to modern Transformers and the sentence level.

### Paper 3: Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production
- **Authors**: Galashov, Jones, Ke, Cao, Nagarajan, Mozer
- **Year**: 2025
- **Source**: arXiv:2510.13879
- **Key Contribution**: Makes pause tokens **adaptive** — the model itself decides when to request additional compute by emitting `<DON'T KNOW>` tokens. Frames this as a sequential decision problem with a time cost.
- **Methodology**: Fine-tuning experiments with three CYB loss variants (anytime prediction, variational, dynamic penalty). Model can request multiple pauses per token.
- **Results**: CYB model needs only 1/3 the training data as no-pause baseline. Model adapts processing time to token-level complexity: pauses after plural nouns ("patients"), never after contractions ("wasn"), high variability for ambiguous tokens ("won").
- **Code Available**: No (Google DeepMind)
- **Relevance**: Provides evidence that adaptive thinking time is beneficial and that models can learn to allocate compute where it's needed. Directly relevant to our hypothesis about per-sentence thinking.

### Paper 4: Training Large Language Models to Reason in a Continuous Latent Space (Coconut)
- **Authors**: Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian
- **Year**: 2024/2025
- **Source**: arXiv:2412.06769
- **Key Contribution**: Introduces Chain of Continuous Thought (Coconut), where the LLM's last hidden state is fed back as the next input embedding, enabling reasoning in a continuous latent space rather than discrete language tokens.
- **Methodology**: GPT-2 base model. Multi-stage training curriculum that gradually replaces language CoT steps with continuous thoughts. `<bot>` and `<eot>` tokens mark latent reasoning mode.
- **Datasets Used**: ProsQA (new graph reasoning dataset), ProntoQA, GSM8k
- **Results**: Outperforms language CoT on ProsQA (requiring search/planning). Continuous thoughts can encode multiple alternative reasoning paths, enabling emergent BFS-like reasoning. Reduces inference token count.
- **Code Available**: Yes — https://github.com/facebookresearch/coconut
- **Relevance**: Shows that latent "thinking" (not visible in output) can outperform explicit CoT. Provides evidence that reasoning can happen in non-language representations. However, focuses on task-specific reasoning rather than general text quality.

### Paper 5: Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning (STAR-LDM)
- **Authors**: Lovelace, Belardi, Zalouk, Polavaram, Kundurthy, Weinberger
- **Year**: 2025/2026 (COLM 2025)
- **Source**: arXiv:2602.20528
- **Key Contribution**: Integrates latent diffusion planning into autoregressive generation. The model pauses generation, performs diffusion-based planning in sentence embedding space, then resumes AR generation guided by the refined plan.
- **Methodology**: Uses Sentence-T5 XL embeddings as latent planning space. Diffusion Transformers (DiTs) translate between embedding space and token space. Joint training of AR and diffusion components.
- **Results**: >70% win rates in LLM-as-judge comparisons for narrative coherence and commonsense reasoning. Also improves standard benchmarks (language understanding). Enables plug-and-play control via lightweight classifiers.
- **Code Available**: Yes — https://github.com/justinlovelace/STAR-LDM
- **Relevance**: Closest to our hypothesis in spirit — pauses generation to "think" about the plan for subsequent text at the sentence level. However, uses diffusion models for planning rather than CoT prompting.

### Paper 6: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou
- **Year**: 2022
- **Source**: arXiv:2201.11903
- **Key Contribution**: Demonstrates that prompting LLMs to generate intermediate reasoning steps dramatically improves performance on reasoning tasks. Foundation paper for the chain-of-thought paradigm.
- **Relevance**: Establishes CoT as a powerful technique. Our hypothesis is essentially a variant: what if CoT reasoning happens between every sentence rather than just before the final answer?

### Paper 7: s1: Simple Test-Time Scaling
- **Authors**: Muennighoff, Yang, Shi, Li, Li, Hajishirzi, Zettlemoyer, Liang, Candès, Hashimoto
- **Year**: 2025
- **Source**: arXiv:2501.19393
- **Key Contribution**: Shows that test-time compute scaling can be achieved simply by appending "Wait" tokens to extend the model's thinking process, often leading the model to double-check and correct its answer.
- **Results**: s1-32B exceeds o1-preview on MATH and AIME24 by up to 27%. Budget forcing (controlling thinking length) enables extrapolation beyond base performance.
- **Code Available**: Yes — https://github.com/simplescaling/s1
- **Relevance**: Demonstrates that simply extending the model's "thinking time" (via "Wait" tokens) improves reasoning. Directly supports our hypothesis that more deliberation leads to better outputs.

### Paper 8: Scaling LLM Test-Time Compute Optimally
- **Authors**: Snell, Lee, Xu, Kumar
- **Year**: 2024
- **Source**: arXiv:2408.03314
- **Key Contribution**: Systematic study of test-time compute scaling strategies. Shows that compute-optimal allocation per prompt can improve efficiency by 4x over best-of-N. A smaller model with optimal test-time compute can outperform a 14x larger model.
- **Relevance**: Provides theoretical and empirical framework for understanding why additional thinking time improves performance. Supports the idea that some prompts benefit more from deliberation than others.

---

## Common Methodologies

1. **Pause/Thinking Token Insertion**: Adding special tokens to the input to give the model more compute steps per output token (Goyal et al., Herel & Mikolov, Galashov et al.)
2. **Latent Space Reasoning**: Feeding hidden states back as input rather than decoding to tokens (Coconut, STAR-LDM, Compressed CoT)
3. **Budget Forcing / Test-Time Scaling**: Controlling the amount of compute at inference time via token budgets (s1, Snell et al.)
4. **Chain-of-Thought Prompting**: Eliciting explicit intermediate reasoning in natural language (Wei et al., and variations)

## Standard Baselines

- **Standard (no thinking)**: Direct generation without any additional computation
- **Chain-of-Thought**: Prompting the model to think step-by-step before answering
- **Few-shot filler tokens**: Appending periods or other characters (shown to NOT help without training)
- **Best-of-N sampling**: Generating N responses and selecting the best one

## Evaluation Metrics

- **Accuracy/Exact Match**: For closed-form tasks (math, QA, classification)
- **Perplexity**: For language modeling quality
- **LLM-as-Judge**: For open-ended generation quality (AlpacaEval, MT-Bench style)
- **ROUGE/BERTScore**: For summarization tasks
- **Factual consistency**: For measuring hallucination reduction (TruthfulQA)
- **Token efficiency**: Ratio of quality improvement to additional tokens used

## Datasets in the Literature

| Dataset | Used In | Task |
|---------|---------|------|
| GSM8k | Pause Tokens, Coconut, s1 | Math reasoning |
| SQuAD | Pause Tokens | Extractive QA |
| CommonSenseQA | Pause Tokens | Commonsense reasoning |
| HellaSwag | Pause Tokens | Sentence completion |
| ProsQA | Coconut | Graph-based logical reasoning |
| ProntoQA | Coconut | Logical reasoning |
| MATH | s1 | Competition math |
| AIME | s1 | Competition math |
| Penn TreeBank | Thinking Tokens | Language modeling |
| WikiText-2 | Thinking Tokens | Language modeling |

## Gaps and Opportunities

1. **No study of sentence-level thinking in prompting**: Existing work focuses on either (a) trained pause tokens at the token level, or (b) CoT before the final answer. No work studies prompting an off-the-shelf LLM to think between every sentence.
2. **Qualitative text analysis missing**: Prior work measures accuracy on benchmarks. No work systematically analyzes how thinking tokens change the *style*, *vocabulary*, *sentence structure*, or *coherence* of generated text.
3. **Off-the-shelf models underexplored**: Goyal et al. showed filler tokens don't help without training. But they only tested simple fillers ('...'), not structured thinking prompts. No work tests whether instructing a model to reason between sentences (using its own CoT capability) produces different text.
4. **Sentence-level granularity**: All existing work operates at the token level or the entire-response level. The sentence level — a natural unit of human writing — is unexplored.
5. **Open-ended generation**: Most benchmarks are closed-form tasks. The effect on open-ended writing, storytelling, and essay generation is unstudied.

## Recommendations for Our Experiment

Based on the literature review:

- **Recommended datasets**: GSM8k (reasoning), TruthfulQA (factual accuracy), CommonSenseQA (commonsense), HellaSwag (coherence) as primary benchmarks. AlpacaEval and MT-Bench for open-ended quality evaluation.
- **Recommended baselines**: Standard generation, standard CoT (think before answering), and filler tokens (periods between sentences) as negative control.
- **Recommended metrics**: Accuracy for closed-form tasks, LLM-as-judge for open-ended quality, plus qualitative analysis of text properties (sentence length, vocabulary diversity, hedging language, etc.).
- **Methodological considerations**:
  - The Pause Tokens paper's finding that filler tokens don't help suggests that *structured* thinking (not just delays) is important for off-the-shelf models.
  - The s1 paper's "Wait" token success suggests that even simple extensions of thinking time can help if the model has been trained to use them.
  - Coconut's BFS-like emergent reasoning suggests that latent thinking can differ qualitatively from explicit CoT.
  - Focus on comparing the *text itself* (not just accuracy) to address the "qualitatively different" part of the hypothesis.
