# Resource Catalog: An LLM That's Careful With Its Words

## Papers

| # | File | Title | Authors | Year | ArXiv | Key Relevance |
|---|------|-------|---------|------|-------|---------------|
| 1 | `goyal2023_pause_tokens.pdf` | Think Before You Speak: Training Language Models With Pause Tokens | Goyal, Ji, Rawat, Menon, Kumar, Nagarajan | 2024 (ICLR) | 2310.02226 | Most directly relevant — learnable pause tokens improve downstream task performance in Transformers |
| 2 | `herel2024_thinking_tokens.pdf` | Thinking Tokens for Language Modeling | Herel, Mikolov | 2024 | 2405.08644 | Earliest thinking token concept; per-word thinking tokens in LSTMs |
| 3 | `galashov2025_catch_your_breath.pdf` | Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production | Galashov, Jones, Ke, Cao, Nagarajan, Mozer | 2025 | 2510.13879 | Adaptive pause tokens — model decides when to pause |
| 4 | `hao2024_coconut.pdf` | Training Large Language Models to Reason in a Continuous Latent Space (Coconut) | Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian | 2024 | 2412.06769 | Latent reasoning via hidden state feedback; emergent BFS-like reasoning |
| 5 | `lovelace2026_star_ldm.pdf` | Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning | Lovelace, Belardi, Zalouk, Polavaram, Kundurthy, Weinberger | 2025 (COLM) | 2602.20528 | Sentence-level latent diffusion planning during AR generation |
| 6 | `wei2022_chain_of_thought.pdf` | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | Wei, Wang, Schuurmans, Bosma, et al. | 2022 | 2201.11903 | Foundational CoT paper |
| 7 | `muennighoff2025_s1_test_time_scaling.pdf` | s1: Simple Test-Time Scaling | Muennighoff, Yang, Shi, et al. | 2025 | 2501.19393 | "Wait" token for test-time compute scaling |
| 8 | `snell2024_scaling_test_time_compute.pdf` | Scaling LLM Test-Time Compute Optimally | Snell, Lee, Xu, Kumar | 2024 | 2408.03314 | Theoretical framework for test-time compute allocation |
| 9 | `cheng2024_compressed_cot.pdf` | Compressed Chain of Thought | Cheng et al. | 2024 | — | Compressing CoT into fewer tokens |
| 10 | `chen2025_latent_cot_survey.pdf` | Latent Chain-of-Thought Survey | Chen et al. | 2025 | — | Survey of latent reasoning approaches |
| 11 | `geiping2025_latent_reasoning_recurrent.pdf` | Latent Reasoning with Recurrent Models | Geiping et al. | 2025 | — | Recurrent architectures for latent reasoning |
| 12 | `gozeten2025_continuous_cot.pdf` | Continuous Chain of Thought | Gozeten et al. | 2025 | — | Continuous-space CoT representations |
| 13 | `hassid2025_dont_overthink.pdf` | Don't Overthink It | Hassid et al. | 2025 | — | When additional thinking hurts performance |
| 14 | `liu2025_thoughtbubbles.pdf` | Thought Bubbles | Liu et al. | 2025 | — | Inline thinking annotations for LLMs |
| 15 | `madaan2023_self_refine.pdf` | Self-Refine | Madaan et al. | 2023 | — | Iterative self-refinement without supervision |
| 16 | `bae2025_mixture_of_recursions.pdf` | Mixture of Recursions | Bae et al. | 2025 | — | Recursive computation for reasoning |
| 17 | `oomerjee2025_bottlenecked_transformers.pdf` | Bottlenecked Transformers | Oomerjee et al. | 2025 | — | Information bottleneck approaches for thinking |
| 18 | `ringel2025_continue_thinking_token.pdf` | Continue Thinking Token | Ringel et al. | 2025 | — | Tokens that extend reasoning chains |
| 19 | `sun2025_latent_tokens.pdf` | Latent Tokens | Sun et al. | 2025 | — | Learnable latent tokens for additional computation |
| 20 | `wang2023_planning_tokens.pdf` | Planning Tokens | Wang et al. | 2023 | — | Planning-oriented special tokens |
| 21 | `wang2025_system15_reasoning.pdf` | System 1.5 Reasoning | Wang et al. | 2025 | — | Hybrid System 1/System 2 reasoning |
| 22 | `zeng2025_ponder_continuous.pdf` | Ponder: Continuous Reasoning | Zeng et al. | 2025 | — | Continuous pondering for improved generation |

**Total**: 22 papers downloaded to `papers/`

---

## Datasets

| # | Directory | HuggingFace ID | Size | Task | Splits | License |
|---|-----------|---------------|------|------|--------|---------|
| 1 | `datasets/gsm8k/` | `openai/gsm8k` | 8,792 examples (~2.7 MB) | Multi-step math reasoning | train (7,473), test (1,319) | MIT |
| 2 | `datasets/truthfulqa/` | `truthfulqa/truthful_qa` | 817 questions (~223 KB) | Factual accuracy (adversarial) | validation (817) | Apache-2.0 |
| 3 | `datasets/commonsense_qa/` | `tau/commonsense_qa` | 12,102 examples (~1.6 MB) | Multiple-choice commonsense reasoning | train (9,741), val (1,221), test (1,140) | MIT |
| 4 | `datasets/hellaswag/` | `Rowan/hellaswag` | ~59,950 examples (~71 MB) | Commonsense NLI / sentence completion | train (39,905), val (10,042), test (10,003) | MIT |

**Loading example**:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

Sample data files are available in `datasets/samples/` for quick inspection without loading full datasets.

### Recommended Additional Datasets (Not Downloaded)

| Dataset | HuggingFace ID | Size | Task | Priority |
|---------|---------------|------|------|----------|
| XSum | `EdinburghNLP/xsum` | ~226K examples | Abstractive summarization | Medium |
| AlpacaEval 2.0 | `tatsu-lab/alpaca_eval` | 805 instructions | Open-ended generation quality | High |
| MT-Bench | `lmsys/mt_bench_human_judgments` | 80 questions | Multi-turn conversation quality | High |

---

## Code Repositories

| # | Directory | Repository | Stars | Key Features | Language |
|---|-----------|-----------|-------|-------------|----------|
| 1 | `code/coconut/` | [facebookresearch/coconut](https://github.com/facebookresearch/coconut) | — | Chain of Continuous Thought; latent reasoning via hidden state feedback; GPT-2 based | Python |
| 2 | `code/s1/` | [simplescaling/s1](https://github.com/simplescaling/s1) | — | Simple test-time scaling with "Wait" tokens; budget forcing; s1-32B model | Python |
| 3 | `code/star-ldm/` | [justinlovelace/STAR-LDM](https://github.com/justinlovelace/STAR-LDM) | — | Latent diffusion planning + AR generation; Sentence-T5 embeddings; DiT architecture | Python |

### Other Notable Repositories (Not Cloned)

| Repository | URL | Relevance |
|-----------|-----|-----------|
| Pause Tokens | Not publicly available (Google internal) | Most relevant paper but no code release |
| Thinking Tokens | Not publicly available | LSTM-based, limited applicability |
| Catch Your Breath | Not publicly available (Google DeepMind) | Adaptive pausing, no code release |

---

## Resource Relevance Matrix

This matrix maps resources to the core research questions:

| Resource | Thinking Between Sentences | Off-the-Shelf Models | Text Quality Analysis | Reasoning Benchmarks |
|----------|---------------------------|---------------------|----------------------|---------------------|
| Pause Tokens paper | High | Low (requires training) | Low | High |
| Thinking Tokens paper | High | Low (requires training) | Medium | Low |
| Catch Your Breath paper | High | Low (requires training) | Low | Medium |
| Coconut paper + code | Medium | Low (requires training) | Low | High |
| STAR-LDM paper + code | High | Low (requires training) | Medium | Medium |
| CoT paper | Medium | High | Low | High |
| s1 paper + code | Medium | High | Low | High |
| Scaling TTC paper | Medium | High | Low | High |
| GSM8k dataset | Low | — | — | High |
| TruthfulQA dataset | Low | — | High | Medium |
| CommonSenseQA dataset | Low | — | — | High |
| HellaSwag dataset | Low | — | Medium | High |

### Key Insight from Resources

The most significant gap identified across all resources: **no prior work studies prompting an off-the-shelf LLM to reason between every sentence using its native CoT capability**. All thinking/pause token work requires training modifications. Our hypothesis — that structured inter-sentence CoT prompting with existing models produces qualitatively different text — is novel and directly testable.

---

## Directory Structure

```
careful-llm-words-aaec-claude/
├── literature_review.md          # Comprehensive literature review
├── resources.md                  # This file — resource catalog
├── papers/                       # Downloaded PDFs (22 papers)
│   ├── README.md                 # Paper documentation
│   ├── pages/                    # Chunked PDFs for deep reading
│   └── *.pdf                     # Individual papers
├── datasets/                     # Downloaded datasets (4 datasets)
│   ├── README.md                 # Dataset documentation with download instructions
│   ├── .gitignore                # Excludes large data files from git
│   ├── samples/                  # Small sample files for quick inspection
│   │   ├── gsm8k_samples.json
│   │   └── truthfulqa_samples.json
│   ├── gsm8k/
│   ├── truthfulqa/
│   ├── commonsense_qa/
│   └── hellaswag/
└── code/                         # Cloned repositories (3 repos)
    ├── README.md                 # Repository documentation
    ├── coconut/
    ├── s1/
    └── star-ldm/
```
