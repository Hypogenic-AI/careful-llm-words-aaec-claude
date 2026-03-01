# Cloned Code Repositories

This directory contains 3 code repositories relevant to the research topic "An LLM That's Careful With Its Words."

## Repositories

### 1. Coconut (Chain of Continuous Thought)
- **Directory**: `coconut/`
- **Source**: https://github.com/facebookresearch/coconut
- **Paper**: Hao et al. (2024) — "Training Large Language Models to Reason in a Continuous Latent Space"
- **Description**: Implements latent reasoning by feeding the LLM's last hidden state back as the next input embedding, enabling reasoning in a continuous latent space rather than discrete tokens. Based on GPT-2.
- **Relevance**: Demonstrates that latent "thinking" can outperform explicit chain-of-thought on tasks requiring search and planning. Provides code for training and evaluating continuous thought models.

### 2. s1 (Simple Test-Time Scaling)
- **Directory**: `s1/`
- **Source**: https://github.com/simplescaling/s1
- **Paper**: Muennighoff et al. (2025) — "s1: Simple Test-Time Scaling"
- **Description**: Achieves test-time compute scaling by appending "Wait" tokens to extend the model's thinking process, enabling budget forcing (controlling thinking length). s1-32B exceeds o1-preview on MATH and AIME24.
- **Relevance**: Directly demonstrates that extending "thinking time" with simple token interventions improves reasoning. The budget forcing technique is conceptually related to our sentence-level thinking approach.

### 3. STAR-LDM (Stop-Think-AutoRegress with Latent Diffusion)
- **Directory**: `star-ldm/`
- **Source**: https://github.com/justinlovelace/STAR-LDM
- **Paper**: Lovelace et al. (2025) — "Stop-Think-AutoRegress: Language Modeling with Latent Diffusion Planning"
- **Description**: Integrates latent diffusion planning into autoregressive generation. The model pauses generation, performs diffusion-based planning in sentence embedding space (Sentence-T5 XL), then resumes AR generation guided by the refined plan.
- **Relevance**: Closest to our hypothesis in spirit — pauses generation to "think" about subsequent text at the sentence level. Achieves >70% win rates in LLM-as-judge for narrative coherence.

## Notes

- These repositories are cloned as-is from GitHub and are not modified
- Each repository has its own license and dependencies — refer to individual READMEs
- Not all relevant papers have public code (notably: Pause Tokens, Thinking Tokens, Catch Your Breath)
