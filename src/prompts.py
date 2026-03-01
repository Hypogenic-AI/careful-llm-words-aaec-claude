"""Prompt templates for the three experimental conditions."""

# ─── System prompts for each condition ───

STANDARD_SYSTEM = "You are a helpful assistant. Answer the user's question directly and clearly."

COT_SYSTEM = "You are a helpful assistant. Think step by step before giving your answer."

INTER_SENTENCE_COT_SYSTEM = """You are a careful, deliberate writer. Before writing EACH sentence of your response, you must first think carefully about what to say next inside <think> tags.

Your response must follow this exact pattern:
<think>[Your internal reasoning about what the first sentence should say]</think>
[First visible sentence.]
<think>[Your reasoning about what the second sentence should say, considering what you just wrote]</think>
[Second visible sentence.]
<think>[Continue this pattern for every sentence...]</think>
[Next sentence.]

Rules:
1. Every single sentence you write must be preceded by a <think> block.
2. In each <think> block, consider: Does this sentence logically follow? Is it accurate? Could I phrase it better? Am I being precise?
3. Your visible sentences (outside <think> tags) form the actual response the reader sees.
4. Be thoughtful and deliberate — this is your chance to carefully consider each claim before committing to it."""


# ─── Open-ended prompts for Experiment 1 ───

OPEN_ENDED_PROMPTS = [
    "Explain why the sky appears blue during the day but red at sunset.",
    "What are the main arguments for and against nuclear energy?",
    "Describe how vaccines work and why herd immunity matters.",
    "Explain the concept of inflation in economics and how it affects ordinary people.",
    "What are the ethical implications of artificial intelligence in healthcare?",
    "Explain why some species go extinct while others thrive.",
    "What are the key differences between democracy and authoritarianism?",
    "How does climate change affect ocean ecosystems?",
    "Explain the relationship between sleep and memory consolidation.",
    "What are the pros and cons of remote work for employees and employers?",
    "How did the invention of the printing press change society?",
    "Explain why antibiotics are becoming less effective over time.",
    "What role does gut microbiome play in human health?",
    "Describe the main causes and consequences of urbanization.",
    "How does the placebo effect work, and what does it tell us about the mind-body connection?",
    "What are the key factors that led to the Industrial Revolution?",
    "Explain the concept of opportunity cost and give real-world examples.",
    "How do black holes form and what happens at the event horizon?",
    "What are the psychological effects of social media on teenagers?",
    "Explain why biodiversity is important for ecosystem stability.",
    "What are the main theories about the origin of consciousness?",
    "How does gerrymandering affect democratic representation?",
    "Explain the difference between correlation and causation with examples.",
    "What are the main challenges of space colonization?",
    "How does the criminal justice system balance punishment and rehabilitation?",
    "Explain why some countries are wealthier than others.",
    "What are the potential benefits and risks of gene editing technologies like CRISPR?",
    "How do languages evolve and why do some die out?",
    "Explain the tragedy of the commons and give modern examples.",
    "What factors contribute to the rise and fall of empires throughout history?",
]


def format_open_ended(prompt: str, condition: str) -> list[dict]:
    """Format an open-ended prompt for a given condition."""
    if condition == "standard":
        return [
            {"role": "system", "content": STANDARD_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    elif condition == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    elif condition == "inter_sentence_cot":
        return [
            {"role": "system", "content": INTER_SENTENCE_COT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def format_truthfulqa(question: str, condition: str) -> list[dict]:
    """Format a TruthfulQA question for a given condition."""
    if condition == "standard":
        return [
            {"role": "system", "content": STANDARD_SYSTEM},
            {"role": "user", "content": question},
        ]
    elif condition == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nThink step by step, then provide your answer."},
        ]
    elif condition == "inter_sentence_cot":
        return [
            {"role": "system", "content": INTER_SENTENCE_COT_SYSTEM},
            {"role": "user", "content": question},
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def format_gsm8k(question: str, condition: str) -> list[dict]:
    """Format a GSM8k question for a given condition."""
    suffix = "\n\nGive your final numerical answer after ####."
    if condition == "standard":
        return [
            {"role": "system", "content": STANDARD_SYSTEM},
            {"role": "user", "content": question + suffix},
        ]
    elif condition == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": question + "\n\nThink step by step. Give your final numerical answer after ####."},
        ]
    elif condition == "inter_sentence_cot":
        return [
            {"role": "system", "content": INTER_SENTENCE_COT_SYSTEM},
            {"role": "user", "content": question + suffix},
        ]
    else:
        raise ValueError(f"Unknown condition: {condition}")
