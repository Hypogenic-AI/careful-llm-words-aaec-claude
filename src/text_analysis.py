"""Text analysis utilities for measuring qualitative differences between conditions."""

import re
from collections import Counter

# Hedging words/phrases
HEDGE_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "likely", "unlikely",
    "probably", "arguably", "potentially", "seemingly", "apparently",
    "roughly", "approximately", "generally", "typically", "tends to",
    "it seems", "it appears", "in some cases", "to some extent",
    "it is possible", "it is likely", "not necessarily", "debatable",
    "uncertain", "some argue", "some believe", "on the other hand",
]

# Self-correction / reconsideration markers
SELF_CORRECTION_MARKERS = [
    "actually", "however", "on second thought", "more precisely",
    "to be more accurate", "that said", "but", "although",
    "nevertheless", "in fact", "correction", "rather",
    "more accurately", "to clarify", "to be fair", "admittedly",
    "it should be noted", "importantly", "notably",
]

# Qualifying/conditional markers
QUALIFYING_MARKERS = [
    "if", "unless", "provided that", "assuming", "depending on",
    "in certain cases", "under some circumstances", "with the caveat",
    "it depends", "context matters", "this varies",
]


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks to get only visible output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def count_sentences(text: str) -> int:
    """Count sentences using simple period/question/exclamation splitting."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def sentence_lengths(text: str) -> list[int]:
    """Get word count per sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [len(s.split()) for s in sentences if s.strip()]


def type_token_ratio(text: str) -> float:
    """Lexical diversity: unique words / total words."""
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def count_pattern_occurrences(text: str, patterns: list[str]) -> int:
    """Count how many times patterns from a list appear in text."""
    text_lower = text.lower()
    count = 0
    for pattern in patterns:
        count += len(re.findall(r'\b' + re.escape(pattern) + r'\b', text_lower))
    return count


def hedge_rate(text: str) -> float:
    """Hedging words per sentence."""
    n_sentences = count_sentences(text)
    if n_sentences == 0:
        return 0.0
    return count_pattern_occurrences(text, HEDGE_WORDS) / n_sentences


def self_correction_rate(text: str) -> float:
    """Self-correction markers per sentence."""
    n_sentences = count_sentences(text)
    if n_sentences == 0:
        return 0.0
    return count_pattern_occurrences(text, SELF_CORRECTION_MARKERS) / n_sentences


def qualifying_rate(text: str) -> float:
    """Qualifying/conditional markers per sentence."""
    n_sentences = count_sentences(text)
    if n_sentences == 0:
        return 0.0
    return count_pattern_occurrences(text, QUALIFYING_MARKERS) / n_sentences


def word_count(text: str) -> int:
    """Total word count."""
    return len(re.findall(r'\b\w+\b', text))


def avg_sentence_length(text: str) -> float:
    """Average words per sentence."""
    lengths = sentence_lengths(text)
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def analyze_text(text: str) -> dict:
    """Compute all text metrics for a single text."""
    visible = strip_think_tags(text)
    return {
        "word_count": word_count(visible),
        "sentence_count": count_sentences(visible),
        "avg_sentence_length": round(avg_sentence_length(visible), 2),
        "type_token_ratio": round(type_token_ratio(visible), 4),
        "hedge_rate": round(hedge_rate(visible), 4),
        "self_correction_rate": round(self_correction_rate(visible), 4),
        "qualifying_rate": round(qualifying_rate(visible), 4),
        "hedge_count": count_pattern_occurrences(visible, HEDGE_WORDS),
        "self_correction_count": count_pattern_occurrences(visible, SELF_CORRECTION_MARKERS),
        "qualifying_count": count_pattern_occurrences(visible, QUALIFYING_MARKERS),
    }


def count_think_blocks(text: str) -> int:
    """Count the number of <think> blocks in a response."""
    return len(re.findall(r'<think>', text))


def extract_think_content(text: str) -> list[str]:
    """Extract the content of all <think> blocks."""
    return re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
