"""Content-based relevance: does a chunk contain the facts a correct answer needs?"""
import re

MIN_FACT_COVERAGE = 0.7  # share of a fact's words that must appear in the chunk


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def supports(chunk_text: str, key_facts: list[str]) -> bool:
    """True if at least half of the key facts are (mostly) present in the chunk."""
    chunk = words(chunk_text)
    covered = 0
    for fact in key_facts:
        fw = words(fact)
        if fw and len(fw & chunk) / len(fw) >= MIN_FACT_COVERAGE:
            covered += 1
    return covered * 2 >= len(key_facts)
