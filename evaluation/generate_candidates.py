"""Draft candidate evaluation questions from manual pages (reviewed by hand before use).

Picks a few content-rich pages per manual and asks an LLM to write one question that the page
answers, with a short reference answer. Output: evaluation/candidates.jsonl

Run from the project root:  python -m evaluation.generate_candidates
"""
import json
import random
import re
import time
from pathlib import Path
from langchain_groq import ChatGroq
from pydantic import BaseModel
from app.core.config import get_settings
from app.services.ingestion import load_file, load_manifest

settings = get_settings()
PAGES_PER_MANUAL = 3
DRAFT_MODEL = "openai/gpt-oss-20b"  # separate rate-limit bucket from the copilot's model
USEFUL = re.compile(r"clean|troubleshoot|fault|error|what to do|program|setting|filter|descal|child|lock|install|temperature|defrost|ice|drain|detergent|salt|rinse", re.I)
OUT = Path(__file__).parent / "candidates.jsonl"


class Candidate(BaseModel):
    skip: bool
    question: str = ""
    reference_answer: str = ""
    key_facts: list[str] = []


def is_content_page(text: str) -> bool:
    toc_dots = text.count("....")
    return len(text) > 900 and toc_dots < 5 and len(USEFUL.findall(text)) >= 3


def main():
    llm = ChatGroq(model=DRAFT_MODEL, temperature=0, api_key=settings.groq_api_key, max_retries=6)
    drafter = llm.with_structured_output(Candidate, method="json_mode")
    kb_dir = Path(settings.kb_dir)
    manifest = load_manifest(kb_dir)
    rng = random.Random(42)

    with open(OUT, "w", encoding="utf-8") as out:
        for file, meta in manifest.items():
            pages = [p for p in load_file(kb_dir / file) if is_content_page(p.page_content)]
            picks = sorted(rng.sample(pages, min(PAGES_PER_MANUAL, len(pages))), key=lambda p: p.metadata["page"])
            for page in picks:
                appliance = f"{meta['brand']} {meta['category'].replace('_', ' ')} ({meta['model']})"
                c = drafter.invoke(f"""
You write test questions for a home-appliance support chatbot.
Below is one page from the user manual of a {appliance}.
Write ONE realistic question a customer might ask that THIS page answers completely and specifically.
Rules:
- Phrase it like a real customer, not like the manual. Mention the brand and appliance type.
- The answer must come only from this page; do not use outside knowledge.
- reference_answer: 1-3 sentences. key_facts: 1-4 short facts copied from the page that a correct answer must contain.
- If the page is only safety boilerplate, a table of contents, or has nothing a customer would ask about, set skip=true.
Return JSON: {{"skip": false, "question": "...", "reference_answer": "...", "key_facts": ["..."]}}

Manual page {page.metadata['page']}:
{page.page_content[:3500]}
""")
                if c.skip or not c.question:
                    print(f"skip  {file} p.{page.metadata['page']}")
                    continue
                row = {"question": c.question, "reference_answer": c.reference_answer, "key_facts": c.key_facts,
                       "expected_path": "kb", "gold_file": file, "gold_page": page.metadata["page"],
                       "category": meta["category"], "brand": meta["brand"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"ok    {file} p.{page.metadata['page']}: {c.question}")
                time.sleep(4)  # stay under the free-tier tokens-per-minute limit


if __name__ == "__main__":
    main()
