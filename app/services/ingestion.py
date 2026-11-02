import csv
import re
from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument


SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

# Manuals ship several languages in one PDF; keep pages where common English words are frequent
ENGLISH_WORDS = set("the and to of a in is you your for with be not on this that are or can it if as from".split())
MIN_ENGLISH_RATIO = 0.12


def is_english(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and sum(w in ENGLISH_WORDS for w in words) / len(words) > MIN_ENGLISH_RATIO


def load_manifest(folder: Path) -> dict[str, dict]:
    manifest = folder / "manifest.csv"
    if not manifest.exists():
        return {}
    with open(manifest, newline="", encoding="utf-8") as fh:
        return {row["file"]: row for row in csv.DictReader(fh)}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = [p for p in PyPDFLoader(str(path)).load() if is_english(p.page_content)]
        meta = load_manifest(path.parent).get(path.name)
        for p in pages:
            p.metadata = {"source": p.metadata["source"], "page": p.metadata["page"] + 1}
            if meta:
                p.metadata.update(brand=meta["brand"], category=meta["category"], model=meta["model"], market=meta["market"], url=meta["url"])
        return pages
    if suffix in {".txt", ".md"}:
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".docx":
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(page_content=text, metadata={"source": str(path)})]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, add_start_index=True)
    chunks = splitter.split_documents(list(docs))
    # Prefix each chunk with its appliance so retrieval can tell similar manuals apart
    for c in chunks:
        if "brand" in c.metadata:
            category = c.metadata["category"].replace("_", " ")
            c.page_content = f"[{c.metadata['brand']} {category} {c.metadata['model']}]\n{c.page_content}"
    return chunks
