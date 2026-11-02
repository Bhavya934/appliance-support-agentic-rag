import time
from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.rag.vectorstore import add_documents

settings = get_settings()
BATCH_SIZE = 90  # Pinecone inference embeds at most 96 inputs per request

folder = Path(settings.kb_dir)
files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
if not files:
    raise SystemExit(f"No documents in {folder}. Run: python download_manuals.py")

all_docs = []
for path in files:
    pages = load_file(path)
    print(f"{path.name}: {len(pages)} English pages")
    all_docs.extend(pages)

chunks = chunk_documents(all_docs)
print(f"{len(files)} files -> {len(all_docs)} pages -> {len(chunks)} chunks")

uploaded = 0
for start in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[start:start + BATCH_SIZE]
    for attempt in range(5):
        try:
            uploaded += len(add_documents(batch))
            break
        except Exception as exc:  # free-tier rate limits: back off and retry
            wait = 15 * (attempt + 1)
            print(f"  batch {start // BATCH_SIZE + 1} failed ({exc.__class__.__name__}), retrying in {wait}s")
            time.sleep(wait)
    else:
        raise SystemExit(f"Gave up after {uploaded} vectors")
    print(f"  {uploaded}/{len(chunks)} vectors uploaded")

print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {uploaded} Pinecone vectors")
