"""Download the appliance manuals listed in data/appliance_kb/manifest.csv.

The PDFs are not committed to Git; run this once after cloning.
"""
import csv
import urllib.request
from pathlib import Path
from app.core.config import get_settings

settings = get_settings()

kb_dir = Path(settings.kb_dir)
with open(kb_dir / "manifest.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

for row in rows:
    dest = kb_dir / row["file"]
    if dest.exists():
        print(f"skip  {row['file']}")
        continue
    urllib.request.urlretrieve(row["url"], dest)
    print(f"saved {row['file']} ({dest.stat().st_size // 1024} KB)")

print(f"{len(rows)} manuals in {kb_dir}")
