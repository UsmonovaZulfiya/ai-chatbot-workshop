import os
import json
import time
import hashlib
from typing import List, Dict
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# 1. LOAD ENV
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

if not PINECONE_HOST:
    raise ValueError("Missing PINECONE_HOST in .env")

# ============================================================
# 2. CONFIG
# ============================================================

JSON_FILE = "tiue_en_pages.json"   # change if your file has another name
NAMESPACE = "tiue-en"
CLEAR_NAMESPACE_FIRST = False      # set True only if you want to wipe this namespace
BATCH_SIZE = 50

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# IMPORTANT:
# This field name should match your Pinecone integrated embedding field_map.
# If your index is configured with field_map {"text": "chunk_text"},
# then keep this as "chunk_text".
TEXT_FIELD_NAME = "text"

# ============================================================
# 3. PINECONE CLIENT
# ============================================================

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# ============================================================
# 4. OPTIONAL: CLEAR NAMESPACE
# ============================================================

if CLEAR_NAMESPACE_FIRST:
    print(f"Clearing namespace '{NAMESPACE}' ...")
    index.delete(delete_all=True, namespace=NAMESPACE)
    time.sleep(3)

# ============================================================
# 5. LOAD JSON
# ============================================================

with open(JSON_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)

print(f"Loaded {len(pages)} pages from JSON.")

# ============================================================
# 6. CHUNKER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ============================================================
# 7. HELPERS
# ============================================================

def make_doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def build_records_from_page(page: Dict) -> List[Dict]:
    content = (page.get("content") or "").strip()
    if not content:
        return []

    url = page.get("url", "")
    title = page.get("title", "")
    meta_description = page.get("meta_description", "")
    page_type = page.get("page_type", "general")
    source = page.get("source", "tiue_en_site")
    language = page.get("language", "en")
    last_crawled_at_unix = page.get("last_crawled_at_unix", None)

    doc_id = make_doc_id(url)

    chunks = text_splitter.split_text(content)

    records = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue

        record = {
            "_id": f"{doc_id}#chunk{i}",
            TEXT_FIELD_NAME: chunk,           # this is the text Pinecone will embed
            "document_id": doc_id,
            "chunk_number": i,
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "page_type": page_type,
            "source": source,
            "language": language,
            "last_crawled_at_unix": last_crawled_at_unix,
        }
        records.append(record)

    return records

def batchify(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# ============================================================
# 8. BUILD RECORDS
# ============================================================

all_records = []

for page in pages:
    page_records = build_records_from_page(page)
    all_records.extend(page_records)

print(f"Prepared {len(all_records)} chunk records for upload.")

if not all_records:
    raise ValueError("No records were created. Check your JSON file and content fields.")

# ============================================================
# 9. UPSERT TO PINECONE
# ============================================================

uploaded = 0

for batch in batchify(all_records, BATCH_SIZE):
    index.upsert_records(NAMESPACE, batch)
    uploaded += len(batch)
    print(f"Uploaded {uploaded}/{len(all_records)}")

print("Ingestion complete.")