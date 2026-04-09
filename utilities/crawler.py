import requests
import re
import trafilatura
import json
import html
import time
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

# ============================================================
# CONFIG
# ============================================================

START_URL = "https://tiue.uz/en/"
OUTPUT_FILE = "tiue_en_pages.json"

# Crawl scope
ALLOWED_DOMAIN = "tiue.uz"
ALLOWED_PATH_PREFIX = "/en/"
MAX_PAGES = 500
REQUEST_TIMEOUT = 20
DELAY_BETWEEN_REQUESTS = 0.5

# Content filtering
MIN_PARAGRAPH_LENGTH = 50
MIN_TOTAL_CONTENT_LENGTH = 200
PARAGRAPH_SIMILARITY_THRESHOLD = 0.92
PAGE_SIMILARITY_THRESHOLD = 0.98
SEEN_PARAGRAPH_BUFFER_SIZE = 300

# Extensions/pages to ignore
IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".doc", ".docx",
    ".xls", ".xlsx", ".css", ".js", ".xml", ".json", ".zip",
    ".mp4", ".mp3", ".svg", ".ico", ".webp", ".avi", ".mov",
    ".ppt", ".pptx", ".rar", ".7z"
)

# URL patterns to avoid (adjust if needed)
IGNORED_PATH_KEYWORDS = (
    "/tag/",
    "/author/",
    "/feed/",
    "/wp-json/",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ============================================================
# STATE
# ============================================================

visited_urls = set()
queued_urls = set()
seen_paragraphs = []
page_content_hashes = []
collected_data = []

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
})


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    """
    Normalize URL:
    - remove query and fragment
    - remove trailing slash except root
    - lowercase scheme and netloc
    """
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    # collapse duplicate slashes in path
    path = re.sub(r"/{2,}", "/", path)

    # remove trailing slash except root
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    normalized = urlunparse((scheme, netloc, path, "", "", ""))
    return normalized


def is_same_domain_or_subdomain(netloc: str, base_domain: str) -> bool:
    """
    Allow tiue.uz and subdomains such as www.tiue.uz
    """
    netloc = netloc.lower()
    return netloc == base_domain or netloc.endswith("." + base_domain)


def is_valid_url(url: str) -> bool:
    """
    Restrict crawl to English TIUE pages only.
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        if not is_same_domain_or_subdomain(parsed.netloc, ALLOWED_DOMAIN):
            return False

        if not parsed.path.startswith(ALLOWED_PATH_PREFIX):
            return False

        if parsed.path.lower().endswith(IGNORED_EXTENSIONS):
            return False

        for keyword in IGNORED_PATH_KEYWORDS:
            if keyword in parsed.path.lower():
                return False

        return True
    except Exception:
        return False


def should_skip_url(url: str) -> bool:
    """
    Additional skip logic for duplicated/undesirable URLs.
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Optional: skip obvious list/archive pages if you later find too much noise.
    # Keep them for now unless they are very thin.
    if path.endswith("/amp"):
        return True

    return False


def fetch_html(url: str) -> str | None:
    """
    Download HTML with requests for better control than direct fetch_url.
    """
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return None

        return response.text
    except Exception as e:
        print(f"[FETCH ERROR] {url} -> {e}")
        return None


def get_html_title(html_content: str) -> str:
    """
    Extract title from HTML <title>.
    """
    if not html_content:
        return "Unknown title"

    match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
    if match:
        raw_title = match.group(1).strip()
        return html.unescape(re.sub(r"\s+", " ", raw_title))
    return "Title not found"


def get_meta_description(html_content: str) -> str:
    """
    Extract meta description if available.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    if tag and tag.get("content"):
        return re.sub(r"\s+", " ", tag["content"]).strip()
    return ""


def extract_main_text(html_content: str) -> str:
    """
    Extract the main readable text using Trafilatura.
    """
    if not html_content:
        return ""

    extracted = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=True,
        deduplicate=False,
        favor_precision=True
    )
    return extracted or ""


def normalize_text(text: str) -> str:
    """
    Clean whitespace and normalize.
    """
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_redundant_paragraph(text: str, similarity_threshold: float = PARAGRAPH_SIMILARITY_THRESHOLD) -> bool:
    """
    Fuzzy duplicate check for paragraph-level repetition.
    """
    check_buffer = seen_paragraphs[-SEEN_PARAGRAPH_BUFFER_SIZE:]
    current_len = len(text)

    for seen_text in check_buffer:
        seen_len = len(seen_text)
        if not (0.8 * seen_len < current_len < 1.2 * seen_len):
            continue

        if SequenceMatcher(None, text, seen_text).ratio() >= similarity_threshold:
            return True

    return False


def clean_and_deduplicate_paragraphs(text_content: str) -> list[str]:
    """
    Split into paragraphs and remove weak/repeated content.
    """
    paragraphs = []
    raw_parts = re.split(r"\n+", text_content)

    for paragraph in raw_parts:
        paragraph = normalize_text(paragraph)

        if len(paragraph) < MIN_PARAGRAPH_LENGTH:
            continue

        # skip very repetitive menu-like lines
        word_count = len(paragraph.split())
        if word_count < 5:
            continue

        if is_redundant_paragraph(paragraph):
            continue

        seen_paragraphs.append(paragraph)
        paragraphs.append(paragraph)

    return paragraphs


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_near_duplicate_page(text: str) -> bool:
    """
    Exact hash check first, then approximate page-level similarity if needed.
    """
    if not text:
        return True

    new_hash = content_hash(text)
    if new_hash in page_content_hashes:
        return True

    # Optional approximate similarity check against recent pages
    recent_pages = page_content_hashes[-100:]
    if new_hash in recent_pages:
        return True

    page_content_hashes.append(new_hash)
    return False


def detect_page_type(url: str, title: str) -> str:
    """
    Simple heuristic page type tagging.
    """
    u = url.lower()
    t = title.lower()

    if "admission" in u or "admission" in t:
        return "admissions"
    if "news" in u or "news" in t:
        return "news"
    if "faculty" in u or "school" in u or "department" in u:
        return "academic"
    if "contact" in u or "contacts" in u:
        return "contact"
    if "about" in u:
        return "about"
    return "general"


def get_links(html_content: str, current_url: str) -> set[str]:
    """
    Extract internal English links from raw HTML.
    """
    links = set()

    if not html_content:
        return links

    soup = BeautifulSoup(html_content, "html.parser")

    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "").strip()
        if not href:
            continue

        absolute_url = urljoin(current_url, href)
        absolute_url = normalize_url(absolute_url)

        if not is_valid_url(absolute_url):
            continue

        if should_skip_url(absolute_url):
            continue

        if absolute_url not in visited_urls and absolute_url not in queued_urls:
            links.add(absolute_url)

    return links


def save_output(data: list[dict], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# MAIN CRAWL
# ============================================================

def crawl():
    start_url = normalize_url(START_URL)

    queue = deque([start_url])
    queued_urls.add(start_url)

    print(f"Starting crawl: {start_url}")
    print("=" * 60)

    while queue and len(visited_urls) < MAX_PAGES:
        current_url = queue.popleft()
        queued_urls.discard(current_url)

        if current_url in visited_urls:
            continue

        print(f"[{len(visited_urls)+1}/{MAX_PAGES}] Processing: {current_url}")
        visited_urls.add(current_url)

        html_content = fetch_html(current_url)
        if not html_content:
            continue

        title = get_html_title(html_content)
        meta_description = get_meta_description(html_content)

        extracted_text = extract_main_text(html_content)
        extracted_text = normalize_text(extracted_text)

        if not extracted_text:
            print("  -> No main content extracted.")
        else:
            clean_paragraphs = clean_and_deduplicate_paragraphs(extracted_text)

            if clean_paragraphs:
                final_content = "\n\n".join(clean_paragraphs)

                if len(final_content) >= MIN_TOTAL_CONTENT_LENGTH and not is_near_duplicate_page(final_content):
                    entry = {
                        "url": current_url,
                        "title": title,
                        "meta_description": meta_description,
                        "content": final_content,
                        "page_type": detect_page_type(current_url, title),
                        "source": "tiue_en_site",
                        "language": "en",
                        "last_crawled_at_unix": int(time.time())
                    }
                    collected_data.append(entry)
                    print(f"  -> Saved page with {len(final_content)} chars.")
                else:
                    print("  -> Skipped: too short or duplicate after cleaning.")
            else:
                print("  -> Skipped: no usable paragraphs after cleaning.")

        # Discover new links
        new_links = get_links(html_content, current_url)
        for link in new_links:
            queue.append(link)
            queued_urls.add(link)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    save_output(collected_data, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print(f"Crawl finished.")
    print(f"Visited URLs: {len(visited_urls)}")
    print(f"Saved pages:  {len(collected_data)}")
    print(f"Output file:  {OUTPUT_FILE}")


if __name__ == "__main__":
    crawl()