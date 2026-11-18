from config.settings import DATA_DIR
import json
import time
from pathlib import Path
import requests
import trafilatura

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


BASE_DOMAIN = "te.eg"
RAW_DATA_DIR = (DATA_DIR / "raw")

EXCLUDED_KEYWORDS = [
    "/about-te/",
    "/corporate-sustainability/",
    "/careers",
    "/board-of-directors",
    "/management-team",
    "/press-releases",
    "/store-locator",
    "/live-chat",
    "/echannel/",
    "/user/login",
    "/devices",
    "/shop",

    # Initial knowledge-base exclusions
    "/personal/sitemap",
    "/personal/promotions",
    "/collect-we-egyptian-premier-league-tickets",
    "/sahel-stores",
    "/real-state-registration-office",
]


INCLUDED_KEYWORDS = [
    "/personal",
    "/5gmobile",
    "/check-availability",
    "/kalamy-ar",
    "/marhaba-ar",
    "/nitro-mifi",
]


def get_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    return response.text


def extract_text(html: str) -> str:
    text = trafilatura.extract(
        html,
        include_links=True,
        include_tables=True,
    )

    return text or ""


def extract_internal_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        full_url = urljoin(base_url, href)

        parsed = urlparse(full_url)

        if (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.endswith(BASE_DOMAIN)
        ):
            clean_url = (
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            )

            links.add(clean_url)

    return sorted(links)


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)

    path = parsed.path.rstrip("/")

    # Telecom Egypt exposes some pages under both:
    # /personal/...
    # /web/guest/personal/...
    if path.startswith("/web/guest/"):
        path = path.replace("/web/guest", "", 1)

    return f"https://www.te.eg{path}"


def is_relevant_url(url: str) -> bool:
    url_lower = url.lower()

    if any(
        keyword in url_lower
        for keyword in EXCLUDED_KEYWORDS
    ):
        return False

    if any(
        keyword in url_lower
        for keyword in INCLUDED_KEYWORDS
    ):
        return True

    return False


def crawl_pages(urls: list[str]) -> list[dict]:
    documents = []

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Crawling: {url}")

        try:
            html = get_html(url)
            text = extract_text(html)

            if not text.strip():
                print("  -> Skipped: no text")
                continue

            document = {
                "document_id": f"te_{index:04d}",
                "source_type": "website",
                "source_name": "Telecom Egypt",
                "url": url,
                "text": text,
            }

            documents.append(document)

            print(
                f"  -> Extracted {len(text):,} characters"
            )

        except requests.RequestException as exc:
            print(f"  -> Failed: {exc}")

        # Be polite to the website
        time.sleep(1)

    return documents


def save_documents(documents: list[dict]) -> None:
    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RAW_DATA_DIR / "te_website.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    sitemap_url = "https://www.te.eg/personal/sitemap/"

    print("Fetching sitemap:", sitemap_url)

    html = get_html(sitemap_url)

    all_links = extract_internal_links(
        html=html,
        base_url=sitemap_url,
    )

    relevant_links = sorted({
        canonicalize_url(link)
        for link in all_links
        if is_relevant_url(link)
    })

    print("\nTotal discovered links:", len(all_links))
    print("Relevant unique links:", len(relevant_links))

    print("\n--- URLS TO CRAWL ---\n")

    for link in relevant_links:
        print(link)

    print("\nStarting crawl...\n")

    documents = crawl_pages(relevant_links)

    print(
        f"\nSuccessfully crawled "
        f"{len(documents)}/{len(relevant_links)} pages."
    )

    save_documents(documents)


