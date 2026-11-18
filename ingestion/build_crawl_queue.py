from config.settings import DATA_DIR
import json
from pathlib import Path
from urllib.parse import urlparse

from ingestion.web_crawler import canonicalize_url


SEED_PATH = (DATA_DIR / "raw/te_website.json")
DISCOVERED_PATH = (DATA_DIR / "raw/te_discovered_links.json")

OUTPUT_PATH = (DATA_DIR / "raw/te_crawl_queue.json")


BLOCKED_SUBSTRINGS = [
    "/main-content",
    "/wps/portal/",
    "/personal/''",
    "/services/''",
    "/check-availability",
]


def should_keep_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.netloc not in {
        "www.te.eg",
        "te.eg",
    }:
        return False

    path = parsed.path.lower().rstrip("/")

    if any(
        blocked in path
        for blocked in BLOCKED_SUBSTRINGS
    ):
        return False

    # Clearly suspicious / invalid routes
    if path.endswith("/delete"):
        return False

    if path.endswith("/max-"):
        return False

    return True


def get_language_hint(url: str) -> str:
    path = urlparse(url).path.lower()

    if path.startswith("/en/"):
        return "en"

    if path.startswith("/ar/"):
        return "ar"

    return "default"


def main():
    with SEED_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        seed_documents = json.load(file)

    with DISCOVERED_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        discovered_links = json.load(file)

    queue = {}
    excluded = []

    # --------------------------------
    # Existing 38 seed URLs
    # --------------------------------

    for document in seed_documents:
        url = canonicalize_url(
            document["url"]
        )

        if not should_keep_url(url):
            excluded.append(url)
            continue

        queue[url] = {
            "url": url,
            "depth": 0,
            "language_hint": get_language_hint(url),
            "discovered_from": [],
        }

    # --------------------------------
    # Newly discovered deep URLs
    # --------------------------------

    for item in discovered_links:
        url = canonicalize_url(
            item["url"]
        )

        if not should_keep_url(url):
            excluded.append(url)
            continue

        if url not in queue:
            queue[url] = {
                "url": url,
                "depth": 1,
                "language_hint": get_language_hint(url),
                "discovered_from": item.get(
                    "discovered_from",
                    [],
                ),
            }

    results = sorted(
        queue.values(),
        key=lambda item: item["url"],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Seed documents:", len(seed_documents))
    print(
        "Discovered links:",
        len(discovered_links),
    )

    print(
        "Final crawl queue:",
        len(results),
    )

    print(
        "Excluded:",
        len(set(excluded)),
    )

    print("\n--- EXCLUDED URLS ---\n")

    for url in sorted(set(excluded)):
        print(url)

    print(
        "\nSaved crawl queue to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()