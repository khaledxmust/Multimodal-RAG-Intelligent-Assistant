from config.settings import DATA_DIR
import json
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from ingestion.web_crawler import (
    canonicalize_url,
    is_relevant_url,
)


INPUT_PATH = (DATA_DIR / "raw/te_website.json")
OUTPUT_PATH = (DATA_DIR / "raw/te_discovered_links.json")


def load_seed_urls() -> set[str]:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        documents = json.load(file)

    return {
        canonicalize_url(document["url"])
        for document in documents
    }


def main():
    seed_urls = load_seed_urls()

    print("Existing URLs:", len(seed_urls))
    print()

    # Store:
    # discovered_url -> pages where we found it
    discovered = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )

        page = browser.new_page(
            viewport={
                "width": 1400,
                "height": 1000,
            }
        )

        for index, source_url in enumerate(
            sorted(seed_urls),
            start=1,
        ):
            print(
                f"[{index}/{len(seed_urls)}] "
                f"Opening: {source_url}"
            )

            try:
                page.goto(
                    source_url,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )

                # Give JavaScript some time to render links
                page.wait_for_timeout(2500)

                hrefs = page.locator(
                    "a[href]"
                ).evaluate_all(
                    """
                    elements =>
                        elements.map(element => element.href)
                    """
                )

                relevant_count = 0

                for href in hrefs:
                    if not href:
                        continue

                    if not href.startswith(
                        ("http://", "https://")
                    ):
                        continue

                    clean_url = canonicalize_url(href)

                    if not is_relevant_url(clean_url):
                        continue

                    # We already crawled this URL
                    if clean_url in seed_urls:
                        continue

                    relevant_count += 1

                    if clean_url not in discovered:
                        discovered[clean_url] = set()

                    discovered[clean_url].add(source_url)

                print(
                    f"   New relevant links found: "
                    f"{relevant_count}"
                )

            except PlaywrightTimeoutError:
                print("   -> Timeout")

            except Exception as error:
                print(
                    "   -> Error:",
                    error,
                )

        browser.close()

    results = []

    for url in sorted(discovered):
        results.append(
            {
                "url": url,
                "discovered_from": sorted(
                    discovered[url]
                ),
            }
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

    print("\n" + "=" * 80)
    print("Existing URLs:", len(seed_urls))
    print(
        "New unique relevant URLs:",
        len(results),
    )

    print("\n--- NEW URLS ---\n")

    for item in results:
        print(item["url"])

    print(
        "\nSaved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

