from config.settings import DATA_DIR
import json
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


INPUT_PATH = (DATA_DIR / "raw/te_crawl_queue.json")
OUTPUT_PATH = (DATA_DIR / "raw/te_website_content.json")


def load_queue() -> list[dict]:
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_documents(documents: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )


def scroll_page(page):
    """
    Scroll gradually to trigger lazy-loaded content.
    """

    for _ in range(6):
        page.evaluate(
            """
            window.scrollBy(
                0,
                window.innerHeight
            )
            """
        )

        page.wait_for_timeout(300)

    page.evaluate(
        "window.scrollTo(0, 0)"
    )


def extract_page_content(page) -> str:
    """
    Extract the useful Telecom Egypt page content while
    avoiding headers, navigation menus, breadcrumbs,
    comparison widgets, and footers.
    """

    # ---------------------------------
    # Strategy 1:
    # Normal TE product/service pages
    # ---------------------------------

    main_template = page.locator(
        ".main_page_template_container"
    )

    if main_template.count() > 0:
        texts = []

        for i in range(main_template.count()):
            try:
                text = (
                    main_template
                    .nth(i)
                    .inner_text()
                    .strip()
                )

                if text:
                    texts.append(text)

            except Exception:
                pass

        if texts:
            return "\n\n".join(texts)

    # ---------------------------------
    # Strategy 2:
    # Fallback for other TE page types
    # ---------------------------------

    main_content = page.locator("#main-content")

    if main_content.count() == 0:
        return ""

    children = main_content.locator(
        ":scope > *"
    )

    useful_parts = []

    excluded_class_keywords = [
        "main-header",
        "secondary-header",
        "breadcrumb",
        "comparison-page-cards",
        "back-to-top",
    ]

    footer_markers = [
        "جميع حقوق النشر محفوظة",
        "All rights reserved",
        "حمل التطبيق من هنا",
    ]

    for i in range(children.count()):
        try:
            element = children.nth(i)

            classes = (
                element.get_attribute("class")
                or ""
            ).lower()

            # Skip known layout/navigation blocks
            if any(
                keyword in classes
                for keyword in excluded_class_keywords
            ):
                continue

            text = (
                element.inner_text()
                .strip()
            )

            if not text:
                continue

            # Skip footer-like blocks
            if any(
                marker.lower() in text.lower()
                for marker in footer_markers
            ):
                continue

            useful_parts.append(text)

        except Exception:
            pass

    return "\n\n".join(
        useful_parts
    ).strip()


def main():
    crawl_queue = load_queue()

    print(
        "Pages to crawl:",
        len(crawl_queue),
    )

    documents = []

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

        for index, item in enumerate(
            crawl_queue,
            start=1,
        ):
            url = item["url"]

            print(
                f"\n[{index}/{len(crawl_queue)}] "
                f"{url}"
            )

            try:
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=60000,
                )

                # Allow JavaScript to render
                page.wait_for_timeout(2000)

                # Trigger lazy-loaded content
                scroll_page(page)

                page.wait_for_timeout(1000)

                # Extract only useful page content
                text = extract_page_content(page)

                title = page.title()

                document = {
                    "document_id": f"te_{index:04d}",
                    "source_type": "website",
                    "source_name": "Telecom Egypt",
                    "url": url,
                    "final_url": page.url,
                    "title": title,
                    "language_hint": item.get(
                        "language_hint"
                    ),
                    "depth": item.get("depth"),
                    "discovered_from": item.get(
                        "discovered_from",
                        [],
                    ),
                    "text": text.strip(),
                }

                documents.append(document)

                print(
                    "   Characters:",
                    len(text.strip()),
                )

                # Save progress after every page
                save_documents(documents)

            except PlaywrightTimeoutError:
                print("   -> Timeout")

            except Exception as error:
                print(
                    "   -> Error:",
                    error,
                )

        browser.close()

    print("\n" + "=" * 80)

    print(
        "Successfully crawled:",
        len(documents),
        "/",
        len(crawl_queue),
    )

    print(
        "Saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

