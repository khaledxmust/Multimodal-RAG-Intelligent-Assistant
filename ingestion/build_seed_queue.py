from config.settings import DATA_DIR
import json
import re

from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
)

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)

from ingestion.dynamic_crawler import (
    classify_url,
    get_route_language,
    get_url_key,
    is_allowed_te_url,
    normalize_te_url,
)


# ============================================================
# CONFIG
# ============================================================

OLD_QUEUE_PATH = (DATA_DIR / "raw/te_crawl_queue.json")

OUTPUT_PATH = (DATA_DIR / "raw/te_seed_queue.json")

REPORT_PATH = (DATA_DIR / "raw/te_seed_queue_report.json")


SITEMAP_URLS = {
    "default_sitemap":
        "https://www.te.eg/personal/sitemap/",

    "arabic_sitemap":
        "https://www.te.eg/ar/personal/sitemap/",

    "english_sitemap":
        "https://www.te.eg/en/personal/sitemap/",
}


# Important roots that should always exist in the seed set
# even if the sitemap presentation changes.
FIXED_SEEDS = [
    "https://www.te.eg/personal",
    "https://www.te.eg/ar/personal",
    "https://www.te.eg/en/personal",

    "https://www.te.eg/personal/home/we-int",
    "https://www.te.eg/ar/personal/home/we-int",
    "https://www.te.eg/en/personal/home/we-int",

    "https://www.te.eg/personal/mobile",
    "https://www.te.eg/ar/personal/mobile",
    "https://www.te.eg/en/personal/mobile",

    "https://www.te.eg/personal/services",
    "https://www.te.eg/ar/personal/services",
    "https://www.te.eg/en/personal/services",

    "https://www.te.eg/personal/devices",
    "https://www.te.eg/ar/personal/devices",
    "https://www.te.eg/en/personal/devices",
]


# Some TE Personal/customer pages live outside /personal/.
#
# We already encountered examples like:
# /5gmobile
# /kalamy-ar
# /marhaba-ar
# /nitro-mifi
#
# Keep these when the sitemap exposes them.
KNOWN_PERSONAL_TOP_LEVEL = {
    "/5gmobile",
    "/kalamy-ar",
    "/marhaba-ar",
    "/nitro-mifi",
}


# ============================================================
# RELEVANCE
# ============================================================

def remove_language_prefix(
    path: str,
) -> str:
    """
    /ar/personal/... -> /personal/...
    /en/personal/... -> /personal/...
    """

    return re.sub(
        r"^/(ar|en)(?=/|$)",
        "",
        path,
        flags=re.IGNORECASE,
    ) or "/"


def is_relevant_seed_url(
    url: str,
) -> bool:
    """
    Decide whether a URL belongs in the bootstrap seed set.

    This is intentionally stricter than the dynamic crawler.

    The seed set supplies breadth.
    The dynamic crawler will discover deeper product/detail URLs.
    """

    if not is_allowed_te_url(
        url
    ):
        return False

    parsed = urlparse(
        url
    )

    path = parsed.path.rstrip("/")

    path_without_language = (
        remove_language_prefix(
            path
        )
    )

    # Main Personal/customer hierarchy.
    if (
        path_without_language == "/personal"
        or path_without_language.startswith(
            "/personal/"
        )
    ):
        # Do not seed the sitemap page itself.
        if path_without_language.startswith(
            "/personal/sitemap"
        ):
            return False

        return True

    # Known Personal/customer top-level pages.
    if (
        path_without_language
        in KNOWN_PERSONAL_TOP_LEVEL
    ):
        return True

    return False


# ============================================================
# SITEMAP EXTRACTION
# ============================================================

def extract_links_from_page(
    page,
    page_url: str,
) -> set[str]:
    """
    Extract all links from a rendered sitemap.

    Unlike ordinary product pages, scanning all sitemap anchors
    is acceptable because relevance filtering happens afterward.
    """

    discovered = set()

    hrefs = page.locator(
        "a[href]"
    ).evaluate_all(
        """
        elements =>
            elements.map(
                element =>
                    element.getAttribute("href")
            )
        """
    )

    for href in hrefs:
        if not href:
            continue

        href = href.strip()

        if not href:
            continue

        href_lower = (
            href.lower()
        )

        if href_lower.startswith(
            (
                "#",
                "javascript:",
                "mailto:",
                "tel:",
                "data:",
            )
        ):
            continue

        absolute_url = urljoin(
            page_url,
            href,
        )

        normalized = normalize_te_url(
            absolute_url
        )

        if not normalized:
            continue

        if not is_relevant_seed_url(
            normalized
        ):
            continue

        discovered.add(
            normalized
        )

    return discovered


# ============================================================
# SEED REGISTRY
# ============================================================

def register_seed(
    registry: dict,
    url: str,
    source: str,
):
    """
    Add a normalized URL and preserve every source
    through which it was discovered.
    """

    normalized = normalize_te_url(
        url
    )

    if not normalized:
        return False

    if not is_relevant_seed_url(
        normalized
    ):
        return False

    key = get_url_key(
        normalized
    )

    if key not in registry:
        registry[key] = {
            "url": normalized,
            "depth": 0,
            "type": classify_url(
                normalized
            ),
            "route_language": (
                get_route_language(
                    normalized
                )
            ),
            "seed_sources": [],
        }

    if source not in registry[
        key
    ]["seed_sources"]:
        registry[
            key
        ]["seed_sources"].append(
            source
        )

    return True


# ============================================================
# OLD QUEUE
# ============================================================

def load_old_queue() -> list[dict]:
    if not OLD_QUEUE_PATH.exists():
        return []

    with OLD_QUEUE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


# ============================================================
# SAVE
# ============================================================

def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    registry = {}

    sitemap_results = {}

    # --------------------------------------------------------
    # 1. Current TE sitemaps
    # --------------------------------------------------------

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )

        for (
            sitemap_name,
            sitemap_url,
        ) in SITEMAP_URLS.items():

            print(
                "\n"
                + "=" * 90
            )

            print(
                "Sitemap:",
                sitemap_name,
            )

            print(
                sitemap_url
            )

            # Fresh context prevents Arabic/English cookies
            # leaking between sitemap versions.
            context = browser.new_context(
                viewport={
                    "width": 1400,
                    "height": 1000,
                }
            )

            page = context.new_page()

            try:
                page.goto(
                    sitemap_url,
                    wait_until=(
                        "domcontentloaded"
                    ),
                    timeout=60000,
                )

                page.wait_for_timeout(
                    2500
                )

                links = extract_links_from_page(
                    page=page,
                    page_url=page.url,
                )

                print(
                    "Relevant links found:",
                    len(links),
                )

                sitemap_results[
                    sitemap_name
                ] = sorted(
                    links
                )

                for url in links:
                    register_seed(
                        registry=registry,
                        url=url,
                        source=sitemap_name,
                    )

            except PlaywrightTimeoutError:
                print(
                    "   -> Timeout"
                )

                sitemap_results[
                    sitemap_name
                ] = []

            except Exception as error:
                print(
                    "   -> Error:",
                    error,
                )

                sitemap_results[
                    sitemap_name
                ] = []

            finally:
                context.close()

        browser.close()

    # --------------------------------------------------------
    # 2. Fixed important roots
    # --------------------------------------------------------

    for url in FIXED_SEEDS:
        register_seed(
            registry=registry,
            url=url,
            source="fixed_seed",
        )

    # --------------------------------------------------------
    # 3. Preserve old useful seeds as fallback
    # --------------------------------------------------------

    old_items = load_old_queue()

    old_normalized = set()

    for item in old_items:
        url = normalize_te_url(
            item["url"]
        )

        if not url:
            continue

        if not is_relevant_seed_url(
            url
        ):
            continue

        old_normalized.add(
            get_url_key(
                url
            )
        )

        register_seed(
            registry=registry,
            url=url,
            source="legacy_seed",
        )

    # --------------------------------------------------------
    # Final seed set
    # --------------------------------------------------------

    seeds = sorted(
        registry.values(),
        key=lambda item: (
            item["route_language"],
            item["url"],
        ),
    )

    new_keys = {
        get_url_key(
            item["url"]
        )
        for item in seeds
    }

    # --------------------------------------------------------
    # Compare old vs rebuilt
    # --------------------------------------------------------

    added_keys = (
        new_keys
        - old_normalized
    )

    old_only_keys = (
        old_normalized
        - new_keys
    )

    new_urls = {
        get_url_key(
            item["url"]
        ): item["url"]
        for item in seeds
    }

    old_urls = {}

    for item in old_items:
        normalized = normalize_te_url(
            item["url"]
        )

        if not normalized:
            continue

        if not is_relevant_seed_url(
            normalized
        ):
            continue

        old_urls[
            get_url_key(
                normalized
            )
        ] = normalized

    newly_added = sorted(
        new_urls[key]
        for key in added_keys
    )

    old_only = sorted(
        old_urls[key]
        for key in old_only_keys
        if key in old_urls
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    arabic_count = sum(
        item["route_language"] == "ar"
        for item in seeds
    )

    english_count = sum(
        item["route_language"] == "en"
        for item in seeds
    )

    default_count = sum(
        item["route_language"] == "default"
        for item in seeds
    )

    state_count = sum(
        item["type"] == "state"
        for item in seeds
    )

    page_count = sum(
        item["type"] == "page"
        for item in seeds
    )

    sitemap_only = [
        item
        for item in seeds
        if (
            "legacy_seed"
            not in item["seed_sources"]
        )
    ]

    legacy_only = [
        item
        for item in seeds
        if (
            item["seed_sources"]
            == ["legacy_seed"]
        )
    ]

    report = {
        "old_queue_total": len(
            old_items
        ),
        "old_relevant_normalized": len(
            old_normalized
        ),
        "new_seed_total": len(
            seeds
        ),
        "normal_pages": page_count,
        "state_urls": state_count,
        "arabic_routes": arabic_count,
        "english_routes": english_count,
        "default_routes": default_count,
        "new_not_in_old_count": len(
            newly_added
        ),
        "old_not_in_new_count": len(
            old_only
        ),
        "sitemap_or_fixed_only_count": len(
            sitemap_only
        ),
        "legacy_only_count": len(
            legacy_only
        ),
        "sitemap_counts": {
            name: len(urls)
            for name, urls
            in sitemap_results.items()
        },
        "new_not_in_old": newly_added,
        "old_not_in_new": old_only,
        "legacy_only": [
            item["url"]
            for item in legacy_only
        ],
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        OUTPUT_PATH,
        seeds,
    )

    save_json(
        REPORT_PATH,
        report,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "SEED QUEUE SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        "Old queue total:",
        len(old_items),
    )

    print(
        "Old relevant normalized:",
        len(old_normalized),
    )

    print(
        "New seed total:",
        len(seeds),
    )

    print(
        "Normal pages:",
        page_count,
    )

    print(
        "State URLs:",
        state_count,
    )

    print(
        "Arabic routes:",
        arabic_count,
    )

    print(
        "English routes:",
        english_count,
    )

    print(
        "Default routes:",
        default_count,
    )

    print(
        "\nCurrent sitemap results:"
    )

    for (
        sitemap_name,
        urls,
    ) in sitemap_results.items():
        print(
            f"   {sitemap_name}: "
            f"{len(urls)}"
        )

    print(
        "\nNew URLs not in old queue:",
        len(newly_added),
    )

    print(
        "Old relevant URLs not in rebuilt queue:",
        len(old_only),
    )

    print(
        "Seeds not backed by legacy queue:",
        len(sitemap_only),
    )

    print(
        "Legacy-only seeds:",
        len(legacy_only),
    )

    print(
        "\n--- SAMPLE NEW URLS ---"
    )

    for url in newly_added[:30]:
        print(
            url
        )

    print(
        "\n--- LEGACY-ONLY URLS ---"
    )

    for item in legacy_only[:30]:
        print(
            item["url"]
        )

    print(
        "\nSaved seed queue to:",
        OUTPUT_PATH,
    )

    print(
        "Saved report to:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()