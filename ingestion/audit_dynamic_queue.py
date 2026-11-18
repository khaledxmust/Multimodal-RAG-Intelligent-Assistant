from config.settings import DATA_DIR
import json
import re

from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


INPUT_PATH = (DATA_DIR / "raw/te_dynamic_queue.json")


def stronger_alias_normalization(url: str) -> str:
    parsed = urlparse(url)

    path = parsed.path

    # /en/web/guest/... -> /en/...
    path = re.sub(
        r"^/(en|ar)/web/guest/",
        r"/\1/",
        path,
        flags=re.IGNORECASE,
    )

    # /web/guest/... -> /...
    path = re.sub(
        r"^/web/guest/",
        "/",
        path,
        flags=re.IGNORECASE,
    )

    path = re.sub(
        r"/+",
        "/",
        path,
    )

    if path != "/":
        path = path.rstrip("/")

    return (
        f"https://www.te.eg"
        f"{path}"
        + (
            f"?{parsed.query}"
            if parsed.query
            else ""
        )
    )


def get_path_family(url: str) -> str:
    path = urlparse(url).path.strip("/")

    parts = path.split("/")

    # Ignore language prefix for summary
    if parts and parts[0] in {
        "ar",
        "en",
    }:
        parts = parts[1:]

    if not parts:
        return "/"

    if len(parts) == 1:
        return "/" + parts[0]

    return "/" + "/".join(parts[:2])


def main():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        items = json.load(file)

    print("Total queue items:", len(items))

    # --------------------------------
    # Exact URL duplicates
    # --------------------------------

    exact_counts = Counter(
        item["url"]
        for item in items
    )

    exact_duplicates = {
        url: count
        for url, count in exact_counts.items()
        if count > 1
    }

    print(
        "Exact duplicate URLs:",
        len(exact_duplicates),
    )

    # --------------------------------
    # Alias duplicate detection
    # --------------------------------

    alias_groups = defaultdict(list)

    for item in items:
        key = stronger_alias_normalization(
            item["url"]
        )

        alias_groups[key].append(
            item["url"]
        )

    collisions = {
        key: sorted(set(urls))
        for key, urls in alias_groups.items()
        if len(set(urls)) > 1
    }

    print(
        "Alias duplicate groups:",
        len(collisions),
    )

    print(
        "\n--- ALIAS DUPLICATES ---\n"
    )

    for key, urls in list(
        collisions.items()
    )[:50]:
        print("=" * 100)

        print("Canonical:")
        print(key)

        print("\nVariants:")

        for url in urls:
            print("  ", url)

    # --------------------------------
    # Route statistics
    # --------------------------------

    depth_counts = Counter(
        item["depth"]
        for item in items
    )

    type_counts = Counter(
        item["type"]
        for item in items
    )

    language_counts = Counter(
        item["route_language"]
        for item in items
    )

    families = Counter(
        get_path_family(
            item["url"]
        )
        for item in items
    )

    print("\n--- DEPTH ---")

    for value, count in sorted(
        depth_counts.items()
    ):
        print(
            f"Depth {value}: {count}"
        )

    print("\n--- TYPE ---")

    for value, count in sorted(
        type_counts.items()
    ):
        print(
            value,
            ":",
            count,
        )

    print("\n--- LANGUAGE ROUTES ---")

    for value, count in sorted(
        language_counts.items()
    ):
        print(
            value,
            ":",
            count,
        )

    print(
        "\n--- TOP PATH FAMILIES ---\n"
    )

    for family, count in families.most_common(
        30
    ):
        print(
            f"{count:4}  {family}"
        )

    # --------------------------------
    # Suspicious routes
    # --------------------------------

    suspicious_markers = [
        "/web/guest/",
        "/documents/",
        "/wps/",
        "/c/",
        "/o/",
    ]

    suspicious = []

    for item in items:
        url_lower = item[
            "url"
        ].lower()

        if any(
            marker in url_lower
            for marker in suspicious_markers
        ):
            suspicious.append(
                item["url"]
            )

    print(
        "\nSuspicious/technical URLs:",
        len(suspicious),
    )

    print(
        "\n--- SAMPLE SUSPICIOUS URLS ---\n"
    )

    for url in suspicious[:50]:
        print(url)


if __name__ == "__main__":
    main()