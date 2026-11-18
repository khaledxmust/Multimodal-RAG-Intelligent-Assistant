from config.settings import DATA_DIR
import html
import json
import re

from collections import Counter, deque
from pathlib import Path
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlparse,
    urlunparse,
)

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (DATA_DIR / "raw/te_seed_queue.json")

OUTPUT_PATH = (DATA_DIR / "raw/te_dynamic_queue.json")

STATE_PATH = (DATA_DIR / "raw/te_dynamic_crawl_state.json")


BASE_URL = "https://www.te.eg/"


ALLOWED_HOSTS = {
    "te.eg",
    "www.te.eg",
}


# Meaningful query parameters that represent
# actual product/content states.
STATE_QUERY_PARAMS = {
    "filterby": "filterBy",
}


# Maximum discovery depth.
#
# Depth 0 = seeds
# Depth 1 = links discovered from seeds
# Depth 2 = deeper links
# Depth 3 = stored, but not expanded further
MAX_DEPTH = 3


# None means:
# keep crawling until the queue is completely exhausted.
#
# MAX_DEPTH remains our safety boundary.
MAX_PAGES = None


# Resume from checkpoint/output if possible.
RESUME = True


# Retry temporary failures before giving up.
MAX_RETRIES = 2


BLOCKED_URL_PARTS = [
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
    "/shop",
    "/personal/sitemap",
    "/personal/promotions",
    "/check-availability",

    # CMS / technical routes
    "/wps/",
    "/main-content",

    # Business area - outside Personal KB
    "/te-business",

    # Document repository will be handled separately
    "/documents/",

    # Invalid routes
    "/personal/''",
    "/services/''",
]


BLOCKED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".xml",
    ".zip",
    ".rar",
    ".pdf",
    ".mp4",
    ".mp3",
    ".woff",
    ".woff2",
    ".ttf",
}


# ============================================================
# JSON HELPERS
# ============================================================

def write_json_atomic(
    path: Path,
    data,
):
    """
    Write JSON safely using a temporary file first.

    This reduces the chance of corrupting our crawl state
    if the process is interrupted during a write.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temp_path.replace(
        path
    )


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_te_url(
    url: str,
    base_url: str = BASE_URL,
) -> str:
    """
    Normalize Telecom Egypt URLs.

    Examples:

    /web/guest/personal/...
        ->
    /personal/...

    /en/web/guest/personal/...
        ->
    /en/personal/...

    /ar/web/guest/personal/...
        ->
    /ar/personal/...

    Meaningful filterBy state parameters are preserved.
    """

    if not url:
        return ""

    url = html.unescape(
        url
    )

    url = url.replace(
        "\\/",
        "/",
    )

    url = url.strip()
    url = url.strip("\"'")

    if not url:
        return ""

    lower_url = url.lower()

    if lower_url.startswith(
        (
            "javascript:",
            "mailto:",
            "tel:",
            "data:",
        )
    ):
        return ""

    # Same-page interaction, not another crawl URL.
    if url.startswith("#"):
        return ""

    absolute_url = urljoin(
        base_url,
        url,
    )

    parsed = urlparse(
        absolute_url
    )

    hostname = parsed.netloc.lower()

    if hostname in ALLOWED_HOSTS:
        hostname = "www.te.eg"

    path = re.sub(
        r"/+",
        "/",
        parsed.path,
    )

    # --------------------------------------------------------
    # Normalize /web/guest aliases
    # --------------------------------------------------------

    # /en/web/guest/... -> /en/...
    # /ar/web/guest/... -> /ar/...
    path = re.sub(
        r"^/(en|ar)/web/guest(?=/|$)",
        r"/\1",
        path,
        flags=re.IGNORECASE,
    )

    # /web/guest/... -> /...
    path = re.sub(
        r"^/web/guest(?=/|$)",
        "",
        path,
        flags=re.IGNORECASE,
    )

    path = re.sub(
        r"/+",
        "/",
        path,
    )

    if not path:
        path = "/"

    if path != "/":
        path = path.rstrip("/")

    # --------------------------------------------------------
    # Preserve only meaningful query-state parameters
    # --------------------------------------------------------

    parsed_query = parse_qsl(
        parsed.query,
        keep_blank_values=False,
    )

    kept_query = []

    for key, value in parsed_query:
        canonical_key = STATE_QUERY_PARAMS.get(
            key.lower()
        )

        if not canonical_key:
            continue

        value = value.strip()

        if not value:
            continue

        kept_query.append(
            (
                canonical_key,
                value,
            )
        )

    query = urlencode(
        kept_query,
        doseq=True,
    )

    # --------------------------------------------------------
    # Clean malformed fragments
    #
    # Example:
    #
    # #learning?filterBy=learning#learning
    #
    # becomes:
    #
    # #learning
    # --------------------------------------------------------

    fragment = ""

    if query and parsed.fragment:
        fragment = re.split(
            r"[?#]",
            parsed.fragment.strip(),
            maxsplit=1,
        )[0].strip()

    return urlunparse(
        (
            "https",
            hostname,
            path,
            "",
            query,
            fragment,
        )
    )


def get_url_key(
    url: str,
) -> str:
    """
    Canonical duplicate key.

    Fragment is ignored because filterBy already identifies
    the meaningful state.
    """

    parsed = urlparse(
        url
    )

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            parsed.query,
            "",
        )
    )


def is_allowed_te_url(
    url: str,
) -> bool:

    if not url:
        return False

    parsed = urlparse(
        url
    )

    if parsed.netloc.lower() not in ALLOWED_HOSTS:
        return False

    path_lower = parsed.path.lower()

    if any(
        blocked.lower() in path_lower
        for blocked in BLOCKED_URL_PARTS
    ):
        return False

    for extension in BLOCKED_EXTENSIONS:
        if path_lower.endswith(
            extension
        ):
            return False

    return True


# ============================================================
# URL METADATA
# ============================================================

def classify_url(
    url: str,
) -> str:
    """
    state:
        a meaningful query-state such as ?filterBy=...

    page:
        normal/path-based page.
    """

    parsed = urlparse(
        url
    )

    query = dict(
        parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
    )

    query_keys = {
        key.lower()
        for key in query
    }

    if any(
        key in query_keys
        for key in STATE_QUERY_PARAMS
    ):
        return "state"

    return "page"


def get_route_language(
    url: str,
) -> str:
    """
    Language suggested by the URL path.

    Actual content language is determined later by
    dynamic_content_extractor.py.
    """

    path = urlparse(
        url
    ).path.lower()

    if path.startswith("/ar/"):
        return "ar"

    if path.startswith("/en/"):
        return "en"

    return "default"


# ============================================================
# QUEUE HELPERS
# ============================================================

def queue_contains_key(
    queue: deque,
    key: str,
) -> bool:

    for item in queue:
        if get_url_key(
            item["url"]
        ) == key:
            return True

    return False


def add_to_queue(
    queue: deque,
    seen: set,
    all_items: dict,
    url: str,
    depth: int,
    discovered_from: str | None = None,
    discovery_method: str | None = None,
    completed_keys: set | None = None,
) -> bool:

    normalized_url = normalize_te_url(
        url
    )

    if not is_allowed_te_url(
        normalized_url
    ):
        return False

    key = get_url_key(
        normalized_url
    )

    # --------------------------------------------------------
    # Already known
    # --------------------------------------------------------

    if key in seen:
        item = all_items.get(
            key
        )

        if not item:
            return False

        old_depth = item[
            "depth"
        ]

        # A shorter route was discovered later.
        if depth < old_depth:
            item[
                "depth"
            ] = depth

            # Important edge case:
            #
            # the page may previously have been skipped because
            # it was discovered at MAX_DEPTH, but now we found a
            # shorter route where it is eligible to expand.
            if (
                item.get(
                    "crawl_status"
                ) == "depth_limit"
                and depth < MAX_DEPTH
            ):
                item[
                    "crawl_status"
                ] = "pending"

                if completed_keys is not None:
                    completed_keys.discard(
                        key
                    )

                if not queue_contains_key(
                    queue,
                    key,
                ):
                    if item[
                        "type"
                    ] == "state":
                        queue.appendleft(
                            item
                        )
                    else:
                        queue.append(
                            item
                        )

        if discovered_from:
            item.setdefault(
                "discovered_from",
                [],
            )

            if (
                discovered_from
                not in item[
                    "discovered_from"
                ]
            ):
                item[
                    "discovered_from"
                ].append(
                    discovered_from
                )

        if discovery_method:
            item.setdefault(
                "discovery_methods",
                [],
            )

            if (
                discovery_method
                not in item[
                    "discovery_methods"
                ]
            ):
                item[
                    "discovery_methods"
                ].append(
                    discovery_method
                )

        return False

    # --------------------------------------------------------
    # New URL
    # --------------------------------------------------------

    seen.add(
        key
    )

    item = {
        "url": normalized_url,
        "depth": depth,
        "type": classify_url(
            normalized_url
        ),
        "route_language": (
            get_route_language(
                normalized_url
            )
        ),
        "discovered_from": (
            [discovered_from]
            if discovered_from
            else []
        ),
        "discovery_methods": (
            [discovery_method]
            if discovery_method
            else []
        ),

        # Resume/checkpoint metadata
        "crawl_status": "pending",
        "crawl_attempts": 0,
    }

    all_items[
        key
    ] = item

    # Query states get priority.
    if item["type"] == "state":
        queue.appendleft(
            item
        )
    else:
        queue.append(
            item
        )

    return True


# ============================================================
# FRESH SEED QUEUE
# ============================================================

def build_initial_queue():

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        seeds = json.load(
            file
        )

    queue = deque()
    seen = set()
    all_items = {}

    for item in seeds:
        add_to_queue(
            queue=queue,
            seen=seen,
            all_items=all_items,
            url=item["url"],
            depth=item.get(
                "depth",
                0,
            ),
            discovery_method="seed",
        )

    return (
        queue,
        seen,
        all_items,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    queue: deque,
    all_items: dict,
    completed_keys: set,
    opened_pages: int,
    dequeued_items: int,
):
    """
    Save exact crawler position.

    Once this file exists, future runs can continue without
    reopening successfully completed pages.
    """

    state = {
        "queue": list(
            queue
        ),
        "all_items": list(
            all_items.values()
        ),
        "completed_keys": sorted(
            completed_keys
        ),
        "opened_pages": (
            opened_pages
        ),
        "dequeued_items": (
            dequeued_items
        ),
        "crawl_complete": (
            len(queue) == 0
        ),
    }

    write_json_atomic(
        STATE_PATH,
        state,
    )


def load_checkpoint():
    """
    Resume from an exact checkpoint created by this version.
    """

    if not STATE_PATH.exists():
        return None

    try:
        with STATE_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(
                file
            )

    except Exception as error:
        print(
            "Could not load checkpoint:",
            error,
        )

        return None

    all_items = {}

    for item in state.get(
        "all_items",
        [],
    ):
        item.setdefault(
            "crawl_status",
            "pending",
        )

        item.setdefault(
            "crawl_attempts",
            0,
        )

        item.setdefault(
            "discovered_from",
            [],
        )

        item.setdefault(
            "discovery_methods",
            [],
        )

        key = get_url_key(
            item["url"]
        )

        all_items[
            key
        ] = item

    seen = set(
        all_items.keys()
    )

    completed_keys = set(
        state.get(
            "completed_keys",
            [],
        )
    )

    queue = deque()

    for queued_item in state.get(
        "queue",
        [],
    ):
        key = get_url_key(
            queued_item["url"]
        )

        if key in completed_keys:
            continue

        canonical_item = all_items.get(
            key
        )

        if canonical_item:
            queue.append(
                canonical_item
            )

    opened_pages = state.get(
        "opened_pages",
        0,
    )

    dequeued_items = state.get(
        "dequeued_items",
        0,
    )

    return (
        queue,
        seen,
        all_items,
        completed_keys,
        opened_pages,
        dequeued_items,
    )


# ============================================================
# ONE-TIME RECOVERY OF OLD PRE-CHECKPOINT RUN
# ============================================================

def recover_from_existing_results():
    """
    Recover as much as possible from the old crawler run.

    Before checkpointing existed, te_dynamic_queue.json saved:
    - discovered URLs
    - discovered_from provenance

    If URL A occurs inside another item's discovered_from list,
    URL A MUST have been opened previously, because it was the
    page that discovered that child.

    Therefore those URLs can safely be marked completed.

    Pages that were scanned but discovered zero new URLs cannot
    be proven from the old output. Those few pages may be
    rescanned once.

    After this recovery creates the first checkpoint, future
    resumes are exact.
    """

    if not OUTPUT_PATH.exists():
        return None

    try:
        with OUTPUT_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            existing_items = (
                json.load(
                    file
                )
            )

    except Exception as error:
        print(
            "Could not recover old crawl:",
            error,
        )

        return None

    if not existing_items:
        return None

    all_items = {}

    for item in existing_items:
        item.setdefault(
            "crawl_attempts",
            0,
        )

        item.setdefault(
            "crawl_status",
            "unknown",
        )

        item.setdefault(
            "discovered_from",
            [],
        )

        item.setdefault(
            "discovery_methods",
            [],
        )

        key = get_url_key(
            item["url"]
        )

        all_items[
            key
        ] = item

    seen = set(
        all_items.keys()
    )

    completed_keys = set()

    # --------------------------------------------------------
    # URLs that definitely acted as discovery parents
    # were definitely scanned.
    # --------------------------------------------------------

    for child in existing_items:
        for source_url in child.get(
            "discovered_from",
            [],
        ):
            normalized_source = (
                normalize_te_url(
                    source_url
                )
            )

            if not normalized_source:
                continue

            source_key = get_url_key(
                normalized_source
            )

            source_item = all_items.get(
                source_key
            )

            if not source_item:
                continue

            completed_keys.add(
                source_key
            )

            source_item[
                "crawl_status"
            ] = "recovered_completed"

    # --------------------------------------------------------
    # Depth-limit URLs do not need opening for discovery.
    #
    # They remain in the final graph and will still be
    # processed by the later content extractor.
    # --------------------------------------------------------

    for key, item in all_items.items():
        if item.get(
            "depth",
            0,
        ) >= MAX_DEPTH:
            completed_keys.add(
                key
            )

            item[
                "crawl_status"
            ] = "depth_limit"

    # --------------------------------------------------------
    # Anything else is conservatively pending.
    #
    # We prefer rescanning a small ambiguous subset rather
    # than accidentally skipping an unprocessed page.
    # --------------------------------------------------------

    pending_items = [
        item
        for key, item
        in all_items.items()
        if (
            key not in completed_keys
            and item.get(
                "depth",
                0,
            ) < MAX_DEPTH
        )
    ]

    # Process shallow pages first.
    # Within a depth, prioritize state URLs.
    pending_items.sort(
        key=lambda item: (
            item.get(
                "depth",
                0,
            ),
            (
                0
                if item.get(
                    "type"
                ) == "state"
                else 1
            ),
            item["url"],
        )
    )

    for item in pending_items:
        item[
            "crawl_status"
        ] = "pending"

    queue = deque(
        pending_items
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RECOVERING PRE-CHECKPOINT RUN"
    )

    print(
        "=" * 80
    )

    print(
        "Known URLs:",
        len(all_items),
    )

    print(
        "Definitely completed:",
        len(completed_keys),
    )

    print(
        "Pending / uncertain:",
        len(queue),
    )

    print(
        "\nNOTE:"
    )

    print(
        "Pages from the old run that produced zero new URLs "
        "cannot be identified exactly."
    )

    print(
        "A small number of those may be scanned once more."
    )

    return (
        queue,
        seen,
        all_items,
        completed_keys,
        0,
        0,
    )


# ============================================================
# INITIALIZE / RESUME
# ============================================================

def initialize_crawl():

    if RESUME:
        # ----------------------------------------------------
        # 1. Exact checkpoint
        # ----------------------------------------------------

        checkpoint = (
            load_checkpoint()
        )

        if checkpoint is not None:
            print(
                "\n"
                + "=" * 80
            )

            print(
                "RESUMING EXACT CHECKPOINT"
            )

            print(
                "=" * 80
            )

            print(
                "Known URLs:",
                len(
                    checkpoint[2]
                ),
            )

            print(
                "Already completed:",
                len(
                    checkpoint[3]
                ),
            )

            print(
                "Remaining queue:",
                len(
                    checkpoint[0]
                ),
            )

            return checkpoint

        # ----------------------------------------------------
        # 2. Recover old run from te_dynamic_queue.json
        # ----------------------------------------------------

        recovered = (
            recover_from_existing_results()
        )

        if recovered is not None:
            # Immediately create the first exact checkpoint.
            save_checkpoint(
                queue=recovered[0],
                all_items=recovered[2],
                completed_keys=(
                    recovered[3]
                ),
                opened_pages=(
                    recovered[4]
                ),
                dequeued_items=(
                    recovered[5]
                ),
            )

            return recovered

    # --------------------------------------------------------
    # 3. Fresh crawl
    # --------------------------------------------------------

    (
        queue,
        seen,
        all_items,
    ) = build_initial_queue()

    completed_keys = set()

    save_checkpoint(
        queue=queue,
        all_items=all_items,
        completed_keys=(
            completed_keys
        ),
        opened_pages=0,
        dequeued_items=0,
    )

    print(
        "\nSTARTING FRESH CRAWL"
    )

    return (
        queue,
        seen,
        all_items,
        completed_keys,
        0,
        0,
    )


# ============================================================
# PAGE RENDERING
# ============================================================

def scroll_page(page):
    """
    Trigger lazy-loaded content and selectors.
    """

    for _ in range(8):
        page.evaluate(
            """
            window.scrollBy(
                0,
                window.innerHeight * 0.85
            )
            """
        )

        page.wait_for_timeout(
            250
        )

    page.evaluate(
        """
        window.scrollTo(
            0,
            document.body.scrollHeight
        )
        """
    )

    page.wait_for_timeout(
        700
    )

    page.evaluate(
        """
        window.scrollTo(0, 0)
        """
    )

    page.wait_for_timeout(
        300
    )


# ============================================================
# CONTENT ROOTS
# ============================================================

def get_discovery_roots(page):
    """
    Return useful content containers while excluding
    global navigation/header/footer.
    """

    roots = []

    main_templates = page.locator(
        ".main_page_template_container"
    )

    if main_templates.count() > 0:
        for i in range(
            main_templates.count()
        ):
            element = (
                main_templates.nth(
                    i
                )
            )

            try:
                text = (
                    element.inner_text()
                    .strip()
                )

                if text:
                    roots.append(
                        element
                    )

            except Exception:
                continue

        if roots:
            return roots

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    main_content = page.locator(
        "#main-content"
    )

    if main_content.count() == 0:
        main = page.locator(
            "main"
        )

        if main.count() > 0:
            return [
                main.first
            ]

        return []

    children = (
        main_content.first
        .locator(
            ":scope > *"
        )
    )

    excluded_class_keywords = [
        "main-header",
        "secondary-header",
        "breadcrumb",
        "comparison-page-cards",
        "back-to-top",
        "footer",
        "navigation",
        "navbar",
    ]

    excluded_tags = {
        "header",
        "footer",
        "nav",
    }

    footer_markers = [
        "جميع حقوق النشر محفوظة",
        "all rights reserved",
        "حمل التطبيق من هنا",
    ]

    for i in range(
        children.count()
    ):
        element = children.nth(
            i
        )

        try:
            tag_name = (
                element.evaluate(
                    """
                    element =>
                        element.tagName.toLowerCase()
                    """
                )
            )

            if tag_name in excluded_tags:
                continue

            classes = (
                element.get_attribute(
                    "class"
                )
                or ""
            ).lower()

            if any(
                keyword in classes
                for keyword
                in excluded_class_keywords
            ):
                continue

            text = (
                element.inner_text()
                .strip()
            )

            if not text:
                continue

            text_lower = (
                text.lower()
            )

            if any(
                marker.lower()
                in text_lower
                for marker
                in footer_markers
            ):
                continue

            roots.append(
                element
            )

        except Exception:
            continue

    return roots


# ============================================================
# NORMAL PAGE DISCOVERY
# ============================================================

def discover_content_links(
    page,
    current_url: str,
) -> set[str]:

    discovered = set()

    roots = get_discovery_roots(
        page
    )

    for root in roots:
        try:
            hrefs = root.locator(
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

        except Exception:
            continue

        for href in hrefs:
            if not href:
                continue

            href = (
                html.unescape(
                    href
                )
                .strip()
            )

            if not href:
                continue

            # Same-page interactive control.
            if href.startswith("#"):
                continue

            href_lower = (
                href.lower()
            )

            if href_lower.startswith(
                (
                    "javascript:",
                    "mailto:",
                    "tel:",
                    "data:",
                )
            ):
                continue

            normalized = normalize_te_url(
                href,
                base_url=current_url,
            )

            if not is_allowed_te_url(
                normalized
            ):
                continue

            discovered.add(
                normalized
            )

    return discovered


# ============================================================
# STATE DISCOVERY
# ============================================================

def extract_state_candidate(
    raw_value: str,
    current_url: str,
) -> set[str]:

    results = set()

    if not raw_value:
        return results

    raw_value = html.unescape(
        raw_value
    )

    raw_value = raw_value.replace(
        "\\/",
        "/",
    )

    if (
        "filterby="
        not in raw_value.lower()
    ):
        return results

    candidates = [
        raw_value.strip()
    ]

    # Quoted URL inside JS/config.
    quoted_candidates = re.findall(
        r"""["']([^"']*filterBy=[^"']+)["']""",
        raw_value,
        flags=re.IGNORECASE,
    )

    candidates.extend(
        quoted_candidates
    )

    # URL-like substring containing filterBy.
    url_like = re.findall(
        r"""(
            (?:
                https?://
                |
                /+
                |
                \.\.?/
                |
                \?
            )
            [^\s"'<>]*?
            filterBy=
            [^\s"'<>;,)]+
        )""",
        raw_value,
        flags=(
            re.IGNORECASE
            | re.VERBOSE
        ),
    )

    candidates.extend(
        url_like
    )

    for candidate in candidates:
        candidate = (
            candidate
            .strip()
            .strip("\"'")
            .rstrip(";,)")
        )

        if not candidate:
            continue

        lower_candidate = (
            candidate.lower()
        )

        looks_like_url = (
            candidate.startswith(
                (
                    "http://",
                    "https://",
                    "/",
                    "./",
                    "../",
                    "?",
                )
            )
            or "?filterby="
            in lower_candidate
            or "&filterby="
            in lower_candidate
        )

        if not looks_like_url:
            continue

        normalized = normalize_te_url(
            candidate,
            base_url=current_url,
        )

        if not normalized:
            continue

        if not is_allowed_te_url(
            normalized
        ):
            continue

        if classify_url(
            normalized
        ) != "state":
            continue

        results.add(
            normalized
        )

    return results


def discover_state_urls(
    page,
    current_url: str,
) -> dict[str, set[str]]:

    discovered = {}

    def add_state(
        state_url: str,
        method: str,
    ):
        if state_url not in discovered:
            discovered[
                state_url
            ] = set()

        discovered[
            state_url
        ].add(
            method
        )

    # --------------------------------------------------------
    # DOM attributes
    # --------------------------------------------------------

    try:
        attribute_values = (
            page.locator(
                "*"
            ).evaluate_all(
                """
                elements => {
                    const values = [];

                    for (const element of elements) {
                        for (
                            const attribute
                            of element.attributes
                        ) {
                            const value =
                                attribute.value;

                            if (
                                value &&
                                value.toLowerCase()
                                    .includes("filterby=")
                            ) {
                                values.push(value);
                            }
                        }
                    }

                    return values;
                }
                """
            )
        )

        for value in attribute_values:
            states = (
                extract_state_candidate(
                    value,
                    current_url,
                )
            )

            for state_url in states:
                add_state(
                    state_url,
                    "dom_attribute",
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # Rendered HTML / embedded configuration
    # --------------------------------------------------------

    try:
        rendered_html = (
            page.content()
        )

        rendered_html = (
            html.unescape(
                rendered_html
            )
        )

        rendered_html = (
            rendered_html.replace(
                "\\/",
                "/",
            )
        )

        embedded_values = (
            re.findall(
                r"""["']([^"']*filterBy=[^"']+)["']""",
                rendered_html,
                flags=re.IGNORECASE,
            )
        )

        for value in embedded_values:
            states = (
                extract_state_candidate(
                    value,
                    current_url,
                )
            )

            for state_url in states:
                add_state(
                    state_url,
                    "embedded_html",
                )

    except Exception:
        pass

    return discovered


# ============================================================
# COMBINED DISCOVERY
# ============================================================

def discover_urls(
    page,
    current_url: str,
):

    discoveries = {}

    def register(
        url: str,
        method: str,
    ):
        if url not in discoveries:
            discoveries[
                url
            ] = set()

        discoveries[
            url
        ].add(
            method
        )

    # Normal links from actual content.
    content_links = discover_content_links(
        page=page,
        current_url=current_url,
    )

    for url in content_links:
        if classify_url(
            url
        ) == "state":
            register(
                url,
                "content_state_link",
            )
        else:
            register(
                url,
                "content_link",
            )

    # Dynamic/query state URLs.
    state_urls = discover_state_urls(
        page=page,
        current_url=current_url,
    )

    for url, methods in (
        state_urls.items()
    ):
        for method in methods:
            register(
                url,
                method,
            )

    return discoveries


# ============================================================
# SAVE FINAL URL REGISTRY
# ============================================================

def save_results(
    all_items: dict,
):

    results = sorted(
        all_items.values(),
        key=lambda item: (
            item.get(
                "depth",
                0,
            ),
            item.get(
                "type",
                "page",
            ),
            item.get(
                "route_language",
                "default",
            ),
            item["url"],
        ),
    )

    write_json_atomic(
        OUTPUT_PATH,
        results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        queue,
        seen,
        all_items,
        completed_keys,
        opened_pages,
        dequeued_items,
    ) = initialize_crawl()

    known_at_startup = len(
        all_items
    )

    opened_this_run = 0

    print(
        "\nKnown URLs at startup:",
        known_at_startup,
    )

    print(
        "Already completed:",
        len(
            completed_keys
        ),
    )

    print(
        "Remaining queue:",
        len(
            queue
        ),
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )

        try:
            while (
                queue
                and (
                    MAX_PAGES is None
                    or opened_this_run
                    < MAX_PAGES
                )
            ):
                item = queue.popleft()

                key = get_url_key(
                    item["url"]
                )

                # Exact resume protection.
                if key in completed_keys:
                    continue

                dequeued_items += 1

                url = item[
                    "url"
                ]

                depth = item.get(
                    "depth",
                    0,
                )

                # ------------------------------------------------
                # Depth boundary
                # ------------------------------------------------

                if depth >= MAX_DEPTH:
                    item[
                        "crawl_status"
                    ] = "depth_limit"

                    completed_keys.add(
                        key
                    )

                    save_results(
                        all_items
                    )

                    save_checkpoint(
                        queue=queue,
                        all_items=all_items,
                        completed_keys=(
                            completed_keys
                        ),
                        opened_pages=(
                            opened_pages
                        ),
                        dequeued_items=(
                            dequeued_items
                        ),
                    )

                    continue

                # ------------------------------------------------
                # We are actually opening this URL
                # ------------------------------------------------

                opened_pages += 1
                opened_this_run += 1

                item[
                    "crawl_attempts"
                ] = (
                    item.get(
                        "crawl_attempts",
                        0,
                    )
                    + 1
                )

                item[
                    "crawl_status"
                ] = "running"

                print(
                    "\n"
                    + "=" * 90
                )

                print(
                    f"[Run {opened_this_run}] "
                    f"[Total {opened_pages}] "
                    f"Depth={depth} "
                    f"Type={item['type']} "
                    f"Language="
                    f"{item['route_language']}"
                )

                print(
                    url
                )

                print(
                    "   Queue remaining:",
                    len(
                        queue
                    ),
                )

                context = (
                    browser.new_context(
                        viewport={
                            "width": 1400,
                            "height": 1000,
                        }
                    )
                )

                page = (
                    context.new_page()
                )

                interrupted = False

                try:
                    page.goto(
                        url,
                        wait_until=(
                            "domcontentloaded"
                        ),
                        timeout=60000,
                    )

                    page.wait_for_timeout(
                        1500
                    )

                    scroll_page(
                        page
                    )

                    discoveries = (
                        discover_urls(
                            page=page,
                            current_url=(
                                page.url
                            ),
                        )
                    )

                    new_pages = 0
                    new_states = 0
                    state_candidates_on_page = 0

                    for (
                        discovered_url,
                        methods,
                    ) in discoveries.items():

                        discovered_type = (
                            classify_url(
                                discovered_url
                            )
                        )

                        if (
                            discovered_type
                            == "state"
                        ):
                            state_candidates_on_page += 1

                        first_method = next(
                            iter(
                                methods
                            )
                        )

                        added = add_to_queue(
                            queue=queue,
                            seen=seen,
                            all_items=(
                                all_items
                            ),
                            url=(
                                discovered_url
                            ),
                            depth=(
                                depth + 1
                            ),
                            discovered_from=url,
                            discovery_method=(
                                first_method
                            ),
                            completed_keys=(
                                completed_keys
                            ),
                        )

                        normalized = (
                            normalize_te_url(
                                discovered_url
                            )
                        )

                        discovered_key = (
                            get_url_key(
                                normalized
                            )
                        )

                        existing_item = (
                            all_items.get(
                                discovered_key
                            )
                        )

                        if existing_item:
                            existing_item.setdefault(
                                "discovery_methods",
                                [],
                            )

                            for method in methods:
                                if (
                                    method
                                    not in
                                    existing_item[
                                        "discovery_methods"
                                    ]
                                ):
                                    existing_item[
                                        "discovery_methods"
                                    ].append(
                                        method
                                    )

                        if not added:
                            continue

                        if (
                            discovered_type
                            == "state"
                        ):
                            new_states += 1
                        else:
                            new_pages += 1

                    item[
                        "crawl_status"
                    ] = "done"

                    completed_keys.add(
                        key
                    )

                    print(
                        "   URLs found on page:",
                        len(
                            discoveries
                        ),
                    )

                    print(
                        "   State candidates:",
                        state_candidates_on_page,
                    )

                    print(
                        "   New pages:",
                        new_pages,
                    )

                    print(
                        "   New states:",
                        new_states,
                    )

                    print(
                        "   Total known URLs:",
                        len(
                            all_items
                        ),
                    )

                # ------------------------------------------------
                # Timeout
                # ------------------------------------------------

                except PlaywrightTimeoutError:
                    print(
                        "   -> Timeout"
                    )

                    item[
                        "crawl_status"
                    ] = "timeout"

                    if (
                        item[
                            "crawl_attempts"
                        ]
                        < MAX_RETRIES
                    ):
                        print(
                            "   -> Will retry later"
                        )

                        item[
                            "crawl_status"
                        ] = "retry_pending"

                        queue.append(
                            item
                        )

                    else:
                        print(
                            "   -> Retry limit reached"
                        )

                        item[
                            "crawl_status"
                        ] = "failed_timeout"

                        completed_keys.add(
                            key
                        )

                # ------------------------------------------------
                # Ctrl+C
                # ------------------------------------------------

                except KeyboardInterrupt:
                    interrupted = True

                    print(
                        "\n   -> Interrupted."
                    )

                    print(
                        "   -> Saving current page "
                        "back to queue."
                    )

                    item[
                        "crawl_status"
                    ] = "pending"

                    queue.appendleft(
                        item
                    )

                    raise

                # ------------------------------------------------
                # Other errors
                # ------------------------------------------------

                except Exception as error:
                    print(
                        "   -> Error:",
                        error,
                    )

                    item[
                        "crawl_status"
                    ] = "error"

                    if (
                        item[
                            "crawl_attempts"
                        ]
                        < MAX_RETRIES
                    ):
                        print(
                            "   -> Will retry later"
                        )

                        item[
                            "crawl_status"
                        ] = "retry_pending"

                        queue.append(
                            item
                        )

                    else:
                        print(
                            "   -> Retry limit reached"
                        )

                        item[
                            "crawl_status"
                        ] = "failed_error"

                        completed_keys.add(
                            key
                        )

                finally:
                    context.close()

                    # Save after EVERY processed page.
                    save_results(
                        all_items
                    )

                    save_checkpoint(
                        queue=queue,
                        all_items=all_items,
                        completed_keys=(
                            completed_keys
                        ),
                        opened_pages=(
                            opened_pages
                        ),
                        dequeued_items=(
                            dequeued_items
                        ),
                    )

        finally:
            browser.close()

    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_results(
        all_items
    )

    save_checkpoint(
        queue=queue,
        all_items=all_items,
        completed_keys=(
            completed_keys
        ),
        opened_pages=(
            opened_pages
        ),
        dequeued_items=(
            dequeued_items
        ),
    )

    results = list(
        all_items.values()
    )

    page_count = sum(
        item.get(
            "type"
        ) == "page"
        for item in results
    )

    state_count = sum(
        item.get(
            "type"
        ) == "state"
        for item in results
    )

    arabic_count = sum(
        item.get(
            "route_language"
        ) == "ar"
        for item in results
    )

    english_count = sum(
        item.get(
            "route_language"
        ) == "en"
        for item in results
    )

    default_count = sum(
        item.get(
            "route_language"
        ) == "default"
        for item in results
    )

    failed_count = sum(
        item.get(
            "crawl_status",
            ""
        ).startswith(
            "failed_"
        )
        for item in results
    )

    state_method_counts = (
        Counter()
    )

    for item in results:
        if (
            item.get(
                "type"
            )
            != "state"
        ):
            continue

        for method in item.get(
            "discovery_methods",
            [],
        ):
            state_method_counts[
                method
            ] += 1

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CRAWL SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        "Total discovered URLs:",
        len(
            results
        ),
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
        "Pages opened this run:",
        opened_this_run,
    )

    print(
        "Pages opened total:",
        opened_pages,
    )

    print(
        "Completed/handled:",
        len(
            completed_keys
        ),
    )

    print(
        "Failed after retries:",
        failed_count,
    )

    print(
        "Still waiting in queue:",
        len(
            queue
        ),
    )

    print(
        "\nState discovery methods:"
    )

    if state_method_counts:
        for (
            method,
            count,
        ) in (
            state_method_counts
            .most_common()
        ):
            print(
                f"   {method}: {count}"
            )
    else:
        print(
            "   No states discovered."
        )

    print(
        "\nSaved URL registry to:",
        OUTPUT_PATH,
    )

    print(
        "Saved crawl checkpoint to:",
        STATE_PATH,
    )

    if not queue:
        print(
            "\nCrawl queue fully exhausted."
        )


if __name__ == "__main__":
    main()

