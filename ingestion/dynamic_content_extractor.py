from config.settings import DATA_DIR
import hashlib
import json
import re

from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (DATA_DIR / "raw/te_dynamic_queue.json")

FULL_OUTPUT_PATH = (DATA_DIR / "raw/te_dynamic_content.json")

TEST_OUTPUT_PATH = (DATA_DIR / "raw/te_dynamic_content_test.json")


# ------------------------------------------------------------
# TEST MODE
#
# Keep this at 25 for the representative validation run.
#
# After validation:
#
# TEST_LIMIT = None
#
# ------------------------------------------------------------

TEST_LIMIT = None


EXTRACTOR_VERSION = 4


# Minimum amount of useful newly revealed text
# before we consider an interaction section valid.
MIN_SECTION_CHARS = 30


# Give dynamic panels enough time to finish switching.
INTERACTION_WAIT_MS = 700


BLOCKED_CONTROL_TEXT = [
    "اشترك",
    "إشترك",
    "أضف للمقارنة",
    "اضف للمقارنة",
    "شراء",
    "اشتري",
    "login",
    "log in",
    "my we",
    "subscribe",
    "buy",
    "add to compare",
    "checkout",
    "next",
    "cancel",
    "submit",
]


# ============================================================
# OUTPUT PATH
# ============================================================

def get_output_path() -> Path:
    if TEST_LIMIT is not None:
        return TEST_OUTPUT_PATH

    return FULL_OUTPUT_PATH


# ============================================================
# SAFE JSON WRITE
# ============================================================

def write_json_atomic(
    path: Path,
    data,
):
    """
    Save through a temporary file first.

    Important for the eventual 771-page extraction:
    if Python/PyCharm is stopped during writing, the main
    output file is less likely to become corrupted.
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
# TEXT HELPERS
# ============================================================

def normalize_line(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def normalize_text(
    text: str,
) -> str:
    """
    Normalize whitespace while preserving meaningful lines.
    """

    if not text:
        return ""

    lines = []

    for line in text.splitlines():

        line = normalize_line(
            line
        )

        if not line:
            continue

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


def normalized_key(
    text: str,
) -> str:

    return normalize_line(
        text
    ).lower()


def text_hash(
    text: str,
) -> str:

    normalized = normalize_text(
        text
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def split_lines(
    text: str,
) -> list[str]:

    return [
        line
        for line in normalize_text(
            text
        ).splitlines()
        if line
    ]


def get_new_lines(
    base_text: str,
    candidate_text: str,
) -> str:
    """
    Remove text already present in the original page.

    Primarily used for target-panel extraction.
    """

    base_lines = split_lines(
        base_text
    )

    candidate_lines = split_lines(
        candidate_text
    )

    base_keys = {
        normalized_key(
            line
        )
        for line in base_lines
    }

    new_lines = []

    seen = set()

    for line in candidate_lines:

        key = normalized_key(
            line
        )

        if not key:
            continue

        if key in base_keys:
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        new_lines.append(
            line
        )

    return "\n".join(
        new_lines
    )


# ============================================================
# LANGUAGE
# ============================================================

def detect_content_language(
    text: str,
) -> str:
    """
    Detect actual rendered language rather than relying
    on /ar/, /en/ or default routing.
    """

    arabic_chars = len(
        re.findall(
            r"[\u0600-\u06FF]",
            text,
        )
    )

    english_chars = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    total = (
        arabic_chars
        + english_chars
    )

    if total == 0:
        return "unknown"

    arabic_ratio = (
        arabic_chars
        / total
    )

    english_ratio = (
        english_chars
        / total
    )

    if arabic_ratio >= 0.75:
        return "ar"

    if english_ratio >= 0.75:
        return "en"

    return "mixed"


def get_route_language(
    url: str,
) -> str:
    """
    Language suggested by the final URL.
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
# PAGE RENDERING
# ============================================================

def scroll_page(page):
    """
    Trigger lazy-loaded TE content.
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

def get_content_roots(page):
    """
    Find useful TE page content while avoiding:
    - header
    - navbar
    - breadcrumb
    - footer
    - comparison widget
    """

    roots = []

    # --------------------------------------------------------
    # Preferred TE template container
    # --------------------------------------------------------

    main_templates = page.locator(
        ".main_page_template_container"
    )

    if main_templates.count() > 0:

        for i in range(
            main_templates.count()
        ):

            element = (
                main_templates.nth(i)
            )

            try:

                text = normalize_text(
                    element.inner_text()
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
    # Fallback: #main-content direct children
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

            tag_name = element.evaluate(
                """
                el =>
                    el.tagName.toLowerCase()
                """
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

            text = normalize_text(
                element.inner_text()
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


def extract_main_text(
    page,
) -> str:
    """
    Extract complete visible main content.
    """

    roots = get_content_roots(
        page
    )

    texts = []

    seen_hashes = set()

    for root in roots:

        try:

            text = normalize_text(
                root.inner_text()
            )

        except Exception:
            continue

        if not text:
            continue

        current_hash = text_hash(
            text
        )

        if current_hash in seen_hashes:
            continue

        seen_hashes.add(
            current_hash
        )

        texts.append(
            text
        )

    return "\n\n".join(
        texts
    )


# ============================================================
# PAGE TITLE
# ============================================================

def extract_meta_content(
    page,
    selector: str,
) -> str:

    try:

        locator = page.locator(
            selector
        )

        if locator.count() == 0:
            return ""

        value = (
            locator.first.get_attribute(
                "content"
            )
            or ""
        )

        return normalize_line(
            value
        )

    except Exception:
        return ""


def extract_title(
    page,
) -> str:
    """
    Title priority:

        1. OpenGraph title
        2. Twitter title
        3. HTML document <title>
        4. Visible H1

    Metadata is intentionally preferred over H1 because
    several One Home pages use section-level H1 elements
    such as "الاتصال الدولى".
    """

    # --------------------------------------------------------
    # 1. OpenGraph
    # --------------------------------------------------------

    title = extract_meta_content(
        page,
        'meta[property="og:title"]',
    )

    if title:
        return title

    # --------------------------------------------------------
    # 2. Twitter metadata
    # --------------------------------------------------------

    title = extract_meta_content(
        page,
        'meta[name="twitter:title"]',
    )

    if title:
        return title

    # --------------------------------------------------------
    # 3. Browser/document title
    # --------------------------------------------------------

    try:

        title = normalize_line(
            page.title()
        )

        if title:
            return title

    except Exception:
        pass

    # --------------------------------------------------------
    # 4. Visible H1 fallback only
    # --------------------------------------------------------

    selectors = [
        ".main_page_template_container h1",
        "#main-content h1",
        "main h1",
        "h1",
    ]

    for selector in selectors:

        locator = page.locator(
            selector
        )

        for i in range(
            min(
                locator.count(),
                5,
            )
        ):

            try:

                text = normalize_line(
                    locator.nth(i)
                    .inner_text()
                )

                if text:
                    return text

            except Exception:
                continue

    return ""


# ============================================================
# VISIBLE CONTENT BLOCK SNAPSHOT
# ============================================================

BLOCK_SELECTOR = (
    "h1, h2, h3, h4, h5, h6, "
    "p, "
    "li, "
    "dt, dd, "
    "tr, "
    ".accordion-body, "
    ".accordion-content, "
    ".tab-pane, "
    '[role="tabpanel"], '
    ".card-body"
)


def extract_visible_blocks(
    page,
) -> list[str]:
    """
    Snapshot visible semantic content blocks.

    This is the fallback for sites such as One Home where
    clicking a tab changes visible content but the control
    does not point to a useful target-panel element.

    We intentionally avoid arbitrary DIVs because they tend
    to duplicate large portions of the whole page.
    """

    roots = get_content_roots(
        page
    )

    collected = []

    seen = set()

    for root in roots:

        try:

            raw_blocks = root.locator(
                BLOCK_SELECTOR
            ).evaluate_all(
                """
                elements => {
                    function visible(el) {
                        const style =
                            window.getComputedStyle(el);

                        if (
                            style.display === "none" ||
                            style.visibility === "hidden"
                        ) {
                            return false;
                        }

                        const rect =
                            el.getBoundingClientRect();

                        return (
                            rect.width > 0 &&
                            rect.height > 0
                        );
                    }

                    return elements
                        .filter(visible)
                        .map(
                            el =>
                                el.innerText || ""
                        );
                }
                """
            )

        except Exception:
            continue

        for raw_text in raw_blocks:

            text = normalize_text(
                raw_text
            )

            if not text:
                continue

            key = (
                normalize_line(
                    text
                ).lower()
            )

            if not key:
                continue

            if key in seen:
                continue

            seen.add(
                key
            )

            collected.append(
                text
            )

    return collected


def block_diff(
    before_blocks: list[str],
    after_blocks: list[str],
    base_text: str,
) -> str:
    """
    Return semantic visible blocks that appeared after a click.

    This is more precise than comparing every line from the
    entire page.
    """

    before_keys = {
        normalize_line(
            block
        ).lower()
        for block in before_blocks
    }

    normalized_base = (
        normalize_line(
            base_text
        ).lower()
    )

    new_blocks = []

    seen = set()

    for block in after_blocks:

        clean_block = normalize_text(
            block
        )

        if not clean_block:
            continue

        key = (
            normalize_line(
                clean_block
            ).lower()
        )

        if not key:
            continue

        # Already visible immediately before click.
        if key in before_keys:
            continue

        # Already part of the original base page.
        if (
            key
            and key in normalized_base
        ):
            continue

        if key in seen:
            continue

        seen.add(
            key
        )

        new_blocks.append(
            clean_block
        )

    # --------------------------------------------------------
    # Remove exact smaller duplicates when the exact same
    # block already exists as another selected block.
    #
    # We deliberately do NOT aggressively remove substring
    # relationships because price/package tables may contain
    # repeated numbers that are still meaningful.
    # --------------------------------------------------------

    output = []

    output_seen = set()

    for block in new_blocks:

        key = normalize_line(
            block
        ).lower()

        if key in output_seen:
            continue

        output_seen.add(
            key
        )

        output.append(
            block
        )

    return "\n".join(
        output
    )


# ============================================================
# INTERACTIVE CONTROL FILTERING
# ============================================================

def is_blocked_control(
    label: str,
) -> bool:

    label_lower = (
        label.lower()
    )

    return any(
        blocked.lower()
        in label_lower
        for blocked
        in BLOCKED_CONTROL_TEXT
    )


def is_machine_control_label(
    label: str,
) -> bool:
    """
    Ignore implementation-generated controls such as:

    collapseId_expandaple_contentId_0_3938221
    """

    label = (
        label
        .strip()
        .lower()
    )

    if not label:
        return False

    machine_patterns = [
        r"^collapseid[_a-z0-9-]*$",
        r"^contentid[_a-z0-9-]*$",
        r"^accordion[_a-z0-9-]*$",
        r"^collapse[_a-z0-9-]*$",
    ]

    return any(
        re.fullmatch(
            pattern,
            label,
        )
        for pattern
        in machine_patterns
    )


def get_control_target_id(
    control: dict,
) -> str:

    aria_controls = (
        control.get(
            "aria_controls",
            "",
        )
        or ""
    ).strip()

    if aria_controls:
        return aria_controls.lstrip(
            "#"
        )

    data_bs_target = (
        control.get(
            "data_bs_target",
            "",
        )
        or ""
    ).strip()

    if data_bs_target.startswith("#"):
        return data_bs_target[1:]

    data_target = (
        control.get(
            "data_target",
            "",
        )
        or ""
    ).strip()

    if data_target.startswith("#"):
        return data_target[1:]

    href = (
        control.get(
            "href",
            "",
        )
        or ""
    ).strip()

    if href.startswith("#"):
        return href[1:]

    return ""


# ============================================================
# INTERACTIVE CONTROL DISCOVERY
# ============================================================

def discover_interactive_controls(
    page,
) -> list[dict]:
    """
    Discover safe informational same-page interactions.

    Supported:

    href="#..."
    aria-controls="..."
    data-bs-target="#..."
    data-target="#..."
    role="tab"
    aria-expanded
    summary
    """

    roots = get_content_roots(
        page
    )

    controls = []

    seen_keys = set()

    counter = 0

    selector = (
        'a[href^="#"], '
        '[aria-controls], '
        '[data-bs-target^="#"], '
        '[data-target^="#"], '
        '[role="tab"], '
        'button[aria-expanded], '
        'summary'
    )

    for root in roots:

        candidates = root.locator(
            selector
        )

        count = candidates.count()

        for i in range(
            count
        ):

            element = candidates.nth(
                i
            )

            try:

                if not element.is_visible():
                    continue

                label = normalize_line(
                    element.inner_text()
                )

                if not label:

                    label = normalize_line(
                        (
                            element.get_attribute(
                                "aria-label"
                            )
                            or element.get_attribute(
                                "title"
                            )
                            or ""
                        )
                    )

                href = (
                    element.get_attribute(
                        "href"
                    )
                    or ""
                ).strip()

                aria_controls = (
                    element.get_attribute(
                        "aria-controls"
                    )
                    or ""
                ).strip()

                data_bs_target = (
                    element.get_attribute(
                        "data-bs-target"
                    )
                    or ""
                ).strip()

                data_target = (
                    element.get_attribute(
                        "data-target"
                    )
                    or ""
                ).strip()

                role = (
                    element.get_attribute(
                        "role"
                    )
                    or ""
                ).strip()

                aria_expanded = (
                    element.get_attribute(
                        "aria-expanded"
                    )
                    or ""
                ).strip()

                tag_name = (
                    element.evaluate(
                        """
                        el =>
                            el.tagName.toLowerCase()
                        """
                    )
                )

                # ------------------------------------------------
                # Unsafe/action controls
                # ------------------------------------------------

                if is_blocked_control(
                    label
                ):
                    continue

                # ------------------------------------------------
                # CMS-generated duplicate control
                # ------------------------------------------------

                if is_machine_control_label(
                    label
                ):
                    continue

                has_safe_target = (
                    href.startswith("#")
                    or bool(
                        aria_controls
                    )
                    or data_bs_target.startswith(
                        "#"
                    )
                    or data_target.startswith(
                        "#"
                    )
                    or role == "tab"
                    or bool(
                        aria_expanded
                    )
                    or tag_name == "summary"
                )

                if not has_safe_target:
                    continue

                target = (
                    aria_controls
                    or data_bs_target
                    or data_target
                    or href
                )

                target_key = (
                    target
                    .lstrip("#")
                    .strip()
                    .lower()
                )

                # ------------------------------------------------
                # Deduplicate controls primarily by target.
                # ------------------------------------------------

                if target_key:

                    key = (
                        "target",
                        target_key,
                    )

                else:

                    key = (
                        "label",
                        label.lower(),
                        role.lower(),
                        tag_name,
                    )

                if key in seen_keys:
                    continue

                seen_keys.add(
                    key
                )

                crawler_id = (
                    f"te-crawler-{counter}"
                )

                counter += 1

                element.evaluate(
                    """
                    (el, crawlerId) => {
                        el.setAttribute(
                            "data-te-crawler-id",
                            crawlerId
                        );
                    }
                    """,
                    crawler_id,
                )

                controls.append(
                    {
                        "crawler_id":
                            crawler_id,

                        "label":
                            label,

                        "href":
                            href,

                        "aria_controls":
                            aria_controls,

                        "data_bs_target":
                            data_bs_target,

                        "data_target":
                            data_target,

                        "role":
                            role,

                        "aria_expanded":
                            aria_expanded,

                        "tag_name":
                            tag_name,

                        "target_id":
                            target.lstrip("#")
                            if target
                            else "",
                    }
                )

            except Exception:
                continue

    return controls


# ============================================================
# TARGET PANEL EXTRACTION
# ============================================================

def extract_target_text(
    page,
    control: dict,
) -> str:
    """
    Try to extract the exact DOM target identified by
    aria-controls / data-target / href.

    If TE's markup does not expose a useful target,
    caller falls back to semantic DOM-block diff.
    """

    target_id = get_control_target_id(
        control
    )

    if not target_id:
        return ""

    try:

        result = page.evaluate(
            """
            targetId => {
                const element =
                    document.getElementById(targetId);

                if (!element) {
                    return "";
                }

                const style =
                    window.getComputedStyle(element);

                if (
                    style.display === "none" ||
                    style.visibility === "hidden"
                ) {
                    return "";
                }

                const rect =
                    element.getBoundingClientRect();

                if (
                    rect.width <= 0 ||
                    rect.height <= 0
                ) {
                    return "";
                }

                return element.innerText || "";
            }
            """,
            target_id,
        )

        return normalize_text(
            result or ""
        )

    except Exception:
        return ""


# ============================================================
# SAFE CONTROL CLICK
# ============================================================

def click_control_safely(
    locator,
) -> str:
    """
    Click a previously-filtered informational control.

    Strategy:

    1. Normal Playwright click.
    2. DOM click fallback when TE's dynamic layout leaves the
       element present but Playwright actionability checks time out.

    Returns the click method used.

    Raises RuntimeError only if both strategies fail.
    """

    first = locator.first

    try:
        first.scroll_into_view_if_needed(
            timeout=3000,
        )

        first.click(
            timeout=4000,
            no_wait_after=True,
        )

        return "playwright_click"

    except Exception as playwright_error:

        try:
            first.evaluate(
                """
                el => el.click()
                """
            )

            return "dom_click"

        except Exception as dom_error:

            raise RuntimeError(
                "Playwright click failed: "
                f"{str(playwright_error)[:180]} | "
                "DOM click failed: "
                f"{str(dom_error)[:180]}"
            ) from dom_error


# ============================================================
# INTERACTION EXTRACTION
# ============================================================

def click_and_extract_sections(
    page,
    base_text: str,
) -> tuple[list[dict], list[dict]]:

    controls = discover_interactive_controls(
        page
    )

    sections = []

    control_errors = []

    seen_section_hashes = set()

    print(
        "   Interactive controls:",
        len(
            controls
        ),
    )

    for control in controls:

        crawler_id = control[
            "crawler_id"
        ]

        display_name = (
            control["label"][:60]
            or control["href"]
            or control["aria_controls"]
            or control["target_id"]
            or "(unnamed control)"
        )

        locator = page.locator(
            f'[data-te-crawler-id="{crawler_id}"]'
        )

        if locator.count() == 0:

            error_text = (
                "Control disappeared from the DOM "
                "before it could be processed."
            )

            control_errors.append(
                {
                    "label":
                        control["label"],

                    "target_id":
                        get_control_target_id(
                            control
                        ),

                    "href":
                        control["href"],

                    "error":
                        error_text,
                }
            )

            print(
                "      Control error:",
                display_name,
                "->",
                error_text,
            )

            continue

        try:

            # ------------------------------------------------
            # Snapshot semantic visible blocks BEFORE click.
            # ------------------------------------------------

            before_blocks = (
                extract_visible_blocks(
                    page
                )
            )

            before_text = (
                extract_main_text(
                    page
                )
            )

            # ------------------------------------------------
            # Click safely.
            #
            # TE sometimes leaves controls in the DOM after
            # another tab/accordion changed the layout. In that
            # case the normal Playwright click may time out even
            # though a DOM click still works correctly.
            # ------------------------------------------------

            click_method = (
                click_control_safely(
                    locator
                )
            )

            page.wait_for_timeout(
                INTERACTION_WAIT_MS
            )

            # ------------------------------------------------
            # Snapshot AFTER click.
            # ------------------------------------------------

            after_text = (
                extract_main_text(
                    page
                )
            )

            after_blocks = (
                extract_visible_blocks(
                    page
                )
            )

            changed_page = (
                text_hash(
                    before_text
                )
                !=
                text_hash(
                    after_text
                )
            )

            # ------------------------------------------------
            # Strategy 1:
            # exact target panel
            # ------------------------------------------------

            target_text = extract_target_text(
                page,
                control,
            )

            new_text = ""

            extraction_method = ""

            if target_text:

                target_new_text = (
                    get_new_lines(
                        base_text,
                        target_text,
                    )
                )

                if (
                    len(
                        target_new_text
                    )
                    >= MIN_SECTION_CHARS
                ):

                    new_text = (
                        target_new_text
                    )

                    extraction_method = (
                        "target_panel"
                    )

            # ------------------------------------------------
            # Strategy 2:
            # before/after page line diff
            #
            # This is preferable when TE dynamically swaps an
            # entire product tab without exposing a useful target.
            # Comparing BEFORE vs AFTER also avoids leaking text
            # from previously opened controls into this section.
            # ------------------------------------------------

            if not new_text:

                page_new_text = (
                    get_new_lines(
                        before_text,
                        after_text,
                    )
                )

                if (
                    len(
                        page_new_text
                    )
                    >= MIN_SECTION_CHARS
                ):

                    new_text = (
                        page_new_text
                    )

                    extraction_method = (
                        "page_diff"
                    )

            # ------------------------------------------------
            # Strategy 3:
            # semantic DOM block diff
            #
            # Emergency fallback only.
            # ------------------------------------------------

            if not new_text:

                block_new_text = (
                    block_diff(
                        before_blocks=(
                            before_blocks
                        ),
                        after_blocks=(
                            after_blocks
                        ),
                        base_text=(
                            base_text
                        ),
                    )
                )

                if (
                    len(
                        block_new_text
                    )
                    >= MIN_SECTION_CHARS
                ):

                    new_text = (
                        block_new_text
                    )

                    extraction_method = (
                        "block_diff"
                    )

            # ------------------------------------------------
            # Nothing useful revealed.
            # ------------------------------------------------

            if not new_text:

                print(
                    "      SKIP:",
                    display_name,
                    "-> already visible / "
                    "no useful new content",
                    f"[{click_method}]",
                )

                continue

            section_hash = text_hash(
                new_text
            )

            # ------------------------------------------------
            # Same content reached through another control.
            # ------------------------------------------------

            if (
                section_hash
                in seen_section_hashes
            ):

                print(
                    "      SKIP:",
                    display_name,
                    "-> duplicate section",
                    f"[{click_method}]",
                )

                continue

            seen_section_hashes.add(
                section_hash
            )

            sections.append(
                {
                    "label":
                        control["label"],

                    "target_id":
                        get_control_target_id(
                            control
                        ),

                    "href":
                        control["href"],

                    "aria_controls":
                        control[
                            "aria_controls"
                        ],

                    "data_bs_target":
                        control[
                            "data_bs_target"
                        ],

                    "data_target":
                        control[
                            "data_target"
                        ],

                    "role":
                        control["role"],

                    "tag_name":
                        control[
                            "tag_name"
                        ],

                    "changed_page":
                        changed_page,

                    "click_method":
                        click_method,

                    "extraction_method":
                        extraction_method,

                    "text":
                        new_text,

                    "text_hash":
                        section_hash,
                }
            )

            print(
                "      CLICK:",
                display_name,
                "->",
                len(
                    new_text
                ),
                "new chars",
                f"[{extraction_method}]",
                f"[{click_method}]",
            )

        except Exception as error:

            error_text = str(
                error
            )

            control_errors.append(
                {
                    "label":
                        control["label"],

                    "target_id":
                        get_control_target_id(
                            control
                        ),

                    "href":
                        control["href"],

                    "error":
                        error_text,
                }
            )

            print(
                "      Control error:",
                display_name,
                "->",
                error_text[:180],
            )

    return (
        sections,
        control_errors,
    )


# ============================================================
# REPRESENTATIVE TEST SELECTION
# ============================================================

def item_matches(
    item: dict,
    contains: str | None = None,
    item_type: str | None = None,
    language: str | None = None,
) -> bool:

    url = item.get(
        "url",
        "",
    ).lower()

    if (
        contains is not None
        and contains.lower()
        not in url
    ):
        return False

    if (
        item_type is not None
        and item.get(
            "type"
        ) != item_type
    ):
        return False

    if (
        language is not None
        and item.get(
            "route_language"
        ) != language
    ):
        return False

    return True


def build_representative_test_selection(
    queue_items: list[dict],
    limit: int,
) -> list[dict]:
    """
    Build a deterministic test set covering:

    - Arabic
    - English
    - default routes
    - state URLs
    - normal pages
    - WE Space
    - One Home
    - 5G
    - eKYC
    - Nitro
    - home
    - mobile
    - services
    - devices

    This replaces the previous "first 10 state +
    first 10 page" approach.
    """

    selected = []

    selected_urls = set()

    def add_first(
        contains=None,
        item_type=None,
        language=None,
    ):
        if len(
            selected
        ) >= limit:
            return

        # Exact profile first.
        for item in queue_items:

            url = item.get(
                "url",
                "",
            )

            if url in selected_urls:
                continue

            if item_matches(
                item=item,
                contains=contains,
                item_type=item_type,
                language=language,
            ):

                selected.append(
                    item
                )

                selected_urls.add(
                    url
                )

                return

        # ----------------------------------------------------
        # If requested language does not exist for that family,
        # relax language but keep family/type.
        # ----------------------------------------------------

        if language is not None:

            for item in queue_items:

                url = item.get(
                    "url",
                    "",
                )

                if url in selected_urls:
                    continue

                if item_matches(
                    item=item,
                    contains=contains,
                    item_type=item_type,
                    language=None,
                ):

                    selected.append(
                        item
                    )

                    selected_urls.add(
                        url
                    )

                    return

    # --------------------------------------------------------
    # High-value dynamic/state pages
    # --------------------------------------------------------

    profiles = [
        # WE Space
        (
            "we-space",
            "state",
            "default",
        ),
        (
            "we-space",
            "state",
            "ar",
        ),
        (
            "we-space",
            "state",
            "en",
        ),

        # One Home
        (
            "one-home",
            "state",
            "ar",
        ),
        (
            "one-home",
            "state",
            "en",
        ),
        (
            "one-home",
            "state",
            "default",
        ),

        # Nitro dynamic states
        (
            "nitro",
            "state",
            "ar",
        ),
        (
            "nitro",
            "state",
            "en",
        ),
        (
            "nitro",
            "state",
            "default",
        ),

        # 5G interactive pages
        (
            "5g",
            "page",
            "ar",
        ),
        (
            "5g",
            "page",
            "en",
        ),
        (
            "5g",
            "page",
            "default",
        ),

        # eKYC FAQ
        (
            "ekyc",
            "page",
            "ar",
        ),
        (
            "ekyc",
            "page",
            "en",
        ),
        (
            "ekyc",
            "page",
            "default",
        ),

        # Mobile
        (
            "/personal/mobile",
            "page",
            "ar",
        ),
        (
            "/personal/mobile",
            "page",
            "en",
        ),

        # Home
        (
            "/personal/home",
            "page",
            "ar",
        ),
        (
            "/personal/home",
            "page",
            "en",
        ),

        # Services
        (
            "/personal/services",
            "page",
            "ar",
        ),
        (
            "/personal/services",
            "page",
            "en",
        ),

        # Devices
        (
            "/personal/devices",
            "page",
            "ar",
        ),
        (
            "/personal/devices",
            "page",
            "en",
        ),

        # Explicit default route
        (
            "/personal",
            "page",
            "default",
        ),
    ]

    for (
        contains,
        item_type,
        language,
    ) in profiles:

        add_first(
            contains=contains,
            item_type=item_type,
            language=language,
        )

        if len(
            selected
        ) >= limit:
            break

    # --------------------------------------------------------
    # Balanced fallback if a profile did not exist.
    # --------------------------------------------------------

    if len(
        selected
    ) < limit:

        combinations = [
            (
                "state",
                "ar",
            ),
            (
                "state",
                "en",
            ),
            (
                "state",
                "default",
            ),
            (
                "page",
                "ar",
            ),
            (
                "page",
                "en",
            ),
            (
                "page",
                "default",
            ),
        ]

        progress = True

        while (
            len(
                selected
            ) < limit
            and progress
        ):

            progress = False

            for (
                item_type,
                language,
            ) in combinations:

                for item in queue_items:

                    url = item.get(
                        "url",
                        "",
                    )

                    if url in selected_urls:
                        continue

                    if (
                        item.get(
                            "type"
                        )
                        != item_type
                    ):
                        continue

                    if (
                        item.get(
                            "route_language"
                        )
                        != language
                    ):
                        continue

                    selected.append(
                        item
                    )

                    selected_urls.add(
                        url
                    )

                    progress = True

                    break

                if len(
                    selected
                ) >= limit:
                    break

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if len(
        selected
    ) < limit:

        for item in queue_items:

            url = item.get(
                "url",
                "",
            )

            if url in selected_urls:
                continue

            selected.append(
                item
            )

            selected_urls.add(
                url
            )

            if len(
                selected
            ) >= limit:
                break

    return selected


# ============================================================
# OUTPUT
# ============================================================

def load_existing_results(
    output_path: Path,
):
    """
    Resume only results produced by this extractor version.
    """

    if not output_path.exists():
        return []

    try:

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            results = json.load(
                file
            )

        return [
            result
            for result
            in results
            if result.get(
                "extractor_version"
            ) == EXTRACTOR_VERSION
        ]

    except Exception:
        return []


def save_results(
    results: list[dict],
    output_path: Path,
):

    write_json_atomic(
        output_path,
        results,
    )


# ============================================================
# TEST SUMMARY HELPERS
# ============================================================

def print_test_distribution(
    selected_items: list[dict],
):

    language_counts = {
        "ar": 0,
        "en": 0,
        "default": 0,
    }

    type_counts = {
        "page": 0,
        "state": 0,
    }

    for item in selected_items:

        language = item.get(
            "route_language",
            "default",
        )

        if language not in language_counts:
            language_counts[
                language
            ] = 0

        language_counts[
            language
        ] += 1

        item_type = item.get(
            "type",
            "page",
        )

        if item_type not in type_counts:
            type_counts[
                item_type
            ] = 0

        type_counts[
            item_type
        ] += 1

    print(
        "\nTest distribution:"
    )

    print(
        "   Arabic routes:",
        language_counts.get(
            "ar",
            0,
        ),
    )

    print(
        "   English routes:",
        language_counts.get(
            "en",
            0,
        ),
    )

    print(
        "   Default routes:",
        language_counts.get(
            "default",
            0,
        ),
    )

    print(
        "   State URLs:",
        type_counts.get(
            "state",
            0,
        ),
    )

    print(
        "   Normal pages:",
        type_counts.get(
            "page",
            0,
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    output_path = (
        get_output_path()
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        queue_items = json.load(
            file
        )

    # --------------------------------------------------------
    # REPRESENTATIVE TEST / FULL RUN
    # --------------------------------------------------------

    if TEST_LIMIT is not None:

        selected_items = (
            build_representative_test_selection(
                queue_items=queue_items,
                limit=TEST_LIMIT,
            )
        )

        # Test run always starts fresh.
        results = []

    else:

        selected_items = (
            queue_items
        )

        # Full extraction is resumable.
        results = (
            load_existing_results(
                output_path
            )
        )

    print(
        "Queue documents:",
        len(
            queue_items
        ),
    )

    print(
        "Selected for extraction:",
        len(
            selected_items
        ),
    )

    print_test_distribution(
        selected_items
    )

    print(
        "\nOutput:",
        output_path,
    )

    completed_urls = {
        result["url"]
        for result in results
        if result.get(
            "status"
        ) in {
            "ok",
            "empty",
        }
    }

    # --------------------------------------------------------
    # PLAYWRIGHT
    # --------------------------------------------------------

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True,
        )

        try:

            for index, item in enumerate(
                selected_items,
                start=1,
            ):

                url = item[
                    "url"
                ]

                if url in completed_urls:
                    continue

                print(
                    "\n"
                    + "=" * 90
                )

                print(
                    f"[{index}/"
                    f"{len(selected_items)}]"
                )

                print(
                    "Type:",
                    item["type"],
                    "| Route language:",
                    item[
                        "route_language"
                    ],
                )

                print(
                    url
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

                document = {
                    "extractor_version":
                        EXTRACTOR_VERSION,

                    "url":
                        url,

                    "final_url":
                        "",

                    "type":
                        item["type"],

                    "depth":
                        item.get(
                            "depth",
                            0,
                        ),

                    "route_language":
                        item[
                            "route_language"
                        ],

                    "final_route_language":
                        "unknown",

                    "content_language":
                        "unknown",

                    "title":
                        "",

                    "base_text":
                        "",

                    "base_hash":
                        "",

                    "sections":
                        [],

                    "control_errors":
                        [],

                    "discovery_methods":
                        item.get(
                            "discovery_methods",
                            [],
                        ),

                    "discovered_from":
                        item.get(
                            "discovered_from",
                            [],
                        ),

                    "status":
                        "error",

                    "error":
                        None,
                }

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

                    final_url = (
                        page.url
                    )

                    final_route_language = (
                        get_route_language(
                            final_url
                        )
                    )

                    title = (
                        extract_title(
                            page
                        )
                    )

                    base_text = (
                        extract_main_text(
                            page
                        )
                    )

                    content_language = (
                        detect_content_language(
                            base_text
                        )
                    )

                    print(
                        "   Final URL:",
                        final_url,
                    )

                    print(
                        "   Final route language:",
                        final_route_language,
                    )

                    print(
                        "   Title:",
                        title[:120],
                    )

                    print(
                        "   Base characters:",
                        len(
                            base_text
                        ),
                    )

                    print(
                        "   Content language:",
                        content_language,
                    )

                    sections = []

                    control_errors = []

                    if base_text:

                        (
                            sections,
                            control_errors,
                        ) = click_and_extract_sections(
                            page=page,
                            base_text=base_text,
                        )

                    document.update(
                        {
                            "final_url":
                                final_url,

                            "final_route_language":
                                final_route_language,

                            "content_language":
                                content_language,

                            "title":
                                title,

                            "base_text":
                                base_text,

                            "base_hash":
                                (
                                    text_hash(
                                        base_text
                                    )
                                    if base_text
                                    else ""
                                ),

                            "sections":
                                sections,

                            "control_errors":
                                control_errors,

                            "status":
                                (
                                    "ok"
                                    if base_text
                                    else "empty"
                                ),

                            "error":
                                None,
                        }
                    )

                    print(
                        "   Sections recorded:",
                        len(
                            sections
                        ),
                    )

                    print(
                        "   Control errors:",
                        len(
                            control_errors
                        ),
                    )

                except PlaywrightTimeoutError:

                    document[
                        "error"
                    ] = "timeout"

                    print(
                        "   -> Timeout"
                    )

                except KeyboardInterrupt:

                    print(
                        "\nInterrupted."
                    )

                    print(
                        "Saving completed extraction "
                        "results before exiting."
                    )

                    raise

                except Exception as error:

                    document[
                        "error"
                    ] = str(
                        error
                    )

                    print(
                        "   -> Error:",
                        error,
                    )

                finally:

                    context.close()

                # Keep exactly one latest result per URL.
                # This prevents duplicate records when an earlier
                # error is retried during a resumed full run.
                results = [
                    result
                    for result in results
                    if result.get(
                        "url"
                    ) != url
                ]

                results.append(
                    document
                )

                save_results(
                    results,
                    output_path,
                )

        finally:

            browser.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    ok_count = sum(
        result.get(
            "status"
        ) == "ok"
        for result in results
    )

    empty_count = sum(
        result.get(
            "status"
        ) == "empty"
        for result in results
    )

    error_count = sum(
        result.get(
            "status"
        ) == "error"
        for result in results
    )

    interactive_count = sum(
        len(
            result.get(
                "sections",
                [],
            )
        )
        for result in results
    )

    target_panel_count = sum(
        1
        for result in results
        for section in result.get(
            "sections",
            [],
        )
        if section.get(
            "extraction_method"
        ) == "target_panel"
    )

    block_diff_count = sum(
        1
        for result in results
        for section in result.get(
            "sections",
            [],
        )
        if section.get(
            "extraction_method"
        ) == "block_diff"
    )

    page_diff_count = sum(
        1
        for result in results
        for section in result.get(
            "sections",
            [],
        )
        if section.get(
            "extraction_method"
        ) == "page_diff"
    )

    control_error_count = sum(
        len(
            result.get(
                "control_errors",
                [],
            )
        )
        for result in results
    )

    documents_with_control_errors = sum(
        bool(
            result.get(
                "control_errors",
                [],
            )
        )
        for result in results
    )

    content_language_counts = {
        "ar": 0,
        "en": 0,
        "mixed": 0,
        "unknown": 0,
    }

    for result in results:

        language = result.get(
            "content_language",
            "unknown",
        )

        if (
            language
            not in content_language_counts
        ):
            content_language_counts[
                language
            ] = 0

        content_language_counts[
            language
        ] += 1

    print(
        "\n"
        + "=" * 90
    )

    print(
        "Extraction complete"
    )

    print(
        "=" * 90
    )

    print(
        "OK:",
        ok_count,
    )

    print(
        "Empty:",
        empty_count,
    )

    print(
        "Errors:",
        error_count,
    )

    print(
        "Interactive sections:",
        interactive_count,
    )

    print(
        "Target-panel sections:",
        target_panel_count,
    )

    print(
        "Block-diff sections:",
        block_diff_count,
    )

    print(
        "Page-diff sections:",
        page_diff_count,
    )

    print(
        "Control errors:",
        control_error_count,
    )

    print(
        "Documents with control errors:",
        documents_with_control_errors,
    )

    print(
        "\nActual content languages:"
    )

    print(
        "   Arabic:",
        content_language_counts.get(
            "ar",
            0,
        ),
    )

    print(
        "   English:",
        content_language_counts.get(
            "en",
            0,
        ),
    )

    print(
        "   Mixed:",
        content_language_counts.get(
            "mixed",
            0,
        ),
    )

    print(
        "   Unknown:",
        content_language_counts.get(
            "unknown",
            0,
        ),
    )

    print(
        "\nSaved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
