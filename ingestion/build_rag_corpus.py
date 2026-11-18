import hashlib
import json
import re

from collections import defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
INGESTION_DIR = Path(__file__).resolve().parent
DATA_DIR = INGESTION_DIR / "data"

# IMPORTANT:
# While the full 771-page extraction is running, use the
# already-finished test extraction.
INPUT_PATH = (
    DATA_DIR
    / "raw"
    / "te_dynamic_content.json"
)

OUTPUT_PATH = (
    DATA_DIR
    / "processed"
    / "te_documents.json"
)

REPORT_PATH = (
    DATA_DIR
    / "processed"
    / "te_documents_report.json"
)


LOW_VALUE_LINES = {
    "الرجوع الي الأعلي",
    "الرجوع إلى الأعلى",
    "الرجوع للأعلى",
    "back to top",

    # Generic UI / navigation text
    "more details",
    "تفاصيل أكتر",

    # App download/device-detection UI
    "لم نتمكن من التعرف على جهازك",
    "تنزيل باستخدام",
    "download using",
}


CORPUS_VERSION = 1

MIN_DOCUMENT_CHARS = 30


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_line(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text or "",
    ).strip()


def normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    lines = []

    previous_key = None

    for line in text.splitlines():

        line = normalize_line(
            line
        )

        if not line:
            continue

        key = line.lower()

        # Remove obvious UI/navigation boilerplate.
        if key in {
            value.lower()
            for value in LOW_VALUE_LINES
        }:
            continue

        # Remove immediately repeated identical lines.
        if key == previous_key:
            continue

        lines.append(
            line
        )

        previous_key = key

    return "\n".join(
        lines
    )


def text_hash(
    text: str,
) -> str:

    normalized = normalize_text(
        text
    ).lower()

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def stable_id(
    *parts,
) -> str:

    raw = "||".join(
        str(part or "")
        for part in parts
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:24]


# ============================================================
# LANGUAGE
# ============================================================

def detect_text_language(
    text: str,
) -> str:

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


# ============================================================
# SECTION LABEL
# ============================================================

def is_machine_label(
    label: str,
) -> bool:
    """
    Detect TE CMS/internal labels such as:

        collapseId_3993735
        contentId_123
        accordion_xyz

    These should not be added to the RAG text.
    """

    label = normalize_line(
        label
    ).lower()

    if not label:
        return True

    patterns = [
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
        for pattern in patterns
    )


def build_section_text(
    label: str,
    section_text: str,
) -> str:
    """
    FAQ example:

        What is eKYC?
        eKYC is ...

    For internal labels like collapseId_123,
    only the useful answer text is retained.
    """

    section_text = normalize_text(
        section_text
    )

    label = normalize_line(
        label
    )

    if (
        label
        and not is_machine_label(
            label
        )
    ):

        # Avoid adding the label twice if it is already
        # the first part of the extracted section.
        normalized_section = (
            section_text.lower()
        )

        if not normalized_section.startswith(
            label.lower()
        ):
            return (
                f"{label}\n"
                f"{section_text}"
            )

    return section_text


# ============================================================
# SOURCE PRIORITY
# ============================================================

def source_priority(
    document: dict,
) -> tuple:
    """
    When the exact same content exists on multiple TE routes,
    prefer an explicit Arabic/English route over a default route.

    We still retain the duplicate URLs as alternate sources.
    """

    route_language = document.get(
        "route_language",
        "default",
    )

    content_language = document.get(
        "content_language",
        "unknown",
    )

    if (
        route_language in {"ar", "en"}
        and route_language == content_language
    ):
        language_score = 0

    elif route_language in {"ar", "en"}:
        language_score = 1

    else:
        language_score = 2

    source_kind = document.get(
        "source_kind",
        "base",
    )

    # Prefer exact interactive section over a large base page
    # if they somehow contain identical text.
    kind_score = (
        0
        if source_kind == "interactive_section"
        else 1
    )

    return (
        language_score,
        kind_score,
        document.get(
            "url",
            "",
        ),
    )


# ============================================================
# DOCUMENT CREATION
# ============================================================

def make_source_metadata(
    page: dict,
) -> dict:

    return {
        "url":
            page.get(
                "url",
                "",
            ),

        "final_url":
            page.get(
                "final_url",
                "",
            ),

        "title":
            page.get(
                "title",
                "",
            ),

        "type":
            page.get(
                "type",
                "",
            ),

        "route_language":
            page.get(
                "route_language",
                "default",
            ),

        "final_route_language":
            page.get(
                "final_route_language",
                "unknown",
            ),
    }


def build_base_document(
    page: dict,
) -> dict | None:

    text = normalize_text(
        page.get(
            "base_text",
            "",
        )
    )

    if len(text) < MIN_DOCUMENT_CHARS:
        return None

    content_language = (
        detect_text_language(
            text
        )
    )

    url = (
        page.get(
            "final_url"
        )
        or page.get(
            "url",
            "",
        )
    )

    document = {
        "corpus_version":
            CORPUS_VERSION,

        "document_id":
            stable_id(
                "te",
                url,
                "base",
                text_hash(
                    text
                ),
            ),

        "source":
            "telecom_egypt_website",

        "source_kind":
            "base",

        "url":
            page.get(
                "url",
                "",
            ),

        "final_url":
            page.get(
                "final_url",
                "",
            ),

        "title":
            normalize_line(
                page.get(
                    "title",
                    "",
                )
            ),

        "page_type":
            page.get(
                "type",
                "",
            ),

        "depth":
            page.get(
                "depth",
                0,
            ),

        "route_language":
            page.get(
                "route_language",
                "default",
            ),

        "final_route_language":
            page.get(
                "final_route_language",
                "unknown",
            ),

        "content_language":
            content_language,

        "section_label":
            "",

        "target_id":
            "",

        "extraction_method":
            "base",

        "click_method":
            "",

        "page_status":
            page.get(
                "status",
                "",
            ),

        "page_has_control_errors":
            bool(
                page.get(
                    "control_errors",
                    [],
                )
            ),

        "text":
            text,

        "text_hash":
            text_hash(
                text
            ),

        "char_count":
            len(
                text
            ),

        "alternate_sources":
            [],
    }

    return document


def build_section_document(
    page: dict,
    section: dict,
    section_index: int,
) -> dict | None:

    raw_text = normalize_text(
        section.get(
            "text",
            "",
        )
    )

    if not raw_text:
        return None

    label = normalize_line(
        section.get(
            "label",
            "",
        )
    )

    text = build_section_text(
        label=label,
        section_text=raw_text,
    )

    if len(text) < MIN_DOCUMENT_CHARS:
        return None

    content_language = (
        detect_text_language(
            text
        )
    )

    url = (
        page.get(
            "final_url"
        )
        or page.get(
            "url",
            "",
        )
    )

    target_id = normalize_line(
        section.get(
            "target_id",
            "",
        )
    )

    document = {
        "corpus_version":
            CORPUS_VERSION,

        "document_id":
            stable_id(
                "te",
                url,
                "section",
                section_index,
                target_id,
                text_hash(
                    text
                ),
            ),

        "source":
            "telecom_egypt_website",

        "source_kind":
            "interactive_section",

        "url":
            page.get(
                "url",
                "",
            ),

        "final_url":
            page.get(
                "final_url",
                "",
            ),

        "title":
            normalize_line(
                page.get(
                    "title",
                    "",
                )
            ),

        "page_type":
            page.get(
                "type",
                "",
            ),

        "depth":
            page.get(
                "depth",
                0,
            ),

        "route_language":
            page.get(
                "route_language",
                "default",
            ),

        "final_route_language":
            page.get(
                "final_route_language",
                "unknown",
            ),

        "content_language":
            content_language,

        "section_label":
            (
                ""
                if is_machine_label(
                    label
                )
                else label
            ),

        # Keep internal ID as metadata even though we do not
        # put it into the natural-language RAG text.
        "target_id":
            target_id,

        "extraction_method":
            section.get(
                "extraction_method",
                "",
            ),

        "click_method":
            section.get(
                "click_method",
                "",
            ),

        "page_status":
            page.get(
                "status",
                "",
            ),

        "page_has_control_errors":
            bool(
                page.get(
                    "control_errors",
                    [],
                )
            ),

        "text":
            text,

        "text_hash":
            text_hash(
                text
            ),

        "char_count":
            len(
                text
            ),

        "alternate_sources":
            [],
    }

    return document


# ============================================================
# EXACT DEDUPLICATION
# ============================================================

def deduplicate_documents(
    documents: list[dict],
) -> tuple[list[dict], int]:
    """
    Deduplicate exact normalized knowledge content.

    Important:
    Arabic and English translations are NOT deduplicated because
    their text hashes are different.

    Exact duplicate Arabic/default routes may collapse into one
    document, while all source URLs remain preserved.
    """

    grouped = defaultdict(
        list
    )

    for document in documents:

        grouped[
            document["text_hash"]
        ].append(
            document
        )

    output = []

    removed_count = 0

    for group in grouped.values():

        if len(group) == 1:

            output.append(
                group[0]
            )

            continue

        group.sort(
            key=source_priority
        )

        canonical = group[0]

        alternate_sources = []

        seen_sources = set()

        for duplicate in group[1:]:

            source_metadata = {
                "url":
                    duplicate.get(
                        "url",
                        "",
                    ),

                "final_url":
                    duplicate.get(
                        "final_url",
                        "",
                    ),

                "title":
                    duplicate.get(
                        "title",
                        "",
                    ),

                "route_language":
                    duplicate.get(
                        "route_language",
                        "default",
                    ),

                "final_route_language":
                    duplicate.get(
                        "final_route_language",
                        "unknown",
                    ),

                "page_type":
                    duplicate.get(
                        "page_type",
                        "",
                    ),
            }

            source_key = (
                source_metadata["url"],
                source_metadata["final_url"],
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            alternate_sources.append(
                source_metadata
            )

        canonical[
            "alternate_sources"
        ] = alternate_sources

        canonical[
            "duplicate_source_count"
        ] = len(
            alternate_sources
        )

        output.append(
            canonical
        )

        removed_count += (
            len(group) - 1
        )

    output.sort(
        key=lambda document: (
            document.get(
                "content_language",
                "",
            ),
            document.get(
                "title",
                "",
            ),
            document.get(
                "source_kind",
                "",
            ),
            document.get(
                "document_id",
                "",
            ),
        )
    )

    return (
        output,
        removed_count,
    )


# ============================================================
# LOAD / SAVE
# ============================================================

def load_json(
    path: Path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def save_json(
    path: Path,
    data,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary.replace(
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    pages = load_json(
        INPUT_PATH
    )

    raw_documents = []

    skipped_empty = 0

    skipped_errors = 0

    accepted_pages = 0

    partial_pages = 0

    base_documents = 0

    section_documents = 0

    skipped_short_documents = 0

    for page in pages:

        status = page.get(
            "status",
            "",
        )

        if status == "empty":

            skipped_empty += 1

            continue

        if status == "error":

            skipped_errors += 1

            continue

        # Support both the current status model and older
        # extraction files where a page with control_errors
        # may still be labelled "ok".
        if status not in {
            "ok",
            "partial",
        }:

            skipped_errors += 1

            continue

        accepted_pages += 1

        if (
            status == "partial"
            or page.get(
                "control_errors",
                [],
            )
        ):
            partial_pages += 1

        base_document = (
            build_base_document(
                page
            )
        )

        if base_document:

            raw_documents.append(
                base_document
            )

            base_documents += 1

        elif page.get(
            "base_text"
        ):

            skipped_short_documents += 1

        sections = page.get(
            "sections",
            [],
        )

        for section_index, section in enumerate(
            sections
        ):

            section_document = (
                build_section_document(
                    page=page,
                    section=section,
                    section_index=section_index,
                )
            )

            if section_document:

                raw_documents.append(
                    section_document
                )

                section_documents += 1

            elif section.get(
                "text"
            ):

                skipped_short_documents += 1

    (
        clean_documents,
        exact_duplicates_removed,
    ) = deduplicate_documents(
        raw_documents
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    language_counts = defaultdict(
        int
    )

    source_kind_counts = defaultdict(
        int
    )

    total_chars = 0

    for document in clean_documents:

        language_counts[
            document.get(
                "content_language",
                "unknown",
            )
        ] += 1

        source_kind_counts[
            document.get(
                "source_kind",
                "unknown",
            )
        ] += 1

        total_chars += document.get(
            "char_count",
            0,
        )

    report = {
        "corpus_version":
            CORPUS_VERSION,

        "input_path":
            str(
                INPUT_PATH
            ),

        "input_pages":
            len(
                pages
            ),

        "accepted_pages":
            accepted_pages,

        "partial_pages":
            partial_pages,

        "skipped_empty_pages":
            skipped_empty,

        "skipped_error_pages":
            skipped_errors,

        "base_documents_before_dedupe":
            base_documents,

        "section_documents_before_dedupe":
            section_documents,

        "raw_documents":
            len(
                raw_documents
            ),

        "exact_duplicates_removed":
            exact_duplicates_removed,

        "final_documents":
            len(
                clean_documents
            ),

        "skipped_short_documents":
            skipped_short_documents,

        "languages":
            dict(
                sorted(
                    language_counts.items()
                )
            ),

        "source_kinds":
            dict(
                sorted(
                    source_kind_counts.items()
                )
            ),

        "total_characters":
            total_chars,

        "average_document_characters":
            (
                round(
                    total_chars
                    / len(
                        clean_documents
                    ),
                    2,
                )
                if clean_documents
                else 0
            ),
    }

    save_json(
        OUTPUT_PATH,
        clean_documents,
    )

    save_json(
        REPORT_PATH,
        report,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "TE RAG CORPUS BUILD"
    )

    print(
        "=" * 80
    )

    print(
        "Input pages:",
        len(
            pages
        ),
    )

    print(
        "Accepted pages:",
        accepted_pages,
    )

    print(
        "Partial pages:",
        partial_pages,
    )

    print(
        "Skipped empty pages:",
        skipped_empty,
    )

    print(
        "Skipped error pages:",
        skipped_errors,
    )

    print()

    print(
        "Base documents:",
        base_documents,
    )

    print(
        "Interactive section documents:",
        section_documents,
    )

    print(
        "Raw documents:",
        len(
            raw_documents
        ),
    )

    print(
        "Exact duplicates removed:",
        exact_duplicates_removed,
    )

    print(
        "Final documents:",
        len(
            clean_documents
        ),
    )

    print(
        "Skipped short documents:",
        skipped_short_documents,
    )

    print()

    print(
        "Languages:"
    )

    for language, count in sorted(
        language_counts.items()
    ):

        print(
            f"   {language}: {count}"
        )

    print()

    print(
        "Source kinds:"
    )

    for source_kind, count in sorted(
        source_kind_counts.items()
    ):

        print(
            f"   {source_kind}: {count}"
        )

    print()

    print(
        "Saved documents:",
        OUTPUT_PATH,
    )

    print(
        "Saved report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()