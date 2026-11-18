import hashlib
import json
import re

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INGESTION_DIR = Path(__file__).resolve().parent

DATA_DIR = INGESTION_DIR / "data"

INPUT_PATH = (
    DATA_DIR
    / "processed"
    / "te_documents.json"
)

OUTPUT_PATH = (
    DATA_DIR
    / "processed"
    / "te_chunks.json"
)

REPORT_PATH = (
    DATA_DIR
    / "processed"
    / "te_chunks_report.json"
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_VERSION = 1

# Character-based for now so we do NOT load the embedding
# model while Playwright extraction is running.
MAX_CHUNK_CHARS = 1400

OVERLAP_CHARS = 180

MIN_CHUNK_CHARS = 40


# ============================================================
# HELPERS
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

    lines = []

    for line in (
        text or ""
    ).splitlines():

        line = normalize_line(
            line
        )

        if line:
            lines.append(
                line
            )

    return "\n".join(
        lines
    )


def text_hash(
    text: str,
) -> str:

    return hashlib.sha256(
        normalize_text(
            text
        ).lower().encode(
            "utf-8"
        )
    ).hexdigest()


def stable_id(
    *parts,
) -> str:

    raw = "||".join(
        str(
            part or ""
        )
        for part in parts
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:24]


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
# CHUNKING
# ============================================================

def split_long_line(
    line: str,
    max_chars: int,
) -> list[str]:
    """
    Emergency split for a single very long line.
    Prefer punctuation boundaries when possible.
    """

    if len(
        line
    ) <= max_chars:

        return [
            line
        ]

    pieces = []

    remaining = line

    while len(
        remaining
    ) > max_chars:

        window = remaining[
            :max_chars
        ]

        split_positions = [
            window.rfind(". "),
            window.rfind("؟ "),
            window.rfind("! "),
            window.rfind("، "),
            window.rfind(", "),
            window.rfind("; "),
            window.rfind("؛ "),
        ]

        split_at = max(
            split_positions
        )

        # If no useful punctuation was found,
        # try the final whitespace.
        if split_at < (
            max_chars * 0.5
        ):

            split_at = (
                window.rfind(
                    " "
                )
            )

        # Absolute fallback.
        if split_at <= 0:

            split_at = max_chars

        else:

            split_at += 1

        piece = normalize_line(
            remaining[
                :split_at
            ]
        )

        if piece:
            pieces.append(
                piece
            )

        remaining = (
            remaining[
                split_at:
            ]
            .strip()
        )

    if remaining:

        pieces.append(
            normalize_line(
                remaining
            )
        )

    return pieces


def prepare_units(
    text: str,
) -> list[str]:
    """
    Preserve source lines as semantic units.

    A very long individual line is split safely.
    """

    units = []

    for line in normalize_text(
        text
    ).splitlines():

        if len(
            line
        ) <= MAX_CHUNK_CHARS:

            units.append(
                line
            )

        else:

            units.extend(
                split_long_line(
                    line,
                    MAX_CHUNK_CHARS,
                )
            )

    return units


def make_overlap_units(
    units: list[str],
) -> list[str]:
    """
    Keep a small amount of previous context for
    continuity between base-page chunks.
    """

    selected = []

    total_chars = 0

    for unit in reversed(
        units
    ):

        additional = (
            len(
                unit
            )
            + (
                1
                if selected
                else 0
            )
        )

        if (
            total_chars
            + additional
            > OVERLAP_CHARS
        ):

            break

        selected.append(
            unit
        )

        total_chars += (
            additional
        )

    selected.reverse()

    return selected


def chunk_text(
    text: str,
) -> list[str]:

    units = prepare_units(
        text
    )

    if not units:
        return []

    chunks = []

    current_units = []

    current_chars = 0

    for unit in units:

        additional_chars = (
            len(
                unit
            )
            + (
                1
                if current_units
                else 0
            )
        )

        if (
            current_units
            and (
                current_chars
                + additional_chars
                > MAX_CHUNK_CHARS
            )
        ):

            chunk = "\n".join(
                current_units
            )

            if len(
                chunk
            ) >= MIN_CHUNK_CHARS:

                chunks.append(
                    chunk
                )

            overlap_units = (
                make_overlap_units(
                    current_units
                )
            )

            current_units = (
                overlap_units
                + [
                    unit
                ]
            )

            current_chars = len(
                "\n".join(
                    current_units
                )
            )

        else:

            current_units.append(
                unit
            )

            current_chars += (
                additional_chars
            )

    if current_units:

        chunk = "\n".join(
            current_units
        )

        if len(
            chunk
        ) >= MIN_CHUNK_CHARS:

            chunks.append(
                chunk
            )

    return chunks


# ============================================================
# DOCUMENT → CHUNKS
# ============================================================

def build_chunks_for_document(
    document: dict,
) -> list[dict]:

    text = normalize_text(
        document.get(
            "text",
            "",
        )
    )

    if not text:
        return []

    source_kind = document.get(
        "source_kind",
        "",
    )

    # --------------------------------------------------------
    # Short interactive sections such as FAQ answers should
    # remain whole whenever possible.
    # --------------------------------------------------------

    if (
        source_kind
        == "interactive_section"
        and len(
            text
        ) <= MAX_CHUNK_CHARS
    ):

        text_chunks = [
            text
        ]

    else:

        text_chunks = (
            chunk_text(
                text
            )
        )

    output = []

    chunk_total = len(
        text_chunks
    )

    for chunk_index, chunk_text_value in enumerate(
        text_chunks
    ):

        chunk_hash = text_hash(
            chunk_text_value
        )

        chunk_id = stable_id(
            document.get(
                "document_id",
                "",
            ),
            chunk_index,
            chunk_hash,
        )

        chunk = {
            "chunk_version":
                CHUNK_VERSION,

            "chunk_id":
                chunk_id,

            "document_id":
                document.get(
                    "document_id",
                    "",
                ),

            "chunk_index":
                chunk_index,

            "chunk_total":
                chunk_total,

            "source":
                document.get(
                    "source",
                    "",
                ),

            "source_kind":
                source_kind,

            "url":
                document.get(
                    "url",
                    "",
                ),

            "final_url":
                document.get(
                    "final_url",
                    "",
                ),

            "title":
                document.get(
                    "title",
                    "",
                ),

            "section_label":
                document.get(
                    "section_label",
                    "",
                ),

            "content_language":
                document.get(
                    "content_language",
                    "unknown",
                ),

            "route_language":
                document.get(
                    "route_language",
                    "default",
                ),

            "final_route_language":
                document.get(
                    "final_route_language",
                    "unknown",
                ),

            "page_type":
                document.get(
                    "page_type",
                    "",
                ),

            "target_id":
                document.get(
                    "target_id",
                    "",
                ),

            "extraction_method":
                document.get(
                    "extraction_method",
                    "",
                ),

            "click_method":
                document.get(
                    "click_method",
                    "",
                ),

            "page_status":
                document.get(
                    "page_status",
                    "",
                ),

            "page_has_control_errors":
                document.get(
                    "page_has_control_errors",
                    False,
                ),

            "alternate_sources":
                document.get(
                    "alternate_sources",
                    [],
                ),

            "text":
                chunk_text_value,

            "text_hash":
                chunk_hash,

            "char_count":
                len(
                    chunk_text_value
                ),
        }

        output.append(
            chunk
        )

    return output


# ============================================================
# EXACT CHUNK DEDUPE
# ============================================================

def deduplicate_chunks(
    chunks: list[dict],
) -> tuple[list[dict], int]:
    """
    Final safety net.

    The corpus builder already deduplicates full documents,
    but chunking two different long documents may still
    produce an identical chunk.
    """

    output = []

    seen_hashes = set()

    duplicates_removed = 0

    for chunk in chunks:

        chunk_hash = chunk[
            "text_hash"
        ]

        if chunk_hash in seen_hashes:

            duplicates_removed += 1

            continue

        seen_hashes.add(
            chunk_hash
        )

        output.append(
            chunk
        )

    return (
        output,
        duplicates_removed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    documents = load_json(
        INPUT_PATH
    )

    raw_chunks = []

    for document in documents:

        document_chunks = (
            build_chunks_for_document(
                document
            )
        )

        raw_chunks.extend(
            document_chunks
        )

    (
        clean_chunks,
        duplicates_removed,
    ) = deduplicate_chunks(
        raw_chunks
    )

    char_counts = [
        chunk[
            "char_count"
        ]
        for chunk in clean_chunks
    ]

    language_counts = {}

    source_kind_counts = {}

    for chunk in clean_chunks:

        language = chunk.get(
            "content_language",
            "unknown",
        )

        language_counts[
            language
        ] = (
            language_counts.get(
                language,
                0,
            )
            + 1
        )

        source_kind = chunk.get(
            "source_kind",
            "unknown",
        )

        source_kind_counts[
            source_kind
        ] = (
            source_kind_counts.get(
                source_kind,
                0,
            )
            + 1
        )

    report = {
        "chunk_version":
            CHUNK_VERSION,

        "input_documents":
            len(
                documents
            ),

        "raw_chunks":
            len(
                raw_chunks
            ),

        "duplicate_chunks_removed":
            duplicates_removed,

        "final_chunks":
            len(
                clean_chunks
            ),

        "max_chunk_chars_setting":
            MAX_CHUNK_CHARS,

        "overlap_chars_setting":
            OVERLAP_CHARS,

        "minimum_chunk_chars":
            (
                min(
                    char_counts
                )
                if char_counts
                else 0
            ),

        "maximum_chunk_chars":
            (
                max(
                    char_counts
                )
                if char_counts
                else 0
            ),

        "average_chunk_chars":
            (
                round(
                    sum(
                        char_counts
                    )
                    / len(
                        char_counts
                    ),
                    2,
                )
                if char_counts
                else 0
            ),

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
    }

    save_json(
        OUTPUT_PATH,
        clean_chunks,
    )

    save_json(
        REPORT_PATH,
        report,
    )

    print(
        "=" * 80
    )

    print(
        "TE DOCUMENT CHUNKING"
    )

    print(
        "=" * 80
    )

    print(
        "Input documents:",
        len(
            documents
        ),
    )

    print(
        "Raw chunks:",
        len(
            raw_chunks
        ),
    )

    print(
        "Duplicate chunks removed:",
        duplicates_removed,
    )

    print(
        "Final chunks:",
        len(
            clean_chunks
        ),
    )

    print()

    if char_counts:

        print(
            "Chunk characters:"
        )

        print(
            "   Minimum:",
            min(
                char_counts
            ),
        )

        print(
            "   Maximum:",
            max(
                char_counts
            ),
        )

        print(
            "   Average:",
            round(
                sum(
                    char_counts
                )
                / len(
                    char_counts
                ),
                2,
            ),
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
        "Saved chunks:",
        OUTPUT_PATH,
    )

    print(
        "Saved report:",
        REPORT_PATH,
    )


if __name__ == "__main__":
    main()